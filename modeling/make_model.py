import torch.nn as nn
import torch.nn.functional as F
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from modeling.backbones.basic_cnn_params.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
from modeling.moe.AttnMOE import GeneralFusion, QuickGELU
from modeling.sdtps import MultiModalSDTPS
from modeling.sacr import SACR
from modeling.multimodal_sacr import MultiModalSACR, MultiModalSACRv2
from modeling.trimodal_lif import TrimodalLIF, TrimodalLIFLoss
from modeling.dual_gated_fusion import DualGatedAdaptiveFusion, DualGatedAdaptiveFusionV2, DualGatedPostFusion, DualGatedAdaptiveFusionV3, DualGatedAdaptiveFusionV4
import torch


# ============================================================================
# DeMoBeiyong: 原始完整版本（备份）
# ============================================================================
class DeMoBeiyong(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(DeMoBeiyong, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.HDM = cfg.MODEL.HDM
        self.ATM = cfg.MODEL.ATM
        self.USE_SACR = cfg.MODEL.USE_SACR
        self.USE_SDTPS = cfg.MODEL.USE_SDTPS
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL
        self.head = cfg.MODEL.HEAD
        self.USE_LIF = cfg.MODEL.USE_LIF
        self.USE_DGAF = cfg.MODEL.USE_DGAF
        self.USE_MULTIMODAL_SACR = cfg.MODEL.USE_MULTIMODAL_SACR

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.rgb_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                        nn.Linear(2 * self.feat_dim, self.feat_dim),QuickGELU())
        self.nir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                        nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())
        self.tir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                        nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())

        # SACR: 在 SDTPS 之前对 patch 特征进行增强
        if self.USE_SACR:
            # 计算 patch grid 尺寸
            h, w = cfg.INPUT.SIZE_TRAIN
            stride_h, stride_w = cfg.MODEL.STRIDE_SIZE
            patch_h = h // stride_h  # 256 / 16 = 16
            patch_w = w // stride_w  # 128 / 16 = 8

            # 三个模态共用一个 SACR 模块（减少参数量）
            self.sacr = SACR(
                token_dim=self.feat_dim,
                height=patch_h,
                width=patch_w,
                dilation_rates=cfg.MODEL.SACR_DILATION_RATES,
            )

        # Trimodal-LIF: Quality-aware multi-modal fusion
        if self.USE_LIF:
            self.lif = TrimodalLIF(beta=cfg.MODEL.LIF_BETA)
            self.lif_loss = TrimodalLIFLoss()
            # 存储温度参数，用于 softmax 加权
            # 注意：原始 M2D-LIF 使用 beta*10 作为温度，这里提供可配置选项
            # 较高温度（如10）会使权重更尖锐（近乎 one-hot）
            # 较低温度（如1-3）会使权重更平滑（软融合）
            self.lif_temperature = cfg.MODEL.LIF_BETA * 10.0  # 默认：0.4 * 10 = 4.0

        # DGAF: Dual-Gated Adaptive Fusion (基于 AGFN 论文)
        # 用于 SDTPS 输出的自适应融合，替代简单的 concat
        if self.USE_DGAF:
            self.DGAF_VERSION = cfg.MODEL.DGAF_VERSION
            if self.DGAF_VERSION == 'v3':
                # V3: 内置 Attention Pooling，直接接受 SDTPS 的 (B, K+1, C) 输出
                self.dgaf = DualGatedAdaptiveFusionV3(
                    feat_dim=self.feat_dim,
                    output_dim=3 * self.feat_dim,
                    tau=cfg.MODEL.DGAF_TAU,
                    init_alpha=cfg.MODEL.DGAF_INIT_ALPHA,
                    num_heads=cfg.MODEL.DGAF_NUM_HEADS,
                )
            else:
                # V1: 需要先 mean pooling，再输入 (B, C)
                self.dgaf = DualGatedPostFusion(
                    feat_dim=self.feat_dim,
                    output_dim=3 * self.feat_dim,
                    tau=cfg.MODEL.DGAF_TAU,
                    init_alpha=cfg.MODEL.DGAF_INIT_ALPHA,
                )

        # MultiModal-SACR: 多模态交互版本的 SACR
        # 新流程: MultiModalSACR (concat→SACR→split) → SDTPS → DGAF
        if self.USE_MULTIMODAL_SACR:
            h, w = cfg.INPUT.SIZE_TRAIN
            stride_h, stride_w = cfg.MODEL.STRIDE_SIZE
            patch_h = h // stride_h  # 256 / 16 = 16
            patch_w = w // stride_w  # 128 / 16 = 8

            if cfg.MODEL.MULTIMODAL_SACR_VERSION == 'v2':
                self.multimodal_sacr = MultiModalSACRv2(
                    token_dim=self.feat_dim,
                    height=patch_h,
                    width=patch_w,
                    dilation_rates=cfg.MODEL.SACR_DILATION_RATES,
                )
            else:
                self.multimodal_sacr = MultiModalSACR(
                    token_dim=self.feat_dim,
                    height=patch_h,
                    width=patch_w,
                    dilation_rates=cfg.MODEL.SACR_DILATION_RATES,
                )

        if self.HDM or self.ATM:
            self.generalFusion = GeneralFusion(feat_dim=self.feat_dim, num_experts=7, head=self.head, reg_weight=0,
                                               cfg=cfg)
            self.classifier_moe = nn.Linear(7 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_moe.apply(weights_init_classifier)
            self.bottleneck_moe = nn.BatchNorm1d(7 * self.feat_dim)
            self.bottleneck_moe.bias.requires_grad_(False)
            self.bottleneck_moe.apply(weights_init_kaiming)

        # SDTPS: 替代 HDM + ATMoE 的 token selection 机制
        if self.USE_SDTPS:
            # 计算 patch 数量
            h, w = cfg.INPUT.SIZE_TRAIN
            stride_h, stride_w = cfg.MODEL.STRIDE_SIZE
            num_patches = (h // stride_h) * (w // stride_w)

            self.sdtps = MultiModalSDTPS(
                embed_dim=self.feat_dim,
                num_patches=num_patches,
                sparse_ratio=cfg.MODEL.SDTPS_SPARSE_RATIO,
                aggr_ratio=cfg.MODEL.SDTPS_AGGR_RATIO,
                use_gumbel=cfg.MODEL.SDTPS_USE_GUMBEL,
                gumbel_tau=cfg.MODEL.SDTPS_GUMBEL_TAU,
                beta=cfg.MODEL.SDTPS_BETA,
                cross_attn_type=cfg.MODEL.SDTPS_CROSS_ATTN_TYPE,
                cross_attn_heads=cfg.MODEL.SDTPS_CROSS_ATTN_HEADS,
            )
            # SDTPS 输出特征维度：每个模态 (K+1) 个 token，K 取决于 sparse_ratio
            # 暂时使用3倍 feat_dim（拼接三个模态的全局特征）
            self.classifier_sdtps = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_sdtps.apply(weights_init_classifier)
            self.bottleneck_sdtps = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_sdtps.bias.requires_grad_(False)
            self.bottleneck_sdtps.apply(weights_init_kaiming)

        # 单独使用 DGAF（不依赖 SDTPS）或 SDTPS+DGAF 组合：都需要 DGAF 分类器
        if self.USE_DGAF:
            self.classifier_dgaf = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_dgaf.apply(weights_init_classifier)
            self.bottleneck_dgaf = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_dgaf.bias.requires_grad_(False)
            self.bottleneck_dgaf.apply(weights_init_kaiming)

        if self.direct:
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        else:
            self.classifier_r = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_r.apply(weights_init_classifier)
            self.bottleneck_r = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_r.bias.requires_grad_(False)
            self.bottleneck_r.apply(weights_init_kaiming)
            self.classifier_n = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_n.apply(weights_init_classifier)
            self.bottleneck_n = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_n.bias.requires_grad_(False)
            self.bottleneck_n.apply(weights_init_kaiming)
            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)
            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
            # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        cam_label = 0
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "The out_proj here is called by the nn.MultiheadAttention, which has been calculated in th .forward(), so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("For the bottleneck or classifier, it is not calculated during inference, so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None):
        # ==========================================================
        # 1. Input Preparation & Missing Modality Simulation (Eval)
        # ==========================================================
        RGB = x['RGB']
        NI = x['NI']
        TI = x['TI']

        # 仅在非训练模式下处理缺失模态模拟
        if not self.training:
            if self.miss_type == 'r': RGB = torch.zeros_like(RGB)
            elif self.miss_type == 'n': NI = torch.zeros_like(NI)
            elif self.miss_type == 't': TI = torch.zeros_like(TI)
            elif self.miss_type == 'rn': RGB = torch.zeros_like(RGB); NI = torch.zeros_like(NI)
            elif self.miss_type == 'rt': RGB = torch.zeros_like(RGB); TI = torch.zeros_like(TI)
            elif self.miss_type == 'nt': NI = torch.zeros_like(NI); TI = torch.zeros_like(TI)

        if 'cam_label' in x:
            cam_label = x['cam_label']

        # ==========================================================
        # 2. Backbone Feature Extraction
        # ==========================================================
        RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
        NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
        TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

        # ==========================================================
        # 3. SACR: Context Enhancement
        # ==========================================================
        if self.USE_MULTIMODAL_SACR:
            RGB_cash, NI_cash, TI_cash = self.multimodal_sacr(RGB_cash, NI_cash, TI_cash)
        elif self.USE_SACR:
            RGB_cash = self.sacr(RGB_cash)
            NI_cash = self.sacr(NI_cash)
            TI_cash = self.sacr(TI_cash)

        # ==========================================================
        # 4. Trimodal-LIF: Quality-aware Weighting
        # ==========================================================
        lif_loss = None
        if self.USE_LIF:
            # 1. Predict Quality & Calculate Loss
            q_rgb, q_nir, q_tir = self.lif.predict_quality(RGB, NI, TI)
            if self.training:
                lif_loss = self.lif_loss(q_rgb, q_nir, q_tir, RGB, NI, TI)['total']

            # 2. Resize to patch grid
            patch_h = self.image_size[0] // self.cfg.MODEL.STRIDE_SIZE[0]
            patch_w = self.image_size[1] // self.cfg.MODEL.STRIDE_SIZE[1]

            q_rgb_patch = F.interpolate(q_rgb, size=(patch_h, patch_w), mode='bilinear')
            q_nir_patch = F.interpolate(q_nir, size=(patch_h, patch_w), mode='bilinear')
            q_tir_patch = F.interpolate(q_tir, size=(patch_h, patch_w), mode='bilinear')

            # 3. Calculate Spatial Weights
            q_logits = torch.cat([q_rgb_patch, q_nir_patch, q_tir_patch], dim=1)
            q_weights_spatial = F.softmax(q_logits * self.lif_temperature, dim=1)

            # 4. Apply Weights (Reshape to match tokens)
            w_rgb = q_weights_spatial[:, 0:1].flatten(2).transpose(1, 2) # (B, 128, 1)
            w_nir = q_weights_spatial[:, 1:2].flatten(2).transpose(1, 2)
            w_tir = q_weights_spatial[:, 2:3].flatten(2).transpose(1, 2)

            RGB_cash = RGB_cash * w_rgb
            NI_cash = NI_cash * w_nir
            TI_cash = TI_cash * w_tir

        # ==========================================================
        # 5. Feature Aggregation (SDTPS / DGAF / Baseline)
        # ==========================================================
        sdtps_feat = None
        sdtps_score = None
        dgaf_feat = None
        dgaf_score = None

        # --- Helper: Fuse Global + Pooled Local Features ---
        def fuse_global_local(feat_cash, feat_global, pool_layer, reduce_layer):
            feat_local = pool_layer(feat_cash.permute(0, 2, 1)).squeeze(-1)
            return reduce_layer(torch.cat([feat_global, feat_local], dim=-1))

        # --- Logic Branching ---

        # A. SDTPS Path
        if self.USE_SDTPS:
            # Token Selection
            RGB_enh, NI_enh, TI_enh, _, _, _ = self.sdtps(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global
            )

            # Feature aggregation: use GLOBAL_LOCAL fusion if enabled, else mean pooling
            if self.GLOBAL_LOCAL:
                # Global-Local Fusion: pool enhanced tokens + concat with backbone global
                RGB_final = fuse_global_local(RGB_enh, RGB_global, self.pool, self.rgb_reduce)
                NI_final  = fuse_global_local(NI_enh, NI_global, self.pool, self.nir_reduce)
                TI_final  = fuse_global_local(TI_enh, TI_global, self.pool, self.tir_reduce)
            else:
                # Simple mean pooling when GLOBAL_LOCAL=False (SDTPS-only baseline)
                RGB_final = RGB_enh.mean(dim=1)  # (B, N, C) -> (B, C)
                NI_final  = NI_enh.mean(dim=1)
                TI_final  = TI_enh.mean(dim=1)

            # ========== 修改：SDTPS+DGAF 组合时，SDTPS 只用于 token selection ==========
            # 总是构建 sdtps_feat（用于推理），但只在 SDTPS-only 时计算分类器
            sdtps_feat = torch.cat([RGB_final, NI_final, TI_final], dim=-1)
            sdtps_score = None
            if self.training and not self.USE_DGAF:
                # SDTPS-only: 计算 SDTPS 分类器
                sdtps_score = self.classifier_sdtps(self.bottleneck_sdtps(sdtps_feat))

        # B. DGAF Path
        if self.USE_DGAF:
            if self.DGAF_VERSION == 'v3':
                # ========== 修改：V3 使用 SDTPS 选择后的 tokens ==========
                # V3 directly uses patch tokens
                if self.USE_SDTPS:
                    # 使用 SDTPS 选择后的 enhanced tokens
                    dgaf_feat = self.dgaf(RGB_enh, NI_enh, TI_enh)
                else:
                    # DGAF-only：使用原始 tokens
                    dgaf_feat = self.dgaf(RGB_cash, NI_cash, TI_cash)
            else:
                # V1 needs aggregated features (B, C)
                if self.USE_SDTPS:
                    # If SDTPS ran, reuse its fused features
                    if self.DGAF_VERSION == 'v3': raise ValueError("SDTPS + DGAF requires V1")
                    if not self.GLOBAL_LOCAL: raise ValueError("SDTPS + DGAF V1 requires GLOBAL_LOCAL")

                    # RGB_final calculated in SDTPS block above
                    dgaf_feat = self.dgaf(RGB_final, NI_final, TI_final)
                else:
                    # Standalone DGAF: Calculate fused features from raw Backbone output
                    if self.GLOBAL_LOCAL:
                        r_in = fuse_global_local(RGB_cash, RGB_global, self.pool, self.rgb_reduce)
                        n_in = fuse_global_local(NI_cash, NI_global, self.pool, self.nir_reduce)
                        t_in = fuse_global_local(TI_cash, TI_global, self.pool, self.tir_reduce)
                    else:
                        r_in, n_in, t_in = RGB_global, NI_global, TI_global

                    dgaf_feat = self.dgaf(r_in, n_in, t_in)

            if self.training:
                dgaf_score = self.classifier_dgaf(self.bottleneck_dgaf(dgaf_feat))

        # C. Baseline / Direct Path
        # Always compute 'ori' for fallback or auxiliary returns
        ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
        ori_score = None

        # Calculate scores if needed (Standard Baseline or Separate Heads)
        if self.training:
            if self.direct:
                ori_global = self.bottleneck(ori)
                ori_score = self.classifier(ori_global)
            else:
                # Individual classifiers per modality
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))


        # ==========================================================
        # 6. Return Logic
        # ==========================================================

        # --- Training Return ---
        if self.training:
            result = ()

            # ========== 修改：所有非 Baseline 模式只返回最后的特征，不包含 ori ==========
            # Priority 1: SDTPS + DGAF (只用 DGAF 特征)
            if self.USE_SDTPS and self.USE_DGAF:
                if self.direct:
                    result = (dgaf_score, dgaf_feat)  # 不包含 ori
                else:
                    result = (dgaf_score, dgaf_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)
            # Priority 2: SDTPS Only (只用 SDTPS 特征)
            elif self.USE_SDTPS:
                if self.direct:
                    result = (sdtps_score, sdtps_feat)  # 不包含 ori
                else:
                    result = (sdtps_score, sdtps_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)
            # Priority 3: DGAF Only (只用 DGAF 特征)
            elif self.USE_DGAF:
                if self.direct:
                    result = (dgaf_score, dgaf_feat)  # 不包含 ori
                else:
                    result = (dgaf_score, dgaf_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)
            # Priority 4: Baseline (只用 ori)
            else:
                if self.direct:
                    result = (ori_score, ori)
                else:
                    result = (RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)

            # Append LIF loss if it exists
            if self.USE_LIF and lif_loss is not None:
                result = result + (lif_loss,)

            return result

        # --- Inference Return ---
        else:
            # ========== 修改：所有非 Baseline 模式只返回最后的特征，不包含 ori ==========
            if self.USE_SDTPS and self.USE_DGAF:
                # SDTPS+DGAF: 只返回 DGAF 特征
                return dgaf_feat
            elif self.USE_SDTPS and not self.USE_DGAF:
                # SDTPS-only: 只返回 SDTPS 特征
                return sdtps_feat
            elif not self.USE_SDTPS and self.USE_DGAF:
                # DGAF-only: 只返回 DGAF 特征
                return dgaf_feat
            else:
                # Baseline: 返回 ori
                return ori


# ============================================================================
# DeMo: 简化版本（去除 SACR/LIF/HDM/ATM，保留 SDTPS/DGAF/Baseline）
# ============================================================================
class DeMo(nn.Module):
    """
    DeMo 模型 - 简化版

    核心模块：
    - Backbone: ViT-based multi-modal feature extraction
    - SDTPS: Token selection with cross-modal attention
    - DGAF: Dual-gated adaptive fusion
    - Baseline: Direct concat of global features

    已移除：SACR, LIF, HDM, ATM（简化代码，保留核心功能）
    """

    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(DeMo, self).__init__()

        # Feature dimension
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512

        # Backbone
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)

        # Basic config
        self.num_classes = num_classes
        self.cfg = cfg
        self.direct = cfg.MODEL.DIRECT
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS

        # Module flags
        self.USE_SDTPS = cfg.MODEL.USE_SDTPS
        self.USE_DGAF = cfg.MODEL.USE_DGAF
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL

        # Global-Local Fusion layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.rgb_reduce = nn.Sequential(
            nn.LayerNorm(2 * self.feat_dim),
            nn.Linear(2 * self.feat_dim, self.feat_dim),
            QuickGELU()
        )
        self.nir_reduce = nn.Sequential(
            nn.LayerNorm(2 * self.feat_dim),
            nn.Linear(2 * self.feat_dim, self.feat_dim),
            QuickGELU()
        )
        self.tir_reduce = nn.Sequential(
            nn.LayerNorm(2 * self.feat_dim),
            nn.Linear(2 * self.feat_dim, self.feat_dim),
            QuickGELU()
        )

        # SDTPS: Token selection module
        if self.USE_SDTPS:
            h, w = cfg.INPUT.SIZE_TRAIN
            stride_h, stride_w = cfg.MODEL.STRIDE_SIZE
            num_patches = (h // stride_h) * (w // stride_w)

            self.sdtps = MultiModalSDTPS(
                embed_dim=self.feat_dim,
                num_patches=num_patches,
                sparse_ratio=cfg.MODEL.SDTPS_SPARSE_RATIO,
                aggr_ratio=cfg.MODEL.SDTPS_AGGR_RATIO,
                use_gumbel=cfg.MODEL.SDTPS_USE_GUMBEL,
                gumbel_tau=cfg.MODEL.SDTPS_GUMBEL_TAU,
                beta=cfg.MODEL.SDTPS_BETA,
                cross_attn_type=cfg.MODEL.SDTPS_CROSS_ATTN_TYPE,
                cross_attn_heads=cfg.MODEL.SDTPS_CROSS_ATTN_HEADS,
            )

            self.classifier_sdtps = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_sdtps.apply(weights_init_classifier)
            self.bottleneck_sdtps = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_sdtps.bias.requires_grad_(False)
            self.bottleneck_sdtps.apply(weights_init_kaiming)

        # DGAF: Adaptive fusion module
        if self.USE_DGAF:
            self.DGAF_VERSION = cfg.MODEL.DGAF_VERSION

            if self.DGAF_VERSION == 'v3':
                # V3: Processes patch tokens directly
                self.dgaf = DualGatedAdaptiveFusionV3(
                    feat_dim=self.feat_dim,
                    output_dim=3 * self.feat_dim,
                    tau=cfg.MODEL.DGAF_TAU,
                    init_alpha=cfg.MODEL.DGAF_INIT_ALPHA,
                    num_heads=cfg.MODEL.DGAF_NUM_HEADS,
                )
            else:
                # V1: Processes aggregated features
                self.dgaf = DualGatedPostFusion(
                    feat_dim=self.feat_dim,
                    output_dim=3 * self.feat_dim,
                    tau=cfg.MODEL.DGAF_TAU,
                    init_alpha=cfg.MODEL.DGAF_INIT_ALPHA,
                )

            self.classifier_dgaf = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_dgaf.apply(weights_init_classifier)
            self.bottleneck_dgaf = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_dgaf.bias.requires_grad_(False)
            self.bottleneck_dgaf.apply(weights_init_kaiming)

        # Baseline classifiers
        if self.direct:
            # Direct concat classifier
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        else:
            # Per-modality classifiers
            self.classifier_r = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_r.apply(weights_init_classifier)
            self.bottleneck_r = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_r.bias.requires_grad_(False)
            self.bottleneck_r.apply(weights_init_kaiming)

            self.classifier_n = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_n.apply(weights_init_classifier)
            self.bottleneck_n = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_n.bias.requires_grad_(False)
            self.bottleneck_n.apply(weights_init_kaiming)

            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)
            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])

        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()

        input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        cam_label = 0
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label}

        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~" * 88)
        print("Note: out_proj is calculated in MultiheadAttention.forward(), ignore it")
        print("Note: bottleneck/classifier not used in inference, ignore it")
        print("~" * 88)

        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None):
        """
        Forward pass - 完全分离的4个独立分支

        Args:
            x: dict with keys 'RGB', 'NI', 'TI'
            label: class labels (for training)
            cam_label: camera labels
            view_label: view labels
            return_pattern: for evaluation (not used in simplified version)
            img_path: image paths (for evaluation)

        Returns:
            Training: (score, feat) or (score, feat, ...) depending on config
            Inference: feat tensor
        """

        # Extract inputs and handle camera label
        RGB, NI, TI = x['RGB'], x['NI'], x['TI']
        if 'cam_label' in x:
            cam_label = x['cam_label']

        # Missing modality simulation (inference only)
        if not self.training:
            if self.miss_type == 'r': RGB = torch.zeros_like(RGB)
            elif self.miss_type == 'n': NI = torch.zeros_like(NI)
            elif self.miss_type == 't': TI = torch.zeros_like(TI)
            elif self.miss_type == 'rn': RGB, NI = torch.zeros_like(RGB), torch.zeros_like(NI)
            elif self.miss_type == 'rt': RGB, TI = torch.zeros_like(RGB), torch.zeros_like(TI)
            elif self.miss_type == 'nt': NI, TI = torch.zeros_like(NI), torch.zeros_like(TI)

        # Backbone feature extraction
        RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
        NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
        TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

        # ========================================================================
        # 分支1: Baseline (无 SDTPS, 无 DGAF)
        # ========================================================================
        if not self.USE_SDTPS and not self.USE_DGAF:
            # 直接拼接全局特征
            ori_feat = torch.cat([RGB_global, NI_global, TI_global], dim=-1)

            if self.training:
                if self.direct:
                    ori_score = self.classifier(self.bottleneck(ori_feat))
                    return (ori_score, ori_feat)
                else:
                    RGB_score = self.classifier_r(self.bottleneck_r(RGB_global))
                    NI_score = self.classifier_n(self.bottleneck_n(NI_global))
                    TI_score = self.classifier_t(self.bottleneck_t(TI_global))
                    return (RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global)
            else:
                return ori_feat

        # ========================================================================
        # 分支2: SDTPS Only (有 SDTPS, 无 DGAF)
        # ========================================================================
        elif self.USE_SDTPS and not self.USE_DGAF:
            # Helper function
            def fuse_global_local(feat_cash, feat_global, pool_layer, reduce_layer):
                feat_local = pool_layer(feat_cash.permute(0, 2, 1)).squeeze(-1)
                return reduce_layer(torch.cat([feat_global, feat_local], dim=-1))

            # SDTPS: Token selection
            RGB_enh, NI_enh, TI_enh, _, _, _ = self.sdtps(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global
            )

            # Feature aggregation
            if self.GLOBAL_LOCAL:
                RGB_final = fuse_global_local(RGB_enh, RGB_global, self.pool, self.rgb_reduce)
                NI_final = fuse_global_local(NI_enh, NI_global, self.pool, self.nir_reduce)
                TI_final = fuse_global_local(TI_enh, TI_global, self.pool, self.tir_reduce)
            else:
                RGB_final = RGB_enh.mean(dim=1)
                NI_final = NI_enh.mean(dim=1)
                TI_final = TI_enh.mean(dim=1)

            sdtps_feat = torch.cat([RGB_final, NI_final, TI_final], dim=-1)

            if self.training:
                sdtps_score = self.classifier_sdtps(self.bottleneck_sdtps(sdtps_feat))
                if self.direct:
                    return (sdtps_score, sdtps_feat)
                else:
                    RGB_score = self.classifier_r(self.bottleneck_r(RGB_global))
                    NI_score = self.classifier_n(self.bottleneck_n(NI_global))
                    TI_score = self.classifier_t(self.bottleneck_t(TI_global))
                    return (sdtps_score, sdtps_feat, RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global)
            else:
                return sdtps_feat

        # ========================================================================
        # 分支3: DGAF Only (无 SDTPS, 有 DGAF)
        # ========================================================================
        elif not self.USE_SDTPS and self.USE_DGAF:
            # Helper function
            def fuse_global_local(feat_cash, feat_global, pool_layer, reduce_layer):
                feat_local = pool_layer(feat_cash.permute(0, 2, 1)).squeeze(-1)
                return reduce_layer(torch.cat([feat_global, feat_local], dim=-1))

            # DGAF: Adaptive fusion
            if self.DGAF_VERSION == 'v3':
                # V3: process patch tokens
                dgaf_feat = self.dgaf(RGB_cash, NI_cash, TI_cash)
            else:
                # V1: process aggregated features
                if self.GLOBAL_LOCAL:
                    RGB_in = fuse_global_local(RGB_cash, RGB_global, self.pool, self.rgb_reduce)
                    NI_in = fuse_global_local(NI_cash, NI_global, self.pool, self.nir_reduce)
                    TI_in = fuse_global_local(TI_cash, TI_global, self.pool, self.tir_reduce)
                else:
                    RGB_in, NI_in, TI_in = RGB_global, NI_global, TI_global
                dgaf_feat = self.dgaf(RGB_in, NI_in, TI_in)

            if self.training:
                dgaf_score = self.classifier_dgaf(self.bottleneck_dgaf(dgaf_feat))
                if self.direct:
                    return (dgaf_score, dgaf_feat)
                else:
                    RGB_score = self.classifier_r(self.bottleneck_r(RGB_global))
                    NI_score = self.classifier_n(self.bottleneck_n(NI_global))
                    TI_score = self.classifier_t(self.bottleneck_t(TI_global))
                    return (dgaf_score, dgaf_feat, RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global)
            else:
                return dgaf_feat

        # ========================================================================
        # 分支4: SDTPS + DGAF (有 SDTPS, 有 DGAF)
        # ========================================================================
        # 优化：优先使用 V3（避免双重信息损失）
        # - SDTPS masking + pool + reduce = 双重压缩（V1的问题）
        # - V3 直接处理 tokens，保留局部细节
        # ========================================================================
        else:  # self.USE_SDTPS and self.USE_DGAF
            # SDTPS: Token selection
            RGB_enh, NI_enh, TI_enh, _, _, _ = self.sdtps(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global
            )

            # DGAF: Adaptive fusion
            if self.DGAF_VERSION == 'v3':
                # ✅ 推荐：V3 直接处理 SDTPS-selected tokens（保留局部细节）
                dgaf_feat = self.dgaf(RGB_enh, NI_enh, TI_enh)

            else:
                # ⚠️ V1: 需要 GLOBAL_LOCAL（双重压缩，效果较差）
                if not self.GLOBAL_LOCAL:
                    raise ValueError("SDTPS + DGAF V1 requires GLOBAL_LOCAL=True")

                def fuse_global_local(feat_cash, feat_global, pool_layer, reduce_layer):
                    feat_local = pool_layer(feat_cash.permute(0, 2, 1)).squeeze(-1)
                    return reduce_layer(torch.cat([feat_global, feat_local], dim=-1))

                RGB_final = fuse_global_local(RGB_enh, RGB_global, self.pool, self.rgb_reduce)
                NI_final = fuse_global_local(NI_enh, NI_global, self.pool, self.nir_reduce)
                TI_final = fuse_global_local(TI_enh, TI_global, self.pool, self.tir_reduce)
                dgaf_feat = self.dgaf(RGB_final, NI_final, TI_final)

            if self.training:
                dgaf_score = self.classifier_dgaf(self.bottleneck_dgaf(dgaf_feat))
                if self.direct:
                    return (dgaf_score, dgaf_feat)
                else:
                    RGB_score = self.classifier_r(self.bottleneck_r(RGB_global))
                    NI_score = self.classifier_n(self.bottleneck_n(NI_global))
                    TI_score = self.classifier_t(self.bottleneck_t(TI_global))
                    return (dgaf_score, dgaf_feat, RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global)
            else:
                return dgaf_feat
__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


# ============================================================================
# DeMo_Parallel: 并行架构，9个分类头
# ============================================================================
class DeMo_Parallel(nn.Module):
    """
    DeMo 并行架构 - 9个独立分类头

    架构：
        Backbone
          ├─→ SDTPS  → RGB_enh.mean(), NI_enh.mean(), TI_enh.mean() (3个特征)
          ├─→ DGAF V4 → RGB_dgaf, NI_dgaf, TI_dgaf (3个特征，输入 global)
          └─→ Fused   → fuse_global_local(cash, global) × 3 (3个特征)

        总计: 9个特征 → 9个分类头

    简化说明：
    - Fused 固定使用 fuse_global_local（无 GLOBAL_LOCAL 可选项）
    - SDTPS 输出直接 mean pooling（不再用 fuse_global_local）
    - DGAF 输入直接用 global features（不用 fused）
    """

    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(DeMo_Parallel, self).__init__()

        # Feature dimension
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512

        # Backbone
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)

        # Basic config
        self.num_classes = num_classes
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS

        # ========== Global-Local Fusion layers (固定使用) ==========
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.rgb_reduce = nn.Sequential(
            nn.LayerNorm(2 * self.feat_dim),
            nn.Linear(2 * self.feat_dim, self.feat_dim),
            QuickGELU()
        )
        self.nir_reduce = nn.Sequential(
            nn.LayerNorm(2 * self.feat_dim),
            nn.Linear(2 * self.feat_dim, self.feat_dim),
            QuickGELU()
        )
        self.tir_reduce = nn.Sequential(
            nn.LayerNorm(2 * self.feat_dim),
            nn.Linear(2 * self.feat_dim, self.feat_dim),
            QuickGELU()
        )

        # ========== SDTPS Module ==========
        h, w = cfg.INPUT.SIZE_TRAIN
        stride_h, stride_w = cfg.MODEL.STRIDE_SIZE
        num_patches = (h // stride_h) * (w // stride_w)

        self.sdtps = MultiModalSDTPS(
            embed_dim=self.feat_dim,
            num_patches=num_patches,
            sparse_ratio=cfg.MODEL.SDTPS_SPARSE_RATIO,
            aggr_ratio=cfg.MODEL.SDTPS_AGGR_RATIO,
            use_gumbel=cfg.MODEL.SDTPS_USE_GUMBEL,
            gumbel_tau=cfg.MODEL.SDTPS_GUMBEL_TAU,
            beta=cfg.MODEL.SDTPS_BETA,
            cross_attn_type=cfg.MODEL.SDTPS_CROSS_ATTN_TYPE,
            cross_attn_heads=cfg.MODEL.SDTPS_CROSS_ATTN_HEADS,
        )

        # ========== DGAF V3 Module ==========
        # V3 接受 tokens (B, N, C)，输出 cat 后的特征 (B, 3C)
        self.dgaf = DualGatedAdaptiveFusionV3(
            feat_dim=self.feat_dim,
            output_dim=3 * self.feat_dim,
            tau=cfg.MODEL.DGAF_TAU,
            init_alpha=cfg.MODEL.DGAF_INIT_ALPHA,
            num_heads=cfg.MODEL.DGAF_NUM_HEADS,
        )

        # ========== 9个分类头 + BatchNorm ==========
        # SDTPS 分支（3个）
        self.classifier_sdtps_rgb = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_sdtps_rgb.apply(weights_init_classifier)
        self.bottleneck_sdtps_rgb = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_sdtps_rgb.bias.requires_grad_(False)
        self.bottleneck_sdtps_rgb.apply(weights_init_kaiming)

        self.classifier_sdtps_nir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_sdtps_nir.apply(weights_init_classifier)
        self.bottleneck_sdtps_nir = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_sdtps_nir.bias.requires_grad_(False)
        self.bottleneck_sdtps_nir.apply(weights_init_kaiming)

        self.classifier_sdtps_tir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_sdtps_tir.apply(weights_init_classifier)
        self.bottleneck_sdtps_tir = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_sdtps_tir.bias.requires_grad_(False)
        self.bottleneck_sdtps_tir.apply(weights_init_kaiming)

        # DGAF 分支（3个）
        self.classifier_dgaf_rgb = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_dgaf_rgb.apply(weights_init_classifier)
        self.bottleneck_dgaf_rgb = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_dgaf_rgb.bias.requires_grad_(False)
        self.bottleneck_dgaf_rgb.apply(weights_init_kaiming)

        self.classifier_dgaf_nir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_dgaf_nir.apply(weights_init_classifier)
        self.bottleneck_dgaf_nir = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_dgaf_nir.bias.requires_grad_(False)
        self.bottleneck_dgaf_nir.apply(weights_init_kaiming)

        self.classifier_dgaf_tir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_dgaf_tir.apply(weights_init_classifier)
        self.bottleneck_dgaf_tir = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_dgaf_tir.bias.requires_grad_(False)
        self.bottleneck_dgaf_tir.apply(weights_init_kaiming)

        # Fused 分支（3个）
        self.classifier_fused_rgb = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_fused_rgb.apply(weights_init_classifier)
        self.bottleneck_fused_rgb = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_fused_rgb.bias.requires_grad_(False)
        self.bottleneck_fused_rgb.apply(weights_init_kaiming)

        self.classifier_fused_nir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_fused_nir.apply(weights_init_classifier)
        self.bottleneck_fused_nir = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_fused_nir.bias.requires_grad_(False)
        self.bottleneck_fused_nir.apply(weights_init_kaiming)

        self.classifier_fused_tir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_fused_tir.apply(weights_init_classifier)
        self.bottleneck_fused_tir = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_fused_tir.bias.requires_grad_(False)
        self.bottleneck_fused_tir.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None):
        """
        并行架构 Forward - 9个独立分类头

        Returns:
            Training: 18个值 (9对 score-feat)
            Inference: (B, 9C) 拼接特征
        """

        # Extract inputs
        RGB, NI, TI = x['RGB'], x['NI'], x['TI']
        if 'cam_label' in x:
            cam_label = x['cam_label']

        # Missing modality simulation (inference only)
        if not self.training:
            if self.miss_type == 'r': RGB = torch.zeros_like(RGB)
            elif self.miss_type == 'n': NI = torch.zeros_like(NI)
            elif self.miss_type == 't': TI = torch.zeros_like(TI)
            elif self.miss_type == 'rn': RGB, NI = torch.zeros_like(RGB), torch.zeros_like(NI)
            elif self.miss_type == 'rt': RGB, TI = torch.zeros_like(RGB), torch.zeros_like(TI)
            elif self.miss_type == 'nt': NI, TI = torch.zeros_like(NI), torch.zeros_like(TI)

        # Backbone
        RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
        NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
        TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

        # ========== 分支1: SDTPS (并行) ==========
        RGB_enh, NI_enh, TI_enh, _, _, _ = self.sdtps(
            RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global
        )
        # 直接 mean pooling
        feat_sdtps_rgb = RGB_enh.mean(dim=1)  # (B, N, C) -> (B, C)
        feat_sdtps_nir = NI_enh.mean(dim=1)
        feat_sdtps_tir = TI_enh.mean(dim=1)

        # ========== 分支2: DGAF V3 (并行) ==========
        # 输入: tokens (B, N, C) - 保留局部信息
        # V3 输出: cat([RGB_out, NI_out, TI_out], dim=-1) -> (B, 3C)
        dgaf_output = self.dgaf(RGB_cash, NI_cash, TI_cash)  # (B, 3C)

        # 按最后一个维度拆分成3个独立特征
        feat_dgaf_rgb = dgaf_output[:, :self.feat_dim]                    # (B, C)
        feat_dgaf_nir = dgaf_output[:, self.feat_dim:2*self.feat_dim]     # (B, C)
        feat_dgaf_tir = dgaf_output[:, 2*self.feat_dim:]                  # (B, C)

        # ========== 分支3: Fused (并行) ==========
        # 固定使用 fuse_global_local
        def fuse_global_local(feat_cash, feat_global, pool_layer, reduce_layer):
            feat_local = pool_layer(feat_cash.permute(0, 2, 1)).squeeze(-1)
            return reduce_layer(torch.cat([feat_global, feat_local], dim=-1))

        feat_fused_rgb = fuse_global_local(RGB_cash, RGB_global, self.pool, self.rgb_reduce)
        feat_fused_nir = fuse_global_local(NI_cash, NI_global, self.pool, self.nir_reduce)
        feat_fused_tir = fuse_global_local(TI_cash, TI_global, self.pool, self.tir_reduce)

        # ========== 训练：9个分类头 ==========
        if self.training:
            # SDTPS 分支
            score_sdtps_rgb = self.classifier_sdtps_rgb(self.bottleneck_sdtps_rgb(feat_sdtps_rgb))
            score_sdtps_nir = self.classifier_sdtps_nir(self.bottleneck_sdtps_nir(feat_sdtps_nir))
            score_sdtps_tir = self.classifier_sdtps_tir(self.bottleneck_sdtps_tir(feat_sdtps_tir))

            # DGAF 分支
            score_dgaf_rgb = self.classifier_dgaf_rgb(self.bottleneck_dgaf_rgb(feat_dgaf_rgb))
            score_dgaf_nir = self.classifier_dgaf_nir(self.bottleneck_dgaf_nir(feat_dgaf_nir))
            score_dgaf_tir = self.classifier_dgaf_tir(self.bottleneck_dgaf_tir(feat_dgaf_tir))

            # Fused 分支
            score_fused_rgb = self.classifier_fused_rgb(self.bottleneck_fused_rgb(feat_fused_rgb))
            score_fused_nir = self.classifier_fused_nir(self.bottleneck_fused_nir(feat_fused_nir))
            score_fused_tir = self.classifier_fused_tir(self.bottleneck_fused_tir(feat_fused_tir))

            # 返回18个值（9对 score-feat）
            return (
                score_sdtps_rgb, feat_sdtps_rgb,
                score_sdtps_nir, feat_sdtps_nir,
                score_sdtps_tir, feat_sdtps_tir,
                score_dgaf_rgb, feat_dgaf_rgb,
                score_dgaf_nir, feat_dgaf_nir,
                score_dgaf_tir, feat_dgaf_tir,
                score_fused_rgb, feat_fused_rgb,
                score_fused_nir, feat_fused_nir,
                score_fused_tir, feat_fused_tir,
            )
        else:
            # 推理：拼接所有9个特征
            return torch.cat([
                feat_sdtps_rgb, feat_sdtps_nir, feat_sdtps_tir,
                feat_dgaf_rgb, feat_dgaf_nir, feat_dgaf_tir,
                feat_fused_rgb, feat_fused_nir, feat_fused_tir,
            ], dim=-1)  # (B, 9C)


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    # 架构选择
    model_arch = cfg.MODEL.get('ARCH', 'DeMo') if hasattr(cfg.MODEL, 'ARCH') else 'DeMo'

    if model_arch == 'DeMo_Parallel':
        model = DeMo_Parallel(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building DeMo_Parallel (9 heads)===========')
    elif model_arch == 'DeMoBeiyong':
        model = DeMoBeiyong(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building DeMoBeiyong (legacy)===========')
    else:
        model = DeMo(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building DeMo===========')

    return model
