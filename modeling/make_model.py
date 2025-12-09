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
from modeling.dual_gated_fusion import DualGatedAdaptiveFusion, DualGatedAdaptiveFusionV2, DualGatedPostFusion, DualGatedAdaptiveFusionV3
import torch


class DeMo(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(DeMo, self).__init__()
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
        if self.GLOBAL_LOCAL:
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
            # Global-Local Fusion for SDTPS features
            RGB_final = fuse_global_local(RGB_enh, RGB_global, self.pool, self.rgb_reduce)
            NI_final  = fuse_global_local(NI_enh, NI_global, self.pool, self.nir_reduce)
            TI_final  = fuse_global_local(TI_enh, TI_global, self.pool, self.tir_reduce)
            
            sdtps_feat = torch.cat([RGB_final, NI_final, TI_final], dim=-1)
            
            if self.training:
                sdtps_score = self.classifier_sdtps(self.bottleneck_sdtps(sdtps_feat))

        # B. DGAF Path
        if self.USE_DGAF:
            if self.DGAF_VERSION == 'v3':
                # V3 directly uses patch tokens
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
            
            # Priority 1: SDTPS + DGAF
            if self.USE_SDTPS and self.USE_DGAF:
                result = (sdtps_score, sdtps_feat, dgaf_score, dgaf_feat)
            # Priority 2: SDTPS Only
            elif self.USE_SDTPS:
                if self.direct:
                    result = (sdtps_score, sdtps_feat)
                else:
                    result = (sdtps_score, sdtps_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)
            # Priority 3: DGAF Only
            elif self.USE_DGAF:
                if self.direct:
                    result = (dgaf_score, dgaf_feat，ori_score, ori)
                else:
                    result = (dgaf_score, dgaf_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)
            # Priority 4: Baseline
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
            # Flexible return based on configuration
            if self.USE_SDTPS and not self.USE_DGAF:
                return sdtps_feat
            elif not self.USE_SDTPS and self.USE_DGAF:
                return torch.cat([ori, dgaf_feat], dim=-1)
            elif self.USE_SDTPS and self.USE_DGAF:
               return torch.cat([sdtps_feat, dgaf_feat], dim=-1)
            else:
                return ori



__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = DeMo(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building DeMo===========')
    return model
