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
from modeling.sdtps_complete import MultiModalSDTPS
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

        # 单独使用 DGAF（不依赖 SDTPS）：DGAF V3 直接处理 backbone 的 patch tokens
        if self.USE_DGAF and not self.USE_SDTPS:
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
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

            # SACR: 对 patch 特征进行多尺度上下文增强
            if self.USE_MULTIMODAL_SACR:
                # MultiModal-SACR: 三模态拼接 → SACR → 拆分（跨模态交互）
                RGB_cash, NI_cash, TI_cash = self.multimodal_sacr(RGB_cash, NI_cash, TI_cash)
            elif self.USE_SACR:
                # 单模态 SACR：三个模态独立处理
                RGB_cash = self.sacr(RGB_cash)  # (B, N, C) → (B, N, C)
                NI_cash = self.sacr(NI_cash)    # (B, N, C) → (B, N, C)
                TI_cash = self.sacr(TI_cash)    # (B, N, C) → (B, N, C)

            # Trimodal-LIF: Quality-aware feature enhancement
            # LIF 计算质量感知权重，对 patch 特征进行逐位置加权
            lif_loss = None
            if self.USE_LIF:
                # 1. 预测质量图（从原始图像）
                q_rgb, q_nir, q_tir = self.lif.predict_quality(RGB, NI, TI)
                # q_rgb: (B, 1, 32, 16) - QualityPredictor 经过 3 次 AvgPool2d(2,2): 256×128 → 32×16

                # 2. 计算 LIF 损失（自监督）
                lif_loss = self.lif_loss(q_rgb, q_nir, q_tir, RGB, NI, TI)['total']

                # 3. 将质量图 resize 到 patch grid 尺寸（16×8）
                patch_h = self.image_size[0] // self.cfg.MODEL.STRIDE_SIZE[0]  # 256/16 = 16
                patch_w = self.image_size[1] // self.cfg.MODEL.STRIDE_SIZE[1]  # 128/16 = 8

                q_rgb_patch = F.interpolate(q_rgb, size=(patch_h, patch_w), mode='bilinear')  # (B, 1, 16, 8)
                q_nir_patch = F.interpolate(q_nir, size=(patch_h, patch_w), mode='bilinear')  # (B, 1, 16, 8)
                q_tir_patch = F.interpolate(q_tir, size=(patch_h, patch_w), mode='bilinear')  # (B, 1, 16, 8)

                # 4. 计算逐位置的模态权重（每个 patch 位置有独立权重）
                q_logits = torch.cat([q_rgb_patch, q_nir_patch, q_tir_patch], dim=1)  # (B, 3, 16, 8)
                # 使用配置的温度参数，而非硬编码
                # LIF_BETA=0.4 时温度=4.0，比硬编码的10.0更平滑
                q_weights_spatial = F.softmax(q_logits * self.lif_temperature, dim=1)  # (B, 3, 16, 8)

                # 5. Reshape 为 token 维度：(B, 1, 16, 8) → (B, 128, 1)
                w_rgb_token = q_weights_spatial[:, 0:1].flatten(2).transpose(1, 2)  # (B, 128, 1)
                w_nir_token = q_weights_spatial[:, 1:2].flatten(2).transpose(1, 2)  # (B, 128, 1)
                w_tir_token = q_weights_spatial[:, 2:3].flatten(2).transpose(1, 2)  # (B, 128, 1)

                # 6. 加权 patch 特征（逐 patch 加权！）
                RGB_cash = RGB_cash * w_rgb_token  # (B, 128, 512) * (B, 128, 1)
                NI_cash = NI_cash * w_nir_token    # 每个 patch 根据其位置的质量被加权
                TI_cash = TI_cash * w_tir_token

                # 效果：图像左边亮（RGB patch权重大），右边暗（NIR/TIR patch权重大）
                # 这是真正的局部质量感知增强！

            # SDTPS 分支：使用 token selection 替代 HDM+ATM
            if self.USE_SDTPS:
                # SDTPS token selection and enhancement
                RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask = self.sdtps(
                    RGB_cash, NI_cash, TI_cash,
                    RGB_global, NI_global, TI_global
                )

                # DGAF: 使用双门控自适应融合替代简单 concat
                if self.USE_DGAF:
                    if self.DGAF_VERSION == 'v3':
                        # V3: 直接输入 tokens，内置 attention pooling
                        sdtps_feat = self.dgaf(RGB_enhanced, NI_enhanced, TI_enhanced)  # (B, 3C)
                    else:
                        # V1: 需要先将 tokens 聚合成 (B, C) 特征
                        if self.GLOBAL_LOCAL:
                            # GLOBAL_LOCAL 模式：pool(enhanced) + backbone_global 融合降维
                            # RGB_enhanced: (B, K+1, C) → permute → (B, C, K+1) → pool → (B, C, 1) → squeeze → (B, C)
                            RGB_local = self.pool(RGB_enhanced.permute(0, 2, 1)).squeeze(-1)  # (B, C)
                            NI_local = self.pool(NI_enhanced.permute(0, 2, 1)).squeeze(-1)    # (B, C)
                            TI_local = self.pool(TI_enhanced.permute(0, 2, 1)).squeeze(-1)    # (B, C)

                            # 融合 backbone 的 global 特征和 enhanced 的 pooled local 特征
                            RGB_sdtps = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))  # (B, C)
                            NI_sdtps = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))     # (B, C)
                            TI_sdtps = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))     # (B, C)
                        else:
                            # 默认方式：mean pooling
                            RGB_sdtps = RGB_enhanced.mean(dim=1)  # (B, K+1, C) → (B, C)
                            NI_sdtps = NI_enhanced.mean(dim=1)
                            TI_sdtps = TI_enhanced.mean(dim=1)

                        sdtps_feat = self.dgaf(RGB_sdtps, NI_sdtps, TI_sdtps)  # (B, 3C)
                else:
                    # 不使用 DGAF：简单拼接（也需要聚合特征）
                    if self.GLOBAL_LOCAL:
                        # GLOBAL_LOCAL 模式：pool(enhanced) + backbone_global 融合降维
                        RGB_local = self.pool(RGB_enhanced.permute(0, 2, 1)).squeeze(-1)  # (B, C)
                        NI_local = self.pool(NI_enhanced.permute(0, 2, 1)).squeeze(-1)    # (B, C)
                        TI_local = self.pool(TI_enhanced.permute(0, 2, 1)).squeeze(-1)    # (B, C)

                        RGB_sdtps = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))  # (B, C)
                        NI_sdtps = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))     # (B, C)
                        TI_sdtps = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))     # (B, C)
                    else:
                        # 默认方式：mean pooling
                        RGB_sdtps = RGB_enhanced.mean(dim=1)  # (B, K+1, C) → (B, C)
                        NI_sdtps = NI_enhanced.mean(dim=1)
                        TI_sdtps = TI_enhanced.mean(dim=1)

                    sdtps_feat = torch.cat([RGB_sdtps, NI_sdtps, TI_sdtps], dim=-1)  # (B, 3C)

                sdtps_score = self.classifier_sdtps(self.bottleneck_sdtps(sdtps_feat))

            # 单独使用 DGAF V3（不依赖 SDTPS）：直接处理 backbone 的 patch tokens
            elif self.USE_DGAF:
                # DGAF V3 直接接受 patch tokens (B, N, C)
                dgaf_feat = self.dgaf(RGB_cash, NI_cash, TI_cash)  # (B, 3C)
                dgaf_score = self.classifier_dgaf(self.bottleneck_dgaf(dgaf_feat))

            if self.HDM or self.ATM:
                moe_feat, loss_moe = self.generalFusion(RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
                moe_score = self.classifier_moe(self.bottleneck_moe(moe_feat))
            if self.direct:
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_score = self.classifier(ori_global)
            else:
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))

            # 构造返回值，可能包含 LIF 损失
            if self.direct:
                if self.USE_SDTPS:
                    # SDTPS 分支：返回 SDTPS 特征和原始特征
                    if self.HDM or self.ATM:
                        result = (sdtps_score, sdtps_feat, ori_score, ori, loss_moe)
                        if self.USE_LIF and lif_loss is not None:
                            result = result + (lif_loss,)
                    else:
                        result = (sdtps_score, sdtps_feat, ori_score, ori)
                        if self.USE_LIF and lif_loss is not None:
                            result = result + (lif_loss,)
                    return result
                elif self.USE_DGAF:
                    # 单独 DGAF 分支：返回 DGAF 特征和原始特征（与 SDTPS 格式一致）
                    result = (dgaf_score, dgaf_feat, ori_score, ori)
                    if self.USE_LIF and lif_loss is not None:
                        result = result + (lif_loss,)
                    return result
                elif self.HDM or self.ATM:
                    result = (moe_score, moe_feat, ori_score, ori, loss_moe)
                    if self.USE_LIF and lif_loss is not None:
                        result = result + (lif_loss,)
                    return result
                else:
                    if self.USE_LIF and lif_loss is not None:
                        return (ori_score, ori, lif_loss)
                    return (ori_score, ori)
            else:
                if self.USE_SDTPS:
                    # SDTPS 分支（非direct模式）
                    result = (sdtps_score, sdtps_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)
                    if self.USE_LIF and lif_loss is not None:
                        result = result + (lif_loss,)
                    return result
                elif self.USE_DGAF:
                    # 单独 DGAF 分支（非direct模式）
                    result = (dgaf_score, dgaf_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)
                    if self.USE_LIF and lif_loss is not None:
                        result = result + (lif_loss,)
                    return result
                elif self.HDM or self.ATM:
                    result = (moe_score, moe_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, loss_moe)
                    if self.USE_LIF and lif_loss is not None:
                        result = result + (lif_loss,)
                    return result
                else:
                    result = (RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)
                    if self.USE_LIF and lif_loss is not None:
                        result = result + (lif_loss,)
                    return result

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            if self.miss_type == 'r':
                RGB = torch.zeros_like(RGB)
            elif self.miss_type == 'n':
                NI = torch.zeros_like(NI)
            elif self.miss_type == 't':
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'rn':
                RGB = torch.zeros_like(RGB)
                NI = torch.zeros_like(NI)
            elif self.miss_type == 'rt':
                RGB = torch.zeros_like(RGB)
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'nt':
                NI = torch.zeros_like(NI)
                TI = torch.zeros_like(TI)

            if 'cam_label' in x:
                cam_label = x['cam_label']
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

            # SACR: 对 patch 特征进行多尺度上下文增强
            if self.USE_MULTIMODAL_SACR:
                # MultiModal-SACR: 三模态拼接 → SACR → 拆分（跨模态交互）
                RGB_cash, NI_cash, TI_cash = self.multimodal_sacr(RGB_cash, NI_cash, TI_cash)
            elif self.USE_SACR:
                # 单模态 SACR：三个模态独立处理
                RGB_cash = self.sacr(RGB_cash)  # (B, N, C) → (B, N, C)
                NI_cash = self.sacr(NI_cash)    # (B, N, C) → (B, N, C)
                TI_cash = self.sacr(TI_cash)    # (B, N, C) → (B, N, C)

            # Trimodal-LIF: Quality-aware feature enhancement (推理时也使用)
            if self.USE_LIF:
                # 1. 预测质量图
                q_rgb, q_nir, q_tir = self.lif.predict_quality(RGB, NI, TI)

                # 2. Resize 到 patch grid 尺寸
                patch_h = self.image_size[0] // self.cfg.MODEL.STRIDE_SIZE[0]
                patch_w = self.image_size[1] // self.cfg.MODEL.STRIDE_SIZE[1]

                q_rgb_patch = F.interpolate(q_rgb, size=(patch_h, patch_w), mode='bilinear')
                q_nir_patch = F.interpolate(q_nir, size=(patch_h, patch_w), mode='bilinear')
                q_tir_patch = F.interpolate(q_tir, size=(patch_h, patch_w), mode='bilinear')

                # 3. 计算逐位置权重
                q_logits = torch.cat([q_rgb_patch, q_nir_patch, q_tir_patch], dim=1)
                q_weights_spatial = F.softmax(q_logits * self.lif_temperature, dim=1)

                # 4. Reshape 为 token 维度并加权 patch 特征
                w_rgb_token = q_weights_spatial[:, 0:1].flatten(2).transpose(1, 2)
                w_nir_token = q_weights_spatial[:, 1:2].flatten(2).transpose(1, 2)
                w_tir_token = q_weights_spatial[:, 2:3].flatten(2).transpose(1, 2)

                RGB_cash = RGB_cash * w_rgb_token
                NI_cash = NI_cash * w_nir_token
                TI_cash = TI_cash * w_tir_token

            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)

            # SDTPS 推理分支
            if self.USE_SDTPS:
                RGB_enhanced, NI_enhanced, TI_enhanced, _, _, _ = self.sdtps(
                    RGB_cash, NI_cash, TI_cash,
                    RGB_global, NI_global, TI_global
                )

                # DGAF: 使用双门控自适应融合替代简单 concat
                if self.USE_DGAF:
                    if self.DGAF_VERSION == 'v3':
                        # V3: 直接输入 tokens，内置 attention pooling
                        sdtps_feat = self.dgaf(RGB_enhanced, NI_enhanced, TI_enhanced)
                    else:
                        # V1: 需要先将 tokens 聚合成 (B, C) 特征
                        if self.GLOBAL_LOCAL:
                            # GLOBAL_LOCAL 模式：pool(enhanced) + backbone_global 融合降维
                            # RGB_enhanced: (B, K+1, C) → permute → (B, C, K+1) → pool → (B, C, 1) → squeeze → (B, C)
                            RGB_local = self.pool(RGB_enhanced.permute(0, 2, 1)).squeeze(-1)  # (B, C)
                            NI_local = self.pool(NI_enhanced.permute(0, 2, 1)).squeeze(-1)    # (B, C)
                            TI_local = self.pool(TI_enhanced.permute(0, 2, 1)).squeeze(-1)    # (B, C)

                            # 融合 backbone 的 global 特征和 enhanced 的 pooled local 特征
                            RGB_sdtps = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))  # (B, C)
                            NI_sdtps = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))     # (B, C)
                            TI_sdtps = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))     # (B, C)
                        else:
                            # 默认方式：mean pooling
                            RGB_sdtps = RGB_enhanced.mean(dim=1)
                            NI_sdtps = NI_enhanced.mean(dim=1)
                            TI_sdtps = TI_enhanced.mean(dim=1)

                        sdtps_feat = self.dgaf(RGB_sdtps, NI_sdtps, TI_sdtps)
                else:
                    # 不使用 DGAF：简单拼接（也需要聚合特征）
                    if self.GLOBAL_LOCAL:
                        # GLOBAL_LOCAL 模式：pool(enhanced) + backbone_global 融合降维
                        RGB_local = self.pool(RGB_enhanced.permute(0, 2, 1)).squeeze(-1)  # (B, C)
                        NI_local = self.pool(NI_enhanced.permute(0, 2, 1)).squeeze(-1)    # (B, C)
                        TI_local = self.pool(TI_enhanced.permute(0, 2, 1)).squeeze(-1)    # (B, C)

                        RGB_sdtps = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))  # (B, C)
                        NI_sdtps = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))     # (B, C)
                        TI_sdtps = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))     # (B, C)
                    else:
                        # 默认方式：mean pooling
                        RGB_sdtps = RGB_enhanced.mean(dim=1)
                        NI_sdtps = NI_enhanced.mean(dim=1)
                        TI_sdtps = TI_enhanced.mean(dim=1)

                    sdtps_feat = torch.cat([RGB_sdtps, NI_sdtps, TI_sdtps], dim=-1)

                if return_pattern == 1:
                    return ori
                elif return_pattern == 2:
                    return sdtps_feat
                elif return_pattern == 3:
                    return torch.cat([ori, sdtps_feat], dim=-1)

            # 单独使用 DGAF V3（不依赖 SDTPS）：直接处理 backbone 的 patch tokens
            elif self.USE_DGAF:
                # DGAF V3 直接接受 patch tokens (B, N, C)
                dgaf_feat = self.dgaf(RGB_cash, NI_cash, TI_cash)  # (B, 3C)
                if return_pattern == 1:
                    return ori
                elif return_pattern == 2:
                    return dgaf_feat
                elif return_pattern == 3:
                    return torch.cat([ori, dgaf_feat], dim=-1)

            if self.HDM or self.ATM:
                moe_feat = self.generalFusion(RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
                if return_pattern == 1:
                    return ori
                elif return_pattern == 2:
                    return moe_feat
                elif return_pattern == 3:
                    return torch.cat([ori, moe_feat], dim=-1)
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
