import torch.nn as nn
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
            )
            # SDTPS 输出特征维度：每个模态 (K+1) 个 token，K 取决于 sparse_ratio
            # 暂时使用3倍 feat_dim（拼接三个模态的全局特征）
            self.classifier_sdtps = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_sdtps.apply(weights_init_classifier)
            self.bottleneck_sdtps = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_sdtps.bias.requires_grad_(False)
            self.bottleneck_sdtps.apply(weights_init_kaiming)
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
            if self.GLOBAL_LOCAL:
                RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
                NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
                TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
                RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))

            # SACR: 对 patch 特征进行多尺度上下文增强（三个模态共用）
            if self.USE_SACR:
                RGB_cash = self.sacr(RGB_cash)  # (B, N, C) → (B, N, C)
                NI_cash = self.sacr(NI_cash)    # (B, N, C) → (B, N, C)
                TI_cash = self.sacr(TI_cash)    # (B, N, C) → (B, N, C)

            # SDTPS 分支：使用 token selection 替代 HDM+ATM
            if self.USE_SDTPS:
                # SDTPS token selection and enhancement
                RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask = self.sdtps(
                    RGB_cash, NI_cash, TI_cash,
                    RGB_global, NI_global, TI_global
                )
                # 对增强的 tokens 进行池化得到全局特征
                RGB_sdtps = RGB_enhanced.mean(dim=1)  # (B, K+1, C) → (B, C)
                NI_sdtps = NI_enhanced.mean(dim=1)    # (B, K+1, C) → (B, C)
                TI_sdtps = TI_enhanced.mean(dim=1)    # (B, K+1, C) → (B, C)

                # 拼接三个模态的增强特征
                sdtps_feat = torch.cat([RGB_sdtps, NI_sdtps, TI_sdtps], dim=-1)  # (B, 3C)
                sdtps_score = self.classifier_sdtps(self.bottleneck_sdtps(sdtps_feat))

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
            if self.direct:
                if self.USE_SDTPS:
                    # SDTPS 分支：返回 SDTPS 特征和原始特征
                    return sdtps_score, sdtps_feat, ori_score, ori
                elif self.HDM or self.ATM:
                    return moe_score, moe_feat, ori_score, ori, loss_moe
                return ori_score, ori
            else:
                if self.USE_SDTPS:
                    # SDTPS 分支（非direct模式）
                    return sdtps_score, sdtps_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global
                elif self.HDM or self.ATM:
                    return moe_score, moe_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, loss_moe
                return RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global

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
            if self.GLOBAL_LOCAL:
                RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
                NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
                TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
                RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))

            # SACR: 对 patch 特征进行多尺度上下文增强（三个模态共用）
            if self.USE_SACR:
                RGB_cash = self.sacr(RGB_cash)  # (B, N, C) → (B, N, C)
                NI_cash = self.sacr(NI_cash)    # (B, N, C) → (B, N, C)
                TI_cash = self.sacr(TI_cash)    # (B, N, C) → (B, N, C)

            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)

            # SDTPS 推理分支
            if self.USE_SDTPS:
                RGB_enhanced, NI_enhanced, TI_enhanced, _, _, _ = self.sdtps(
                    RGB_cash, NI_cash, TI_cash,
                    RGB_global, NI_global, TI_global
                )
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
