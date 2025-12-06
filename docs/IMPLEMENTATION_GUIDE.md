# SACR、SDTPS、Trimodal-LIF 集成实现指南

## 第一部分：当前方案（SACR → SDTPS）的完整实现

### 文件清单

| 文件路径 | 功能 | 状态 |
|---------|------|------|
| `/modeling/sacr.py` | SACR模块实现 | ✓ 完成 |
| `/modeling/sdtps_complete.py` | SDTPS完整实现 | ✓ 完成 |
| `/modeling/make_model.py` | 模型集成 | ✓ 完成 |
| `/configs/RGBNT201/DeMo_SACR_SDTPS.yml` | 配置文件 | ✓ 完成 |
| `/test_sacr_sdtps.py` | 集成测试 | ✓ 完成 |

---

## 第二部分：SACR模块详细实现

### 代码位置
`/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/sacr.py`

### 核心实现

```python
class SACR(nn.Module):
    """
    Scale-Adaptive Contextual Refinement

    两种输入支持:
    1. Transformer tokens: (B, N, D) → (B, N, D)
    2. 特征图: (B, C, H, W) → (B, C, H, W)
    """

    def __init__(self, token_dim, height=None, width=None,
                 dilation_rates=[6, 12, 18]):
        super().__init__()

        self.token_dim = token_dim
        self.height = height
        self.width = width

        # Part 1: 多尺度空洞卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, 1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True)
        )

        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(token_dim, token_dim, 3,
                         padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(token_dim),
                nn.ReLU(inplace=True)
            ) for r in dilation_rates
        ])

        # 融合多个分支
        num_branches = 1 + len(dilation_rates)
        self.fusion = nn.Sequential(
            nn.Conv2d(token_dim * num_branches, token_dim, 1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True)
        )

        # Part 2: 自适应通道注意力
        k = int(abs((math.log2(token_dim) + 1) / 2))
        k = k if k % 2 else k + 1
        k = max(k, 3)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Conv1d(1, 1, kernel_size=k,
                                      padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 自动处理3D和4D输入
        if x.dim() == 3:
            # Transformer特征: (B, N, D) → (B, D, H, W)
            B, N, D = x.shape
            assert self.height is not None and self.width is not None
            assert self.height * self.width == N
            x = x.permute(0, 2, 1).view(B, D, self.height, self.width)
            reshape_back = True
        else:
            reshape_back = False
            B = x.shape[0]

        # 多尺度空洞卷积
        feat_1x1 = self.conv1x1(x)
        feat_atrous = [conv(x) for conv in self.atrous_convs]
        feat_cat = torch.cat([feat_1x1] + feat_atrous, dim=1)
        feat = self.fusion(feat_cat)

        # 通道注意力
        b, c, _, _ = feat.shape
        attn = self.gap(feat).view(b, 1, c)
        attn = self.sigmoid(self.channel_attn(attn)).view(b, c, 1, 1)

        out = feat * attn

        # 转换回原始形式
        if reshape_back:
            out = out.view(B, D, -1).permute(0, 2, 1)

        return out
```

### 使用示例

```python
# 在模型中的使用（make_model.py）
if self.USE_SACR:
    self.sacr = SACR(
        token_dim=cfg.MODEL.BACKBONE_DIM,
        height=8,  # patch grid height
        width=16,  # patch grid width
        dilation_rates=cfg.MODEL.SACR_DILATION_RATES,
    )

# 前向传播中的使用
if self.USE_SACR:
    RGB_cash = self.sacr(RGB_cash)  # (B, 128, 512) → (B, 128, 512)
    NI_cash = self.sacr(NI_cash)
    TI_cash = self.sacr(TI_cash)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| token_dim | 512 | token维度或通道数 |
| height | 8 | patch grid高度 |
| width | 16 | patch grid宽度 |
| dilation_rates | [6,12,18] | 膨胀卷积膨胀率 |

### 关键设计决策

```python
# 1. 膨胀率选择的原理
#    感受野大小 = kernel_size + (kernel_size-1)×(dilation-1) - 1
#    对于3×3卷积:
dilation=6:   感受野 = 3 + 2×5 = 13
dilation=12:  感受野 = 3 + 2×11 = 25
dilation=18:  感受野 = 3 + 2×17 = 37
#    这样可以在patch grid (8×16=128patches)上捕获不同尺度

# 2. 通道注意力的内核大小
#    使用ECA-Net的自适应内核大小计算
k = int(abs((math.log2(token_dim) + 1) / 2))
#    对于token_dim=512: k = log2(512)≈9, 确保≥3且为奇数

# 3. Reshape操作的正确性
#    输入: (B, N, D)  N=128=8×16
#    步骤: permute(0,2,1) → (B, D, N)
#    然后: view(B, D, H, W) → (B, D, 8, 16)
#    处理后: view(B, D, N) → (B, D, 128)
#    最后: permute(0, 2, 1) → (B, 128, D)
```

---

## 第三部分：SDTPS模块详细实现

### 代码位置
`/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/sdtps_complete.py`

### 核心实现（简化版）

```python
class TokenSparse(nn.Module):
    """Token稀疏选择 - 单个模态"""

    def __init__(self, embed_dim=512, sparse_ratio=0.5,
                 use_gumbel=False, gumbel_tau=1.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau

        # MLP Score Predictor
        self.score_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens, self_attention,
                cross_attention_m2, cross_attention_m3,
                beta=0.25):
        """
        Args:
            tokens: (B, N, C) - 128个patch特征
            self_attention: (B, N) - RGB自注意力分数
            cross_attention_m2: (B, N) - NIR交叉注意力
            cross_attention_m3: (B, N) - TIR交叉注意力
            beta: 权重参数

        Returns:
            select_tokens: (B, K, C) - 选中的patches (K≈64)
            extra_token: (B, 1, C) - 冗余patches融合
            score_mask: (B, N) - 完整决策矩阵
        """
        B, N, C = tokens.size()

        # Step 1: 计算综合得分
        # 公式: s = (1-2β)·s_pred + β·(s_m2 + s_m3 + 2·s_im)

        s_pred = self.score_predictor(tokens).squeeze(-1)  # (B, N)

        # Min-Max归一化
        def normalize_score(s):
            s_min = s.min(dim=-1, keepdim=True)[0]
            s_max = s.max(dim=-1, keepdim=True)[0]
            return (s - s_min) / (s_max - s_min + 1e-8)

        s_im = normalize_score(self_attention)
        s_m2 = normalize_score(cross_attention_m2)
        s_m3 = normalize_score(cross_attention_m3)

        # 综合得分
        score = (1 - 2*beta)*s_pred + beta*(s_m2 + s_m3 + 2*s_im)

        # Step 2: Top-K选择
        num_keep = max(1, math.ceil(N * self.sparse_ratio))  # ≈64

        score_sorted, score_indices = torch.sort(score, dim=1, descending=True)
        keep_policy = score_indices[:, :num_keep]  # (B, K)

        # Step 3: 决策矩阵
        if self.training and self.use_gumbel:
            # Gumbel-Softmax（已禁用，因为数值不稳定）
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)
            soft_mask = F.softmax((score + gumbel_noise) / self.gumbel_tau, dim=1)
            hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)
            score_mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            # 标准Top-K
            score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

        # Step 4: 提取选中patches
        select_tokens = torch.gather(
            tokens, dim=1,
            index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, K, C)

        # Step 5: 融合被丢弃的patches
        non_keep_policy = score_indices[:, num_keep:]
        non_tokens = torch.gather(tokens, dim=1,
                                 index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))

        non_keep_score = score_sorted[:, num_keep:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)

        return select_tokens, extra_token, score_mask


class TokenAggregation(nn.Module):
    """Token聚合 - 进一步压缩"""

    def __init__(self, dim=512, keeped_patches=26, dim_ratio=0.2):
        super().__init__()

        hidden_dim = int(dim * dim_ratio)  # 512×0.2=102

        # MLP生成聚合权重
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )

        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x, keep_policy=None):
        """
        Args:
            x: (B, N_s, C) - 选中的tokens (N_s≈64)
            keep_policy: (B, N_s) - 可选mask

        Returns:
            (B, N_c, C) - 聚合后的tokens (N_c=26)

        过程:
            x (B, 64, 512)
              ↓ MLP
            weight (B, 64, 26)
              ↓ transpose
            weight (B, 26, 64)
              ↓ softmax
            weight (B, 26, 64) - 每行和为1
              ↓ bmm with x
            output (B, 26, 512)
        """
        # 生成聚合权重
        weight = self.weight(x)  # (B, 64, 512) → (B, 64, 26)
        weight = weight.transpose(2, 1)  # (B, 26, 64)
        weight = weight * self.scale

        # 可选的mask
        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)  # (B, 1, 64)
            weight = weight - (1 - keep_policy) * 1e10

        # Softmax归一化
        weight = F.softmax(weight, dim=2)  # (B, 26, 64)

        # 加权求和
        return torch.bmm(weight, x)  # (B, 26, 512)


class MultiModalSDTPS(nn.Module):
    """多模态SDTPS - 完整流程"""

    def __init__(self, embed_dim=512, num_patches=128,
                 sparse_ratio=0.5, aggr_ratio=0.4,
                 use_gumbel=False, gumbel_tau=1.0,
                 beta=0.25, dim_ratio=0.2):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # 计算聚合后的patch数
        self.keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)
        # = int(128 × 0.4 × 0.5) = 25

        # RGB模态
        self.rgb_sparse = TokenSparse(embed_dim, sparse_ratio, use_gumbel, gumbel_tau)
        self.rgb_aggr = TokenAggregation(embed_dim, self.keeped_patches, dim_ratio)

        # NIR模态
        self.nir_sparse = TokenSparse(embed_dim, sparse_ratio, use_gumbel, gumbel_tau)
        self.nir_aggr = TokenAggregation(embed_dim, self.keeped_patches, dim_ratio)

        # TIR模态
        self.tir_sparse = TokenSparse(embed_dim, sparse_ratio, use_gumbel, gumbel_tau)
        self.tir_aggr = TokenAggregation(embed_dim, self.keeped_patches, dim_ratio)

    def _compute_self_attention(self, patches, global_feat):
        """计算self-attention score: patch与全局特征的相似度"""
        if global_feat.dim() == 2:
            global_feat = global_feat.unsqueeze(1)  # (B, C) → (B, 1, C)

        patches_norm = F.normalize(patches, dim=-1)
        global_norm = F.normalize(global_feat, dim=-1)

        return (patches_norm * global_norm).sum(dim=-1)  # (B, N)

    def _compute_cross_attention(self, patches, cross_global):
        """计算cross-attention score: patch与其他模态特征的相似度"""
        if cross_global.dim() == 2:
            cross_global = cross_global.unsqueeze(1)

        patches_norm = F.normalize(patches, dim=-1)
        cross_norm = F.normalize(cross_global, dim=-1)

        return (patches_norm * cross_norm).sum(dim=-1)  # (B, N)

    def forward(self, RGB_cash, NI_cash, TI_cash,
                RGB_global, NI_global, TI_global):
        """
        完整的多模态token选择和聚合

        输入: (B, 128, 512) × 3 + (B, 512) × 3
        输出: (B, 27, 512) × 3

        处理流程（以RGB为例）:
            RGB_cash (B, 128, 512)
              ├─ TokenSparse (sparse_ratio=0.5)
              │  → (B, 64, 512) select + (B, 1, 512) extra
              ├─ TokenAggregation (keeped=25)
              │  → (B, 25, 512)
              └─ Concat
                 → (B, 26, 512)  ← 最终输出
        """

        # ==================== RGB 模态 ====================
        # 自注意力
        rgb_self_attn = self._compute_self_attention(RGB_cash, RGB_global)

        # 交叉注意力（NIR和TIR对RGB的引导）
        rgb_nir_cross = self._compute_cross_attention(RGB_cash, NI_global)
        rgb_tir_cross = self._compute_cross_attention(RGB_cash, TI_global)

        # Token选择
        rgb_select, rgb_extra, rgb_mask = self.rgb_sparse(
            tokens=RGB_cash,
            self_attention=rgb_self_attn,
            cross_attention_m2=rgb_nir_cross,
            cross_attention_m3=rgb_tir_cross,
            beta=0.25,
        )  # (B, K, C), (B, 1, C), (B, N)

        # Token聚合
        rgb_aggr = self.rgb_aggr(rgb_select)  # (B, K, C) → (B, 25, C)

        # 拼接
        RGB_enhanced = torch.cat([rgb_aggr, rgb_extra], dim=1)  # (B, 26, C)

        # ==================== NIR 模态 ====================
        nir_self_attn = self._compute_self_attention(NI_cash, NI_global)
        nir_rgb_cross = self._compute_cross_attention(NI_cash, RGB_global)
        nir_tir_cross = self._compute_cross_attention(NI_cash, TI_global)

        nir_select, nir_extra, nir_mask = self.nir_sparse(
            tokens=NI_cash,
            self_attention=nir_self_attn,
            cross_attention_m2=nir_rgb_cross,
            cross_attention_m3=nir_tir_cross,
            beta=0.25,
        )

        nir_aggr = self.nir_aggr(nir_select)
        NI_enhanced = torch.cat([nir_aggr, nir_extra], dim=1)

        # ==================== TIR 模态 ====================
        tir_self_attn = self._compute_self_attention(TI_cash, TI_global)
        tir_rgb_cross = self._compute_cross_attention(TI_cash, RGB_global)
        tir_nir_cross = self._compute_cross_attention(TI_cash, NI_global)

        tir_select, tir_extra, tir_mask = self.tir_sparse(
            tokens=TI_cash,
            self_attention=tir_self_attn,
            cross_attention_m2=tir_rgb_cross,
            cross_attention_m3=tir_nir_cross,
            beta=0.25,
        )

        tir_aggr = self.tir_aggr(tir_select)
        TI_enhanced = torch.cat([tir_aggr, tir_extra], dim=1)

        return RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask
```

---

## 第四部分：模型集成（make_model.py）

### 关键代码片段

```python
class DeMo(nn.Module):
    def __init__(self, num_class, cfg, camera_num, view_num, factory_T_type):
        super().__init__()

        # ... 其他初始化 ...

        self.USE_SACR = cfg.MODEL.USE_SACR
        self.USE_SDTPS = cfg.MODEL.USE_SDTPS

        # SACR初始化
        if self.USE_SACR:
            self.sacr = SACR(
                token_dim=cfg.MODEL.BACKBONE_DIM,  # 512
                height=8,
                width=16,
                dilation_rates=cfg.MODEL.SACR_DILATION_RATES,
            )

        # SDTPS初始化
        if self.USE_SDTPS:
            self.sdtps = MultiModalSDTPS(
                embed_dim=cfg.MODEL.BACKBONE_DIM,
                num_patches=128,
                sparse_ratio=cfg.MODEL.SDTPS_SPARSE_RATIO,
                aggr_ratio=cfg.MODEL.SDTPS_AGGR_RATIO,
                use_gumbel=cfg.MODEL.SDTPS_USE_GUMBEL,
                gumbel_tau=cfg.MODEL.SDTPS_GUMBEL_TAU,
                beta=cfg.MODEL.SDTPS_BETA,
            )

            # SDTPS的分类头
            self.bottleneck_sdtps = nn.BatchNorm1d(cfg.MODEL.BACKBONE_DIM * 3)
            self.classifier_sdtps = nn.Linear(cfg.MODEL.BACKBONE_DIM * 3, num_class)

    def forward(self, x, label=None, cam_label=None, ...):
        # 训练模式
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

            # Step 1: Backbone提取特征
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, ...)
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, ...)
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, ...)

            # ... GLOBAL_LOCAL融合 ...

            # Step 2: SACR增强
            if self.USE_SACR:
                RGB_cash = self.sacr(RGB_cash)
                NI_cash = self.sacr(NI_cash)
                TI_cash = self.sacr(TI_cash)

            # Step 3: SDTPS选择和聚合
            if self.USE_SDTPS:
                RGB_enh, NI_enh, TI_enh, rgb_mask, nir_mask, tir_mask = self.sdtps(
                    RGB_cash, NI_cash, TI_cash,
                    RGB_global, NI_global, TI_global
                )

                # 池化
                RGB_feat = RGB_enh.mean(dim=1)  # (B, 512)
                NI_feat = NI_enh.mean(dim=1)
                TI_feat = TI_enh.mean(dim=1)

                # 拼接
                sdtps_feat = torch.cat([RGB_feat, NI_feat, TI_feat], dim=-1)  # (B, 1536)

                # 分类
                sdtps_score = self.classifier_sdtps(
                    self.bottleneck_sdtps(sdtps_feat)
                )

            # ... 返回值 ...
            if self.USE_SDTPS:
                return sdtps_score, sdtps_feat, ori_score, ori
```

---

## 第五部分：配置文件

### 完整的DeMo_SACR_SDTPS.yml

位置: `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/configs/RGBNT201/DeMo_SACR_SDTPS.yml`

```yaml
MODEL:
  TRANSFORMER_TYPE: 'ViT-B-16'
  STRIDE_SIZE: [16, 16]
  SIE_CAMERA: True
  DIRECT: 1
  SIE_COE: 1.0

  # 损失权重
  ID_LOSS_WEIGHT: 0.25
  TRIPLET_LOSS_WEIGHT: 1.0
  GLOBAL_LOCAL: True

  # 禁用旧的模块
  HDM: False
  ATM: False

  # SACR配置 (多尺度上下文增强)
  USE_SACR: True
  SACR_DILATION_RATES: [6, 12, 18]

  # SDTPS配置 (Token选择和聚合)
  USE_SDTPS: True
  SDTPS_SPARSE_RATIO: 0.7         # 保留比例
  SDTPS_AGGR_RATIO: 0.5           # 聚合比例
  SDTPS_BETA: 0.25                 # 得分权重
  SDTPS_USE_GUMBEL: False          # 禁用Gumbel
  SDTPS_GUMBEL_TAU: 5.0
  SDTPS_LOSS_WEIGHT: 2.0           # 辅助损失权重

  HEAD: 4

INPUT:
  SIZE_TRAIN: [256, 128]
  SIZE_TEST: [256, 128]
  PROB: 0.5
  RE_PROB: 0.5
  PADDING: 10

DATALOADER:
  SAMPLER: 'softmax_triplet'
  NUM_INSTANCE: 8
  NUM_WORKERS: 14

DATASETS:
  NAMES: ('RGBNT201')
  ROOT_DIR: '..'

SOLVER:
  BASE_LR: 0.00035
  WARMUP_ITERS: 10
  MAX_EPOCHS: 50
  OPTIMIZER_NAME: 'Adam'
  IMS_PER_BATCH: 64
  EVAL_PERIOD: 1

TEST:
  IMS_PER_BATCH: 128
  RE_RANKING: 'no'
  WEIGHT: ''
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'
  MISS: "nothing"

OUTPUT_DIR: '..'
```

---

## 第六部分：测试和验证

### 运行集成测试

```bash
python test_sacr_sdtps.py
```

预期输出：
```
================================================================================
SACR + SDTPS 集成测试
================================================================================

配置信息:
  USE_SACR: True
  SACR_DILATION_RATES: [6, 12, 18]
  USE_SDTPS: True
  SDTPS_SPARSE_RATIO: 0.7
  SDTPS_AGGR_RATIO: 0.5

创建模型...

总参数量: xxx,xxx (xxM)
SACR (共用): xxx,xxx (xxxM) - xxx%
SDTPS: xxx,xxx (xxxM) - xxx%

测试训练模式...
  输出数量: 4
  output[0]: shape=torch.Size([B, num_class])
  output[1]: shape=torch.Size([B, 1536])
  output[2]: shape=torch.Size([B, num_class])
  output[3]: shape=torch.Size([B, 1536])

测试推理模式...
  return_pattern=1: torch.Size([B, 1536])
  return_pattern=2: torch.Size([B, 1536])
  return_pattern=3: torch.Size([B, 3072])

✓ 测试通过！SACR + SDTPS 集成成功！
```

### 形状验证检查清单

```python
# 检查点清单
def verify_shapes():
    """验证各阶段的形状正确性"""

    # 输入
    assert RGB_cash.shape == (B, 128, 512)
    assert RGB_global.shape == (B, 512)

    # SACR后
    RGB_sacr = sacr(RGB_cash)
    assert RGB_sacr.shape == RGB_cash.shape

    # SDTPS后
    RGB_enh, NI_enh, TI_enh, masks = sdtps(
        RGB_sacr, NI_sacr, TI_sacr,
        RGB_global, NI_global, TI_global
    )

    # 验证SDTPS输出
    for enh in [RGB_enh, NI_enh, TI_enh]:
        assert enh.shape[0] == B
        assert enh.shape[2] == 512
        assert 25 <= enh.shape[1] <= 30  # 应该在26±4

    # 池化后
    RGB_feat = RGB_enh.mean(dim=1)
    assert RGB_feat.shape == (B, 512)

    # 拼接后
    sdtps_feat = torch.cat([RGB_feat, NI_feat, TI_feat], dim=-1)
    assert sdtps_feat.shape == (B, 1536)

    # 分类
    logits = classifier(bottleneck(sdtps_feat))
    assert logits.shape == (B, num_class)

    return True
```

---

## 第七部分：训练和推理

### 训练命令

```bash
# 基础训练
python train_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml

# 多GPU训练
python -m torch.distributed.launch --nproc_per_node=4 train_net.py \
    --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml

# 自定义参数训练
python train_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml \
    MODEL.SDTPS_BETA 0.15 \
    SOLVER.MAX_EPOCHS 60
```

### 推理命令

```bash
# 基础推理
python test_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml

# 缺失模态推理
python test_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml TEST.MISS r  # 缺RGB
python test_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml TEST.MISS n  # 缺NIR
python test_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml TEST.MISS t  # 缺TIR

# 不同特征模式
python test_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml \
    TEST.RETURN_PATTERN 1  # 仅原始拼接特征
python test_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml \
    TEST.RETURN_PATTERN 2  # 仅SDTPS特征
python test_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml \
    TEST.RETURN_PATTERN 3  # 两者拼接（默认）
```

---

**实现指南完成**
版本: 1.0
日期: 2025-12-06

