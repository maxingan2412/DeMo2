# SEPS任务详解：为什么用MSE Loss而不用CE Loss？

## 目录
1. [MSE Loss vs CE Loss 对比](#mse-loss-vs-ce-loss-对比)
2. [为什么SEPS不用CE Loss](#为什么seps不用ce-loss)
3. [SEPS任务完整说明](#seps任务完整说明)
4. [数据集详解](#数据集详解)
5. [输入输出示例](#输入输出示例)

---

## MSE Loss vs CE Loss 对比

### 1. 数学定义

#### MSE Loss (Mean Squared Error)

```python
L_MSE = (1/N) Σ (y_pred - y_true)²

# PyTorch
loss = nn.MSELoss()
output = loss(predictions, targets)
```

**特点**:
- 输出: 连续值（回归）
- 目标: 连续值
- 范围: [0, +∞)

#### CE Loss (Cross Entropy)

```python
L_CE = -Σ y_true * log(y_pred)

# PyTorch (多分类)
loss = nn.CrossEntropyLoss()
output = loss(logits, class_labels)  # logits: (N, C), labels: (N,)

# PyTorch (二分类)
loss = nn.BCELoss()
output = loss(probs, targets)  # probs: (N,), targets: (N,)
```

**特点**:
- 输出: 概率分布（分类）
- 目标: 类别标签（离散）
- 范围: [0, +∞)

### 2. 适用场景对比

| 维度 | MSE Loss | CE Loss |
|-----|----------|---------|
| **任务类型** | 回归、拟合 | 分类、概率估计 |
| **输出类型** | 连续值 | 类别/概率 |
| **目标类型** | 连续值 | 离散标签 |
| **典型应用** | 预测房价、温度、比例 | 分类图像、文本分类 |
| **梯度特性** | 线性 | 指数（接近0/1时梯度小） |

### 3. 具体示例

#### MSE Loss示例

```python
# 任务：预测选择了多少比例的patch
predictions = torch.tensor([0.48, 0.52, 0.45])  # 预测比例
targets = torch.tensor([0.5, 0.5, 0.5])          # 目标比例

mse_loss = ((predictions - targets) ** 2).mean()
# = ((0.48-0.5)² + (0.52-0.5)² + (0.45-0.5)²) / 3
# = (0.0004 + 0.0004 + 0.0025) / 3
# = 0.0011
```

#### CE Loss示例

```python
# 任务：分类这个patch是否应该保留
logits = torch.tensor([[2.0, 0.1],    # patch 0: 更可能是类别0(保留)
                       [0.3, 1.5]])    # patch 1: 更可能是类别1(丢弃)
labels = torch.tensor([0, 1])          # ground truth

ce_loss = nn.CrossEntropyLoss()(logits, labels)
# = -log(softmax([2.0, 0.1])[0]) - log(softmax([0.3, 1.5])[1])
# ≈ 0.126 + 0.379 = 0.505
```

---

## 为什么SEPS不用CE Loss？

### SEPS中的两个损失分析

#### 1. L_align (对比损失) - 为什么不用CE?

**任务性质**: **排序/检索任务**，不是分类任务

```
┌────────────────────────────────────────────────────────┐
│            分类 vs 排序的本质区别                        │
└────────────────────────────────────────────────────────┘

分类任务:
    输入: 一个图像
    输出: 属于哪个类别 (狗/猫/鸟/...)
    目标: 预测正确的类别标签
    损失: CE Loss

    Example:
    - 图像 → 模型 → [0.1, 0.8, 0.1] (狗/猫/鸟)
    - 真实标签: "猫" (index=1)
    - CE Loss: -log(0.8) = 0.223

排序/检索任务:
    输入: 一个图像 + 多个文本
    输出: 哪个文本与图像最匹配 (相似度分数)
    目标: 正确的文本排在前面
    损失: Ranking Loss / Triplet Loss

    Example:
    - 图像I与文本[T1, T2, T3]的相似度
    - sims = [0.9, 0.3, 0.5]
    - 真实匹配: T1
    - Triplet Loss: [α - 0.9 + max(0.3, 0.5)]_+
                  = [0.2 - 0.9 + 0.5]_+ = 0
```

**为什么不用CE Loss?**

❌ **CE Loss的问题**:
```python
# 如果用CE Loss
sims = [0.9, 0.3, 0.5]  # 与3个文本的相似度
labels = 0               # 第0个文本是正确的

# Softmax归一化
probs = softmax(sims) = [0.56, 0.16, 0.28]

# CE Loss
ce_loss = -log(0.56) = 0.58

# 问题1: 必须归一化为概率分布（和=1）
#       但相似度本身没有这个约束
# 问题2: 只关心正确类别的概率，不关心负样本的排序
#       但检索需要所有负样本都排在正样本后面
# 问题3: 不能处理一图多文的情况
```

✅ **Triplet Loss的优势**:
```python
# Triplet Loss
sims = [0.9, 0.3, 0.5]  # 不需要归一化
positive = 0.9
hardest_negative = max(0.3, 0.5) = 0.5

triplet_loss = [margin - positive + hardest_negative]_+
             = [0.2 - 0.9 + 0.5]_+ = 0

# 优势1: 直接优化相对排序，不需要概率归一化
# 优势2: 明确要求 positive - negative >= margin
# 优势3: 可处理一图多文（多个正样本）
```

**对比表**:

| 维度 | CE Loss | Triplet Loss (SEPS使用) |
|-----|---------|------------------------|
| **任务** | 分类 | 排序/检索 |
| **输出** | 类别概率 | 相似度分数 |
| **约束** | Σp_i = 1 | 无约束 |
| **优化目标** | 最大化正确类别概率 | 拉大正负样本间隔 |
| **多正样本** | ❌ 不支持 | ✅ 支持 |
| **梯度** | 指数衰减 | 线性 |

#### 2. L_ratio (比例损失) - 为什么用MSE?

**任务性质**: **回归任务** - 预测选择比例

```
┌────────────────────────────────────────────────────────┐
│          比例约束：回归任务 vs 分类任务                   │
└────────────────────────────────────────────────────────┘

实际任务:
    预测: "应该选择多少比例的patch？"
    目标: 50% (target_ratio = 0.5)
    实际: 48% (actual_ratio = 0.48)

这是一个回归问题，不是分类问题！
```

**如果用MSE Loss (正确)**:
```python
target_ratio = 0.5
actual_ratio = score_mask.mean()  # 0.48

mse_loss = (actual_ratio - target_ratio) ** 2
         = (0.48 - 0.5) ** 2
         = 0.0004

# 优势：
# 1. 直接优化连续值
# 2. 梯度平滑，训练稳定
# 3. 误差越大，惩罚越重（平方）
```

**如果用CE Loss (错误)**:
```python
# 必须把连续的比例值转换为离散类别
# 例如：0-10%, 10-20%, ..., 90-100% (10个类别)

actual_ratio = 0.48
# 属于哪个类别？40-50%? 还是45-55%?
# 类别边界如何定义？

# 问题：
# 1. 人为离散化，丢失精度
# 2. 类别定义主观
# 3. 0.49和0.51应该损失相近，但离散化后可能差别很大
```

**对比表**:

| 损失函数 | 适用场景 | SEPS的L_ratio使用 |
|---------|---------|------------------|
| **MSE Loss** | ✅ 预测连续值（比例、角度、坐标） | 预测patch选择比例 (0.48 vs 0.5) |
| **CE Loss** | ✅ 预测离散类别（猫/狗、是/否） | ❌ 不适用（比例是连续的） |

---

## SEPS任务完整说明

### 任务定义：跨模态检索 (Cross-Modal Retrieval)

**任务**: Image-Text Matching / Cross-Modal Retrieval
**中文**: 图文匹配 / 跨模态检索

### 两个子任务

#### 1. Image-to-Text Retrieval (I2T)
给定一个图像，从文本库中检索最相关的文本描述

```
输入: 一张图像
文本库: ["A dog running", "A cat sleeping", "A bird flying", ...]
输出: 根据相似度排序的文本列表
评估: 正确文本是否在Top-1, Top-5, Top-10中
```

#### 2. Text-to-Image Retrieval (T2I)
给定一段文本，从图像库中检索最相关的图像

```
输入: 一段文本描述 "A dog running on grass"
图像库: [img1, img2, img3, ...]
输出: 根据相似度排序的图像列表
评估: 正确图像是否在Top-1, Top-5, Top-10中
```

### 任务流程图

```
┌─────────────────────────────────────────────────────────┐
│              跨模态检索任务流程                           │
└─────────────────────────────────────────────────────────┘

训练阶段:
━━━━━━━
输入Batch:
- Images: (B, 3, 224, 224)     例如32张图像
- Captions: (B, L_s)           每张图像配1-5个文本描述
- Dense Captions: (B, L_d)     MLLM生成的详细描述

    ↓ 编码器

特征:
- img_embs: (B, N+1, C)
- cap_embs: (B, L_s, C)
- long_cap_embs: (B, L_d, C)

    ↓ SEPS模型

相似度矩阵:
- sims: (B, B)
- sims[i,j] = S(Image_i, Text_j)

    ↓ 损失函数

目标:
- sims[i,i] 应该是第i行和第i列的最大值
  (正确匹配的图文对相似度最高)

    ↓ 反向传播

优化模型参数

推理阶段:
━━━━━━━
输入:
- 查询图像/文本
- 候选文本库/图像库

    ↓ 编码 + SEPS

相似度分数:
- 对每个候选计算相似度

    ↓ 排序

检索结果:
- Top-K相似度最高的结果
```

---

## 数据集详解

### Flickr30K

**来源**: Flickr图片平台
**论文**: Young et al., "From image descriptions to visual denotations" (2014)

**数据规模**:
```
总图像数: 31,784张
总文本数: 158,915条 (每张图像5条描述)

数据划分:
├─ 训练集: 29,000张图像, 145,000条文本
├─ 验证集: 1,000张图像, 5,000条文本
└─ 测试集: 1,014张图像, 5,070条文本
```

**单个样本示例**:

```json
{
  "image_id": 12345,
  "image_path": "flickr30k-images/12345.jpg",
  "captions": [
    "A dog running on the grass.",
    "A brown dog is playing outside.",
    "A pet running in a park.",
    "Dog enjoying outdoor time.",
    "An animal in a grassy field."
  ],
  "dense_caption": "A medium-sized brown dog with floppy ears is running energetically on green grass. The dog appears to be a mixed breed, with a happy expression and its tongue hanging out. The background shows a sunny park with trees and blue sky. The grass is well-maintained and bright green, suggesting it's during spring or summer."
}
```

**文本特点**:
- **Sparse Caption (原始)**: 5-15个词，简洁描述
- **Dense Caption (MLLM生成)**: 50-200个词，详细描述

### MS-COCO

**来源**: Microsoft COCO (Common Objects in Context)
**论文**: Lin et al., "Microsoft COCO: Common objects in context" (2014)

**数据规模**:
```
总图像数: 123,287张
总文本数: 616,435条 (每张图像5条描述)

数据划分:
├─ 训练集: 113,287张图像, 566,435条文本
├─ 验证集: 5,000张图像, 25,000条文本
└─ 测试集: 5,000张图像, 25,000条文本

评估方式:
├─ 1K test: 5-fold cross-validation (每fold 1,000图像, 5,000文本)
└─ 5K test: 全部5,000图像, 25,000文本
```

**单个样本示例**:

```json
{
  "image_id": 78901,
  "image_path": "coco/train2014/COCO_train2014_000000078901.jpg",
  "captions": [
    "A woman playing tennis on a court.",
    "Female tennis player hitting a ball.",
    "A person holding a racket on a tennis court.",
    "Woman in athletic wear playing tennis.",
    "Tennis match with a female player."
  ],
  "dense_caption": "A woman in her mid-20s wearing a white tennis outfit consisting of a sleeveless top and short skirt is positioned on a blue hard court. She is in mid-swing with a white and red tennis racket, preparing to hit a yellow tennis ball. Her long brown hair is tied back in a ponytail. The court has clear white boundary lines, and there are advertising boards visible in the background. The lighting suggests it's daytime with good weather conditions."
}
```

**数据集对比**:

| 特征 | Flickr30K | MS-COCO |
|-----|-----------|---------|
| **图像来源** | 日常生活照片 | 日常场景 + 物体检测 |
| **图像复杂度** | 🟢 简单-中等 | 🟡 中等-复杂 |
| **场景多样性** | 🟢 多样 | 🟢🟢 非常多样 |
| **物体数量** | 1-3个主要物体 | 2-10个物体 |
| **训练集大小** | 29K | 113K |
| **测试难度** | 🟢 中等 | 🔴 困难 |

---

## 输入输出示例

### 完整数据流示例

```python
# ========================================
# 训练时的一个batch
# ========================================

batch_size = 32

# 1. 图像输入
images = torch.randn(32, 3, 224, 224)
# 32张RGB图像，分辨率224×224

# 2. 稀疏文本输入 (原始caption)
captions = torch.tensor([
    [101, 1037, 3899, 2770, 102, 0, 0, ...],  # "A dog running" + padding
    [101, 1037, 4937, 5437, 102, 0, 0, ...],  # "A cat sleeping" + padding
    ...  # 32条caption
])  # (32, L_s) L_s=最大长度，例如30

cap_lens = torch.tensor([5, 5, 7, 6, ...])  # (32,) 实际有效长度

# 3. 稠密文本输入 (MLLM生成)
long_captions = torch.tensor([
    [101, 1037, 2512, 1011, 5048, ...],  # 详细描述 (100+ tokens)
    [101, 1037, 4937, 2007, 6081, ...],  # 详细描述
    ...  # 32条dense caption
])  # (32, L_d) L_d=最大长度，例如200

long_cap_lens = torch.tensor([156, 189, 178, ...])  # (32,)

# 4. 图像ID (用于一图多文)
img_ids = torch.tensor([0, 0, 0, 0, 0,   # 图像0的5个caption
                        1, 1, 1, 1, 1,   # 图像1的5个caption
                        ...])            # (32,)

# ========================================
# 模型处理
# ========================================

# 编码
img_embs = vision_encoder(images)         # (32, 197, 512)
cap_embs = text_encoder(captions, cap_lens)  # (32, 30, 512)
long_cap_embs = text_encoder(long_captions, long_cap_lens)  # (32, 200, 512)

# SEPS处理
sims, score_mask = model(
    img_embs, cap_embs, cap_lens,
    long_cap_embs, long_cap_lens
)

# 输出
sims: (32, 32)  # 相似度矩阵
# sims[i,j] = 图像i与文本j的相似度

score_mask: (32, 32, 196) 或 tuple
# 每个图文对的patch选择决策

# ========================================
# 损失计算
# ========================================

total_loss, align_loss, ratio_loss = criterion(sims, score_mask, img_ids)

# total_loss: 反向传播用
# align_loss: 监控图文匹配质量
# ratio_loss: 监控patch选择比例
```

### 真实数据示例

#### Flickr30K样本

```
Image: flickr30k-images/3012345.jpg
└─ 一个女孩在海滩玩耍的照片

Sparse Captions (5条):
1. "A young girl playing on the beach."
2. "A child building a sandcastle."
3. "Girl in a red dress at the seaside."
4. "A kid having fun on sandy beach."
5. "Young child enjoying beach time."

Dense Caption (MLLM生成):
"A young girl approximately 5-7 years old is kneeling on a sandy beach,
building a sandcastle with a small red plastic bucket and shovel. She is
wearing a bright red summer dress with white polka dots and a white sun
hat. Her hair is blonde and appears windblown. The background shows calm
blue ocean waves and a clear sky. The sand is light beige and appears
fine-grained. The girl's expression shows concentration and joy. Several
other beachgoers can be seen in the distant background."

词数对比:
- Sparse: 平均8个词/caption
- Dense: 95个词

语义密度对比:
- Sparse: 主要物体（girl, beach）
- Dense: 详细特征（年龄、动作、服装、表情、环境）
```

#### MS-COCO样本

```
Image: COCO_val2014_000000123456.jpg
└─ 棒球比赛场景

Sparse Captions (5条):
1. "A baseball player swinging at a pitch."
2. "A man hitting a baseball during a game."
3. "Baseball player at bat in a stadium."
4. "Batter attempting to hit the ball."
5. "A person playing baseball on a field."

Dense Caption (MLLM生成):
"A professional baseball game in progress at a large outdoor stadium.
The batter is a right-handed player wearing a white uniform with red
pinstripes and a red helmet, number 27 visible on the back. He is in
mid-swing position, having just made contact with a white baseball.
The catcher, wearing dark blue protective gear, is crouched behind home
plate. An umpire in black attire stands behind the catcher. The stadium
features green artificial turf, white bases, and advertising boards along
the outfield walls. Crowd can be seen in the stands, mostly wearing red
and white team colors. The sky is clear and blue, suggesting a day game."

词数对比:
- Sparse: 平均7个词/caption
- Dense: 132个词

细节增强:
- Sparse: 基本动作（playing baseball）
- Dense: 球员姿势、装备细节、场地特征、观众、天气
```

---

## 为什么这个任务设计适合MSE而不是CE？

### 原因1: 任务本质是排序，不是分类

```
┌────────────────────────────────────────────────────────┐
│     图像检索 ≠ 图像分类                                  │
└────────────────────────────────────────────────────────┘

图像分类 (用CE Loss):
─────────────────
输入: 一张图像
输出: 类别 {猫, 狗, 鸟, ...}
目标: 预测正确类别
损失: CE Loss

图像检索 (用Triplet/Ranking Loss):
────────────────────────────────
输入: 一张图像 + 候选文本集合
输出: 相似度分数 [0.9, 0.3, 0.5, ...]
目标: 正确匹配的相似度最高
损失: Ranking Loss (Triplet Loss)

关键区别:
- 分类: 预测属于哪个固定类别
- 检索: 计算与任意候选的相似度
```

### 原因2: 相似度是连续值，不是概率

```python
# ========================================
# 相似度的性质
# ========================================

# 相似度是连续的实数
sims = torch.tensor([
    [0.92, 0.35, 0.47, ...],  # 图像0与各文本
    [0.28, 0.88, 0.41, ...],  # 图像1与各文本
    ...
])

# 特点:
# 1. 不需要归一化为概率（不要求Σ=1）
# 2. 可以都很高，也可以都很低
# 3. 关心的是相对排序，不是绝对概率

# 如果强行用CE Loss
probs = softmax(sims, dim=1)
# 问题：破坏了原始相似度的绝对大小信息
```

### 原因3: 支持一图多文

```python
# ========================================
# 一图多文场景 (COCO数据集)
# ========================================

# 每张图像有5个caption
img_ids = [0, 0, 0, 0, 0,  # 图像0的5个描述
           1, 1, 1, 1, 1,  # 图像1的5个描述
           ...]

# Triplet Loss可以处理
mask = (img_ids.unsqueeze(0) == img_ids.unsqueeze(1))
# mask[0,0-4] = True  (都是正样本)
# mask[0,5-31] = False (都是负样本)

# CE Loss无法处理
# CE要求每个样本只有1个正确类别
# 但这里有5个正确类别（5个caption）
```

### 原因4: L_ratio约束的是连续比例

```python
# ========================================
# 比例约束任务
# ========================================

# 目标: 选择patch的比例接近50%
target_ratio = 0.5

# 实际选择的比例（连续值）
actual_ratio = 0.483  # 48.3%

# MSE Loss (正确)
mse_loss = (0.483 - 0.5) ** 2 = 0.000289
# 直接优化连续值，梯度平滑

# CE Loss (不适用)
# 必须离散化: {0-20%, 20-40%, 40-60%, 60-80%, 80-100%}
# 0.483属于类别2 (40-60%)
# 问题: 0.483和0.517应该损失相近，但离散化后都是类别2，CE无法区分
```

---

## 任务目标与评估

### 评估指标

#### Recall@K (R@K)

**定义**: 正确结果出现在Top-K中的比例

```python
# Image-to-Text Retrieval
# 给定图像，检索文本

npts = 1000  # 1000张测试图像
sims = (1000, 5000)  # 每个图像与5000个文本的相似度

for i in range(npts):
    scores = sims[i]  # 第i个图像与所有文本的相似度
    sorted_indices = argsort(scores, descending=True)

    # 找到5个正确caption的排名
    gt_indices = [i*5, i*5+1, i*5+2, i*5+3, i*5+4]
    ranks = [where(sorted_indices == gt)[0] for gt in gt_indices]
    best_rank = min(ranks)  # 最好的排名

    if best_rank < 1:   # Top-1
        r1_count += 1
    if best_rank < 5:   # Top-5
        r5_count += 1
    if best_rank < 10:  # Top-10
        r10_count += 1

R@1 = 100 * r1_count / npts
R@5 = 100 * r5_count / npts
R@10 = 100 * r10_count / npts
```

**示例**:
```
测试集: 1000张图像, 5000条文本

Image-to-Text Retrieval:
- R@1 = 86.1%  → 861张图像的正确caption在Top-1
- R@5 = 93.7%  → 937张图像的正确caption在Top-5
- R@10 = 96.9% → 969张图像的正确caption在Top-10

Text-to-Image Retrieval:
- R@1 = 86.9%  → 4345条文本 (86.9% of 5000) 的正确图像在Top-1
- R@5 = 98.1%
- R@10 = 99.2%
```

#### rSum (Recall Sum)

**定义**: 6个R@K的总和

```python
rSum = R@1_i2t + R@5_i2t + R@10_i2t
     + R@1_t2i + R@5_t2i + R@10_t2i

# SEPS在Flickr30K的结果
rSum = 86.1 + 93.7 + 96.9 + 86.9 + 98.1 + 99.2
     = 560.9
```

**意义**: 综合评估双向检索性能的单一指标

---

## 完整任务流程示例

### 训练流程

```python
# ========================================
# Epoch 1: Batch 1
# ========================================

# 输入数据
images = load_images([
    "flickr30k/12345.jpg",
    "flickr30k/12346.jpg",
    ...  # 32张图像
])  # (32, 3, 224, 224)

sparse_captions = [
    "A dog running on grass",
    "A cat sleeping on sofa",
    ...  # 每张图像1个caption，共32条
]

dense_captions = [
    "A medium-sized brown dog with...",  # 详细描述
    "A gray tabby cat curled up on...",
    ...  # 32条detailed caption
]

# Tokenize
cap_tokens = tokenizer(sparse_captions)  # (32, 30)
long_cap_tokens = tokenizer(dense_captions)  # (32, 200)

# 图像ID（如果一图多文）
img_ids = torch.arange(32)  # [0, 1, 2, ..., 31]

# ========================================
# 模型前向传播
# ========================================

# 特征编码
img_embs = vision_encoder(images)  # (32, 197, 512)
cap_embs = text_encoder(cap_tokens, cap_lens)  # (32, 30, 512)
long_cap_embs = text_encoder(long_cap_tokens, long_lens)  # (32, 200, 512)

# SEPS处理
sims, score_mask = model(img_embs, cap_embs, cap_lens,
                          long_cap_embs, long_cap_lens)

# 相似度矩阵
sims = [
    [0.92, 0.31, 0.28, ...],  # 图像0与所有文本
    [0.35, 0.89, 0.33, ...],  # 图像1与所有文本
    ...
]  # (32, 32)

# 理想情况: 对角线最大
# sims[0,0]=0.92 > sims[0,1-31]
# sims[1,1]=0.89 > sims[1,0,2-31]

# ========================================
# 损失计算
# ========================================

total_loss, align_loss, ratio_loss = criterion(sims, score_mask, img_ids)

# align_loss计算过程:
diagonal = [0.92, 0.89, 0.91, ...]  # 正样本对
hardest_negative_per_image = [0.35, 0.35, ...]  # 每张图像的最难负样本

triplet_loss_i2t = sum([margin - 0.92 + 0.35]_+)  # Image→Text
                 = sum([0.2 - 0.92 + 0.35]_+)
                 = sum([-0.37]_+) = 0  # 正样本已超过margin

# ratio_loss计算过程:
actual_selection = score_mask.mean() = 0.487
ratio_loss = (0.487 - 0.5) ** 2 = 0.000169

# 总损失
total_loss = align_loss + 2.0 * ratio_loss

# 反向传播
total_loss.backward()
optimizer.step()
```

### 推理流程

```python
# ========================================
# 测试集检索
# ========================================

# Flickr30K测试集
test_images = 1014张图像
test_captions = 5070条文本 (每图5条)

# Step 1: 编码所有数据
img_embs_all = []
for img in test_images:
    img_emb = vision_encoder(img)
    img_embs_all.append(img_emb)
img_embs_all = torch.stack(img_embs_all)  # (1014, 197, 512)

cap_embs_all = []
for cap in test_captions:
    cap_emb = text_encoder(cap)
    cap_embs_all.append(cap_emb)
cap_embs_all = torch.stack(cap_embs_all)  # (5070, L, 512)

# Step 2: 计算相似度矩阵
sims = torch.zeros(1014, 5070)
for i in range(1014):
    for j in range(5070):
        sims[i,j] = model.forward_sim(
            img_embs_all[i:i+1],
            cap_embs_all[j:j+1],
            cap_lens[j:j+1],
            long_cap_embs_all[j:j+1],
            long_cap_lens[j:j+1]
        )

# Step 3: Image-to-Text检索
for i in range(1014):
    scores = sims[i]  # 第i个图像与所有5070个文本的相似度
    sorted_indices = argsort(scores, descending=True)

    # 检查5个正确caption是否在Top-K
    gt_captions = [i*5, i*5+1, i*5+2, i*5+3, i*5+4]

    # 计算最佳排名
    ranks = [where(sorted_indices == gt)[0] for gt in gt_captions]
    best_rank = min(ranks)

    if best_rank == 0:  # Top-1
        r1 += 1
    if best_rank < 5:   # Top-5
        r5 += 1
    if best_rank < 10:  # Top-10
        r10 += 1

R@1 = 100 * r1 / 1014  # 例如: 86.1%
R@5 = 100 * r5 / 1014  # 例如: 93.7%
R@10 = 100 * r10 / 1014  # 例如: 96.9%

# Step 4: Text-to-Image检索（类似）
# ...

# Step 5: 计算rSum
rSum = R@1_i2t + R@5_i2t + R@10_i2t + R@1_t2i + R@5_t2i + R@10_t2i
```

---

## 数据集目录结构

### Flickr30K

```
data/f30k/
├── train_caps.txt           # 训练集caption (145,000行)
├── train_ids.txt            # 训练集图像ID (145,000行)
├── test_caps.txt            # 测试集caption (5,070行)
├── test_ids.txt             # 测试集图像ID (5,070行)
├── id_mapping.json          # 图像ID到文件路径映射
├── f30k_train.jsonl         # MLLM生成的dense caption (训练集)
└── f30k_test.jsonl          # MLLM生成的dense caption (测试集)

flickr30k-images/            # 图像文件夹
├── 1000092795.jpg
├── 10002456.jpg
└── ...
```

**文件格式示例**:

```
train_caps.txt:
─────────────
A dog running on the grass.
A brown dog is playing outside.
A pet running in a park.
...

train_ids.txt:
─────────────
1000092795
1000092795
1000092795
1000092795
1000092795
10002456
...

f30k_train.jsonl:
────────────────
{"image_id": 1000092795, "text": "A medium-sized brown dog with..."}
{"image_id": 10002456, "text": "A young girl approximately..."}
...
```

### MS-COCO

```
data/coco/
├── train_caps.txt           # 训练集caption (566,435行)
├── train_ids.txt            # 训练集图像ID (566,435行)
├── testall_caps.txt         # 测试集caption (25,000行)
├── testall_ids.txt          # 测试集图像ID (25,000行)
├── id_mapping.json
├── coco_train.jsonl         # Dense captions
└── coco_testall.jsonl

coco/                        # COCO图像
├── train2014/
│   ├── COCO_train2014_000000000009.jpg
│   └── ...
└── val2014/
    ├── COCO_val2014_000000000042.jpg
    └── ...
```

---

## 为什么不能用CE Loss？技术细节

### 场景1: 用CE Loss做L_align

**假设**: 把图文匹配当作分类任务

```python
# ❌ 错误的CE Loss使用
# 把"找到正确文本"当作"分类问题"

# 输入
img_emb = vision_encoder(image)  # (1, 512)
candidate_texts = 5000  # 候选文本数量

# 如果用分类头
logits = nn.Linear(512, 5000)(img_emb)  # (1, 5000)
# 每个候选文本是一个类别

# CE Loss
label = 237  # 假设第237个文本是正确的
ce_loss = nn.CrossEntropyLoss()(logits, label)

# 问题:
# 1. 需要为每个数据集训练专门的分类头（5000个类别）
# 2. 新增文本就要重新训练
# 3. 无法泛化到未见过的文本
# 4. 一图多文时，label只能是一个数字，无法表示多个正确答案
```

**正确的Triplet Loss**:

```python
# ✅ 正确的Ranking Loss使用
# 计算图像与每个候选文本的相似度

img_emb = vision_encoder(image)  # (1, 512)
text_embs = text_encoder(candidate_texts)  # (5000, 512)

# 计算相似度（不需要分类头）
sims = img_emb @ text_embs.T  # (1, 5000)

# Triplet Loss
positive_sim = sims[0, 237]  # 正确文本的相似度
negative_sims = sims[0, [0-236, 238-4999]]  # 其他文本
hardest_negative = negative_sims.max()

triplet_loss = [margin - positive_sim + hardest_negative]_+

# 优势:
# 1. 无需为每个数据集训练分类头
# 2. 可泛化到任意新文本
# 3. 支持一图多文
# 4. 直接优化排序
```

### 场景2: 用CE Loss做L_ratio

**假设**: 把比例预测当作分类任务

```python
# ❌ 错误的CE Loss使用
# 把"预测选择比例"当作"分类问题"

# 离散化比例为10个类别
# 0-10%, 10-20%, ..., 90-100%

# 实际选择比例
actual_ratio = 0.483  # 48.3%

# 转换为类别
class_label = int(actual_ratio * 10)  # = 4 (40-50%)

# CE Loss
logits = [...]  # 10个类别的logits
ce_loss = nn.CrossEntropyLoss()(logits, class_label)

# 问题:
# 1. 48.3%和49.7%应该损失相近，但都属于类别4，CE无法区分
# 2. 49.9%和50.1%很接近，但分属类别4和5，CE认为它们差距大
# 3. 离散化丢失精度
# 4. 类别边界人为定义，不合理
```

**正确的MSE Loss**:

```python
# ✅ 正确的MSE Loss使用
# 直接预测连续的比例值

actual_ratio = score_mask.mean()  # 0.483
target_ratio = 0.5

mse_loss = (actual_ratio - target_ratio) ** 2
         = (0.483 - 0.5) ** 2
         = 0.000289

# 优势:
# 1. 直接优化连续值
# 2. 48.3%和49.7%的损失差异正确反映实际差异
# 3. 无需离散化
# 4. 梯度平滑
```

---

## 损失函数选择决策树

```
任务是什么?
    │
    ├─ 预测类别 (离散) → 用 CE Loss
    │   例如: 图像分类、文本分类
    │
    ├─ 预测连续值 (回归) → 用 MSE Loss 或 MAE Loss
    │   例如: 预测房价、预测比例、预测角度
    │
    └─ 排序/检索 → 用 Ranking Loss (Triplet/Contrastive)
        例如: 图像检索、推荐系统、相似度学习

SEPS任务:
    │
    ├─ L_align: 图文检索 → Triplet Loss ✅
    │
    └─ L_ratio: 预测比例 → MSE Loss ✅
```

---

## 任务对比表

### 图像检索 vs 图像分类

| 维度 | 图像检索 (SEPS) | 图像分类 (ResNet) |
|-----|----------------|------------------|
| **任务** | 找到与query最相似的图像 | 预测图像属于哪个类别 |
| **输出** | 相似度分数 (连续) | 类别概率 (离散) |
| **候选集合** | 动态（任意图像/文本） | 固定（预定义类别） |
| **损失函数** | Triplet Loss | CE Loss |
| **评估指标** | R@1, R@5, R@10, rSum | Accuracy, Top-5 Accuracy |
| **泛化性** | ✅ 可泛化到新图文 | ❌ 只能分类已知类别 |

### 示例对比

**图像分类**:
```
输入: 一张猫的图片
输出: [0.05, 0.90, 0.05]  (狗/猫/鸟)
      └─ Softmax归一化，和=1
目标: 类别标签 = 1 (猫)
损失: CE Loss = -log(0.90) = 0.105
```

**图像检索**:
```
输入: 一张猫的图片 + 候选文本库
     ["A dog running", "A cat sleeping", "A bird flying"]
输出: [0.3, 0.9, 0.2]  (相似度，不需要和=1)
目标: "A cat sleeping" 排第一
损失: Triplet Loss = [0.2 - 0.9 + 0.3]_+ = 0
```

---

## 总结

### ✅ SEPS使用的损失函数（完整且正确）

```python
SEPSLoss = ContrastiveLoss (Triplet Loss) + RatioLoss (MSE Loss)
           └─────────┬────────┘              └──────┬──────┘
                     │                              │
              L_align (公式6)                  L_ratio (公式7)
              图文匹配排序任务                   比例预测回归任务
              不能用CE Loss!                    不能用CE Loss!
```

### 🎯 为什么不用CE Loss？

**L_align (Triplet Loss)**:
1. ✅ 任务是**排序**，不是分类
2. ✅ 输出是**相似度**，不是概率
3. ✅ 支持**一图多文**，CE不支持
4. ✅ 可**泛化到新文本**，CE只能分类固定类别

**L_ratio (MSE Loss)**:
1. ✅ 任务是预测**连续比例**，不是离散类别
2. ✅ 0.48和0.52应该损失接近，CE的离散化做不到
3. ✅ 梯度平滑，训练稳定

### 📊 SEPS任务核心

**任务**: 跨模态检索 (Cross-Modal Retrieval)
- Image-to-Text: 给图找文本
- Text-to-Image: 给文本找图

**数据集**:
- Flickr30K: 31K图像, 155K文本
- MS-COCO: 123K图像, 615K文本
- 每图5个caption

**输入**:
- Image: (B, 3, H, W)
- Sparse Text: (B, L_s) - 原始caption
- Dense Text: (B, L_d) - MLLM生成

**输出**:
- Similarity Matrix: (B_v, B_t)
- 用于排序和检索

**评估**:
- R@1, R@5, R@10 (Recall)
- rSum (综合指标)

**损失**:
- ✅ Triplet Loss (排序任务)
- ✅ MSE Loss (比例回归)
- ❌ 不用CE Loss (任务不匹配)

---

**结论**: SEPS的损失函数设计完全正确，不需要也不应该使用CE Loss！
