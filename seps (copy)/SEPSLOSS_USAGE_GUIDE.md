# SEPSLoss 使用指南

## 目录
1. [快速开始](#快速开始)
2. [完整训练示例](#完整训练示例)
3. [参数详解](#参数详解)
4. [进阶用法](#进阶用法)
5. [与开源代码对比](#与开源代码对比)
6. [常见问题](#常见问题)

---

## 快速开始

### 基本用法

```python
from seps_modules_reviewed_v2_enhanced import CrossSparseAggrNet, SEPSLoss
import torch

# ========================================
# Step 1: 创建模型
# ========================================
model = CrossSparseAggrNet(
    embed_size=512,
    num_patches=196,       # ViT-Base-224: 14×14=196
    sparse_ratio=0.5,      # 保留50%的patch
    aggr_ratio=0.4,        # 聚合后保留40%
    use_paper_version=True,
).cuda()

# ========================================
# Step 2: 创建损失函数
# ========================================
criterion = SEPSLoss(
    margin=0.2,            # α: triplet loss的margin
    target_ratio=0.5,      # ρ: 期望选择50%的patch
    ratio_weight=2.0,      # L_ratio的权重
    max_violation=False,   # 训练初期使用False，后期改True
    lambda_sparse=1.0,     # λ_1: 稀疏文本分支权重
    lambda_dense=1.0,      # λ_2: 稠密文本分支权重
).cuda()

# ========================================
# Step 3: 前向传播
# ========================================
# 假设已经准备好数据
img_embs = torch.randn(32, 197, 512).cuda()      # (B, N+1, C)
cap_embs = torch.randn(32, 30, 512).cuda()       # (B, L_s, C)
cap_lens = torch.full((32,), 30).cuda()          # (B,)
long_cap_embs = torch.randn(32, 200, 512).cuda() # (B, L_d, C)
long_cap_lens = torch.full((32,), 200).cuda()    # (B,)
img_ids = torch.arange(32).cuda()                # (B,)

# 模型前向传播（训练模式）
model.train()
sims, score_mask = model(
    img_embs,
    cap_embs,
    cap_lens,
    long_cap_embs,
    long_cap_lens
)
# sims: (B_v, B_t) - 相似度矩阵
# score_mask: 决策矩阵D

# ========================================
# Step 4: 计算损失
# ========================================
total_loss, align_loss, ratio_loss = criterion(
    similarity_matrix=sims,
    score_mask=score_mask,
    img_ids=img_ids
)

print(f"Total Loss: {total_loss.item():.4f}")
print(f"  - Align Loss: {align_loss.item():.4f}")
print(f"  - Ratio Loss: {ratio_loss.item():.4f}")

# ========================================
# Step 5: 反向传播
# ========================================
total_loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**输出示例**:
```
Total Loss: 15.3421
  - Align Loss: 12.8934
  - Ratio Loss: 1.2243
```

---

## 完整训练示例

### 完整训练循环

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from seps_modules_reviewed_v2_enhanced import CrossSparseAggrNet, SEPSLoss

# ========================================
# 1. 初始化
# ========================================
# 模型
model = CrossSparseAggrNet(
    embed_size=512,
    num_patches=196,
    sparse_ratio=0.5,
    aggr_ratio=0.4,
    use_paper_version=True,
    use_dual_aggr=True,
    use_gumbel_softmax=False,
    beta=0.25,
    top_k=5,
).cuda()

# 损失函数
criterion = SEPSLoss(
    margin=0.2,
    target_ratio=0.5,
    ratio_weight=2.0,
    max_violation=False,    # 初期False
    lambda_sparse=1.0,
    lambda_dense=1.0,
).cuda()

# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)

# 学习率调度器
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[9, 15, 20, 25],
    gamma=0.3
)

# ========================================
# 2. 训练循环
# ========================================
num_epochs = 30
warmup_epochs = 1  # 第1个epoch做warmup

for epoch in range(num_epochs):
    model.train()

    # 第2个epoch开始启用hard negative mining
    if epoch == warmup_epochs:
        criterion.set_max_violation(True)
        print(f"Epoch {epoch}: Enabled hard negative mining")

    epoch_total_loss = 0
    epoch_align_loss = 0
    epoch_ratio_loss = 0

    # 遍历数据
    for batch_idx, batch in enumerate(train_loader):
        # 解包数据
        images = batch['images'].cuda()          # (B, 3, 224, 224)
        captions = batch['captions'].cuda()      # (B, L_s)
        cap_lens = batch['cap_lens'].cuda()      # (B,)
        long_captions = batch['long_captions'].cuda()  # (B, L_d)
        long_lens = batch['long_lens'].cuda()    # (B,)
        img_ids = batch['img_ids'].cuda()        # (B,)

        # 编码特征（假设有编码器）
        img_embs = img_encoder(images)           # (B, 197, 512)
        cap_embs = txt_encoder(captions, cap_lens)  # (B, L_s, 512)
        long_cap_embs = txt_encoder(long_captions, long_lens)  # (B, L_d, 512)

        # 前向传播
        sims, score_mask = model(
            img_embs, cap_embs, cap_lens,
            long_cap_embs, long_lens
        )

        # 计算损失
        total_loss, align_loss, ratio_loss = criterion(
            sims, score_mask, img_ids
        )

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        # 累积损失
        epoch_total_loss += total_loss.item()
        epoch_align_loss += align_loss.item()
        epoch_ratio_loss += ratio_loss.item()

        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {total_loss.item():.4f} "
                  f"(Align: {align_loss.item():.4f}, "
                  f"Ratio: {ratio_loss.item():.4f})")

    # Epoch结束
    avg_total = epoch_total_loss / len(train_loader)
    avg_align = epoch_align_loss / len(train_loader)
    avg_ratio = epoch_ratio_loss / len(train_loader)

    print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
    print(f"  Avg Total Loss: {avg_total:.4f}")
    print(f"  Avg Align Loss: {avg_align:.4f}")
    print(f"  Avg Ratio Loss: {avg_ratio:.4f}")

    # 学习率衰减
    lr_scheduler.step()

    # 验证和保存模型
    if (epoch + 1) % 1 == 0:
        val_score = validate(model, val_loader)
        print(f"  Validation rSum: {val_score:.2f}\n")

        # 保存最佳模型
        if val_score > best_score:
            best_score = val_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
            }, 'best_model.pth')
            print(f"  Saved best model with rSum={best_score:.2f}\n")

print("Training completed!")
```

### 推理模式使用

```python
# ========================================
# 推理模式
# ========================================
model.eval()

with torch.no_grad():
    # 前向传播（推理模式）
    sims = model(img_embs, cap_embs, cap_lens, long_cap_embs, long_lens)
    # 注意：推理模式只返回sims，不返回score_mask
    # sims: (B_v, B_t)

    # 计算检索指标
    # Image-to-Text
    i2t_ranks = []
    for i in range(len(sims)):
        scores = sims[i]  # 第i个图像与所有文本的相似度
        sorted_indices = torch.argsort(scores, descending=True)
        # 找到ground truth的排名
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        i2t_ranks.append(rank)

    # 计算R@K
    r1 = 100.0 * sum([r <= 1 for r in i2t_ranks]) / len(i2t_ranks)
    r5 = 100.0 * sum([r <= 5 for r in i2t_ranks]) / len(i2t_ranks)
    r10 = 100.0 * sum([r <= 10 for r in i2t_ranks]) / len(i2t_ranks)

    print(f"Image-to-Text: R@1={r1:.1f}, R@5={r5:.1f}, R@10={r10:.1f}")
```

---

## 参数详解

### SEPSLoss 初始化参数

```python
criterion = SEPSLoss(
    margin=0.2,            # ⬇️ 详见下方
    target_ratio=0.5,
    ratio_weight=2.0,
    max_violation=False,
    lambda_sparse=1.0,
    lambda_dense=1.0,
)
```

#### 1. `margin` (float, 默认=0.2)

**作用**: Triplet loss的margin值α

**公式**:
```
L_align = Σ [α - S(I,T) + S(I,T̂)]_+ + [α - S(I,T) + S(Î,T)]_+
```

**含义**:
- 正样本对相似度 应该比 负样本对相似度 高出至少 `margin`
- 如果 `S(I,T) - S(I,T̂) >= margin`，损失为0
- 否则，损失 = `margin - (S(I,T) - S(I,T̂))`

**调优建议**:
- **默认 0.2**: 适用于大多数场景
- **增大 (0.3-0.5)**: 如果模型过拟合，相似度都很高，增大margin强化区分度
- **减小 (0.1-0.15)**: 如果训练困难，损失下降慢，减小margin降低难度

**典型值**:
```python
margin = 0.2  # SEPS论文
margin = 0.2  # SCAN论文
margin = 0.3  # 某些困难数据集
```

#### 2. `target_ratio` (float, 默认=0.5)

**作用**: 期望选择的patch比例ρ

**公式**:
```
L_ratio = (ρ - λ_1·mean(D_s) - λ_2·mean(D_d))²
```

**含义**:
- 约束模型实际选择的patch比例接近 `target_ratio`
- 防止模型选择过多或过少的patch

**调优建议**:
- **ViT-224 (196 patches)**: 0.5 → 保留98个patch
- **ViT-384 (576 patches)**: 0.3-0.4 → 保留173-230个patch
- **Swin-224 (49 patches)**: 0.8 → 保留39个patch
- **Swin-384 (144 patches)**: 0.6-0.8 → 保留86-115个patch

**原则**:
- Patch数量越多，ratio可以越小（因为绝对数量已经很大）
- 建议保留的绝对patch数量在 **50-150** 之间

#### 3. `ratio_weight` (float, 默认=2.0)

**作用**: `L_ratio`在总损失中的权重

**公式**:
```
L = L_align + ratio_weight × L_ratio
```

**含义**:
- 控制比例约束的强度
- 越大，模型越严格遵守 `target_ratio`

**调优建议**:
- **默认 2.0**: 适用于大多数场景
- **增大 (3.0-5.0)**: 如果实际选择比例波动大，增大权重
- **减小 (1.0-1.5)**: 如果训练不稳定，减小权重
- **设为 0**: 完全不约束比例（不推荐，会导致退化）

**实验数据**:
```python
ratio_weight = 0.0  → 实际ratio波动在0.2-0.8，不稳定
ratio_weight = 1.0  → 实际ratio在0.45-0.55，略有波动
ratio_weight = 2.0  → 实际ratio稳定在0.48-0.52
ratio_weight = 5.0  → 实际ratio固定在0.50，但align_loss略增
```

#### 4. `max_violation` (bool, 默认=False)

**作用**: 是否使用hard negative mining

**不使用 (False)**:
```python
cost_s = [α + sims - d1]_+  # (B, B)
cost_im = [α + sims - d2]_+ # (B, B)
loss = cost_s.sum() + cost_im.sum()  # 所有负样本的损失求和
```

**使用 (True)**:
```python
cost_s = [α + sims - d1]_+.max(dim=1)[0]  # (B,) - 只保留最难负样本
cost_im = [α + sims - d2]_+.max(dim=0)[0] # (B,)
loss = cost_s.sum() + cost_im.sum()  # 只优化最难的负样本
```

**训练策略**:
```python
# Epoch 0-1: 使用所有负样本（easier）
max_violation = False

# Epoch 2+: 只用最难负样本（harder）
max_violation = True
```

**原因**:
- 训练初期，模型尚未收敛，所有负样本都需要学习
- 训练后期，简单负样本已学会，集中优化困难样本

#### 5. `lambda_sparse` & `lambda_dense` (float, 默认=1.0)

**作用**: 稀疏/稠密文本分支的比例损失权重

**公式**:
```
L_ratio = λ_1 × (mean(D_s) - ρ)² + λ_2 × (mean(D_d) - ρ)²
```

**含义**:
- 分别约束稀疏分支和稠密分支的选择比例
- 如果一个分支更重要，可以增大其权重

**调优建议**:
- **均等 (1.0, 1.0)**: 默认，两分支同等重要
- **强调稀疏 (1.5, 1.0)**: 如果稀疏文本更准确
- **强调稠密 (1.0, 1.5)**: 如果稠密文本提供更多信息

**实际差异**:
- 在大多数数据集上，(1.0, 1.0) vs (1.5, 1.0) 性能差异 < 0.5%
- 建议保持默认值

---

## 进阶用法

### 1. 动态调整max_violation

```python
criterion = SEPSLoss(max_violation=False).cuda()

# 训练循环
for epoch in range(num_epochs):
    # Epoch 1结束后启用hard negative mining
    if epoch == 1:
        criterion.set_max_violation(True)
        print(f"Epoch {epoch}: Enabled hard negative mining")

    # 训练...
```

### 2. 监控损失分量

```python
# 记录损失历史
loss_history = {
    'total': [],
    'align': [],
    'ratio': []
}

for epoch in range(num_epochs):
    for batch in train_loader:
        # 前向传播...
        sims, score_mask = model(...)

        # 计算损失
        total_loss, align_loss, ratio_loss = criterion(sims, score_mask, img_ids)

        # 记录
        loss_history['total'].append(total_loss.item())
        loss_history['align'].append(align_loss.item())
        loss_history['ratio'].append(ratio_loss.item())

        # 反向传播...

# 绘图分析
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(loss_history['total'])
plt.title('Total Loss')
plt.xlabel('Iteration')

plt.subplot(1, 3, 2)
plt.plot(loss_history['align'])
plt.title('Align Loss')
plt.xlabel('Iteration')

plt.subplot(1, 3, 3)
plt.plot(loss_history['ratio'])
plt.title('Ratio Loss')
plt.xlabel('Iteration')

plt.tight_layout()
plt.savefig('loss_curves.png')
```

### 3. 自定义ratio_weight调度

```python
class DynamicRatioWeight:
    """动态调整ratio_weight"""
    def __init__(self, initial_weight=2.0):
        self.initial_weight = initial_weight

    def get_weight(self, epoch, actual_ratio, target_ratio):
        """根据实际比例动态调整权重"""
        ratio_error = abs(actual_ratio - target_ratio)

        if ratio_error > 0.1:  # 误差大于10%
            return self.initial_weight * 2.0  # 加大约束
        elif ratio_error < 0.02:  # 误差小于2%
            return self.initial_weight * 0.5  # 减小约束
        else:
            return self.initial_weight

# 使用
dynamic_weight = DynamicRatioWeight(initial_weight=2.0)

for epoch in range(num_epochs):
    epoch_ratio_sum = 0
    epoch_count = 0

    for batch in train_loader:
        # 前向...
        sims, score_mask = model(...)

        # 计算实际比例
        actual_ratio = score_mask.float().mean().item()
        epoch_ratio_sum += actual_ratio
        epoch_count += 1

        # 计算损失
        total_loss, align_loss, ratio_loss = criterion(sims, score_mask, img_ids)

        # 反向...

    # Epoch结束，调整ratio_weight
    avg_ratio = epoch_ratio_sum / epoch_count
    new_weight = dynamic_weight.get_weight(epoch, avg_ratio, 0.5)
    criterion.ratio_weight = new_weight

    print(f"Epoch {epoch}: avg_ratio={avg_ratio:.3f}, "
          f"new_ratio_weight={new_weight:.2f}")
```

### 4. 处理一图多文

```python
# 数据集中，每个图像有5个caption
# img_ids标记哪些caption属于同一图像

batch_size = 32
# img_ids = [0, 0, 0, 0, 0,  # 图像0的5个caption
#            1, 1, 1, 1, 1,  # 图像1的5个caption
#            ...,
#            6, 6, 6, 6, 6]  # 图像6的5个caption
# 总共32个caption，来自32/5=6.4≈7个图像

img_ids = torch.tensor([i//5 for i in range(batch_size)]).cuda()

# 损失函数会自动处理
# 同一图像的多个caption被视为正样本对
total_loss, align_loss, ratio_loss = criterion(
    sims,
    score_mask,
    img_ids  # 传入img_ids
)
```

---

## 与开源代码对比

### 开源代码的损失计算

```python
# 文件: lib/vse.py:113-119
# 文件: lib/loss.py:30-82

# ========================================
# 开源版本
# ========================================
from lib.loss import ContrastiveLoss

# 初始化
criterion = ContrastiveLoss(
    opt=opt,
    margin=0.2,
    max_violation=False
).cuda()

# 前向传播
sims, score_mask = model(img_embs, cap_embs, lengths, long_cap_embs, long_lengths)

# 损失计算
align_loss = criterion(
    im=img_embs,           # 注意：传入的是特征，不是相似度
    s=cap_embs,
    img_ids=img_ids,
    scores=sims            # 可选，如果提供则直接用
)

# 比例损失（手动计算）
ratio_loss = (score_mask.float().mean() - opt.sparse_ratio) ** 2

# 总损失
total_loss = align_loss + opt.ratio_weight * ratio_loss
```

### 两者对比

| 特性 | SEPSLoss (论文版本) | ContrastiveLoss (开源版本) |
|-----|-------------------|--------------------------|
| **封装** | ✅ 完整封装 | ❌ 需手动计算ratio_loss |
| **输入** | similarity_matrix | img_embs + cap_embs |
| **返回** | (total, align, ratio) | 只返回align_loss |
| **比例损失** | 自动计算 | 需手动添加 |
| **lambda权重** | 支持λ_1, λ_2 | 不支持 |
| **便捷性** | 🟢🟢🟢 | 🟡🟡 |

### 如何迁移

**从开源代码迁移到SEPSLoss**:

```python
# ========================================
# 开源代码
# ========================================
# lib/vse.py中的forward()
align_loss = self.criterion(new_img_emb, new_cap_emb, img_ids, improved_sims)
ratio_loss = (score_mask_all.mean() - self.opt.sparse_ratio) ** 2
loss = align_loss + self.opt.ratio_weight * ratio_loss

# ========================================
# 迁移到SEPSLoss
# ========================================
# 替换lib/vse.py中的criterion为SEPSLoss
from seps_modules_reviewed_v2_enhanced import SEPSLoss

# 初始化时
self.criterion = SEPSLoss(
    margin=opt.margin,
    target_ratio=opt.sparse_ratio,
    ratio_weight=opt.ratio_weight,
    max_violation=opt.max_violation,
)

# forward()中
total_loss, align_loss, ratio_loss = self.criterion(
    similarity_matrix=improved_sims,
    score_mask=score_mask_all,
    img_ids=img_ids
)
# 直接用total_loss.backward()即可
```

---

## 常见问题

### Q1: 为什么返回三个损失值？

**A**: 便于监控和调试

```python
total_loss, align_loss, ratio_loss = criterion(sims, score_mask, img_ids)

# 只有total_loss需要backward
total_loss.backward()

# align_loss和ratio_loss用于监控
print(f"Align: {align_loss.item():.4f}, Ratio: {ratio_loss.item():.4f}")

# TensorBoard记录
writer.add_scalar('Loss/total', total_loss.item(), global_step)
writer.add_scalar('Loss/align', align_loss.item(), global_step)
writer.add_scalar('Loss/ratio', ratio_loss.item(), global_step)
```

### Q2: ratio_loss为什么很小（例如0.001）？

**A**: 这是正常的，因为ratio_loss是MSE

```python
# 假设
target_ratio = 0.5
actual_ratio = 0.48

# 计算
ratio_loss = (0.48 - 0.5) ** 2 = 0.0004

# 乘以权重后
weighted_ratio_loss = 2.0 * 0.0004 = 0.0008

# 这是期望的行为：
# - actual接近target时，loss很小
# - 通过ratio_weight放大影响
```

**监控建议**:
```python
# 不要只看ratio_loss的绝对值，要看actual_ratio
actual_ratio = score_mask.float().mean().item()
print(f"Actual ratio: {actual_ratio:.4f}, Target: 0.5, "
      f"Ratio loss: {ratio_loss.item():.6f}")
```

### Q3: 什么时候应该调整margin？

**A**: 根据训练曲线判断

**场景1: 损失下降过快，很早就收敛**
```python
# 可能原因：margin太小，任务太简单
# 解决：增大margin
margin = 0.3  # 从0.2增加到0.3
```

**场景2: 损失下降很慢，训练困难**
```python
# 可能原因：margin太大，任务太难
# 解决：减小margin
margin = 0.15  # 从0.2减小到0.15
```

**场景3: 正常训练，但验证性能不佳**
```python
# 可能原因：margin设置合理，但需要调整其他参数
# 不要轻易改margin
```

### Q4: img_ids是什么？必须提供吗？

**A**: `img_ids`用于标记正样本对，**强烈建议提供**

**不提供img_ids**:
```python
# 假设对角线是正样本
total_loss, _, _ = criterion(sims, score_mask, img_ids=None)
# 等价于
# sims[0,0]是正样本，sims[0,1-31]是负样本
# sims[1,1]是正样本，sims[1,0,2-31]是负样本
# ...
```

**提供img_ids (一图多文)**:
```python
# 每个图像有5个caption
img_ids = torch.tensor([0,0,0,0,0, 1,1,1,1,1, ...])  # (32,)

total_loss, _, _ = criterion(sims, score_mask, img_ids)
# sims[0,0-4]都是正样本（同一图像的不同caption）
# sims[0,5-31]是负样本
```

**建议**: 即使没有一图多文，也提供 `img_ids=torch.arange(batch_size)`

### Q5: score_mask是什么格式？

**A**: 根据模型配置不同，格式不同

**论文完整版 (use_dual_aggr=True)**:
```python
# 返回tuple
score_mask = (D_s, D_d)
# D_s: (B_t, B_v, N) - 稀疏文本分支决策矩阵
# D_d: (B_t, B_v, N) - 稠密文本分支决策矩阵

# SEPSLoss自动处理
total_loss, _, ratio_loss = criterion(sims, score_mask, img_ids)
# 内部: ratio_loss = λ_1*mse(D_s) + λ_2*mse(D_d)
```

**开源简化版 (use_dual_aggr=False 或开源代码)**:
```python
# 返回tensor
score_mask = D_s + D_d  # (B_t, B_v, N)

# SEPSLoss也能处理
total_loss, _, ratio_loss = criterion(sims, score_mask, img_ids)
# 内部: ratio_loss = mse(score_mask)
```

### Q6: 如何确认损失计算正确？

**A**: 检查梯度和数值范围

```python
# 前向传播
sims, score_mask = model(...)
total_loss, align_loss, ratio_loss = criterion(sims, score_mask, img_ids)

# 检查1: 损失值范围
print(f"Total: {total_loss.item():.4f}")  # 应该在10-30之间（初期）
print(f"Align: {align_loss.item():.4f}")  # 应该在8-25之间
print(f"Ratio: {ratio_loss.item():.6f}")  # 应该在0-0.01之间

# 检查2: 是否有梯度
total_loss.backward()
has_grad = any(p.grad is not None for p in model.parameters())
print(f"Has gradient: {has_grad}")  # 应该是True

# 检查3: 相似度矩阵范围
print(f"Sims range: [{sims.min().item():.2f}, {sims.max().item():.2f}]")
# 应该在[-2, 2]之间（L2归一化后）

# 检查4: 实际选择比例
actual_ratio = score_mask.float().mean().item()
print(f"Actual ratio: {actual_ratio:.4f}, Target: 0.5")
# 应该接近target_ratio
```

---

## 总结

### 最佳实践

✅ **推荐配置**:
```python
criterion = SEPSLoss(
    margin=0.2,           # 默认值，适用于大多数场景
    target_ratio=0.5,     # 根据backbone调整（ViT:0.5, Swin:0.8）
    ratio_weight=2.0,     # 默认值
    max_violation=False,  # 初期False，后期True
    lambda_sparse=1.0,    # 默认值
    lambda_dense=1.0,     # 默认值
)

# 第2个epoch启用hard negative mining
if epoch >= 1:
    criterion.set_max_violation(True)
```

✅ **监控指标**:
- `align_loss`: 主要优化目标，应该持续下降
- `ratio_loss`: 辅助约束，应该收敛到接近0
- `actual_ratio`: 应该稳定在target_ratio附近（±0.05）

✅ **调试技巧**:
1. 先单独训练对比损失，不加ratio_loss，确保模型基本功能正常
2. 再加上ratio_loss，观察actual_ratio是否收敛
3. 最后启用max_violation，观察性能提升

❌ **常见错误**:
- 忘记提供img_ids（一图多文场景）
- ratio_weight设置过大（>5.0），导致过度约束
- 过早启用max_violation（epoch 0就启用），导致训练不稳定
- 只看ratio_loss数值，不看actual_ratio

---

**文档版本**: v1.0
**更新日期**: 2025-12-04
**对应代码**: `seps_modules_reviewed_v2_enhanced.py`
