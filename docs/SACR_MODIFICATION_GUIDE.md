# SACR 膨胀率修改实操指南

## 目录

1. [问题识别](#问题识别)
2. [快速修复](#快速修复)
3. [详细步骤](#详细步骤)
4. [验证检查](#验证检查)
5. [性能监控](#性能监控)
6. [故障排除](#故障排除)

---

## 问题识别

### 症状检查

在修改前，确认您遇到以下问题之一：

```
[ ] 训练 loss 收敛缓慢
[ ] 最终性能低于预期
[ ] 模型在 SACR 处出现数值不稳定
[ ] 想要优化多模态特征融合
[ ] 代码注释提到 SACR 是从 UAV 场景借用的
```

### 自动诊断

```bash
# 快速诊断当前配置是否有问题
cd /home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2
python verify_sacr_config.py

# 查看诊断结果，如果显示 "✗ Error"，继续执行修复步骤
```

---

## 快速修复

### 3 分钟快速修复

```bash
# 1. 进入项目目录
cd /home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2

# 2. 备份原始配置
cp config/defaults.py config/defaults.py.bak

# 3. 查看当前第 38 行
sed -n '36,40p' config/defaults.py

# 4. 修改膨胀率（三选一）

# 方案 A：推荐（最平衡）
sed -i "38s/.*/\_C.MODEL.SACR_DILATION_RATES = [2, 3, 4]  # 优化用于 16×8 特征图/" config/defaults.py

# 或方案 B：平衡
# sed -i "38s/.*/\_C.MODEL.SACR_DILATION_RATES = [3, 5, 7]/" config/defaults.py

# 或方案 C：标准 ASPP
# sed -i "38s/.*/\_C.MODEL.SACR_DILATION_RATES = [1, 2, 4]/" config/defaults.py

# 5. 验证修改
sed -n '36,40p' config/defaults.py

# 6. 开始训练
python train_net.py --config_file configs/RGBNT201/DeMo.yml MODEL.USE_SACR True
```

---

## 详细步骤

### 步骤 1：准备工作

```bash
# 1a. 进入项目目录
cd /home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2

# 1b. 检查当前目录结构
ls -la config/
# 应该看到：defaults.py 和其他配置文件

# 1c. 检查 Python 环境
python --version  # 应该是 Python 3.8+
python -c "import torch; print(torch.__version__)"  # 应该是 1.13.1+
```

### 步骤 2：备份原始文件

```bash
# 创建备份，以防需要回退
cp config/defaults.py config/defaults.py.backup.2025-12-06

# 验证备份成功
ls -la config/defaults.py*
```

### 步骤 3：定位要修改的行

```bash
# 查看当前配置
grep -n "SACR_DILATION_RATES" config/defaults.py

# 应该输出类似：
# 38:_C.MODEL.SACR_DILATION_RATES = [6, 12, 18] # dilation rates for atrous convolutions

# 查看该行及周围内容
sed -n '35,42p' config/defaults.py
```

### 步骤 4：进行修改

#### 方法 A：使用 Python（推荐）

```python
# 创建文件 modify_sacr.py

# 读取文件
with open('config/defaults.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到并修改第 38 行（注意：行号从 0 开始）
for i, line in enumerate(lines):
    if 'SACR_DILATION_RATES' in line and '_C.MODEL' in line:
        lines[i] = '_C.MODEL.SACR_DILATION_RATES = [2, 3, 4]  # 优化用于 16×8 特征图\n'
        print(f"修改了第 {i+1} 行")
        print(f"原始: {lines[i-1].strip()}")
        print(f"现在: {lines[i].strip()}")
        break

# 写回文件
with open('config/defaults.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✓ 修改完成")
```

运行：
```bash
python modify_sacr.py
```

#### 方法 B：使用文本编辑器（Visual Studio Code）

```
1. 打开文件：config/defaults.py
2. 按 Ctrl+G 跳转到第 38 行
3. 找到：_C.MODEL.SACR_DILATION_RATES = [6, 12, 18]
4. 改为：_C.MODEL.SACR_DILATION_RATES = [2, 3, 4]
5. 保存：Ctrl+S
```

#### 方法 C：使用 sed（命令行）

```bash
# 方案 A（推荐）
sed -i '38s/\[6, 12, 18\]/[2, 3, 4]/' config/defaults.py

# 验证修改
grep "SACR_DILATION_RATES" config/defaults.py
```

### 步骤 5：验证修改

```bash
# 方法 1：直接查看
sed -n '38p' config/defaults.py
# 应该显示：_C.MODEL.SACR_DILATION_RATES = [2, 3, 4]

# 方法 2：使用 Python 加载测试
python << 'EOF'
from config.defaults import _C
print("膨胀率设置:", _C.MODEL.SACR_DILATION_RATES)
# 应该输出：膨胀率设置: [2, 3, 4]
EOF

# 方法 3：使用诊断脚本
python verify_sacr_config.py
# 应该看到改进的评估结果
```

---

## 验证检查

### 检查 1：配置加载

```bash
python -c "
from config.defaults import _C
dilation_rates = _C.MODEL.SACR_DILATION_RATES
print(f'✓ 膨胀率加载成功: {dilation_rates}')
assert dilation_rates == [2, 3, 4], f'期望 [2, 3, 4]，实际 {dilation_rates}'
print('✓ 膨胀率值正确')
"
```

### 检查 2：模型初始化

```bash
python << 'EOF'
import torch
from modeling.sacr import SACR

# 创建 SACR 模块（与实际使用相同的参数）
sacr = SACR(
    token_dim=512,
    height=16,
    width=8,
    dilation_rates=[2, 3, 4]  # 新的膨胀率
)

# 创建测试输入
x = torch.randn(2, 128, 512)

# 前向传播测试
with torch.no_grad():
    y = sacr(x)

print(f"✓ SACR 模块初始化成功")
print(f"✓ 输入形状: {x.shape}")
print(f"✓ 输出形状: {y.shape}")
print(f"✓ 参数量: {sum(p.numel() for p in sacr.parameters()):,}")
EOF
```

### 检查 3：完整模型初始化

```bash
python << 'EOF'
import torch
from config.defaults import _C
from config import cfg
from modeling.make_model import make_model

# 加载配置
cfg.merge_from_file("configs/RGBNT201/DeMo.yml")
cfg.MODEL.USE_SACR = True  # 启用 SACR

# 创建模型
model = make_model(
    cfg=cfg,
    num_classes=201,  # RGBNT201 的身份数
    camera_num=1,
    view_num=1
)

print(f"✓ 完整模型初始化成功")
print(f"✓ 模型包含 SACR 模块: {hasattr(model, 'sacr')}")

# 验证 SACR 膨胀率
if hasattr(model, 'sacr'):
    sacr_config = cfg.MODEL.SACR_DILATION_RATES
    print(f"✓ SACR 膨胀率: {sacr_config}")
EOF
```

---

## 性能监控

### 训练前的基准

修改前，记录以下信息：

```bash
# 可选：运行一个小规模的对比实验
python train_net.py \
    --config_file configs/RGBNT201/DeMo.yml \
    MODEL.USE_SACR False \  # 禁用 SACR 作为基准
    SOLVER.MAX_EPOCHS 5
```

### 修改后的训练

```bash
# 开始训练（使用新的膨胀率）
python train_net.py \
    --config_file configs/RGBNT201/DeMo.yml \
    MODEL.USE_SACR True \
    SOLVER.MAX_EPOCHS 40  # 完整训练周期

# 或使用分布式训练
python -m torch.distributed.launch --nproc_per_node=4 \
    train_net.py \
    --config_file configs/RGBNT201/DeMo.yml \
    MODEL.USE_SACR True
```

### 监控指标

在训练过程中，重点关注：

```
第 1-10 个 epoch:
  - Loss 曲线是否正常下降（不应该有跳跃）
  - 梯度是否稳定（无 NaN 或 Inf）
  - GPU 显存占用是否正常

第 10-20 个 epoch:
  - mAP 是否开始改善
  - rank-1 准确率是否提升
  - 损失是否继续单调下降

第 20-40 个 epoch:
  - 最终 mAP 是否好于预期
  - 模型是否过拟合
  - 是否需要继续训练
```

### 日志文件位置

```bash
# 训练日志通常保存在：
logs/
├── RGBNT201/
│   ├── best_model.pth
│   ├── last_model.pth
│   └── training_log.txt

# 查看实时日志
tail -f logs/RGBNT201/training_log.txt

# 或使用 tensorboard（如果配置了）
tensorboard --logdir logs/
```

---

## 故障排除

### 问题 1：修改后模型无法加载

**症状**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'config/defaults.py'
```

**解决方案**：
```bash
# 1. 检查文件是否存在
ls -la config/defaults.py

# 2. 确保在正确的目录
pwd
# 应该输出：/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2

# 3. 恢复备份
cp config/defaults.py.backup.2025-12-06 config/defaults.py

# 4. 重新进行修改
```

### 问题 2：语法错误

**症状**：
```
SyntaxError: invalid syntax
```

**解决方案**：
```bash
# 1. 检查修改后的文件语法
python -m py_compile config/defaults.py

# 2. 如果有错误，查看文件内容
sed -n '35,42p' config/defaults.py

# 3. 确保格式正确（看起来像）：
# _C.MODEL.SACR_DILATION_RATES = [2, 3, 4]

# 4. 恢复备份重新尝试
cp config/defaults.py.backup.2025-12-06 config/defaults.py
```

### 问题 3：SACR 模块不被使用

**症状**：
```
模型初始化时未看到 SACR 模块
```

**解决方案**：
```bash
# 1. 检查 SACR 是否启用
python -c "from config.defaults import _C; print(_C.MODEL.USE_SACR)"
# 应该输出：False（默认）或改为 True

# 2. 在训练命令中启用
python train_net.py \
    --config_file configs/RGBNT201/DeMo.yml \
    MODEL.USE_SACR True  # ← 确保这行存在
```

### 问题 4：性能反而下降

**症状**：
```
修改膨胀率后 mAP 反而降低 0.5% 以上
```

**故障排除步骤**：
```bash
# 1. 检查训练是否充分
# - 确保训练了足够的 epoch（至少 40）
# - 观察 loss 是否继续下降

# 2. 检查是否使用了旧的模型权重
python train_net.py \
    --config_file configs/RGBNT201/DeMo.yml \
    MODEL.USE_SACR True \
    MODEL.PRETRAIN_PATH_T ""  # 清空预训练权重，从零开始

# 3. 尝试微调学习率
python train_net.py \
    --config_file configs/RGBNT201/DeMo.yml \
    MODEL.USE_SACR True \
    SOLVER.BASE_LR 0.0001  # 降低学习率

# 4. 尝试其他膨胀率方案
python train_net.py \
    --config_file configs/RGBNT201/DeMo.yml \
    MODEL.USE_SACR True \
    MODEL.SACR_DILATION_RATES "[1, 2, 4]"  # 尝试标准 ASPP

# 5. 收集 100+ 个 batch 后再判断，不要过早下结论
```

### 问题 5：显存不足

**症状**：
```
CUDA out of memory
```

**解决方案**：
```bash
# SACR 本身不增加显存，但如果有问题：

# 1. 检查 SACR 参数量
python verify_sacr_config.py | grep "参数量"

# 2. 如果是显存问题，减小 batch size
python train_net.py \
    --config_file configs/RGBNT201/DeMo.yml \
    DATALOADER.NUM_INSTANCE 2  # 从 4 减到 2
    SOLVER.IMS_PER_BATCH 4     # 从 8 减到 4
```

---

## 对比实验模板

如果想进行科学的对比实验，使用以下模板：

```bash
#!/bin/bash

# 实验脚本：对比不同膨胀率的效果

PROJECT_ROOT="/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2"
cd $PROJECT_ROOT

# 配置
CONFIG_FILE="configs/RGBNT201/DeMo.yml"
EPOCHS=50

echo "=========================================="
echo "SACR 膨胀率对比实验"
echo "=========================================="

# 实验 1：推荐方案 [2, 3, 4]
echo ""
echo "【实验 1】推荐方案 [2, 3, 4]"
python train_net.py \
    --config_file $CONFIG_FILE \
    MODEL.USE_SACR True \
    MODEL.SACR_DILATION_RATES "[2, 3, 4]" \
    SOLVER.MAX_EPOCHS $EPOCHS \
    OUTPUT_DIR "./outputs/exp1_[2,3,4]"

# 实验 2：平衡方案 [3, 5, 7]
echo ""
echo "【实验 2】平衡方案 [3, 5, 7]"
python train_net.py \
    --config_file $CONFIG_FILE \
    MODEL.USE_SACR True \
    MODEL.SACR_DILATION_RATES "[3, 5, 7]" \
    SOLVER.MAX_EPOCHS $EPOCHS \
    OUTPUT_DIR "./outputs/exp2_[3,5,7]"

# 实验 3：标准 ASPP [1, 2, 4]
echo ""
echo "【实验 3】标准 ASPP [1, 2, 4]"
python train_net.py \
    --config_file $CONFIG_FILE \
    MODEL.USE_SACR True \
    MODEL.SACR_DILATION_RATES "[1, 2, 4]" \
    SOLVER.MAX_EPOCHS $EPOCHS \
    OUTPUT_DIR "./outputs/exp3_[1,2,4]"

# 实验 4：原始方案 [6, 12, 18]（参考）
echo ""
echo "【实验 4】原始方案 [6, 12, 18]（对照组）"
python train_net.py \
    --config_file $CONFIG_FILE \
    MODEL.USE_SACR True \
    MODEL.SACR_DILATION_RATES "[6, 12, 18]" \
    SOLVER.MAX_EPOCHS $EPOCHS \
    OUTPUT_DIR "./outputs/exp4_[6,12,18]"

echo ""
echo "=========================================="
echo "所有实验完成"
echo "=========================================="
echo ""
echo "结果对比："
echo "exp1_[2,3,4] vs exp2_[3,5,7] vs exp3_[1,2,4] vs exp4_[6,12,18]"
echo ""
```

保存为 `compare_experiments.sh` 并运行：
```bash
bash compare_experiments.sh
```

---

## 回退计划

如果修改导致严重问题，快速回退：

```bash
# 方法 1：恢复备份
cp config/defaults.py.backup.2025-12-06 config/defaults.py

# 方法 2：手动恢复
sed -i "38s/.*/\_C.MODEL.SACR_DILATION_RATES = [6, 12, 18]/" config/defaults.py

# 方法 3：使用 git（如果有版本控制）
git checkout config/defaults.py

# 验证恢复
grep "SACR_DILATION_RATES" config/defaults.py
```

---

## 总结检查清单

修改前：
- [ ] 阅读本指南
- [ ] 运行诊断脚本 `verify_sacr_config.py`
- [ ] 备份原始配置文件
- [ ] 确认修改位置（第 38 行）

修改中：
- [ ] 选择修改方法（Python / 编辑器 / sed）
- [ ] 执行修改
- [ ] 验证修改成功

修改后：
- [ ] 运行诊断脚本验证
- [ ] 测试模型初始化
- [ ] 开始训练

训练中：
- [ ] 监控前 10 个 epoch 的 loss
- [ ] 监控 mAP 和 rank-1 指标
- [ ] 收集完整数据后再评估

---

**版本**: 1.0
**最后更新**: 2025-12-06
**适用项目**: DeMo 多模态重识别框架
**预期收益**: +0.5-1.5% mAP 提升
