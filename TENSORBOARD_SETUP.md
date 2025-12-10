# TensorBoard 集成指南

## 安装 TensorBoard

```bash
# 激活 DeMo 环境
conda activate DeMo

# 安装 tensorboard
pip install tensorboard
```

## 使用方法

### 1. 训练时自动记录

TensorBoard 已集成到训练流程中，无需额外操作。训练时会自动记录：

**训练指标（每 log_period 迭代）：**
- `Train/Loss`: 训练损失
- `Train/Acc`: 训练准确率
- `Train/LR`: 学习率

**验证指标（每 eval_period 轮次）：**
- `Val/mAP`: 当前 mAP
- `Val/Rank-1/5/10`: 当前 Rank 指标
- `Val_Best/mAP`: 历史最佳 mAP
- `Val_Best/Rank-1/5/10`: 历史最佳 Rank 指标

### 2. 日志目录结构

```
logs/tensorboard/
├── DeMo_SDTPS_DGAF_ablation_20251210_143022/  # 实验ID（配置名+时间戳）
│   └── events.out.tfevents.*                   # TensorBoard 事件文件
├── baseline_20251210_150000/
│   └── events.out.tfevents.*
└── ...
```

### 3. 启动 TensorBoard

```bash
# 从项目根目录启动
tensorboard --logdir logs/tensorboard --port 6006

# 或指定特定实验
tensorboard --logdir logs/tensorboard/DeMo_SDTPS_DGAF_ablation_20251210_143022 --port 6006
```

然后在浏览器打开：`http://localhost:6006`

### 4. 多实验对比

TensorBoard 会自动识别 `logs/tensorboard/` 下的所有实验，可以在界面上选择性查看和对比。

### 5. 分布式训练

只有 rank 0 进程会记录 TensorBoard 日志，避免重复。

### 6. 与现有日志的关系

- **文本日志**（`.log` 文件）：详细的训练输出，适合查看具体信息
- **TensorBoard**：可视化监控，适合查看趋势和对比实验

两者互补，互不影响。

## 示例

### 运行训练
```bash
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml --exp_name my_experiment
```

训练开始时会显示：
```
TensorBoard logging to: logs/tensorboard/my_experiment
Start TensorBoard with: tensorboard --logdir logs/tensorboard --port 6006
```

### 查看监控
```bash
tensorboard --logdir logs/tensorboard --port 6006
```

## 未来扩展（可选）

如需记录更多指标，可以在 `engine/processor.py` 中添加：

```python
# 记录各模块的损失
writer.add_scalar('Train/SDTPS_Loss', sdtps_loss, global_step)
writer.add_scalar('Train/DGAF_Loss', dgaf_loss, global_step)
writer.add_scalar('Train/LIF_Loss', lif_loss, global_step)

# 记录参数分布（每 epoch 一次）
for name, param in model.named_parameters():
    writer.add_histogram(f'Params/{name}', param, epoch)
    if param.grad is not None:
        writer.add_histogram(f'Grads/{name}', param.grad, epoch)
```
