# 顺序训练脚本使用指南

## 场景

您只有一块 GPU，想要按顺序运行多个实验配置，每个实验完成后自动开始下一个。

## 解决方案

提供了 3 种脚本：

---

## 方案1：简单等待脚本（推荐）⭐

**脚本**: `wait_and_run.sh`

**用法**：
```bash
# 在后台运行，等待当前训练完成后自动开始下一个
nohup bash wait_and_run.sh configs/RGBNT201/DeMo_SDTPS.yml > wait_run.log 2>&1 &
```

**特点**：
- ✅ 简单易用
- ✅ 自动检测当前训练是否完成
- ✅ 完成后自动开始指定的配置
- ✅ 后台运行，不影响当前训练

**示例**：
```bash
# 现在正在运行: DeMo_SACR_SDTPS.yml
# 想让它完成后自动运行 DeMo_SDTPS.yml

# 在另一个终端执行：
nohup bash wait_and_run.sh configs/RGBNT201/DeMo_SDTPS.yml > wait_sdtps.log 2>&1 &

# 脚本会：
# 1. 检测 train_net.py 进程
# 2. 等待进程结束
# 3. 自动运行 DeMo_SDTPS.yml
```

---

## 方案2：批量顺序训练（多个实验）

**脚本**: `run_sequential_experiments.py` 或 `run_sequential_experiments.sh`

**用法**：

### Python 版本（推荐）

```bash
# 编辑脚本，修改实验列表
vim run_sequential_experiments.py

# EXPERIMENTS = [
#     {'name': 'exp1', 'config': 'configs/xxx.yml', ...},
#     {'name': 'exp2', 'config': 'configs/yyy.yml', ...},
# ]

# 运行
python run_sequential_experiments.py
```

### Bash 版本

```bash
# 编辑脚本
vim run_sequential_experiments.sh

# experiments=(
#     "exp1:configs/xxx.yml"
#     "exp2:configs/yyy.yml"
# )

# 运行
bash run_sequential_experiments.sh
```

**特点**：
- ✅ 自动运行多个配置
- ✅ 记录每个实验的时间和状态
- ✅ 失败时可选择继续/停止
- ✅ 生成汇总报告

**示例**：

当前默认配置的实验序列：
```
1. SACR_SDTPS (完整版)
   ↓ 完成后
2. SDTPS_only (只用 SDTPS)
   ↓ 完成后
3. Original_DeMo (HDM+ATM baseline)
```

---

## 方案3：手动链式运行

**最简单但需要修改命令**：

```bash
# 使用 && 连接多个命令
python train_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml && \
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml && \
python train_net.py --config_file configs/RGBNT201/DeMo.yml

# 或者后台运行
nohup bash -c "
python train_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml && \
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml && \
python train_net.py --config_file configs/RGBNT201/DeMo.yml
" > all_experiments.log 2>&1 &
```

**特点**：
- ✅ 最简单
- ❌ 中间失败会中断
- ❌ 没有进度提示

---

## 📋 推荐使用方式

### 场景1：当前训练完成后运行一个实验

**使用**: `wait_and_run.sh`

```bash
# 在新终端执行（不影响当前训练）
nohup bash wait_and_run.sh configs/RGBNT201/DeMo_SDTPS.yml > next_exp.log 2>&1 &

# 查看等待状态
tail -f next_exp.log

# 或者查看进程
ps aux | grep wait_and_run
```

### 场景2：计划运行多个实验

**使用**: `run_sequential_experiments.py`

```bash
# 1. 编辑实验列表
vim run_sequential_experiments.py

# 2. 运行（会等待确认）
python run_sequential_experiments.py

# 或者后台运行（自动确认需修改代码）
nohup python run_sequential_experiments.py < /dev/null > seq_exp.log 2>&1 &
```

### 场景3：临时快速链式运行

```bash
# 直接用 && 连接
python train_net.py --config_file cfg1.yml && \
python train_net.py --config_file cfg2.yml
```

---

## 📁 日志管理

所有脚本都会在 `experiment_logs/` 目录下保存日志：

```
experiment_logs/
  ├── sequential_run_20251205_143000.log  # 主日志
  ├── SACR_SDTPS_20251205_143000.log      # 实验1日志
  ├── SDTPS_only_20251205_163000.log      # 实验2日志
  ├── Original_DeMo_20251205_183000.log   # 实验3日志
  └── summary_20251205_143000.txt         # 汇总报告
```

## 🔧 自定义实验

### 修改 run_sequential_experiments.py 的实验列表

```python
EXPERIMENTS = [
    {
        'name': '你的实验名',
        'config': 'configs/你的配置.yml',
        'description': '实验描述'
    },
    # 添加更多...
]
```

### 修改 run_sequential_experiments.sh 的实验列表

```bash
experiments=(
    "实验名:configs/配置文件.yml"
    "实验名2:configs/配置文件2.yml"
)
```

---

## ⚠️ 注意事项

1. **确保有足够的磁盘空间**
   - 每个实验会保存 checkpoint（每个约几百MB）
   - 日志文件也会占用空间

2. **及时清理旧的 checkpoints**
   ```bash
   rm ../DeMo_*.pth  # 清理旧模型
   ```

3. **监控训练状态**
   ```bash
   # 查看当前训练
   tail -f experiment_logs/最新日志文件.log

   # 查看 GPU 使用
   watch -n 1 nvidia-smi
   ```

4. **中断恢复**
   - 脚本支持 Ctrl+C 中断
   - 可以修改脚本从中间某个实验开始

---

## 🚀 快速开始

### 当前场景：等待 SACR_SDTPS 完成后运行 SDTPS_only

**方法1（推荐）**：
```bash
nohup bash wait_and_run.sh configs/RGBNT201/DeMo_SDTPS.yml > wait_sdtps.log 2>&1 &
```

**方法2**：
```bash
# 修改 run_sequential_experiments.py，只保留后两个实验
# 然后运行
python run_sequential_experiments.py
```

---

## 📊 查看进度

```bash
# 查看等待脚本状态
tail -f wait_sdtps.log

# 查看所有训练进程
ps aux | grep train_net

# 查看GPU占用
nvidia-smi

# 查看实验日志目录
ls -lth experiment_logs/
```

---

**现在您可以让脚本在后台等待，当前训练完成后自动开始下一个！** 🚀
