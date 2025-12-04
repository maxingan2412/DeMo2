# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 项目概览

**DeMo**（Decoupled Feature-Based Mixture of Experts，基于解耦特征的专家混合模型）是一个多模态目标重识别框架，已被 AAAI 2025 接收。该系统处理 RGB、NIR（近红外）和 TIR（热红外）三种模态，用于行人和车辆重识别，在缺失模态条件下也具有鲁棒的性能。

## 核心架构组件

### 核心模型架构 (modeling/)

DeMo 模型由三个主要组件构成：

1. **PIFE（Patch-Integrated Feature Extractor，块集成特征提取器）**：使用 Vision Transformer 骨干网络（ViT-B 或基于 CLIP）进行多粒度特征提取。骨干网络分别处理每个模态，并输出块级特征（`*_cash`）和全局特征（`*_global`）。

2. **HDM（Hierarchical Decoupling Module，层次解耦模块）**（`modeling/moe/AttnMOE.py:GeneralFusion.forward_HDM`）：将特征解耦为 7 个组件：
   - 3 个模态特定特征（RGB特定、NIR特定、TIR特定）
   - 3 个双模态共享特征（RGB-NIR、RGB-TIR、NIR-TIR）
   - 1 个三模态共享特征（RGB-NIR-TIR）

   使用可学习的查询令牌和交叉注意力机制来提取这些解耦表示。

3. **ATMoE（Attention-Triggered Mixture of Experts，注意力触发的专家混合）**（`modeling/moe/AttnMOE.py:MoM`）：使用基于多头注意力的门控机制动态加权和组合 7 个解耦特征。门控网络使用拼接特征（查询）和堆叠特征（键）之间的交叉注意力来计算重要性权重。

### 模型变体

模型支持两种训练模式，由 `cfg.MODEL.DIRECT` 控制：
- `DIRECT=1`：直接拼接所有模态特征（3 * feat_dim）
- `DIRECT=0`：为每个模态使用独立的分类器

当启用 `cfg.MODEL.HDM` 或 `cfg.MODEL.ATM` 时，这两种模式都可以与 HDM/ATMoE 特征结合使用。

### 配置系统

使用 YACS 进行配置管理（`config/defaults.py`）。所有超参数在 `configs/{DATASET}/DeMo.yml` 的 YAML 文件中定义。主要配置部分：
- `MODEL`：架构设置（骨干网络类型、损失权重、HDM/ATM 标志）
- `INPUT`：图像尺寸和数据增强参数
- `DATALOADER`：批大小、采样器类型、工作线程数
- `SOLVER`：优化器、学习率、训练轮数、调度器
- `TEST`：评估设置、重排序、缺失模态模拟

### 数据集结构 (data/datasets/)

支持三个主要数据集：
- **RGBNT201**：带 RGB、NIR、TIR 模态的行人重识别（201 个身份）
- **RGBNT100**：带 RGB、NIR、TIR 模态的车辆重识别（100 个身份）
- **MSVR310**：多光谱车辆重识别（310 个身份）

`data/datasets/` 中的每个数据集加载器都继承自 `BaseDataSet`，并实现标准格式返回：
- `train`：训练集的 (img_path, pid, camid) 元组列表
- `query`/`gallery`：用于评估的列表

`make_dataloader.py` 处理多模态数据加载，为每个样本创建包含键 `{'RGB', 'NI', 'TI'}` 的字典。

### 损失函数 (layers/)

框架使用组合损失（`layers/make_loss.py`）：
- **ID Loss（身份损失）**：带标签平滑的交叉熵损失用于分类
- **Triplet Loss（三元组损失）**：可配置边界的度量学习损失
- **Center Loss（中心损失）**：可选，用于学习类中心
- **MoE 正则化损失**：当启用 ATM 时从 GeneralFusion 返回

当同时启用直接拼接特征和 MoE 特征时，损失计算分别进行。

### 训练流程 (engine/processor.py)

`do_train` 函数处理：
1. 使用直接特征和 MoE 特征的混合训练（当启用 HDM/ATM 时）
2. 每 `SOLVER.EVAL_PERIOD` 轮进行一次评估
3. 每 `SOLVER.CHECKPOINT_PERIOD` 轮保存一次模型检查点
4. 当 `MODEL.DIST_TRAIN=True` 时通过 `torch.distributed` 支持分布式训练

训练时模型前向传播根据配置返回不同的输出：
- 使用 HDM/ATM 且 DIRECT=1：`(moe_score, moe_feat, ori_score, ori_feat, loss_moe)`
- 不使用 HDM/ATM 且 DIRECT=1：`(ori_score, ori_feat)`

### 推理和缺失模态 (test_net.py)

推理过程中，根据 `cfg.TEST.MISS` 通过将输入置零来模拟缺失模态：
- `'r'`：缺失 RGB
- `'n'`：缺失 NIR
- `'t'`：缺失 TIR
- `'rn'`、`'rt'`、`'nt'`：缺失两个模态
- `'nothing'`：所有模态都存在

`do_inference` 中的 `return_pattern` 参数控制特征输出：
- `1`：仅原始拼接特征
- `2`：仅 MoE 特征
- `3`：两者拼接（默认）

## 开发命令

### 环境搭建

```bash
conda create -n DeMo python=3.8.12 -y
conda activate DeMo
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### 训练

```bash
# 在 RGBNT201（行人重识别）上训练
python train_net.py --config_file configs/RGBNT201/DeMo.yml

# 在 RGBNT100（车辆重识别）上训练
python train_net.py --config_file configs/RGBNT100/DeMo.yml

# 在 MSVR310（车辆重识别）上训练
python train_net.py --config_file configs/MSVR310/DeMo.yml

# 通过命令行覆盖配置选项
python train_net.py --config_file configs/RGBNT201/DeMo.yml MODEL.BASE_LR 0.0005 SOLVER.MAX_EPOCHS 60

# 分布式训练（多GPU）
python -m torch.distributed.launch --nproc_per_node=4 train_net.py --config_file configs/RGBNT201/DeMo.yml
```

### 测试

```bash
# 基础测试（需要修改 test_net.py 第43行以指定模型路径）
python test_net.py --config_file configs/RGBNT201/DeMo.yml

# 带缺失模态模拟的测试
python test_net.py --config_file configs/RGBNT201/DeMo.yml TEST.MISS r  # 缺失 RGB
python test_net.py --config_file configs/RGBNT201/DeMo.yml TEST.MISS n  # 缺失 NIR
python test_net.py --config_file configs/RGBNT201/DeMo.yml TEST.MISS t  # 缺失 TIR
python test_net.py --config_file configs/RGBNT201/DeMo.yml TEST.MISS rn  # 缺失 RGB+NIR

# 带重排序的测试
python test_net.py --config_file configs/RGBNT201/DeMo.yml TEST.RE_RANKING yes
```

注意：`test_net.py:43` 包含占位符路径 `"your_model.pth"`，必须更新为指向训练好的模型检查点。

### 可视化工具

仓库在 `visualize/` 中包含可视化工具：

```bash
# 为多模态特征生成 Grad-CAM 可视化
python visualize/GradCAM.py --config_file configs/RGBNT201/DeMo.yml --model_path /path/to/checkpoint.pth
```

## 重要实现细节

### 预训练模型

模型需要在配置中指定预训练权重：
- **ViT-B 骨干网络**：将 `MODEL.PRETRAIN_PATH_T` 设置为 `vitb_16_224_21k.pth` 的路径
- **CLIP 骨干网络**：使用 `MODEL.TRANSFORMER_TYPE: 'ViT-B-16'` 并提供 CLIP 权重

下载链接在 README.md 中提供（百度网盘）。

### 内存管理

默认配置针对 <24GB 显存的 GPU 进行了优化：
- `DATALOADER.NUM_INSTANCE: 4`（可从 16 减少）
- `SOLVER.IMS_PER_BATCH: 8`（可从 128 减少）
- 根据 CPU 核心数调整 `DATALOADER.NUM_WORKERS`

### 数据集路径

在配置文件中更新 `DATASETS.ROOT_DIR` 以指向数据集位置。预期的目录结构：
```
ROOT_DIR/
  RGBNT201/
    train/
    query/
    gallery/
```

### 特征维度

- ViT-B-16 (CLIP)：`feat_dim = 512`
- vit_base_patch16_224：`feat_dim = 768`

模型根据 `MODEL.TRANSFORMER_TYPE` 自动设置特征维度。

### 冻结参数

代码库保留了来自 MambaPro（前身项目）的提示和适配器调优功能，但默认禁用：
- `MODEL.PROMPT: False`
- `MODEL.ADAPTER: False`
- `MODEL.FROZEN: False`

可以启用这些选项来冻结骨干网络，仅调优提示/适配器。

### 随机种子

训练使用固定种子（`SOLVER.SEED: 1111`）以确保可重复性。在 `train_net.py:44` 通过 `set_seed()` 设置。

### 相机和视角嵌入

模型使用空间实例嵌入（SIE）：
- `MODEL.SIE_CAMERA: True` - 将相机 ID 嵌入到特征中
- `MODEL.SIE_VIEW: False` - 视角嵌入（未使用）
- `MODEL.SIE_COE: 1.0` - 缩放系数

## 代码组织模式

修改代码库时：
1. **新骨干网络**：添加到 `modeling/backbones/` 并在 `make_model.py:__factory_T_type` 中注册
2. **新损失函数**：在 `layers/` 中实现并集成到 `layers/make_loss.py`
3. **新数据集**：继承 `data/datasets/bases.py` 中的 `BaseDataSet` 并在 `make_dataloader.py` 中注册
4. **新优化器**：添加到 `solver/make_optimizer.py`
5. **配置更改**：在 `config/defaults.py` 中更新新参数

## 遗留代码

根目录中的 `seps_modules_reviewed_v2_enhanced.py` 文件似乎是一个独立的模块审查/增强文件，不是主训练流程的一部分。
