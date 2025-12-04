"""
测试 SDTPS 模块集成
验证模块是否能正常工作，输出形状是否正确
"""

import torch
from config import cfg
from modeling import make_model

def test_sdtps_model():
    """测试 SDTPS 模块"""

    print("=" * 70)
    print("测试 SDTPS 模块集成")
    print("=" * 70)

    # 加载配置
    cfg.merge_from_file("configs/RGBNT201/DeMo_SDTPS.yml")
    cfg.freeze()

    print(f"\n配置信息:")
    print(f"  USE_SDTPS: {cfg.MODEL.USE_SDTPS}")
    print(f"  HDM: {cfg.MODEL.HDM}")
    print(f"  ATM: {cfg.MODEL.ATM}")
    print(f"  GLOBAL_LOCAL: {cfg.MODEL.GLOBAL_LOCAL}")
    print(f"  SDTPS_SPARSE_RATIO: {cfg.MODEL.SDTPS_SPARSE_RATIO}")
    print(f"  SDTPS_BETA: {cfg.MODEL.SDTPS_BETA}")
    print(f"  SDTPS_USE_GUMBEL: {cfg.MODEL.SDTPS_USE_GUMBEL}")

    # 创建模型
    print("\n创建模型...")
    num_classes = 201
    camera_num = 15
    view_num = 1

    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model = model.to(device)  # 确保模型完全在设备上
    model.eval()

    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    # 准备测试数据
    print("\n准备测试数据...")
    batch_size = 4
    img_size = (3, 256, 128)

    RGB = torch.randn(batch_size, *img_size).to(device)
    NI = torch.randn(batch_size, *img_size).to(device)
    TI = torch.randn(batch_size, *img_size).to(device)
    cam_label = torch.zeros(batch_size, dtype=torch.long).to(device)  # 添加 camera label

    x = {'RGB': RGB, 'NI': NI, 'TI': TI}

    # 测试训练模式
    print("\n测试训练模式...")
    model.train()

    with torch.no_grad():
        outputs = model(x, cam_label=cam_label)

    print(f"  输出数量: {len(outputs)}")
    for i, out in enumerate(outputs):
        if isinstance(out, torch.Tensor):
            print(f"  输出 {i}: shape = {out.shape}")
        else:
            print(f"  输出 {i}: {type(out)}")

    # 测试推理模式
    print("\n测试推理模式...")
    model.eval()

    with torch.no_grad():
        # return_pattern = 1: 仅原始特征
        feat1 = model(x, cam_label=cam_label, return_pattern=1)
        print(f"  return_pattern=1: shape = {feat1.shape}")

        # return_pattern = 2: 仅 SDTPS 特征
        feat2 = model(x, cam_label=cam_label, return_pattern=2)
        print(f"  return_pattern=2: shape = {feat2.shape}")

        # return_pattern = 3: 拼接特征
        feat3 = model(x, cam_label=cam_label, return_pattern=3)
        print(f"  return_pattern=3: shape = {feat3.shape}")

    print("\n" + "=" * 70)
    print("✓ 测试通过！SDTPS 模块集成成功！")
    print("=" * 70)

    return model


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    try:
        model = test_sdtps_model()
        print("\n所有测试通过！可以开始训练了。")
        print("\n使用以下命令启动训练:")
        print("python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
