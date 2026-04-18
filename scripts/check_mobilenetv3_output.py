import torch
import sys
# 确保你的环境能导入 ultralytics.nn.modules.mobilenetv3
from ultralytics.nn.modules.mobilenetv3 import MobileNetV3_Small

def check_mobilenetv3():
    # 实例化模型
    model = MobileNetV3_Small(num_classes=1000)  # num_classes 可以随便给，不影响特征提取
    
    # 创建一个模拟输入：batch_size=1, 3通道, 高度=640, 宽度=640
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # 前向传播
    outputs = model(dummy_input)
    
    # 检查输出类型和长度
    print(f"输出类型: {type(outputs)}")
    if isinstance(outputs, (tuple, list)):
        print(f"输出长度: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"输出 {i}: shape = {out.shape}")
    else:
        print("输出不是 tuple/list，请检查 forward 返回值")
    
    # 根据输出形状判断是否符合预期
    # 预期：三个特征图，通道数分别为 24, 40, 96，空间尺寸依次减小
    # 例如输入 640x640，经过 stride 8,16,32 后，尺寸约为 80x80, 40x40, 20x20
    # 实际尺寸可能因 padding 略有差异，但通道数必须匹配
    expected_channels = [24, 40, 96]
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        for i, out in enumerate(outputs):
            if out.shape[1] == expected_channels[i]:
                print(f"✓ 输出 {i} 通道数正确: {expected_channels[i]}")
            else:
                print(f"✗ 输出 {i} 通道数错误: 实际 {out.shape[1]}, 期望 {expected_channels[i]}")
    else:
        print("输出数量或类型不正确")

if __name__ == "__main__":
    check_mobilenetv3()
