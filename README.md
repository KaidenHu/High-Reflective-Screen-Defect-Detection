# 手机屏幕表面缺陷检测系统（基于改进 YOLOv8s）

## 项目结构说明
- `data/`: 存放原始和预处理后的数据集（MSD, SSGD）
- `models/`: 自定义改进模型（CA注意力、MobileNetV3、Dynamic-LSKA）
- `scripts/`: 训练、验证、检测、数据转换脚本
- `config/`: 数据集和训练参数配置
- `system/`: PyQt5 桌面检测系统原型
- `utils/`: 辅助评估工具

## 快速开始
1. 准备数据：`python scripts/convert_dataset.py`
2. 训练基线：`python scripts/train.py --data config/msd.yaml --model yolov8s.pt`
3. 启动系统：`python system/main_window.py`
