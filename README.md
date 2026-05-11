# 手机屏幕表面缺陷检测系统（基于改进 YOLOv8s）

## 项目结构说明
- `data/`: 存放原始和预处理后的数据集（MSD, SSGD）
- `models/`: 自定义改进模型（CA注意力、MobileNetV3、Dynamic-LSKA）
- `scripts/`: 训练、验证、检测、数据转换脚本
- `config/`: 数据集和训练参数配置
- `system/`: PyQt5 桌面检测系统原型
- `utils/`: 辅助评估工具

## 快速开始
python system/ui.py
