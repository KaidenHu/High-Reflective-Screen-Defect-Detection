# 可选：生成简单的 README 模板
cat > README.md << 'EOF'
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
EOF

# 可选：生成 requirements.txt 基础内容
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.7.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
pyqt5>=5.15.0
pandas>=1.5.0
scikit-learn>=1.2.0
tqdm
EOF

echo "项目文件夹结构已创建在：$(pwd)"
tree -L 3  # 如果系统没有 tree 命令，可注释掉或安装：sudo apt install tree
