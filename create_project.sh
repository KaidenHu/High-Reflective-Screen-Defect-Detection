#!/bin/bash
# 2. 模型定义目录
mkdir -p models
touch models/__init__.py
touch models/yolo8s_ca.py
touch models/yolo8s_mobilenet.py
touch models/yolo8s_dynamic_lska.py

# 3. 脚本目录
mkdir -p scripts
touch scripts/train.py
touch scripts/val.py
touch scripts/detect.py
touch scripts/convert_dataset.py

# 4. 配置目录
mkdir -p config
touch config/msd.yaml
touch config/ssgd.yaml
touch config/hybrid.yaml

# 5. 训练输出目录（自动生成，但先创建占位）
mkdir -p runs/train runs/detect

# 6. 系统原型目录
mkdir -p system/resources
touch system/main_window.py
touch system/detector.py
touch system/utils.py

# 7. 辅助工具目录
mkdir -p utils
touch utils/__init__.py
touch utils/metrics.py
touch utils/plot_utils.py
touch utils/augmentations.py
