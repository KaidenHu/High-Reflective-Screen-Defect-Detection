#!/bin/bash
# 这个脚本用于按顺序执行多个模型的训练任务。

# 设置CUDA可见设备，如果你的机器有多张显卡，可以在这里指定。这里用0表示使用第一张显卡。
export CUDA_VISIBLE_DEVICES=0

# --- 1. 训练YOLOv8s基线模型 ---
echo "============================================"
echo "Starting training: YOLOv8s Baseline"
date
echo "============================================"
yolo detect train data=dataset/SSGD_lb101_yolo/data.yaml model=yolov8s.pt epochs=300 imgsz=640 batch=16 optimizer=SGD lr0=0.01 project=runs/train name=ssgd_baseline_300e
# 检查上一条命令是否执行成功，如果失败则退出脚本，避免浪费资源继续训练
if [ $? -ne 0 ]; then
    echo "YOLOv8s Baseline training failed! Exiting."
    exit 1
fi

# --- 2. 训练YOLOv8s_GhostNet模型 ---
echo "============================================"
echo "Starting training: YOLOv8s + GhostNet"
date
echo "============================================"
yolo detect train data=dataset/SSGD_lb101_yolo/data.yaml model=models/yolov8_ghost.yaml epochs=300 imgsz=640 batch=16 optimizer=SGD lr0=0.01 project=runs/train name=ssgd_baseline_hyper_ghost

if [ $? -ne 0 ]; then
    echo "YOLOv8s_GhostNet training failed! Exiting."
    exit 1
fi

# --- 3. 训练YOLOv8s_GhostNet+CA模型 ---
echo "============================================"
echo "Starting training: YOLOv8s + GhostNet + CA"
date
echo "============================================"
yolo detect train data=dataset/SSGD_lb101_yolo/data.yaml model=models/yolov8_ghost_ca.yaml epochs=300 imgsz=640 batch=16 optimizer=SGD lr0=0.01 project=runs/train name=ssgd_baseline_hyper_ghost_ca

if [ $? -ne 0 ]; then
    echo "YOLOv8s_GhostNet+CA training failed! Exiting."
    exit 1
fi

echo "============================================"
echo "All training jobs completed successfully!"
date
echo "============================================"
