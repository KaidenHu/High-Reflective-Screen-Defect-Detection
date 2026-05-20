#FPS测试脚本
# =========================
# 命令行参数
# 运行方式
# python fps_test.py --weights runs/detect/train/weights/best.pt

# python fps_test.py \
# --weights runs/detect/train/weights/best.pt \
# --imgsz 640 \
# --warmup 50 \
# --testtime 200
# =========================

import time
import argparse
import torch
from ultralytics import YOLO

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--warmup', type=int, default=50)
parser.add_argument('--testtime', type=int, default=300)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO(args.weights).model.to(device)
model.eval()

dummy = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)

# warmup
print('Warmup...')
with torch.no_grad():
    for _ in range(args.warmup):
        _ = model(dummy)

torch.cuda.synchronize()

# FPS test
print('Testing...')
start = time.time()

with torch.no_grad():
    for _ in range(args.testtime):
        _ = model(dummy)

torch.cuda.synchronize()

end = time.time()

fps = args.testtime / (end - start)
latency = 1000 / fps

print(f'FPS: {fps:.2f}')
print(f'Latency: {latency:.2f} ms')