import torch
from ultralytics import YOLO

# Load dataset
dataset_path = "dataset/tobacco.yaml"

# Load YOLOv9 model
model = YOLO("yolov9")

# Train model
model.train(data=dataset_path, epochs=50, imgsz=640)

# Save model
model.export(format="torchscript")
print("Training Complete!")
