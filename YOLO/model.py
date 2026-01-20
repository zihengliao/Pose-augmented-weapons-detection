# Disable broken disk space check
import ultralytics.utils.downloads as downloads

def dummy_check_disk_space(*args, **kwargs):
    pass  # Skip the check entirely

downloads.check_disk_space = dummy_check_disk_space

# import wandb
from ultralytics import YOLO
import os
import torch

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, 'src/data/Threat-Detection-1', 'data.yaml')

# Initialize wandb
# wandb.init(
#     project="Pose-augmented weapon detection using machine learning",
#     name="yolo11s with guns dataset (43K images)",
#     config={
#         "epochs": 100,
#         "batch_size": 4,
#         "image_size": 640,
#         "model": "yolov11s",
#         "dataset": "Guns Dataset (43K images)"
#     }
# )

# Load a model
model = YOLO("src/yolo11s.pt")  # Load the default YOLOv11s small model

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use cuda if available
print(f"Using device: {device}")

# Train the model
train_results = model.train(
    data=DATA_YAML,  # path to your data.yaml
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device=device,  # device to run on
    batch=16,  # batch size
    save=True,  # save checkpoints
    project="src/runs/train",  # save results to project/name
    name="train-threat-detection-1",
    exist_ok=True,
    plots=True  # save plots
)

# Log validation metrics to wandb
metrics = model.val()
# wandb.log({
#     "val/mAP50": metrics.box.map50,
#     "val/mAP50-95": metrics.box.map,
#     "val/precision": metrics.box.mp,
#     "val/recall": metrics.box.mr
# })

# Export the model and log it to wandb
path = model.export(format="onnx")
# wandb.save(path)

# Finish wandb run
# wandb.finish()