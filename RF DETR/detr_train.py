from roboflow import Roboflow
import wandb
from rfdetr import RFDETRBase
import torch
import os

# Initialize wandb

wandb.login()
wandb.init(
    project="Pose-augmented weapon detection using machine learning",
    config={
        "epochs": 20,
        "batch_size": 10,
        "image_size": 7780,
        "model": "rf-detr",
        "dataset": "Monash Gun Dataset"
    }
)


rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("arms").project("the-monash-guns-dataset")
version = project.version(2)
dataset = version.download("coco")
                

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


model = RFDETRBase()

model.train(dataset_dir=dataset.location, epochs=20, batch_size=10, grad_accum_steps=1, lr=1e-4)

metrics = model.val()
wandb.log({
    "val/mAP50": metrics.box.map50,
    "val/mAP50-95": metrics.box.map,
    "val/precision": metrics.box.mp,
    "val/recall": metrics.box.mr
})

# Export the model and log it to wandb
path = model.export(format="onnx")
wandb.save(path)

# Finish wandb run
wandb.finish()

