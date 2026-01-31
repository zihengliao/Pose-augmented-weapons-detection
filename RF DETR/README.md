# RF-DETR Training (Roboflow + COCO)

This folder provides a minimal **RF-DETR** training script using the
Roboflow dataset download flow and COCO annotations.

There are two main steps:

1. **Download the dataset** from Roboflow in COCO format
2. **Train + evaluate** RF-DETR and export to ONNX

---

## Files

### `detr_train.py`
- Downloads the Roboflow dataset (COCO)
- Trains RF-DETR
- Runs validation and logs metrics
- Exports the model to ONNX

### `detr_train.ipynb`
- Notebook version of the same pipeline

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages include:

- `rfdetr`
- `roboflow`
- `torch`
- `wandb` (optional for experiment tracking)

---

## Roboflow Setup

The script expects:

- A **Roboflow API key**
- Access to the workspace and project

Update these lines in `detr_train.py`:

```python
wandb.login(key="<WANDB_API_KEY>")
rf = Roboflow(api_key="<ROBOFLOW_API_KEY>")
project = rf.workspace("arms").project("the-monash-guns-dataset")
version = project.version(2)
```

---

## Usage

Run training:

```bash
python detr_train.py
```

Outputs:

- Trained checkpoint and ONNX export are written by RF-DETR
- Validation metrics are logged to W&B (if enabled)

---

## Notes

- Dataset is downloaded in **COCO** format.
- Training hyperparameters are defined directly in `detr_train.py`.
- For custom datasets, change the Roboflow workspace/project/version.
