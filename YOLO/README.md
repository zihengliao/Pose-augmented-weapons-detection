# YOLO Training (Ultralytics)

This folder provides a minimal **Ultralytics YOLO** training script and notebook.
It trains a YOLOv11 model on a dataset defined by `data.yaml` and exports to ONNX.

---

## Files

### `model.py`
- Loads YOLO weights
- Trains using a local `data.yaml`
- Runs validation
- Exports the model to ONNX

### `model.ipynb`
- Notebook version of the same pipeline

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages include:

- `ultralytics`
- `torch`
- `opencv-python`

---

## Dataset Setup

`model.py` expects a dataset YAML at:

```
src/data/Threat-Detection-1/data.yaml
```

Update `DATA_YAML` in `model.py` if your dataset lives elsewhere.

---

## Usage

Run training:

```bash
python model.py
```

The script:

- Loads `src/yolo11s.pt` as the starting checkpoint
- Trains for 100 epochs at 640px
- Saves results to `src/runs/train/train-threat-detection-1`

---

## Notes

- W&B logging is present but commented out in `model.py`.
- The disk space check is disabled in `model.py` due to an upstream issue.
- Adjust `batch`, `epochs`, and `imgsz` in `model.py` to match your hardware.
