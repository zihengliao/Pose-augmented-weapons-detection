# Pose-augmented-weapons-detection

This repository collects multiple training pipelines for **weapon detection**, including
pose/caption-augmented variants. Each model family is self-contained with its own
scripts and requirements.

---

## Project Layout

- `LLMDET/` - GroundingDINO (LLMDet) training + inference framework
- `RF DETR/` - RF-DETR training on COCO datasets (Roboflow pipeline)
- `YOLO/` - Ultralytics YOLO training + export
- `Captioning/` - Caption generation utilities
- `Qwen/` - Qwen fine-tuning utilities
- `demo/` - Demo assets / experiments

---

## Getting Started

Pick a model family and follow its README:

- LLMDet: `LLMDET/README_LLMDet_Framework.md`
- RF-DETR: `RF DETR/README.md`
- YOLO: `YOLO/README.md`
- QWEN: `Qwen/fine-tuningREADME.md`

---

## Common Data Conventions

- All training performed used the **COCO** dataset format. Feel free to change as needed

---

## Notes

- Many scripts assume local paths for datasets and pretrained weights.
- API keys for Roboflow/W&B should be provided via environment variables or
  inserted into the scripts as needed.
