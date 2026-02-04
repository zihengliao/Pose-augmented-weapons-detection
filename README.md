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

## Data

All training performed used the **COCO** dataset format. Feel free to change as needed

- **Firearms Dataset** (Hao et al., 2018)  
  https://arxiv.org/abs/1806.02984

- **Monash Guns Dataset** (Lim et al., 2019)  
  https://github.com/MarcusLimJunYi/Monash-Guns-Dataset

- **LinkSprite Dataset** (Qi et al., 2021)  
  https://arxiv.org/abs/2105.01058

- **Threat Detection Dataset** (Roboflow)  
  https://universe.roboflow.com/threat-detection-k7wmf/threat-detection-m8dvh

---

## Notes

- Many scripts assume local paths for datasets and pretrained weights.
- API keys for Roboflow/W&B should be provided via environment variables or
  inserted into the scripts as needed.
