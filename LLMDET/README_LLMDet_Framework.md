# LLMDet (GroundingDINO) COCO Training + Inference Framework

This repository provides a lightweight training + evaluation framework built on **LLMDet / GroundingDINO**, using **COCO-format datasets**, with optional **image caption auxiliary losses**.

There are two main steps to use this repo:

1. **Customise your dataset** (COCO JSON + optional captions)
2. **Run training or inference scripts** (locally or on any Slurm/HPC system)

---

## Repository Structure

This repo contains two standalone scripts:

### `train_llmdet.py`
- Main training driver for LLMDet GroundingDINO
- Builds COCO dataloaders directly inside the script
- Supports optional caption losses via `--enable-caption-losses`

### `infer_llmdet.py`
- Main evaluation / inference driver
- Runs COCO bbox metrics from a trained checkpoint
- Supports subset evaluation via `--cat-ids`
- Supports custom grounding prompts via `--prompt-names`

### Example launch scripts (optional)
- `slurm_train.sbatch`
- `slurm_infer.sbatch`

These are optional helpers for cluster execution, but the scripts work on any machine.

---

## Requirements

This framework assumes you already have a working **LLMDet + MMDet + MMEngine** environment.

Typical required packages:

- `torch` (CUDA build recommended)
- `mmcv`
- `mmengine`
- `mmdet` (from the LLMDet fork)
- `transformers`
- `huggingface_hub`
- `pycocotools`

---

## Setup: Download LLMDet

You must have the upstream [LLMDet](https://github.com/iSEE-Laboratory/LLMDet) repo available, because it provides:


- `mmdet` modules
- GroundingDINO model definitions
- Base configs under `configs/`

Example layout:

```
LLMDet-main/
  configs/
    grounding_dino_swin_t.py
  mmdet/
  huggingface/
```

Clone:

```bash
git clone <LLMDET_REPO_URL> LLMDet-main
```

Add to your Python path:

```bash
export PYTHONPATH=/path/to/LLMDet-main:$PYTHONPATH
```

---

## HuggingFace Local Models (IMPORTANT)

The upstream LLMDet configs may reference local HuggingFace assets using relative paths like:

```
../huggingface/bert-base-uncased/
```

Therefore, you must ensure the following exist:

```
LLMDet-main/huggingface/
  bert-base-uncased/
  my_llava-onevision-qwen2-0.5b-ov-2/
  siglip-so400m-patch14-384/
```

If missing, download them using:

```python
from huggingface_hub import snapshot_download
snapshot_download("bert-base-uncased", local_dir="huggingface/bert-base-uncased")
```

---

## Critical Note: Run From `configs/`

Because configs contain relative HF paths, training/inference should be launched from:

```bash
cd LLMDet-main/configs
```

Otherwise you may see:

```
Incorrect path_or_model_id: '../huggingface/bert-base-uncased/'
```

---

## Custom Dataset Configuration

This framework uses **COCO JSON datasets**.

### Required COCO Fields

Your dataset must contain:

- `images[]`
- `annotations[]`
- `categories[]`

Example:

```json
{
  "images": [
    {"id": 1, "file_name": "img001.jpg"}
  ],
  "annotations": [
    {"id": 10, "image_id": 1, "category_id": 1, "bbox": [x,y,w,h]}
  ],
  "categories": [
    {"id": 1, "name": "gun"}
  ]
}
```

Bounding boxes must be in COCO format:

```
[x_min, y_min, width, height]
```

---

## Caption Loss Support (Optional)

To enable caption-based auxiliary losses, the **training COCO JSON** must include:

```json
images[].caption
```

Example:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img001.jpg",
      "caption": "A person holding a handgun."
    }
  ]
}
```

---

## Enabling Caption Loss During Training

Caption supervision is disabled by default.

To enable:

```bash
--enable-caption-losses
```

This activates:

- Image-level caption loss
- Contrastive text-image loss

If captions are missing, training still runs but uses a fallback caption string.

---

## Usage

## Training

### Basic Detection Training

```bash
cd /path/to/LLMDet-main/configs

python /path/to/train_llmdet_github.py   --cfg grounding_dino_swin_t.py   --train-images /data/train/images   --train-ann /data/train/train_coco.json   --val-images /data/val/images   --val-ann /data/val/val_coco.json   --work-dir work_dirs/run_train   --load-from grounding_dino_pretrain.pth   --lmm-tokenizer ../huggingface/my_llava-onevision-qwen2-0.5b-ov-2   --batch-size 1   --workers 2   --lr 5e-5   --max-iters 30000   --val-interval 5000   --ckpt-interval 5000
```

Outputs are written under:

```
work_dirs/run_train/
```

---

### Training With Caption Losses Enabled

```bash
cd /path/to/LLMDet-main/configs

python /path/to/train_llmdet_github.py   --cfg grounding_dino_swin_t.py   --train-images /data/train/images   --train-ann /data/train/train_coco_with_captions.json   --val-images /data/val/images   --val-ann /data/val/val_coco.json   --work-dir work_dirs/run_train   --load-from grounding_dino_pretrain.pth   --lmm-tokenizer ../huggingface/my_llava-onevision-qwen2-0.5b-ov-2   --enable-caption-losses   --batch-size 1   --workers 2   --lr 5e-5   --max-iters 30000   --val-interval 5000   --ckpt-interval 5000
```

---

## Inference / Evaluation

```bash
cd /path/to/LLMDet-main/configs

python /path/to/infer_llmdet_github.py   --cfg grounding_dino_swin_t.py   --ckpt work_dirs/run_train/best_coco_bbox_mAP.pth   --images /data/test/images   --ann /data/test/test_coco.json   --cat-ids 1   --prompt-names gun   --work-dir work_dirs/run_test   --batch-size 1   --workers 4
```

---

## Inference Notes

- `--cat-ids` must match the dataset’s `categories[].id`
- `--prompt-names` overrides grounding prompt entities
- Metrics + predictions are written under:

```
work_dirs/run_test/
```

---

## Common Issues

### HuggingFace tokenizer path errors

```
Incorrect path_or_model_id: ../huggingface/bert-base-uncased/
```

Fix:

```bash
cd LLMDet-main/configs
```

or edit the config to use absolute HF paths.

---

### Captions not being used

Fix:
- Ensure training JSON contains `images[].caption`
- Pass:

```bash
--enable-caption-losses
```

---

### Category mismatch in evaluation

Check your COCO categories:

```json
"categories": [{"id": 5, "name": "gun"}]
```

Then run inference with:

```bash
--cat-ids 5
```

---

## Summary

This repo provides a minimal reproducible framework for:

- COCO GroundingDINO training with LLMDet
- Optional caption supervision via COCO captions
- COCO evaluation/inference with grounding prompts
- Compatibility with local machines or cluster environments

---
