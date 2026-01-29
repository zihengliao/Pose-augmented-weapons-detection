"""
train_llmdet.py
======================

Standalone COCO training script for LLMDet / GroundingDINO.

This script is designed to be a minimal "single file" training driver
that works with the upstream LLMDet repo + MMEngine runner.

It supports:

  - Training a COCO detector (here fixed to 1 class: "gun")
  - Using GroundingDINO-style grounding prompts ("entities")
  - OPTIONAL caption-based auxiliary losses:
        --enable-caption-losses

Caption loss support works by reading captions from:

    COCO_JSON["images"][i]["caption"]

If captions are missing, training still runs, but uses a fallback caption.

IMPORTANT PATH NOTE
-------------------
The upstream LLMDet config may contain relative HuggingFace paths like:

    ../huggingface/bert-base-uncased/

Therefore, we typically run from:

    cd LLMDet-main/configs

so these relative references resolve correctly.

Usage Example
-------------
python train_llmdet_github.py \
  --cfg grounding_dino_swin_t.py \
  --train-images /data/train \
  --train-ann train.json \
  --val-images /data/val \
  --val-ann val.json \
  --work-dir work_dirs/run1 \
  --load-from pretrained.pth \
  --lmm-tokenizer huggingface/my_llava... \
  --enable-caption-losses
"""

import os
import sys
import json
import argparse

from mmengine.config import Config
from mmengine.runner import Runner

# Optional: if your environment needs default scope init.
# from mmengine.registry import init_default_scope
# init_default_scope('llmdet')

# IMPORTANT:
# This script assumes LLMDet/MMDet is installed in your environment
# OR you run it from a repo where `mmdet` is importable.
from mmdet.registry import TRANSFORMS

# -------------------------------------------------------------------------
# Fixed detection class list for this project.
# If you want multi-class training, modify DET_CLASSES accordingly.
# -------------------------------------------------------------------------
DET_CLASSES = ('gun',)
# Extra grounding context entities used as prompts during training.
# These help GroundingDINO learn richer grounding beyond just "gun".
CONTEXT_ENTITIES = ('gun', 'pistol', 'long gun', 'rifle', 'knife', 'hand', 'other weapon')



# =========================================================================
# Custom TRANSFORMS
# =========================================================================
@TRANSFORMS.register_module()
class SetGroundingEntities:
        """
    GroundingDINO expects text entity prompts in the sample metadata.

    This transform injects entity strings into results["text"] so the model
    treats them as grounding labels.

    Example:
        prompt=("gun","pistol")  → results["text"] = ("gun","pistol")
    """
    def __init__(self, prompt):
        self.prompt = prompt

    def _to_entity_list(self, t):
        if isinstance(t, (list, tuple)):
            return [str(s).strip() for s in t if str(s).strip()]
        s = str(t)
        s = s.replace(' .', '.').replace('. ', '.')
        parts = [p.strip() for p in (s.split(',') if ',' in s else s.split('.'))]
        return [p.strip(' .') for p in parts if p.strip(' .')]

    def transform(self, results):
        entities = self._to_entity_list(self.prompt)
        assert entities and all(isinstance(x, str) for x in entities)
        results['text'] = tuple(entities)
        results['custom_entities'] = True
        return results

    __call__ = transform


@TRANSFORMS.register_module()
class AttachImageCaptionAsConversation:
    """
    Caption Loss Support Transform
    ------------------------------

    LLMDet can optionally compute caption-based auxiliary losses.

    This transform:
      1. Reads captions from the COCO JSON file:
            images[].caption
      2. Tokenizes them with the given LMM tokenizer
      3. Stores them in results["conversations"]

    The model later consumes:
        results["conversations"]["input_id"]
        results["conversations"]["label"]

    If an image has no caption, a fallback caption is used.
    """
    def __init__(
        self,
        ann_file,
        prompt="Describe the scene.",
        lmm_tokenizer_path="",
        max_len=256,
        ignore_index=-100
    ):
        from transformers import AutoTokenizer

        if not lmm_tokenizer_path:
            raise ValueError("lmm_tokenizer_path must be set (HF model name or local directory).")

        self.prompt = prompt
        self.max_len = max_len
        self.ignore_index = ignore_index

        # load captions from COCO json
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.fn2cap = {}
        for im in data.get('images', []):
            fn = os.path.basename(im.get('file_name', '')).lower()
            cap = im.get('caption', '')
            if fn and isinstance(cap, str) and cap.strip():
                self.fn2cap[fn] = cap.strip()

        # load tokenizer once
        self.tok = AutoTokenizer.from_pretrained(lmm_tokenizer_path, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token or "<|pad|>"

    def _pack_ids_and_labels(self, prompt_text, answer_text):
        """
        Construct input_ids + labels for caption supervision.

        Prompt tokens are ignored (-100),
        Answer tokens contribute to loss.
        """
        prompt_ids = self.tok(prompt_text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        answer_ids = self.tok(answer_text, add_special_tokens=False, return_attention_mask=False)["input_ids"]

        eos = [] if self.tok.eos_token_id is None else [self.tok.eos_token_id]
        input_ids = (prompt_ids + answer_ids + eos)[:self.max_len]
        labels = ([self.ignore_index] * len(prompt_ids) + answer_ids + eos)[:self.max_len]

        if len(labels) < len(input_ids):
            labels += [self.ignore_index] * (len(input_ids) - len(labels))
        return input_ids, labels

    def transform(self, results):
        fn = os.path.basename(results.get('img_path', '')).lower()
        cap = self.fn2cap.get(fn, "")

        prompt_text = "<image>\n" + self.prompt
        if cap:
            input_ids, labels = self._pack_ids_and_labels(prompt_text, cap)
        else:
            input_ids, labels = self._pack_ids_and_labels(prompt_text, "A photo of a scene.")

        results['conversations'] = {'input_id': input_ids, 'label': labels}
        results['region_conversations'] = {'conversations': []}
        results['dataset_mode'] = 'OD'
        return results

    __call__ = transform


# =========================================================================
# Dataset + Config patching
# =========================================================================

def make_dataset_cfg(img_root, ann_path, tokenizer_path, test_mode=False):
    """
    Build a CocoDataset config dict with the correct pipeline.

    - Training pipeline includes caption attachment (if enabled)
    - Validation pipeline uses fixed resize + no caption supervision
    """
    entities = ('gun',) if test_mode else CONTEXT_ENTITIES

    pipe = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        (dict(type='FilterAnnotations',
              min_gt_bbox_wh=(1e-2, 1e-2),
              keep_empty=False) if not test_mode else None),

        (dict(type='AttachImageCaptionAsConversation',
              ann_file=os.path.abspath(ann_path),
              prompt="Describe the scene.",
              lmm_tokenizer_path=tokenizer_path) if not test_mode else None),

        dict(type='SetGroundingEntities', prompt=entities),

        (dict(type='RandomFlip', prob=0.5) if not test_mode else None),

        (dict(
            type='RandomChoiceResize',
            scales=[(512, 640), (576, 640), (640, 640)],
            keep_ratio=True
        ) if not test_mode else dict(
            type='FixScaleResize', scale=(640, 640), keep_ratio=True, backend='pillow'
        )),

        dict(
            type='PackDetInputs',
            meta_keys=(
                'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                'flip', 'flip_direction', 'text', 'custom_entities',
                'conversations', 'region_conversations', 'dataset_mode'
            ),
        ),
    ]
    pipe = [t for t in pipe if t is not None]

    return dict(
        type='CocoDataset',
        metainfo=dict(classes=DET_CLASSES),
        ann_file=os.path.abspath(ann_path),
        data_prefix=dict(img=os.path.abspath(img_root)),
        filter_cfg=dict(filter_empty_gt=not test_mode),
        test_mode=test_mode,
        pipeline=pipe,
    )


def _force_num_classes(model_cfg, n: int):
    def _apply(x):
        if isinstance(x, dict):
            x['num_classes'] = n
        elif isinstance(x, (list, tuple)):
            for h in x:
                if isinstance(h, dict):
                    h['num_classes'] = n

    if isinstance(model_cfg.get('bbox_head'), (dict, list, tuple)):
        _apply(model_cfg['bbox_head'])
    if isinstance(model_cfg.get('roi_head'), dict):
        if isinstance(model_cfg['roi_head'].get('bbox_head'), (dict, list, tuple)):
            _apply(model_cfg['roi_head']['bbox_head'])
    if isinstance(model_cfg.get('rpn_head'), (dict, list, tuple)):
        _apply(model_cfg['rpn_head'])


def patch_cfg(cfg, args):
    """
    Patch the upstream LLMDet config into a runnable single-dataset training job.
    """
    cfg.work_dir = os.path.abspath(args.work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.randomness = dict(seed=args.seed)

    if args.load_from:
        cfg.load_from = os.path.abspath(args.load_from)

    if 'freeze_backbone' in cfg.model:
        cfg.model['freeze_backbone'] = False

    # Caption loss flags in train_cfg only
    cfg.model.setdefault('train_cfg', {})
    if args.enable_caption_losses:
        cfg.model['train_cfg'].update(dict(
            use_caption_loss=True,
            use_region_caption=False,
            use_llm_contrastive_loss=True,
            use_region_conversation=False,
        ))
    cfg.model['num_region_caption'] = 0
    cfg.model.setdefault('lmm_region_loss_weight', 0.0)
    cfg.model['train_cfg']['use_region_caption'] = False
    cfg.model['train_cfg']['use_region_conversation'] = False

    # test cfg knobs
    cfg.model.setdefault('test_cfg', {})
    cfg.model.test_cfg['chunked_size'] = -1
    cfg.model.test_cfg['custom_entities'] = True
    cfg.model.test_cfg.pop('tokens_positive', None)

    _force_num_classes(cfg.model, n=len(DET_CLASSES))

    train_ds = make_dataset_cfg(args.train_images, args.train_ann, args.lmm_tokenizer, test_mode=False)
    val_ds = make_dataset_cfg(args.val_images, args.val_ann, args.lmm_tokenizer, test_mode=True)

    cfg.train_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=train_ds
    )
    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=args.workers,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=val_ds
    )
    cfg.test_dataloader = cfg.val_dataloader

    cfg.val_evaluator = dict(type='CocoMetric', ann_file=os.path.abspath(args.val_ann), metric='bbox')
    cfg.test_evaluator = cfg.val_evaluator

    # opt & schedule
    if 'optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper:
        cfg.optim_wrapper['optimizer']['lr'] = args.lr
    else:
        cfg.optim_wrapper = dict(
            type='OptimWrapper',
            optimizer=dict(type='AdamW', lr=args.lr, weight_decay=1e-4),
            clip_grad=dict(max_norm=0.1, norm_type=2),
        )

    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=args.max_iters, val_interval=args.val_interval)
    cfg.param_scheduler = [
        dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
        dict(type='MultiStepLR', by_epoch=False, begin=0, end=args.max_iters,
             milestones=[int(args.max_iters * 0.8), int(args.max_iters * 0.93)], gamma=0.1),
    ]

    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook',
            by_epoch=False,
            interval=args.ckpt_interval,
            max_keep_ckpts=10,
            save_best='coco/bbox_mAP'
        ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook'),
    )

    cfg.log_processor = dict(by_epoch=False)
    cfg.env_cfg = dict(dist_cfg=dict(backend='nccl', timeout=36000))
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)

    return cfg


def _patch_flatten_lm_forward(runner):
    """Flatten any nested prompt structure before LM forward."""
    def _to_string_list(x):
        if isinstance(x, str):
            return [x]
        if isinstance(x, (list, tuple)):
            out = []
            for item in x:
                out.append(', '.join(map(str, item)) if isinstance(item, (list, tuple)) else str(item))
            return out
        return [str(x)]

    lm = getattr(runner.model, 'language_model', None) or getattr(runner.model, 'language_model_lmm', None)
    if lm is None:
        return

    _orig = lm.forward

    def _patched(self, text_prompts, *args, **kwargs):
        return _orig(_to_string_list(text_prompts), *args, **kwargs)

    lm.forward = _patched.__get__(lm, type(lm))


# =========================================================================
# Main entry
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, required=True, help='Base config (upstream LLMDet/MMDet config)')
    p.add_argument('--train-images', type=str, required=True)
    p.add_argument('--train-ann', type=str, required=True)
    p.add_argument('--val-images', type=str, required=True)
    p.add_argument('--val-ann', type=str, required=True)
    p.add_argument('--work-dir', type=str, default='work_dirs/run_train')
    p.add_argument('--load-from', type=str, default='', help='Optional pretrained .pth')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-iters', type=int, default=60000)
    p.add_argument('--val-interval', type=int, default=5000)
    p.add_argument('--ckpt-interval', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)


    p.add_argument('--lmm-tokenizer', type=str, required=True, help='HF model name or local dir')
    p.add_argument('--enable-caption-losses', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.cfg)
    cfg = patch_cfg(cfg, args)

    print("\n=== Training summary ===")
    print("Work dir      :", cfg.work_dir)
    print("Load from     :", getattr(cfg, 'load_from', None))
    print("Train ann     :", os.path.abspath(args.train_ann))
    print("Val ann       :", os.path.abspath(args.val_ann))
    print("Batch size    :", cfg.train_dataloader['batch_size'])
    print("LR            :", cfg.optim_wrapper['optimizer']['lr'])
    print("Max iters     :", cfg.train_cfg['max_iters'])
    print("Val interval  :", cfg.train_cfg['val_interval'])
    print("========================\n")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    runner = Runner.from_cfg(cfg)
    _patch_flatten_lm_forward(runner)
    runner.train()


if __name__ == '__main__':
    main()
