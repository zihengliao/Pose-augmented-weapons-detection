"""
infer_llmdet_github.py
======================

Standalone evaluation / inference script for LLMDet / GroundingDINO.

This script runs MMEngine Runner.test() on a COCO-format dataset.

Supports:
  - Selecting specific COCO category IDs (--cat-ids)
  - Overriding grounding prompts (--prompt-names)
  - Producing COCO bbox metrics + saving predictions

IMPORTANT PATH NOTE
-------------------
Upstream configs often reference HuggingFace assets with relative paths like:

    ../huggingface/bert-base-uncased/

Therefore, inference should also be launched from:

    cd LLMDet-main/configs/

Usage Example
-------------
python infer_llmdet_github.py \
  --cfg grounding_dino_swin_t.py \
  --ckpt best.pth \
  --images test/images \
  --ann test.json \
  --cat-ids 1 \
  --prompt-names gun \
  --work-dir work_dirs/test_run
"""

import os, json, argparse
from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SetGroundingEntities:
    """
    Inject grounding prompt entities into the sample metadata.

    This ensures GroundingDINO uses the provided prompt strings
    as the grounding text targets.
    """
    def __init__(self, prompt): self.prompt = prompt
    def _to_entity_list(self, t):
        if isinstance(t, (list, tuple)):
            return [str(s).strip() for s in t if str(s).strip()]
        s = str(t).replace(' .', '.').replace('. ', '.')
        parts = [p.strip() for p in (s.split(',') if ',' in s else s.split('.'))]
        return [p.strip(' .') for p in parts if p.strip(' .')]
    def transform(self, results):
        entities = self._to_entity_list(self.prompt)
        assert entities and all(isinstance(x, str) for x in entities)
        results['text'] = tuple(entities)
        results['custom_entities'] = True
        return results
    __call__ = transform


def extract_classes_from_ann(ann_path, cat_id_list=None):
    """
    Read COCO categories from annotation JSON.

    If cat_ids is provided, only those category IDs are evaluated.
    """
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    id2name = {int(c['id']): c['name'] for c in data.get('categories', [])}
    if cat_id_list:
        names = []
        for s in cat_id_list:
            cid = int(s)
            if cid not in id2name:
                raise ValueError(f"Category id {cid} not found in {ann_path}")
            names.append(id2name[cid])
        return tuple(names)
    return tuple(c['name'] for c in data.get('categories', []))


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
    if isinstance(model_cfg.get('panoptic_head'), (dict, list, tuple)):
        _apply(model_cfg['panoptic_head'])
    if isinstance(model_cfg.get('rpn_head'), (dict, list, tuple)):
        _apply(model_cfg['rpn_head'])


def _patch_flatten_lm_forward(runner):
    def _to_string_list(x):
        if isinstance(x, str): return [x]
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


def make_test_dataset_cfg(img_root, ann_path, class_names, prompt_names):
    """
    Build the test-only CocoDataset config.
    No augmentation, fixed resize.
    """
    pipe = [
        dict(type='LoadImageFromFile'),
        dict(type='SetGroundingEntities', prompt=prompt_names),
        dict(type='FixScaleResize', scale=(640, 640), keep_ratio=True, backend='pillow'),
        dict(type='PackDetInputs',
             meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','text','custom_entities')),
    ]
    return dict(
        type='CocoDataset',
        metainfo=dict(classes=tuple(class_names)),
        ann_file=os.path.abspath(ann_path),
        data_prefix=dict(img=os.path.abspath(img_root)),
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True,
        pipeline=pipe,
    )


def build_cfg_for_dataset(base_cfg_path, ckpt, img_root, ann_path, class_names, prompt_names,
                          work_dir, batch_size, workers):
                          
    """
    Patch upstream config into a single COCO evaluation job.
    """
    from os.path import abspath, join
    cfg = Config.fromfile(base_cfg_path)
    os.makedirs(work_dir, exist_ok=True)
    cfg.work_dir = abspath(work_dir)

    cfg.model.setdefault('test_cfg', {})
    cfg.model.test_cfg['custom_entities'] = True
    cfg.model.test_cfg['chunked_size'] = -1
    cfg.model.test_cfg.pop('tokens_positive', None)

    _force_num_classes(cfg.model, n=len(class_names))

    ds = make_test_dataset_cfg(img_root, ann_path, class_names, prompt_names)
    cfg.test_dataloader = dict(
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=ds
    )
    cfg.val_dataloader = cfg.test_dataloader

    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=abspath(ann_path),
        metric='bbox',
        outfile_prefix=join(work_dir, 'preds')
    )
    cfg.val_evaluator = cfg.test_evaluator

    if ckpt:
        cfg.load_from = abspath(ckpt)

    return cfg


def parse_args():
    p = argparse.ArgumentParser("LLMDet / GroundingDINO inference on one COCO dataset")
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--ann', required=True)
    p.add_argument('--cat-ids', type=str, default='',
                   help='Optional comma-separated category IDs to evaluate (subset)')
    p.add_argument('--work-dir', required=True)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prompt-names', type=str, default='',
                   help='Optional comma-separated prompts. If omitted, uses category names.')
    return p.parse_args()


def main():
    args = parse_args()

    cat_ids = [s.strip() for s in args.cat_ids.split(',') if s.strip()] if args.cat_ids else None
    class_names = extract_classes_from_ann(args.ann, cat_ids)

    if args.prompt_names:
        prompt_names = tuple(s.strip() for s in args.prompt_names.split(',') if s.strip())
    else:
        prompt_names = class_names

    print("\n=== TEST ===")
    print("Work dir     :", os.path.abspath(args.work_dir))
    print("Images root  :", os.path.abspath(args.images))
    print("Ann file     :", os.path.abspath(args.ann))
    print("Eval classes :", class_names)
    print("Prompts      :", prompt_names)
    print("=============\n")

    cfg = build_cfg_for_dataset(
        base_cfg_path=args.cfg,
        ckpt=args.ckpt,
        img_root=args.images,
        ann_path=args.ann,
        class_names=class_names,
        prompt_names=prompt_names,
        work_dir=args.work_dir,
        batch_size=args.batch_size,
        workers=args.workers
    )

    runner = Runner.from_cfg(cfg)
    _patch_flatten_lm_forward(runner)
    runner.test()


if __name__ == "__main__":
    main()
