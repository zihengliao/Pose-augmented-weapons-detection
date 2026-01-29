# -*- coding: utf-8 -*-
"""
Qwen VLM Fine-tuning Evaluation Script

This script evaluates fine-tuned Qwen2.5-VL models on object detection tasks
using COCO evaluation metrics.
"""

# Standard library imports
import argparse
import builtins
import io
import json
import os
import re
import tempfile
import time
import warnings
from contextlib import redirect_stdout
from math import nan
from typing import Dict, List, Optional, Tuple

# Third-party library imports
import matplotlib.lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from roboflow import Roboflow
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Jinja2 compatibility patch (must be before jinja2 import)
try:
    import jinja2  # noqa: WPS433
    if not hasattr(jinja2, "pass_eval_context") and hasattr(jinja2, "evalcontextfilter"):
        jinja2.pass_eval_context = jinja2.evalcontextfilter  # type: ignore[assignment]
except Exception as _jinja_error:  # pragma: no cover
    pass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Module-level verbosity control
_VERBOSITY = int(os.getenv("VERBOSITY", "0"))  # 0 = quiet, 1 = normal, 2 = verbose
_builtin_print = builtins.print


def print(*args, **kwargs):
    """Module-scoped print respecting VERBOSITY.

    - Use kwarg level to mark importance: level=0 always prints, level=1 prints in normal, level=2 in verbose
    - Any message containing ERROR or WARNING is treated as critical and always printed
    """
    level = kwargs.pop("level", 1)
    try:
        text = " ".join(str(a) for a in args)
    except Exception:
        text = ""
    auto_critical = ("ERROR" in text) or ("WARNING" in text)
    if auto_critical or _VERBOSITY >= level:
        _builtin_print(*args, **kwargs)


# Constants
FLASH_ATTN_AVAILABLE = False

# Roboflow configuration
ROBOFLOW_API_KEY = "your_roboflow_api_key_here"
WORKSPACE = "your_workspace_name"
PROJECT = "your_project_name"
VERSION = 1

# Model configuration
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_FINETUNED_DIR = os.path.join(_SCRIPT_DIR, "output")
_HAS_LOCAL_FINETUNE = os.path.isdir(_LOCAL_FINETUNED_DIR)
if _HAS_LOCAL_FINETUNE:
    DEFAULT_MODEL_PATH = _LOCAL_FINETUNED_DIR
else:
    DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

# Inference configuration
MAX_IMAGES = None
CONFIDENCE_THRESHOLD = 0.8
EVAL_CONFIDENCE_THRESHOLD = 0.8
ENABLE_VISUALIZATION = True
SAVE_RESULTS = True
MAX_PREDICTIONS_PER_IMAGE = 100
DETECTION_MAX_NEW_TOKENS = 8192

# Evaluation behavior
COLLAPSE_CLASSES_TO_WEAPON = False
COLLAPSED_CLASS_NAME = "gun"

# Cache configuration
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", None)
if HF_CACHE_DIR:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")


def _initialize_model_path():
    """Initialize model path from command-line arguments or environment."""
    global MODEL_PATH
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--model_path", type=str, help="Hugging Face model id")
        args, _ = parser.parse_known_args()
        if args.model_path:
            MODEL_PATH = args.model_path
            print(f"Overriding model via --model_path: {MODEL_PATH}", level=2)
        elif os.getenv("MODEL_PATH"):
            print(f"Overriding model via env MODEL_PATH: {MODEL_PATH}", level=2)
    except Exception:
        pass
    if _HAS_LOCAL_FINETUNE and MODEL_PATH == _LOCAL_FINETUNED_DIR:
        print(f'Using local fine-tuned model: {MODEL_PATH}', level=1)


def _initialize_output_directories():
    """Initialize output directories for results and visualizations."""
    global OUTPUT_ROOT_DIR, FIGURES_DIR, DETECTIONS_DIR
    model_name_safe = MODEL_PATH.split("/")[-1].replace(" ", "_").replace("/", "_")
    OUTPUT_ROOT_DIR = os.path.join(os.getcwd(), "outputs", model_name_safe)
    FIGURES_DIR = os.path.join(OUTPUT_ROOT_DIR, "figures")
    DETECTIONS_DIR = os.path.join(OUTPUT_ROOT_DIR, "detections")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DETECTIONS_DIR, exist_ok=True)
    print(f"Saving visualizations to: {OUTPUT_ROOT_DIR}", level=2)


# Initialize configuration
_initialize_model_path()
_initialize_output_directories()


class QwenVLMInference:
    """Simplified Qwen VLM inference class for object detection"""

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Qwen VLM: {model_path}")
        print(f"Using device: {self.device}")

        # Load model with optimal settings
        try:
            if torch.cuda.is_available() and FLASH_ATTN_AVAILABLE:
                print("Attempting to use Flash Attention...")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
                print("Flash Attention enabled")
            else:
                hf_kwargs = dict(torch_dtype="auto", device_map="auto")
            if HF_CACHE_DIR:
                hf_kwargs["cache_dir"] = HF_CACHE_DIR
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **hf_kwargs)
            print("Default attention enabled")
        except Exception as e:
            print(f"WARNING: Flash attention failed: {e}")
            hf_kwargs = dict(torch_dtype="auto", device_map="auto")
            if HF_CACHE_DIR:
                hf_kwargs["cache_dir"] = HF_CACHE_DIR
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **hf_kwargs)
            print("Fallback to default attention")

        proc_kwargs = {}
        if HF_CACHE_DIR:
            proc_kwargs["cache_dir"] = HF_CACHE_DIR
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)

        # Ensure we can build chat prompts even if the fine-tuned processor lacks a template
        self._chat_template_processor = self.processor if hasattr(self.processor, "apply_chat_template") else None
        if self._chat_template_processor is None:
            fallback_kwargs = {}
            if HF_CACHE_DIR:
                fallback_kwargs["cache_dir"] = HF_CACHE_DIR
            try:
                print("WARNING: Processor missing chat template; loading fallback template from base model", level=1)
                self._chat_template_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", **fallback_kwargs)
                if hasattr(self._chat_template_processor, "apply_chat_template"):
                    print("Fallback chat template loaded", level=1)
                else:
                    print("WARNING: Fallback processor also missing chat template; will use basic template", level=1)
                    self._chat_template_processor = None
            except Exception as template_exc:
                print(f"WARNING: Failed to load fallback chat template: {template_exc}")
                self._chat_template_processor = None

        # Warm up the model
        print("Warming up model...")
        self._warmup()
        print("Model ready for inference")

    def _build_chat_prompt(self, messages):
        """Return chat-formatted text compatible with Qwen processors."""
        if self._chat_template_processor and hasattr(self._chat_template_processor, "apply_chat_template"):
            try:
                return self._chat_template_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as template_exc:
                print(f"WARNING: Chat template unavailable on processor: {template_exc}", level=1)
                self._chat_template_processor = None
        return self._basic_chat_template(messages)

    def _basic_chat_template(self, messages):
        """Minimal fallback chat template covering text/image content."""
        segments = []
        for message in messages:
            role = message.get("role", "user")
            role = role if role in {"user", "assistant", "system"} else "user"
            content_bits = []
            for item in message.get("content", []):
                if item.get("type") == "text":
                    content_bits.append(str(item.get("text", "")))
                elif item.get("type") == "image":
                    content_bits.append("<|vision_start|><|image_pad|><|vision_end|>")
            if not content_bits:
                content_bits.append("")
            content = "\n".join(content_bits)
            segments.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        segments.append("<|im_start|>assistant\n")
        return "\n".join(segments)

    def _warmup(self):
        """Warm up the model with a dummy inference"""
        try:
            dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            dummy_img.save("/tmp/dummy.jpg")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/dummy.jpg"},
                    {"type": "text", "text": "What do you see?"}
                ]
            }]

            text = self._build_chat_prompt(messages)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                _ = self.model.generate(**inputs, do_sample=False, max_new_tokens=5)

            os.remove("/tmp/dummy.jpg")
        except Exception as e:
            print(f"Warmup failed: {e} (this is okay)")

    def detect_objects(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """Detect objects in an image"""
        image = Image.open(image_path)
        img_width, img_height = image.size

        prompt = "find the different objects in the image"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }]

        text = self._build_chat_prompt(messages)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=DETECTION_MAX_NEW_TOKENS,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._parse_detections(output_text, confidence_threshold, img_width, img_height)

    def _parse_detections(self, output_text: str, confidence_threshold: float, img_width: int, img_height: int) -> List[Dict]:
        """Parse model output to extract object detections with improved error handling"""
        detections = []
        print(f"Raw model output: {output_text[:200]}...", level=2)

        try:
            json_str = None

            # Strategy 1: Look for complete JSON array
            json_match = re.search(r'\[(?:[^\[\]]|(?:\[[^\]]*\]))*\]', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)

            # Strategy 2: Look for incomplete JSON and try to complete it
            if not json_str:
                incomplete_match = re.search(r'\[\s*\{[^}]*"bbox":\s*\[[^\]]*', output_text)
                if incomplete_match:
                    partial = incomplete_match.group(0)
                    if not partial.endswith(']'):
                        partial += ']'
                    if not partial.endswith('}]'):
                        partial += '}]'
                    json_str = partial

            # Strategy 3: Extract just the first object if array parsing fails
            if not json_str:
                object_match = re.search(r'\{[^}]*"class"[^}]*"confidence"[^}]*"bbox"[^}]*\}', output_text)
                if object_match:
                    json_str = '[' + object_match.group(0) + ']'

            if json_str:
                print(f"Extracted JSON: {json_str}", level=2)

                try:
                    parsed = json.loads(json_str)

                    if isinstance(parsed, list):
                        for detection in parsed:
                            if (isinstance(detection, dict) and
                                all(key in detection for key in ['class', 'confidence', 'bbox'])):

                                confidence = float(detection['confidence'])
                                bbox = detection['bbox']

                                if (confidence >= confidence_threshold and
                                    isinstance(bbox, list) and len(bbox) == 4):

                                    bbox_values = [float(x) for x in bbox]

                                    if any(val > 1.0 for val in bbox_values):
                                        print(f"WARNING: Detected pixel coordinates: {bbox_values}")
                                        print("Converting to normalized coordinates...", level=2)

                                        bbox_clean = [
                                            bbox_values[0] / img_width,
                                            bbox_values[1] / img_height,
                                            bbox_values[2] / img_width,
                                            bbox_values[3] / img_height
                                        ]

                                        bbox_clean = [max(0.0, min(1.0, val)) for val in bbox_clean]
                                        print(f"Converted to normalized: {bbox_clean}", level=2)
                                    else:
                                        bbox_clean = [max(0.0, min(1.0, val)) for val in bbox_values]

                                    x1, y1, x2, y2 = bbox_clean
                                    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
                                    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
                                    bbox_clean = [x_min, y_min, x_max, y_max]

                                    detections.append({
                                        'class': str(detection['class']).strip(),
                                        'confidence': confidence,
                                        'bbox': bbox_clean
                                    })

                except json.JSONDecodeError as je:
                    print(f"WARNING: JSON decode error: {je}")
                    print(f"Problematic JSON: {json_str}", level=2)
                    print("WARNING: Falling back to fine-tune style parsing", level=2)
            else:
                print("WARNING: No JSON pattern found; attempting fine-tune format parsing", level=2)

            if not detections:
                print("Attempting to parse fine-tuning style annotations (label: [x, y, w, h])", level=2)
                pattern = re.compile(
                    r'(?P<class>[A-Za-z0-9_\-]+)\s*(?:\((?P<conf>0?\.\d+|1(?:\.0+)?)\))?\s*:\s*\[\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*,\s*(?P<w>-?\d+(?:\.\d+)?)\s*,\s*(?P<h>-?\d+(?:\.\d+)?)\s*\]',
                    re.IGNORECASE
                )
                matches = list(pattern.finditer(output_text))

                for match in matches:
                    class_label = str(match.group('class')).strip()
                    conf_str = match.group('conf')
                    try:
                        confidence = float(conf_str) if conf_str is not None else 1.0
                    except (TypeError, ValueError):
                        confidence = 1.0

                    if confidence < confidence_threshold:
                        continue

                    try:
                        x1 = float(match.group('x'))
                        y1 = float(match.group('y'))
                        width = float(match.group('w'))
                        height = float(match.group('h'))
                    except (TypeError, ValueError):
                        continue

                    x2 = x1 + width
                    y2 = y1 + height

                    bbox_norm = [
                        max(0.0, min(1.0, coord))
                        for coord in (
                            x1 / img_width,
                            y1 / img_height,
                            x2 / img_width,
                            y2 / img_height,
                        )
                    ]

                    x_min, x_max = sorted((bbox_norm[0], bbox_norm[2]))
                    y_min, y_max = sorted((bbox_norm[1], bbox_norm[3]))

                    detections.append({
                        'class': class_label.strip(),
                        'confidence': confidence,
                        'bbox': [x_min, y_min, x_max, y_max],
                    })

                if detections:
                    print(f"Parsed {len(detections)} detections from fine-tune format", level=2)
                else:
                    print("WARNING: Fallback parsing did not yield any detections", level=2)

        except Exception as e:
            print(f"WARNING: Error parsing detections: {e}")

        print(f"Successfully parsed {len(detections)} detections", level=2)
        return detections


class RoboflowDataset:
    """Enhanced Roboflow dataset handler with ground truth support"""

    def __init__(self, api_key: str):
        self.rf = Roboflow(api_key=api_key)
        self.dataset_path = None
        self.class_names = []
        self.coco_data = None
        self.image_annotations = {}
        self.class_id_to_name = {}
        self.coco_annotation_path = None

    def download_dataset(self, workspace: str, project: str, version: int) -> str:
        """Download dataset from Roboflow"""
        print(f"Downloading dataset: {workspace}/{project} v{version}", level=1)

        try:
            project_obj = self.rf.workspace(workspace).project(project)
            dataset = project_obj.version(version).download("coco")

            self.dataset_path = dataset.location
            print(f"Dataset downloaded to: {self.dataset_path}", level=1)

            self._load_annotations()
            return self.dataset_path

        except Exception as e:
            print(f"ERROR: Failed to download dataset: {e}")
            raise

    def _load_annotations(self):
        """Load COCO annotations and extract class information"""
        try:
            annotation_path = None

            for split in ['test', 'valid']:
                split_dir = os.path.join(self.dataset_path, split)
                if os.path.exists(split_dir):
                    for file in os.listdir(split_dir):
                        if file.endswith('_annotations.coco.json'):
                            annotation_path = os.path.join(split_dir, file)
                            break
                    if annotation_path:
                        break

            if not annotation_path:
                raise FileNotFoundError("No COCO annotation file found!")

            print(f"Loading annotations from: {annotation_path}", level=2)
            self.coco_annotation_path = annotation_path

            with open(annotation_path, 'r') as f:
                self.coco_data = json.load(f)

            self.class_names = [cat['name'] for cat in self.coco_data['categories']]
            self.class_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
            self.filename_to_image = {img['file_name']: img for img in self.coco_data['images']}

            self.image_annotations = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.image_annotations:
                    self.image_annotations[img_id] = []
                self.image_annotations[img_id].append(ann)

            print(f"Found classes: {self.class_names}", level=1)
            print(f"Loaded {len(self.coco_data['images'])} images with {len(self.coco_data['annotations'])} annotations", level=2)

        except Exception as e:
            print(f"WARNING: Could not load annotations: {e}")
            raise

    def get_test_images_with_gt(self, max_images: Optional[int] = None) -> List[Tuple[str, List[Dict]]]:
        """Get list of test images with their ground truth annotations"""
        if not self.dataset_path or not self.coco_data:
            raise ValueError("Dataset not downloaded or annotations not loaded!")

        results = []

        for split in ['test', 'valid']:
            split_dir = os.path.join(self.dataset_path, split)
            if os.path.exists(split_dir):
                for file in os.listdir(split_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(split_dir, file)

                        gt_annotations = []
                        if hasattr(self, 'filename_to_image') and file in self.filename_to_image:
                            img_info = self.filename_to_image[file]
                            img_id = img_info['id']

                            if img_id in self.image_annotations:
                                for ann in self.image_annotations[img_id]:
                                    x, y, w, h = ann['bbox']
                                    x_min = x / img_info['width']
                                    y_min = y / img_info['height']
                                    x_max = (x + w) / img_info['width']
                                    y_max = (y + h) / img_info['height']

                                    gt_annotations.append({
                                        'class': self.class_id_to_name[ann['category_id']],
                                        'bbox': [x_min, y_min, x_max, y_max]
                                    })

                        results.append((image_path, gt_annotations))
                break

        if max_images and len(results) > max_images:
            results = results[:max_images]

        print(f"Found {len(results)} images with ground truth for evaluation", level=1)
        return results


class COCOObjectDetectionEvaluator:
    """COCO-standard object detection evaluation using pycocotools"""

    def __init__(self, class_names: List[str], original_coco_path: str = None):
        self.class_names = class_names
        self.original_coco_path = original_coco_path

        self.collapse_classes = False
        self.collapsed_class_name = 'gun'
        try:
            self.collapse_classes = bool(COLLAPSE_CLASSES_TO_WEAPON)
            self.collapsed_class_name = str(COLLAPSED_CLASS_NAME)
        except Exception:
            pass

        self.class_to_id = None
        self.id_to_class = None
        if not self.collapse_classes:
            try:
                if original_coco_path and os.path.exists(original_coco_path):
                    with open(original_coco_path, 'r') as f:
                        coco_json = json.load(f)
                    name_to_orig_id = {c['name']: c['id'] for c in coco_json.get('categories', [])}
                    mapped = {name: name_to_orig_id[name] for name in class_names if name in name_to_orig_id}
                    if len(mapped) == len(class_names):
                        self.class_to_id = mapped
                        self.id_to_class = {v: k for k, v in mapped.items()}
            except Exception:
                pass

            if self.class_to_id is None:
                self.class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}
                self.id_to_class = {idx + 1: name for idx, name in enumerate(class_names)}
        else:
            self.class_to_id = {self.collapsed_class_name: 1}
            self.id_to_class = {1: self.collapsed_class_name}

    def _create_coco_gt_data(self, all_ground_truth: List[Dict], image_info: List[Dict]) -> Dict:
        """Create COCO ground truth format"""
        categories = [
            {"id": cat_id, "name": class_name, "supercategory": "object"}
            for class_name, cat_id in self.class_to_id.items()
        ]

        if not image_info:
            unique_images = set()
            for gt in all_ground_truth:
                if 'image_id' in gt:
                    unique_images.add(gt['image_id'])

            images = [
                {"id": img_id, "width": 640, "height": 640, "file_name": f"image_{img_id}.jpg"}
                for img_id in unique_images
            ]
        else:
            images = image_info

        annotations = []
        ann_id = 1

        for gt in all_ground_truth:
            img_id = gt.get('image_id', 1)
            img_width = gt.get('image_width', 640)
            img_height = gt.get('image_height', 640)

            bbox_norm = gt['bbox']
            x1, y1, x2, y2 = bbox_norm

            x1_pixel = x1 * img_width
            y1_pixel = y1 * img_height
            x2_pixel = x2 * img_width
            y2_pixel = y2 * img_height

            bbox_coco = [
                x1_pixel,
                y1_pixel,
                x2_pixel - x1_pixel,
                y2_pixel - y1_pixel
            ]

            if self.collapse_classes:
                category_id = list(self.class_to_id.values())[0]
            else:
                category_id = self.class_to_id[gt['class']]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox_coco,
                "area": bbox_coco[2] * bbox_coco[3],
                "iscrowd": 0
            })
            ann_id += 1

        return {
            "info": {
                "description": "COCO-format dataset for Qwen VLM evaluation",
                "url": "https://github.com/QwenLM/Qwen-VL",
                "version": "1.0",
                "year": 2024,
                "contributor": "Qwen VLM Evaluation",
                "date_created": "2024-01-01 00:00:00"
            },
            "licenses": [
                {
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License"
                }
            ],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

    def _create_coco_predictions(self, all_predictions: List[Dict]) -> List[Dict]:
        """Create COCO predictions format"""
        coco_predictions = []

        for pred in all_predictions:
            img_id = pred.get('image_id', 1)
            img_width = pred.get('image_width', 640)
            img_height = pred.get('image_height', 640)

            bbox_norm = pred['bbox']
            x1, y1, x2, y2 = bbox_norm

            x1_pixel = x1 * img_width
            y1_pixel = y1 * img_height
            x2_pixel = x2 * img_width
            y2_pixel = y2 * img_height

            bbox_coco = [
                x1_pixel,
                y1_pixel,
                x2_pixel - x1_pixel,
                y2_pixel - y1_pixel
            ]

            if self.collapse_classes:
                category_id = list(self.class_to_id.values())[0]
            else:
                category_id = self.class_to_id.get(pred['class'], list(self.class_to_id.values())[0])

            coco_predictions.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox_coco,
                "score": pred['confidence']
            })

        return coco_predictions

    def evaluate_dataset(self, all_predictions: List[Dict], all_ground_truth: List[Dict] = None,
                        image_info: List[Dict] = None) -> Dict:
        """Evaluate entire dataset using COCO evaluation"""
        print("Converting data to COCO format...")

        if (not self.collapse_classes) and self.original_coco_path and os.path.exists(self.original_coco_path):
            print("Using original COCO annotation file (more efficient & accurate)")
            gt_file_path = self.original_coco_path
            cleanup_gt_file = False
        else:
            print("Creating COCO ground truth from processed data (class-agnostic: {collapsed})...".format(
                collapsed=self.collapse_classes
            ))
            coco_gt_data = self._create_coco_gt_data(all_ground_truth, image_info)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
                json.dump(coco_gt_data, gt_file)
                gt_file_path = gt_file.name
            cleanup_gt_file = True

        coco_predictions = self._create_coco_predictions(all_predictions)

        try:
            cat_ids = set(self.class_to_id.values())
            bad_pred_cats = [p for p in coco_predictions if p.get('category_id') not in cat_ids]
            if bad_pred_cats:
                print(f"WARNING: {len(bad_pred_cats)} predictions have unknown category_id; check class mapping")
            if self.original_coco_path and os.path.exists(self.original_coco_path):
                with open(self.original_coco_path, 'r') as f:
                    _coco_json = json.load(f)
                gt_img_ids = {img['id'] for img in _coco_json.get('images', [])}
                bad_pred_imgs = [p for p in coco_predictions if p.get('image_id') not in gt_img_ids]
                if bad_pred_imgs:
                    print(f"WARNING: {len(bad_pred_imgs)} predictions reference unknown image_id; they will be ignored by COCO")
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
            json.dump(coco_predictions, pred_file)
            pred_file_path = pred_file.name

        try:
            print("Running COCO evaluation...", level=2)
            print(f"Ground truth file: {gt_file_path}", level=2)
            print(f"Predictions file: {pred_file_path}", level=2)

            if _VERBOSITY == 0:
                _sink = io.StringIO()
                with redirect_stdout(_sink):
                    coco_gt = COCO(gt_file_path)
                    coco_dt = coco_gt.loadRes(pred_file_path)
                    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                    coco_eval.params.maxDets = [1, 10, 100]
            else:
                print("Loading COCO ground truth data...", level=2)
                coco_gt = COCO(gt_file_path)
                print("Loading COCO predictions data...", level=2)
                coco_dt = coco_gt.loadRes(pred_file_path)
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.params.maxDets = [1, 10, 100]

            try:
                if image_info:
                    eval_img_ids = [img.get('id') for img in image_info if 'id' in img]
                else:
                    eval_img_ids = list({
                        *(p.get('image_id') for p in all_predictions if 'image_id' in p),
                        *(g.get('image_id') for g in all_ground_truth if 'image_id' in g)
                    })
                gt_img_ids = set(coco_gt.getImgIds())
                eval_img_ids = [img_id for img_id in eval_img_ids if img_id in gt_img_ids]
                if not eval_img_ids:
                    eval_img_ids = coco_dt.getImgIds()
                coco_eval.params.imgIds = eval_img_ids
                coco_eval.params.catIds = list(self.class_to_id.values())
                print(f"Restricting COCO eval to {len(eval_img_ids)} images and {len(coco_eval.params.catIds)} categories", level=2)
            except Exception:
                pass

            if _VERBOSITY == 0:
                _sink = io.StringIO()
                with redirect_stdout(_sink):
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                print("Summary:", level=2)
                coco_eval.summarize()

            stats = coco_eval.stats
            try:
                stats = list(stats) if stats is not None else []
            except Exception:
                stats = []
            if len(stats) < 12:
                stats = stats + [nan] * (12 - len(stats))

            map_095 = float('nan')
            try:
                precision = coco_eval.eval.get('precision') if hasattr(coco_eval, 'eval') else None
                if isinstance(precision, np.ndarray) and precision.ndim == 5:
                    iou_thrs = coco_eval.params.iouThrs
                    iou_idx = np.where(np.isclose(iou_thrs, 0.95))[0]
                    if iou_idx.size > 0:
                        precision_095 = precision[iou_idx[0]]
                        valid_precision = precision_095[precision_095 > -1]
                        if valid_precision.size > 0:
                            map_095 = float(valid_precision.mean())
            except Exception:
                pass

            total_predictions = len(all_predictions)
            total_ground_truth = len(all_ground_truth)

            pred_class_counts = {}
            gt_class_counts = {}

            for pred in all_predictions:
                class_name = str(pred['class'])
                pred_class_counts[class_name] = pred_class_counts.get(class_name, 0) + 1

            for gt in all_ground_truth:
                class_name = str(gt['class'])
                gt_class_counts[class_name] = gt_class_counts.get(class_name, 0) + 1

            evaluation_summary = {
                'mAP50': stats[1],
                'mAP75': stats[2],
                'mAP95': map_095,
                'average_mAP': stats[0],
                'mAP_0.5_0.95': stats[0],
                'mAP_small': stats[3],
                'mAP_medium': stats[4],
                'mAP_large': stats[5],
                'recall@1': stats[6],
                'recall@10': stats[7],
                'recall@100': stats[8],
                'recall_small': stats[9],
                'recall_medium': stats[10],
                'recall_large': stats[11],
                'total_predictions': total_predictions,
                'total_ground_truth': total_ground_truth,
                'pred_class_counts': pred_class_counts,
                'gt_class_counts': gt_class_counts,
                'full_stats': stats
            }

            print("COCO evaluation completed", level=0)
            return evaluation_summary

        except Exception as e:
            print(f"ERROR: COCO evaluation failed: {e}")
            print(f"Ground truth items: {len(all_ground_truth)}", level=0)
            print(f"Prediction items: {len(all_predictions)}", level=0)

            if all_ground_truth:
                print(f"Sample GT structure: {all_ground_truth[0]}")
            if all_predictions:
                print(f"Sample prediction structure: {all_predictions[0]}")

            raise

        finally:
            try:
                if cleanup_gt_file:
                    os.unlink(gt_file_path)
                os.unlink(pred_file_path)
            except Exception:
                pass


def visualize_detections(image_path: str, detections: List[Dict], ground_truth: List[Dict] = None,
                        title: str = "Object Detection Results", save_path: Optional[str] = None):
    """Enhanced visualization with optional ground truth overlay"""
    image = Image.open(image_path)
    img_width, img_height = image.size

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    class_colors = {}

    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']

        if class_name not in class_colors:
            class_colors[class_name] = colors[len(class_colors) % len(colors)]

        x_min = bbox[0] * img_width
        y_min = bbox[1] * img_height
        width = (bbox[2] - bbox[0]) * img_width
        height = (bbox[3] - bbox[1]) * img_height

        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2,
            edgecolor=class_colors[class_name],
            facecolor='none',
            linestyle='-'
        )
        ax.add_patch(rect)

        label = f"PRED: {class_name}: {confidence:.2f}"
        ax.text(
            x_min, y_min - 5, label,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=class_colors[class_name], alpha=0.7),
            fontsize=9, color='white', weight='bold'
        )

    if ground_truth:
        for gt in ground_truth:
            bbox = gt['bbox']
            class_name = gt['class']

            if class_name not in class_colors:
                class_colors[class_name] = colors[len(class_colors) % len(colors)]

            x_min = bbox[0] * img_width
            y_min = bbox[1] * img_height
            width = (bbox[2] - bbox[0]) * img_width
            height = (bbox[3] - bbox[1]) * img_height

            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2,
                edgecolor=class_colors[class_name],
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)

            label = f"GT: {class_name}"
            ax.text(
                x_min, y_min - 25, label,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=class_colors[class_name], alpha=0.5),
                fontsize=9, color='white', weight='bold'
            )

    gt_info = f", {len(ground_truth)} GT" if ground_truth else ""
    ax.set_title(f"{title}\n{len(detections)} predictions{gt_info} found", fontsize=14, weight='bold')
    ax.axis('off')

    if ground_truth:
        legend_elements = [
            matplotlib.lines.Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='Predictions'),
            matplotlib.lines.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Ground Truth')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=200)
            print(f"Saved visualization: {save_path}", level=1)
        except Exception as _e:
            print(f"WARNING: Failed to save visualization to {save_path}: {_e}", level=1)
    plt.show()


def create_results_summary(all_results: List[Dict]) -> Dict:
    """Create a summary of all inference results"""
    if not all_results:
        return {}

    total_detections = sum(len(result['detections']) for result in all_results)
    total_images = len(all_results)

    class_counts = {}
    confidence_scores = []

    for result in all_results:
        for detection in result['detections']:
            class_name = detection['class']
            confidence = detection['confidence']

            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_scores.append(confidence)

    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    avg_detections_per_image = total_detections / total_images if total_images > 0 else 0

    return {
        'total_images': total_images,
        'total_detections': total_detections,
        'avg_detections_per_image': avg_detections_per_image,
        'avg_confidence': avg_confidence,
        'class_counts': class_counts,
        'confidence_range': {
            'min': min(confidence_scores) if confidence_scores else 0,
            'max': max(confidence_scores) if confidence_scores else 0
        }
    }


def main():
    """Main execution function"""
    # Validate configuration
    if ROBOFLOW_API_KEY == "your_roboflow_api_key_here":
        raise ValueError("ERROR: Please update ROBOFLOW_API_KEY with your actual API key!")

    if WORKSPACE == "your_workspace_name" or PROJECT == "your_project_name":
        raise ValueError("ERROR: Please update WORKSPACE and PROJECT with your actual values!")

    print("Configuration validated", level=1)
    print("Configuration loaded:", level=2)
    print(f"   Model: {MODEL_PATH}", level=2)
    print(f"   Dataset: {WORKSPACE}/{PROJECT} v{VERSION}", level=2)
    print(f"   Max images: {MAX_IMAGES if MAX_IMAGES else 'All'}", level=2)
    print(f"   Viz confidence threshold: {CONFIDENCE_THRESHOLD}", level=2)
    print(f"   Eval confidence threshold: {EVAL_CONFIDENCE_THRESHOLD}", level=2)
    print(f"   Visualization: {ENABLE_VISUALIZATION}", level=2)
    print(f"   Max predictions/image: {MAX_PREDICTIONS_PER_IMAGE}", level=2)
    print(f"   Max new tokens for detection: {DETECTION_MAX_NEW_TOKENS}", level=2)
    print(f"   Collapse classes to single '{COLLAPSED_CLASS_NAME}' for eval: {COLLAPSE_CLASSES_TO_WEAPON}", level=2)
    if HF_CACHE_DIR:
        print(f"   HF cache: {HF_CACHE_DIR}", level=1)

    print("Libraries imported successfully", level=2)
    print(f"PyTorch version: {torch.__version__}", level=2)
    print(f"CUDA available: {torch.cuda.is_available()}", level=2)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}", level=2)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", level=2)

    # Initialize dataset manager
    print("\nSetting up Roboflow dataset...", level=1)
    dataset = RoboflowDataset(ROBOFLOW_API_KEY)

    # Download dataset
    dataset_path = dataset.download_dataset(WORKSPACE, PROJECT, VERSION)
    class_names = dataset.class_names

    if not class_names:
        raise ValueError("ERROR: Could not extract class names from dataset!")

    print(f"Dataset classes: {class_names}", level=1)

    # Initialize Qwen VLM
    print("\nLoading Qwen VLM...", level=1)
    model = QwenVLMInference(MODEL_PATH)

    print("\nModel loaded successfully - Ready for inference", level=1)

    # Get test images with ground truth
    print("\nLoading test images with ground truth annotations...", level=1)
    image_gt_pairs = dataset.get_test_images_with_gt(MAX_IMAGES)

    if not image_gt_pairs:
        raise ValueError("ERROR: No images found in dataset!")

    print(f"Ready to process {len(image_gt_pairs)} images with ground truth for evaluation", level=1)

    # Initialize evaluator
    print("Initializing COCO object detection evaluator...", level=1)
    print("Using original COCO annotations when available", level=2)
    evaluator = COCOObjectDetectionEvaluator(class_names, dataset.coco_annotation_path)

    # Run inference and evaluation
    print("\nRunning object detection inference and evaluation...", level=1)
    print("=" * 70, level=2)

    all_results = []
    all_predictions = []
    all_ground_truth = []
    image_info_list = []
    start_time = time.time()

    _use_tqdm = _VERBOSITY >= 1
    _iterable = tqdm(image_gt_pairs, desc="Processing images", disable=(not _use_tqdm)) if _use_tqdm else image_gt_pairs
    for i, (image_path, gt_annotations) in enumerate(_iterable):
        try:
            print(f"\nImage {i+1}/{len(image_gt_pairs)}: {os.path.basename(image_path)}", level=2)

            image = Image.open(image_path)
            img_width, img_height = image.size
            basename = os.path.basename(image_path)
            if hasattr(dataset, 'filename_to_image') and basename in dataset.filename_to_image:
                image_id = dataset.filename_to_image[basename]['id']
            else:
                image_id = i + 1

            image_info_list.append({
                "id": image_id,
                "width": img_width,
                "height": img_height,
                "file_name": os.path.basename(image_path)
            })

            image_start = time.time()
            detections = model.detect_objects(image_path, EVAL_CONFIDENCE_THRESHOLD)

            if detections:
                detections = sorted(
                    detections,
                    key=lambda d: float(d.get('confidence', 0.0)),
                    reverse=True,
                )
                if MAX_PREDICTIONS_PER_IMAGE is not None:
                    detections = detections[:MAX_PREDICTIONS_PER_IMAGE]
            inference_time = time.time() - image_start

            print(f"Inference time: {inference_time:.2f}s", level=2)
            print(f"Using {len(detections)} predictions (top {MAX_PREDICTIONS_PER_IMAGE}), {len(gt_annotations)} ground truth", level=2)

            for j, detection in enumerate(detections[:3]):
                print(f"   PRED {j+1}. {detection['class']} (confidence: {detection['confidence']:.3f})", level=2)

            if len(detections) > 3:
                print(f"   ... and {len(detections) - 3} more predictions", level=2)

            for j, gt in enumerate(gt_annotations[:3]):
                print(f"   GT {j+1}. {gt['class']}", level=2)

            if len(gt_annotations) > 3:
                print(f"   ... and {len(gt_annotations) - 3} more ground truth", level=2)

            for detection in detections:
                detection['image_id'] = image_id
                detection['image_width'] = img_width
                detection['image_height'] = img_height

            for gt in gt_annotations:
                gt['image_id'] = image_id
                gt['image_width'] = img_width
                gt['image_height'] = img_height

            result = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'image_id': image_id,
                'image_width': img_width,
                'image_height': img_height,
                'detections': detections,
                'ground_truth': gt_annotations,
                'inference_time': inference_time,
                'num_detections': len(detections),
                'num_ground_truth': len(gt_annotations)
            }
            all_results.append(result)

            all_predictions.extend(detections)
            all_ground_truth.extend(gt_annotations)

            if ENABLE_VISUALIZATION:
                print("Saving detection visualization with ground truth...", level=1)
                viz_detections = [d for d in detections if float(d.get('confidence', 0.0)) >= CONFIDENCE_THRESHOLD]
                overlay_filename = f"{i+1:04d}_" + os.path.splitext(os.path.basename(image_path))[0] + "_overlay.png"
                overlay_path = os.path.join(DETECTIONS_DIR, overlay_filename)
                visualize_detections(
                    image_path,
                    viz_detections,
                    gt_annotations,
                    f"Image {i+1}: {os.path.basename(image_path)}",
                    save_path=overlay_path,
                )

        except Exception as e:
            print(f"ERROR: Error processing {image_path}: {e}")
            continue

    total_time = time.time() - start_time

    print("\n" + "=" * 70, level=1)
    print("INFERENCE AND EVALUATION COMPLETE", level=1)
    print("=" * 70, level=1)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", level=1)
    print(f"Average time per image: {total_time/len(image_gt_pairs):.2f}s", level=1)
    print(f"Successfully processed: {len(all_results)}/{len(image_gt_pairs)} images", level=1)
    print(f"Total predictions: {len(all_predictions)}", level=1)
    print(f"Total ground truth: {len(all_ground_truth)}", level=1)

    # Calculate evaluation metrics
    print("\nCalculating evaluation metrics (COCO)...", level=1)
    evaluation_results = evaluator.evaluate_dataset(all_predictions, all_ground_truth, image_info_list)

    summary = create_results_summary(all_results)

    print("\nCOCO EVALUATION RESULTS", level=0)
    print("=" * 70, level=0)

    print("COCO OBJECT DETECTION METRICS:", level=0)
    print(f"   mAP@[0.5:0.95]: {evaluation_results.get('mAP_0.5_0.95', 0):.4f}  (Primary COCO mAP)")
    print(f"   mAP@0.50:       {evaluation_results.get('mAP50', 0):.4f}")
    print(f"   mAP@0.75:       {evaluation_results.get('mAP75', 0):.4f}")
    print(f"   mAP (small):    {evaluation_results.get('mAP_small', 0):.4f}")
    print(f"   mAP (medium):   {evaluation_results.get('mAP_medium', 0):.4f}")
    print(f"   mAP (large):    {evaluation_results.get('mAP_large', 0):.4f}")

    print(f"\nCOCO RECALL METRICS:", level=0)
    print(f"   AR@1:           {evaluation_results.get('recall@1', 0):.4f}")
    print(f"   AR@10:          {evaluation_results.get('recall@10', 0):.4f}")
    print(f"   AR@100:         {evaluation_results.get('recall@100', 0):.4f}")
    print(f"   AR (small):     {evaluation_results.get('recall_small', 0):.4f}")
    print(f"   AR (medium):    {evaluation_results.get('recall_medium', 0):.4f}")
    print(f"   AR (large):     {evaluation_results.get('recall_large', 0):.4f}")

    print(f"\nDATASET STATISTICS:", level=0)
    print(f"   Images processed: {summary['total_images']}")
    print(f"   Total predictions: {evaluation_results['total_predictions']}")
    print(f"   Total ground truth: {evaluation_results['total_ground_truth']}")
    print(f"   Avg predictions per image: {summary['avg_detections_per_image']:.1f}")
    print(f"   Average confidence: {summary['avg_confidence']:.3f}")

    print(f"\nPER-CLASS ANALYSIS:", level=2)
    pred_classes = evaluation_results.get('pred_class_counts', {})
    gt_classes = evaluation_results.get('gt_class_counts', {})

    for class_name in set(list(pred_classes.keys()) + list(gt_classes.keys())):
        pred_count = pred_classes.get(class_name, 0)
        gt_count = gt_classes.get(class_name, 0)
        print(f"   {class_name}: {pred_count} predictions, {gt_count} ground truth")

    print("\nCreating evaluation visualizations...", level=2)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    map_scores = [
        evaluation_results.get('mAP_0.5_0.95', 0),
        evaluation_results.get('mAP50', 0),
        evaluation_results.get('mAP75', 0),
        evaluation_results.get('mAP95', 0)
    ]
    map_labels = ['mAP@0.5:0.95', 'mAP@0.50', 'mAP@0.75', 'mAP@0.95']

    axes[0, 0].bar(map_labels, map_scores, color=['seagreen', 'green', 'orange', 'red'])
    axes[0, 0].set_title('mAP at Different IoU Thresholds', fontweight='bold')
    axes[0, 0].set_ylabel('mAP Score')
    axes[0, 0].set_ylim(0, 1)

    for i, (label, score) in enumerate(zip(map_labels, map_scores)):
        axes[0, 0].text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')

    recall_metrics = [
        evaluation_results.get('recall@1', 0),
        evaluation_results.get('recall@10', 0),
        evaluation_results.get('recall@100', 0)
    ]
    recall_labels = ['AR@1', 'AR@10', 'AR@100']

    axes[0, 1].bar(recall_labels, recall_metrics, color=['lightblue', 'blue', 'darkblue'])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel('Average Recall')
    axes[0, 1].set_title('COCO Average Recall Metrics', fontweight='bold')

    for i, (label, score) in enumerate(zip(recall_labels, recall_metrics)):
        axes[0, 1].text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')

    if pred_classes:
        classes = list(pred_classes.keys())
        pred_counts = [pred_classes.get(c, 0) for c in classes]
        gt_counts = [gt_classes.get(c, 0) for c in classes]

        x = np.arange(len(classes))
        width = 0.35

        axes[1, 0].bar(x - width/2, pred_counts, width, label='Predictions', color='skyblue')
        axes[1, 0].bar(x + width/2, gt_counts, width, label='Ground Truth', color='lightcoral')

        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Predictions vs Ground Truth by Class', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(classes, rotation=45)
        axes[1, 0].legend()

    if all_results:
        image_map50_scores = []
        for result in all_results:
            metrics = result.get('metrics', {})
            map50 = metrics.get('mAP50', 0)
            image_map50_scores.append(map50)

        axes[1, 1].hist(image_map50_scores, bins=20, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('mAP@0.50 Score')
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].set_title('Distribution of Per-Image mAP@0.50 Scores', fontweight='bold')
        axes[1, 1].axvline(np.mean(image_map50_scores), color='red', linestyle='--',
                          label=f'Mean: {np.mean(image_map50_scores):.3f}')
        axes[1, 1].legend()

    plt.tight_layout()
    overview_filename = f"evaluation_overview_{time.strftime('%Y%m%d_%H%M%S')}.png"
    overview_path = os.path.join(FIGURES_DIR, overview_filename)
    try:
        fig.savefig(overview_path, bbox_inches='tight', dpi=200)
        print(f"Saved evaluation overview: {overview_path}", level=1)
    except Exception as _e:
        print(f"WARNING: Failed to save evaluation overview to {overview_path}: {_e}", level=1)
    plt.show()

    print("\nDETAILED COCO METRICS TABLE:", level=2)

    metrics_data = {
        'Metric': [
            'mAP@[0.5:0.95] (Primary COCO)', 'mAP@0.50', 'mAP@0.75',
            'mAP (small)', 'mAP (medium)', 'mAP (large)',
            'AR@1', 'AR@10', 'AR@100'
        ],
        'Score': [
            f"{evaluation_results.get('mAP_0.5_0.95', 0):.4f}",
            f"{evaluation_results.get('mAP50', 0):.4f}",
            f"{evaluation_results.get('mAP75', 0):.4f}",
            f"{evaluation_results.get('mAP_small', 0):.4f}",
            f"{evaluation_results.get('mAP_medium', 0):.4f}",
            f"{evaluation_results.get('mAP_large', 0):.4f}",
            f"{evaluation_results.get('recall@1', 0):.4f}",
            f"{evaluation_results.get('recall@10', 0):.4f}",
            f"{evaluation_results.get('recall@100', 0):.4f}"
        ]
    }

    df_metrics = pd.DataFrame(metrics_data)
    print(df_metrics.to_string(index=False))

    if SAVE_RESULTS and all_results:
        results_data = {
            'dataset_info': {
                'workspace': WORKSPACE,
                'project': PROJECT,
                'version': VERSION,
                'classes': class_names,
                'total_images_processed': len(all_results),
                'total_predictions': len(all_predictions),
                'total_ground_truth': len(all_ground_truth)
            },
            'model_info': {
                'model_path': MODEL_PATH,
                'confidence_threshold': CONFIDENCE_THRESHOLD
            },
            'evaluation_metrics': evaluation_results,
            'summary': summary,
            'detailed_results': all_results,
            'processing_info': {
                'total_time_seconds': total_time,
                'avg_time_per_image': total_time / len(image_gt_pairs),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        results_filename = f"qwen_vlm_evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"

        with open(results_filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\nComplete evaluation results saved to: {results_filename}", level=1)
        print(f"File size: {os.path.getsize(results_filename) / 1024:.1f} KB", level=2)

        csv_filename = f"qwen_vlm_metrics_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        summary_df = pd.DataFrame([{
            'Model': MODEL_PATH,
            'Dataset': f"{WORKSPACE}/{PROJECT}",
            'mAP_COCO_Primary': evaluation_results.get('mAP_0.5_0.95', 0),
            'mAP@0.50': evaluation_results.get('mAP50', 0),
            'mAP@0.75': evaluation_results.get('mAP75', 0),
            'mAP_small': evaluation_results.get('mAP_small', 0),
            'mAP_medium': evaluation_results.get('mAP_medium', 0),
            'mAP_large': evaluation_results.get('mAP_large', 0),
            'AR@10': evaluation_results.get('recall@10', 0),
            'AR@100': evaluation_results.get('recall@100', 0),
            'Total_Predictions': evaluation_results['total_predictions'],
            'Total_GroundTruth': evaluation_results['total_ground_truth'],
            'Avg_Confidence': summary['avg_confidence'],
            'Processing_Time_Minutes': total_time / 60
        }])

        summary_df.to_csv(csv_filename, index=False)
        print(f"Metrics summary saved to: {csv_filename}", level=1)

    print("\nEvaluation complete - Key results:", level=0)
    print(f"   mAP@[0.5:0.95]: {evaluation_results.get('mAP_0.5_0.95', 0):.4f}", level=0)
    print(f"   mAP@0.50: {evaluation_results.get('mAP50', 0):.4f}", level=0)
    print(f"   mAP@0.75: {evaluation_results.get('mAP75', 0):.4f}", level=0)
    print(f"   AR@10: {evaluation_results.get('recall@10', 0):.4f}", level=0)
    print(f"   AR@100: {evaluation_results.get('recall@100', 0):.4f}", level=0)
    print("Use VERBOSITY=1 or 2 for more logs.", level=0)


if __name__ == "__main__":
    main()
