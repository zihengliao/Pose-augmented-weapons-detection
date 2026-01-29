"""Qwen VLM Inference with Roboflow Integration and COCO Evaluation.

This module provides an interface for running object detection inference using
Qwen Vision-Language Models on Roboflow datasets with industry-standard COCO
evaluation metrics.
"""

# Standard library imports
import argparse
import builtins
import io
import json
import math
import os
import re
import tempfile
import time
import warnings
from contextlib import redirect_stdout
from typing import Dict, List, Optional, Tuple

# Third-party imports
import jinja2
import matplotlib.lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from qwen_vl_utils import process_vision_info
from roboflow import Roboflow
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Global verbosity control (quiet by default)
_VERBOSITY = int(os.getenv("VERBOSITY", "0"))  # 0 = quiet, 1 = normal, 2 = verbose
_builtin_print = builtins.print


def print(*args, **kwargs):
    """Module-scoped print respecting VERBOSITY.

    Use kwarg level to mark importance:
    - level=0 always prints
    - level=1 prints in normal mode
    - level=2 prints in verbose mode
    """
    level = kwargs.pop("level", 1)
    if _VERBOSITY >= level or level == 0:
        _builtin_print(*args, **kwargs)


# Ensure compatibility with older Jinja2 versions used in some environments
try:
    if not hasattr(jinja2, "pass_eval_context") and hasattr(jinja2, "evalcontextfilter"):
        # Fall back to evalcontextfilter which provides similar behavior
        jinja2.pass_eval_context = jinja2.evalcontextfilter  # type: ignore[assignment]
except Exception as _jinja_error:  # pragma: no cover - defensive guard
    print(f"Warning: Unable to patch jinja2 for pass_eval_context: {_jinja_error}", level=2)

# Configuration constants
FLASH_ATTN_AVAILABLE = False

# ROBOFLOW CONFIGURATION - UPDATE THESE VALUES!
ROBOFLOW_API_KEY = "your_roboflow_api_key_here"  # Get from roboflow.com
WORKSPACE = "threat-detection-k7wmf"            # Your Roboflow workspace name
PROJECT = "threat-detection-m8dvh"                # Your project name
VERSION = 1                             # Dataset version number

# MODEL CONFIGURATION (runtime-overridable)
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_ADAPTER_PATH: Optional[str] = None  # Optional PEFT adapter (e.g., LoRA checkpoint directory)

# Allow environment variable override
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
ADAPTER_PATH = os.getenv("ADAPTER_PATH", DEFAULT_ADAPTER_PATH)

# INFERENCE CONFIGURATION
MAX_IMAGES = None                         # Maximum number of images to process (set to None for all)
CONFIDENCE_THRESHOLD = 0.8              # Visualization threshold for drawing predictions
EVAL_CONFIDENCE_THRESHOLD = 0.8         # Evaluation threshold (accept all; cap to top-K later)
ENABLE_VISUALIZATION = True             # Show detection results
SAVE_RESULTS = True                     # Save results to JSON file
MAX_PREDICTIONS_PER_IMAGE = 100         # Cap predictions per image (aligns with COCO AR@100)
REQUEST_TOP_K_DETECTIONS = 100          # Ask the model to return up to this many detections per image
DETECTION_MAX_NEW_TOKENS = 8192         # Allow enough tokens for large detection lists

# Evaluation behavior
COLLAPSE_CLASSES_TO_WEAPON = False      # Treat all classes as a single class for evaluation
COLLAPSED_CLASS_NAME = "gun"            # The single class name used when collapsing

# Canonical threat detection classes derived from COCO annotations
CLASS_SPEC = [
    ("80", "small gun", "Compact hand-held firearms such as pistols or revolvers."),
    ("81", "large rifle", "Long firearms like rifles or shotguns with stocks or long barrels."),
    ("82", "hand", "Human hands or fingers that may interact with objects."),
    ("83", "mobile phone", "Handheld phones, including smartphones and cell phones.")
]

CANONICAL_CLASS_LABELS = [label for _, label, _ in CLASS_SPEC]
ORIGINAL_NAME_TO_LABEL = {orig_name: label for orig_name, label, _ in CLASS_SPEC}
CLASS_LABEL_TO_DEFINITION = {label: definition for _, label, definition in CLASS_SPEC}

CLASS_SYNONYMS = {
    "small gun": {
        "small gun", "small guns", "gun", "pistol", "pistols", "handgun", "handguns",
        "revolver", "revolvers", "sidearm", "sidearms", "small firearm", "small firearms"
    },
    "large rifle": {
        "large rifle", "large rifles", "rifle", "rifles", "long gun", "long guns",
        "assault rifle", "assault rifles", "carbine", "carbines", "shotgun", "shotguns",
        "long firearm", "long firearms"
    },
    "hand": {
        "hand", "hands", "palm", "palms", "finger", "fingers", "thumb", "thumbs", "fist", "fists"
    },
    "mobile phone": {
        "mobile phone", "mobile phones", "cell phone", "cell phones", "cellphone", "cellphones",
        "phone", "phones", "smartphone", "smartphones", "mobile", "mobiles"
    }
}


def normalize_class_label(raw_label: str) -> Optional[str]:
    """Convert a raw class label from the model into a canonical class name or None."""
    if not raw_label:
        return None

    label = str(raw_label).strip().lower()
    if not label:
        return None

    for canonical, synonyms in CLASS_SYNONYMS.items():
        if label in synonyms:
            return canonical

    # Handle very simple plural forms (e.g., "hands")
    if label.endswith('s'):
        label_singular = label[:-1]
        for canonical, synonyms in CLASS_SYNONYMS.items():
            if label_singular in synonyms:
                return canonical

    # Keyword heuristics as a fallback
    firearm_large_keywords = ("rifle", "shotgun", "long gun", "long firearm", "carbine", "sniper")
    firearm_small_keywords = ("pistol", "handgun", "revolver", "sidearm", "small gun", "small firearm")

    if any(keyword in label for keyword in firearm_large_keywords):
        return "large rifle"
    if any(keyword in label for keyword in firearm_small_keywords):
        return "small gun"
    if "gun" in label and "large" in label:
        return "large rifle"
    if "gun" in label:
        return "small gun"
    if any(keyword in label for keyword in ("hand", "finger", "palm", "thumb", "fist")):
        return "hand"
    if any(keyword in label for keyword in ("phone", "mobile", "cell")):
        return "mobile phone"

    return None


def _initialize_model_paths():
    """Initialize model paths from command-line arguments or environment variables."""
    global MODEL_PATH, ADAPTER_PATH
    try:
        _parser = argparse.ArgumentParser(add_help=False)
        _parser.add_argument("--model_path", type=str, help="Hugging Face model id")
        _parser.add_argument("--adapter_path", type=str, help="Optional PEFT adapter path")
        _args, _unknown = _parser.parse_known_args()
        if _args.model_path:
            MODEL_PATH = _args.model_path
            print(f"Overriding model via --model_path: {MODEL_PATH}", level=2)
        elif os.getenv("MODEL_PATH"):
            print(f"Overriding model via env MODEL_PATH: {MODEL_PATH}", level=2)

        if _args.adapter_path:
            ADAPTER_PATH = _args.adapter_path
            print(f"Overriding adapter via --adapter_path: {ADAPTER_PATH}", level=2)
        elif os.getenv("ADAPTER_PATH"):
            print(f"Overriding adapter via env ADAPTER_PATH: {ADAPTER_PATH}", level=2)
    except Exception:
        pass


def _initialize_cache_directories():
    """Initialize Hugging Face cache directories if specified."""
    if HF_CACHE_DIR:
        os.environ["HF_HOME"] = HF_CACHE_DIR
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")


def _initialize_output_directories():
    """Initialize output directories for results and visualizations."""
    model_name_safe = MODEL_PATH.split("/")[-1].replace(" ", "_").replace("/", "_")
    if ADAPTER_PATH:
        adapter_name_safe = os.path.basename(os.path.normpath(ADAPTER_PATH)).replace(" ", "_")
        model_output_name = f"{model_name_safe}__{adapter_name_safe}"
    else:
        model_output_name = model_name_safe

    output_root_dir = os.path.join(os.getcwd(), "outputs", model_output_name)
    figures_dir = os.path.join(output_root_dir, "figures")
    detections_dir = os.path.join(output_root_dir, "detections")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_root_dir}", level=2)
    return output_root_dir, figures_dir, detections_dir

class QwenVLMInference:
    """Simplified Qwen VLM inference class for object detection"""

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", adapter_path: Optional[str] = None):
        self.model_path = model_path
        self.adapter_path = adapter_path
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
                print("Flash Attention enabled!")
            else:
                hf_kwargs = dict(torch_dtype="auto", device_map="auto")
            if HF_CACHE_DIR:
                hf_kwargs["cache_dir"] = HF_CACHE_DIR
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **hf_kwargs)
            print("Default attention enabled")
        except Exception as e:
            print(f"Warning: Flash attention failed: {e}")
            hf_kwargs = dict(torch_dtype="auto", device_map="auto")
            if HF_CACHE_DIR:
                hf_kwargs["cache_dir"] = HF_CACHE_DIR
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **hf_kwargs)
            print("Fallback to default attention")

        if self.adapter_path:
            print(f"Loading PEFT adapter from: {self.adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as _import_error:
                raise ImportError(
                    "Adapter path provided but the `peft` package is not installed."
                    " Install it with `pip install peft`."
                ) from _import_error

            try:
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                # Attempt to merge the adapter for inference efficiency when supported
                try:
                    merged_model = self.model.merge_and_unload()
                    if merged_model is not None:
                        self.model = merged_model
                        print("Adapter merged into base model weights")
                    else:
                        print("Adapter kept in PEFT mode (merge returned None)")
                except AttributeError:
                    print("merge_and_unload not available; using adapter in PEFT mode")
            except Exception as adapter_exc:
                raise RuntimeError(f"Failed to load PEFT adapter from {self.adapter_path}: {adapter_exc}")

        proc_kwargs = {}
        if HF_CACHE_DIR:
            proc_kwargs["cache_dir"] = HF_CACHE_DIR
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)

        # Warm up the model
        print("Warming up model...")
        self._warmup()
        print("Model ready for inference!")

    def _warmup(self):
        """Warm up the model with a dummy inference"""
        try:
            # Create a small dummy image
            dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

            # Save temporarily
            dummy_img.save("/tmp/dummy.jpg")

            # Run quick inference
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/dummy.jpg"},
                    {"type": "text", "text": "What do you see?"}
                ]
            }]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

            # Clean up
            os.remove("/tmp/dummy.jpg")

        except Exception as e:
            print(f"Warmup failed: {e} (this is okay)")

    def detect_objects(self, image_path: str, class_names: List[str],
                      confidence_threshold: float = 0.5) -> List[Dict]:
        """Detect objects in an image"""

        # Get image dimensions for potential coordinate conversion
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Create optimized prompt for object detection
        class_instructions = []
        for name in class_names:
            definition = CLASS_LABEL_TO_DEFINITION.get(name, "").strip()
            if definition:
                class_instructions.append(f'- "{name}": {definition}')
            else:
                class_instructions.append(f'- "{name}"')
        class_instructions_text = "\n".join(class_instructions)
        allowed_label_list = ", ".join(f'"{name}"' for name in class_names)

        prompt = f"""You are an object detection system. Detect objects that belong ONLY to the allowed classes listed below. Use the exact label text in quotes when returning detections.

Allowed classes:
{class_instructions_text}

Do not invent or report any other object types. Every detection must use one of: {allowed_label_list}.

Return ONLY a JSON array. NO explanations, NO reasoning, NO additional text.

Format: [{{"class": "class_name", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}}]

Critical requirements:
1. Bounding boxes use normalized coordinates between 0.0 and 1.0
2. (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
3. Confidence scores are between 0.0 and 1.0
4. Report as many valid detections as possible up to {REQUEST_TOP_K_DETECTIONS}, ordered by confidence
5. Output an empty array [] if no objects are present

JSON array:"""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }]

        # Prepare inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=DETECTION_MAX_NEW_TOKENS,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Parse the output to extract detections (pass image dimensions for coordinate conversion)
        return self._parse_detections(output_text, confidence_threshold, img_width, img_height, class_names)

    def _parse_detections(self, output_text: str, confidence_threshold: float,
                          img_width: int, img_height: int,
                          allowed_classes: Optional[List[str]] = None) -> List[Dict]:
        """Parse model output to extract object detections with improved error handling"""
        detections = []

        print(f"Raw model output: {output_text[:200]}...", level=2)  # Debug output
        allowed_set = set(allowed_classes) if allowed_classes else set()

        try:
            # Multiple JSON extraction strategies
            json_str = None

            # Strategy 1: Look for complete JSON array
            json_match = re.search(r'\[(?:[^[\]]|(?:\[[^\]]*\]))*\]', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)

            # Strategy 2: Look for incomplete JSON and try to complete it
            if not json_str:
                incomplete_match = re.search(r'\[\s*\{[^}]*"bbox":\s*\[[^\]]*', output_text)
                if incomplete_match:
                    partial = incomplete_match.group(0)
                    # Try to complete the JSON
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

                                # Validate detection
                                confidence = float(detection['confidence'])
                                bbox = detection['bbox']

                                if (confidence >= confidence_threshold and
                                    isinstance(bbox, list) and len(bbox) == 4):

                                    # Handle both normalized (0-1) and pixel coordinates
                                    bbox_values = [float(x) for x in bbox]

                                    # Check if coordinates are likely pixel values (>1.0)
                                    if any(val > 1.0 for val in bbox_values):
                                        print(f"Warning: Detected pixel coordinates: {bbox_values}")
                                        print("Converting to normalized coordinates...", level=2)

                                        # Convert pixel coordinates to normalized
                                        bbox_clean = [
                                            bbox_values[0] / img_width,   # x1
                                            bbox_values[1] / img_height,  # y1
                                            bbox_values[2] / img_width,   # x2
                                            bbox_values[3] / img_height   # y2
                                        ]

                                        # Ensure coordinates are within valid range
                                        bbox_clean = [max(0.0, min(1.0, val)) for val in bbox_clean]
                                        print(f"Converted to normalized: {bbox_clean}", level=2)
                                    else:
                                        # Already normalized, just ensure valid range
                                        bbox_clean = [max(0.0, min(1.0, val)) for val in bbox_values]

                                    # Ensure coordinate ordering: (x1,y1) is top-left and (x2,y2) is bottom-right
                                    x1, y1, x2, y2 = bbox_clean
                                    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
                                    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
                                    bbox_clean = [x_min, y_min, x_max, y_max]

                                    canonical_class = normalize_class_label(detection['class'])
                                    if canonical_class is None:
                                        print(f"Warning: Skipping detection with unknown class: {detection.get('class')}", level=2)
                                        continue
                                    if allowed_set and canonical_class not in allowed_set:
                                        print(f"Warning: Skipping detection outside allowed classes: {canonical_class}", level=2)
                                        continue

                                    detections.append({
                                        'class': canonical_class,
                                        'confidence': confidence,
                                        'bbox': bbox_clean
                                    })

                except json.JSONDecodeError as je:
                    print(f"Warning: JSON decode error: {je}")
                    print(f"Problematic JSON: {json_str}", level=2)
                    print("Error: No fallback parsing - using strict COCO evaluation standards")
            else:
                print("Warning: No JSON pattern found - no fallback parsing for strict evaluation")

        except Exception as e:
            print(f"Warning: Error parsing detections: {e}")

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
        self.alias_to_category_id = {}
        self.class_definitions = {}
        self.coco_annotation_path = None  # Store path to original COCO file

    def download_dataset(self, workspace: str, project: str, version: int) -> str:
        """Download dataset from Roboflow"""
        print(f"Downloading dataset: {workspace}/{project} v{version}", level=1)

        try:
            project_obj = self.rf.workspace(workspace).project(project)
            dataset = project_obj.version(version).download("coco")

            self.dataset_path = dataset.location
            print(f"Dataset downloaded to: {self.dataset_path}", level=1)

            # Load COCO annotations and class names
            self._load_annotations()

            return self.dataset_path

        except Exception as e:
            print(f"Error: Failed to download dataset: {e}")
            raise

    def _load_annotations(self):
        """Load COCO annotations and extract class information"""
        try:
            # Look for annotation files (prefer test, then valid)
            annotation_path = None
            test_dir = None

            for split in ['test', 'valid']:
                split_dir = os.path.join(self.dataset_path, split)
                if os.path.exists(split_dir):
                    for file in os.listdir(split_dir):
                        if file.endswith('_annotations.coco.json'):
                            annotation_path = os.path.join(split_dir, file)
                            test_dir = split_dir
                            break
                    if annotation_path:
                        break

            if not annotation_path:
                raise FileNotFoundError("No COCO annotation file found!")

            print(f"Loading annotations from: {annotation_path}", level=2)
            self.coco_annotation_path = annotation_path  # Store the path

            with open(annotation_path, 'r') as f:
                self.coco_data = json.load(f)

            # Extract class information with canonical aliases
            self.class_names = []
            self.class_id_to_name = {}
            self.alias_to_category_id = {}

            for cat in self.coco_data['categories']:
                raw_name = str(cat.get('name', '')).strip()
                if not raw_name or raw_name.lower() == "object-detection" or cat.get('id') == 0:
                    continue

                alias = ORIGINAL_NAME_TO_LABEL.get(raw_name, raw_name).lower()
                if alias not in self.alias_to_category_id:
                    self.alias_to_category_id[alias] = cat['id']
                self.class_id_to_name[cat['id']] = alias

            # Preserve canonical ordering defined in CLASS_SPEC
            ordered_aliases = []
            for _, label, _ in CLASS_SPEC:
                alias = label.lower()
                if alias in self.alias_to_category_id:
                    ordered_aliases.append(alias)
            # Fallback to discovered aliases if none matched expected spec
            if not ordered_aliases:
                ordered_aliases = list(self.alias_to_category_id.keys())

            self.class_names = ordered_aliases
            self.class_definitions = {
                alias: CLASS_LABEL_TO_DEFINITION.get(alias, "")
                for alias in self.class_names
            }
            # Map filename to image info for consistent COCO image IDs
            self.filename_to_image = {img['file_name']: img for img in self.coco_data['images']}

            # Group annotations by image ID for easy lookup
            self.image_annotations = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.image_annotations:
                    self.image_annotations[img_id] = []
                self.image_annotations[img_id].append(ann)

            print(f"Found classes: {self.class_names}", level=1)
            print(f"Loaded {len(self.coco_data['images'])} images with {len(self.coco_data['annotations'])} annotations", level=2)

        except Exception as e:
            print(f"Warning: Could not load annotations: {e}")
            raise

    def get_test_images_with_gt(self, max_images: Optional[int] = None) -> List[Tuple[str, List[Dict]]]:
        """Get list of test images with their ground truth annotations"""
        if not self.dataset_path or not self.coco_data:
            raise ValueError("Dataset not downloaded or annotations not loaded!")

        # Create mapping from filename to image info
        filename_to_image = {img['file_name']: img for img in self.coco_data['images']}

        results = []

        # Look for test or valid directories
        for split in ['test', 'valid']:
            split_dir = os.path.join(self.dataset_path, split)
            if os.path.exists(split_dir):
                for file in os.listdir(split_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(split_dir, file)

                        # Get ground truth annotations for this image
                        gt_annotations = []
                        if file in filename_to_image:
                            img_info = filename_to_image[file]
                            img_id = img_info['id']

                            if img_id in self.image_annotations:
                                for ann in self.image_annotations[img_id]:
                                    # Convert COCO format to normalized coordinates
                                    x, y, w, h = ann['bbox']
                                    x_min = x / img_info['width']
                                    y_min = y / img_info['height']
                                    x_max = (x + w) / img_info['width']
                                    y_max = (y + h) / img_info['height']

                                    class_label = self.class_id_to_name.get(ann['category_id'])
                                    if not class_label:
                                        continue

                                    gt_annotations.append({
                                        'class': class_label,
                                        'bbox': [x_min, y_min, x_max, y_max]
                                    })

                        results.append((image_path, gt_annotations))
                break

        # Limit number of images if specified
        if max_images and len(results) > max_images:
            results = results[:max_images]

        print(f"Found {len(results)} images with ground truth for evaluation", level=1)
        return results

class COCOObjectDetectionEvaluator:
    """COCO-standard object detection evaluation using pycocotools"""

    def __init__(self, class_names: List[str], original_coco_path: str = None,
                 alias_to_category_id: Optional[Dict[str, int]] = None):
        self.class_names = class_names
        self.original_coco_path = original_coco_path  # Path to original COCO annotation file
        self.alias_to_category_id = alias_to_category_id or {}

        # Class-collapsing behavior (class-agnostic evaluation)
        self.collapse_classes = False
        self.collapsed_class_name = 'gun'
        try:
            # Use global config if available
            self.collapse_classes = bool(COLLAPSE_CLASSES_TO_WEAPON)
            self.collapsed_class_name = str(COLLAPSED_CLASS_NAME)
        except Exception:
            pass

        # If we have the original COCO file, align category IDs to it to avoid mismatches
        self.class_to_id = None
        self.id_to_class = None
        if not self.collapse_classes:
            if self.alias_to_category_id:
                mapped_aliases = {
                    name: self.alias_to_category_id[name]
                    for name in class_names
                    if name in self.alias_to_category_id
                }
                if len(mapped_aliases) == len(class_names):
                    self.class_to_id = mapped_aliases
                    self.id_to_class = {v: k for k, v in mapped_aliases.items()}

            try:
                if self.class_to_id is None and original_coco_path and os.path.exists(original_coco_path):
                    with open(original_coco_path, 'r') as f:
                        coco_json = json.load(f)
                    # Build mapping from category name to original COCO id
                    name_to_orig_id = {c['name']: c['id'] for c in coco_json.get('categories', [])}
                    # Only keep classes we care about; if a name is missing, fallback later
                    mapped = {}
                    for name in class_names:
                        if name in name_to_orig_id:
                            mapped[name] = name_to_orig_id[name]
                            continue
                        for orig_name, alias in ORIGINAL_NAME_TO_LABEL.items():
                            if alias.lower() == name and orig_name in name_to_orig_id:
                                mapped[name] = name_to_orig_id[orig_name]
                                break
                    if len(mapped) == len(class_names):
                        self.class_to_id = mapped
                        self.id_to_class = {v: k for k, v in mapped.items()}
            except Exception:
                # Fallback to sequential IDs below if any error
                pass

            # Fallback: sequential IDs starting from 1 (COCO-style) if original mapping unavailable
            if self.class_to_id is None:
                self.class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}
                self.id_to_class = {idx + 1: name for idx, name in enumerate(class_names)}
        else:
            # Collapsed class mapping: single category
            self.class_to_id = {self.collapsed_class_name: 1}
            self.id_to_class = {1: self.collapsed_class_name}

    def _create_coco_gt_data(self, all_ground_truth: List[Dict], image_info: List[Dict]) -> Dict:
        """Create COCO ground truth format"""
        # Create categories (single category if collapsed)
        categories = [
            {"id": cat_id, "name": class_name, "supercategory": "object"}
            for class_name, cat_id in self.class_to_id.items()
        ]
        
        # Create images (use provided image info or create dummy)
        if not image_info:
            # Create dummy image info if not provided
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
        
        # Create annotations
        annotations = []
        ann_id = 1
        
        for gt in all_ground_truth:
            # Get image dimensions for bbox conversion
            img_id = gt.get('image_id', 1)
            img_width = gt.get('image_width', 640)
            img_height = gt.get('image_height', 640)
            
            # Convert normalized bbox to COCO format [x, y, width, height] in pixels
            bbox_norm = gt['bbox']  # [x1, y1, x2, y2] normalized
            x1, y1, x2, y2 = bbox_norm
            
            # Convert to pixel coordinates
            x1_pixel = x1 * img_width
            y1_pixel = y1 * img_height
            x2_pixel = x2 * img_width
            y2_pixel = y2 * img_height
            
            # COCO format: [x, y, width, height]
            bbox_coco = [
                x1_pixel,
                y1_pixel,
                x2_pixel - x1_pixel,
                y2_pixel - y1_pixel
            ]
            
            # Category id: collapsed or use provided class mapping
            if self.collapse_classes:
                category_id = list(self.class_to_id.values())[0]
            else:
                category_id = self.class_to_id.get(gt['class'])
                if category_id is None:
                    continue

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
            # Get image dimensions for bbox conversion
            img_id = pred.get('image_id', 1)
            img_width = pred.get('image_width', 640)
            img_height = pred.get('image_height', 640)
            
            # Convert normalized bbox to COCO format [x, y, width, height] in pixels
            bbox_norm = pred['bbox']  # [x1, y1, x2, y2] normalized
            x1, y1, x2, y2 = bbox_norm
            
            # Convert to pixel coordinates
            x1_pixel = x1 * img_width
            y1_pixel = y1 * img_height
            x2_pixel = x2 * img_width
            y2_pixel = y2 * img_height
            
            # COCO format: [x, y, width, height]
            bbox_coco = [
                x1_pixel,
                y1_pixel,
                x2_pixel - x1_pixel,
                y2_pixel - y1_pixel
            ]
            
            # Category id: collapsed or use provided class mapping
            if self.collapse_classes:
                category_id = list(self.class_to_id.values())[0]
            else:
                category_id = self.class_to_id.get(pred['class'])
                if category_id is None:
                    continue

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
        
        # Use original COCO file if available and not collapsing classes; otherwise create from scratch
        if (not self.collapse_classes) and self.original_coco_path and os.path.exists(self.original_coco_path):
            print("Using original COCO annotation file (more efficient & accurate)")
            gt_file_path = self.original_coco_path
            cleanup_gt_file = False  # Don't delete original file
        else:
            print("Creating COCO ground truth from processed data (class-agnostic: {collapsed})...".format(
                collapsed=self.collapse_classes
            ))
            # Create COCO ground truth data
            coco_gt_data = self._create_coco_gt_data(all_ground_truth, image_info)
            
            # Write ground truth to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
                json.dump(coco_gt_data, gt_file)
                gt_file_path = gt_file.name
            cleanup_gt_file = True  # Clean up temporary file
        
        # Create COCO predictions
        coco_predictions = self._create_coco_predictions(all_predictions)

        # Optional: quick validation to detect common mismatches before eval
        try:
            # Validate category ids exist
            cat_ids = set(self.class_to_id.values())
            bad_pred_cats = [p for p in coco_predictions if p.get('category_id') not in cat_ids]
            if bad_pred_cats:
                print(f"Warning: {len(bad_pred_cats)} predictions have unknown category_id; check class mapping")
            # Validate image ids exist in GT (if using original COCO file)
            if self.original_coco_path and os.path.exists(self.original_coco_path):
                with open(self.original_coco_path, 'r') as f:
                    _coco_json = json.load(f)
                gt_img_ids = {img['id'] for img in _coco_json.get('images', [])}
                bad_pred_imgs = [p for p in coco_predictions if p.get('image_id') not in gt_img_ids]
                if bad_pred_imgs:
                    print(f"Warning: {len(bad_pred_imgs)} predictions reference unknown image_id; they will be ignored by COCO")
        except Exception:
            pass
        
        # Write predictions to temporary file
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
                # Load COCO ground truth
                print("Loading COCO ground truth data...", level=2)
                coco_gt = COCO(gt_file_path)
                # Load COCO predictions  
                print("Loading COCO predictions data...", level=2)
                coco_dt = coco_gt.loadRes(pred_file_path)
                # Initialize COCO evaluation
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.params.maxDets = [1, 10, 100]

            # Restrict evaluation to the images we actually processed
            try:
                if image_info:
                    eval_img_ids = [img.get('id') for img in image_info if 'id' in img]
                else:
                    eval_img_ids = list({
                        *(p.get('image_id') for p in all_predictions if 'image_id' in p),
                        *(g.get('image_id') for g in all_ground_truth if 'image_id' in g)
                    })
                # Keep only IDs present in GT to avoid mismatches
                gt_img_ids = set(coco_gt.getImgIds())
                eval_img_ids = [img_id for img_id in eval_img_ids if img_id in gt_img_ids]
                # Fallback: if nothing left, use the IDs known to the detection results
                if not eval_img_ids:
                    eval_img_ids = coco_dt.getImgIds()
                coco_eval.params.imgIds = eval_img_ids
                # Also restrict categories to our classes
                coco_eval.params.catIds = list(self.class_to_id.values())
                print(f"Restricting COCO eval to {len(eval_img_ids)} images and {len(coco_eval.params.catIds)} categories", level=2)
            except Exception as _:
                pass

            # Run evaluation
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

            # Extract metrics
            # COCO stats: [AP@0.5:0.95, AP@0.5, AP@0.75, AP_small, AP_medium, AP_large, AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large]
            stats = coco_eval.stats
            # Safety: ensure stats is a list of length 12 to avoid indexing errors
            try:
                stats = list(stats) if stats is not None else []
            except Exception:
                stats = []
            if len(stats) < 12:
                # pad with NaNs so downstream indexing doesn't crash
                stats = stats + [math.nan] * (12 - len(stats))
            
            # Count statistics
            total_predictions = len(all_predictions)
            total_ground_truth = len(all_ground_truth)

            # Count per-class statistics
            pred_class_counts = {}
            gt_class_counts = {}

            for pred in all_predictions:
                class_name = pred['class'].lower()
                pred_class_counts[class_name] = pred_class_counts.get(class_name, 0) + 1

            for gt in all_ground_truth:
                class_name = gt['class'].lower()
                gt_class_counts[class_name] = gt_class_counts.get(class_name, 0) + 1

            # Create comprehensive results dictionary
            evaluation_summary = {
                # Main COCO metrics (converted to match previous format)
                'mAP50': stats[1],  # AP@IoU=0.50
                'mAP75': stats[2],  # AP@IoU=0.75
                'mAP95': stats[0],  # AP@IoU=0.50:0.95 (this is the main COCO mAP)
                'average_mAP': stats[0],  # Use COCO's main mAP as average
                
                # Additional COCO metrics
                'mAP_0.5_0.95': stats[0],  # AP@IoU=0.50:0.95
                'mAP_small': stats[3],     # AP for small objects
                'mAP_medium': stats[4],    # AP for medium objects  
                'mAP_large': stats[5],     # AP for large objects
                
                # Recall metrics
                'recall@1': stats[6],      # AR@maxDets=1
                'recall@10': stats[7],     # AR@maxDets=10
                'recall@100': stats[8],    # AR@maxDets=100
                'recall_small': stats[9],  # AR for small objects
                'recall_medium': stats[10], # AR for medium objects
                'recall_large': stats[11],  # AR for large objects
                

                
                # Dataset statistics
                'total_predictions': total_predictions,
                'total_ground_truth': total_ground_truth,
                'pred_class_counts': pred_class_counts,
                'gt_class_counts': gt_class_counts,
                
                'full_stats': stats
            }
            
            print("COCO evaluation completed.", level=0)

            return evaluation_summary
            
        except Exception as e:
            print(f"Error: COCO evaluation failed: {e}")
            print(f"Ground truth items: {len(all_ground_truth)}", level=0)
            print(f"Prediction items: {len(all_predictions)}", level=0)
            
            # For debugging, print sample data structure
            if all_ground_truth:
                print(f"Sample GT structure: {all_ground_truth[0]}")
            if all_predictions:
                print(f"Sample prediction structure: {all_predictions[0]}")
            
            raise  # Re-raise the exception for debugging
            
        finally:
            # Clean up temporary files
            try:
                if cleanup_gt_file:  # Only delete if we created a temporary GT file
                    os.unlink(gt_file_path)
                os.unlink(pred_file_path)
            except:
                pass



def visualize_detections(image_path: str, detections: List[Dict], ground_truth: List[Dict] = None,
                        title: str = "Object Detection Results", save_path: Optional[str] = None):
    """Enhanced visualization with optional ground truth overlay"""

    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # Colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    class_colors = {}

    # Draw predictions in solid lines
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']

        # Assign color to class
        if class_name not in class_colors:
            class_colors[class_name] = colors[len(class_colors) % len(colors)]

        # Convert normalized coordinates to pixel coordinates
        x_min = bbox[0] * img_width
        y_min = bbox[1] * img_height
        width = (bbox[2] - bbox[0]) * img_width
        height = (bbox[3] - bbox[1]) * img_height

        # Draw prediction bounding box
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2,
            edgecolor=class_colors[class_name],
            facecolor='none',
            linestyle='-'
        )
        ax.add_patch(rect)

        # Add prediction label
        label = f"PRED: {class_name}: {confidence:.2f}"
        ax.text(
            x_min, y_min - 5, label,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=class_colors[class_name], alpha=0.7),
            fontsize=9, color='white', weight='bold'
        )

    # Draw ground truth in dashed lines
    if ground_truth:
        for gt in ground_truth:
            bbox = gt['bbox']
            class_name = gt['class']

            # Assign color to class
            if class_name not in class_colors:
                class_colors[class_name] = colors[len(class_colors) % len(colors)]

            # Convert normalized coordinates to pixel coordinates
            x_min = bbox[0] * img_width
            y_min = bbox[1] * img_height
            width = (bbox[2] - bbox[0]) * img_width
            height = (bbox[3] - bbox[1]) * img_height

            # Draw ground truth bounding box
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2,
                edgecolor=class_colors[class_name],
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)

            # Add ground truth label
            label = f"GT: {class_name}"
            ax.text(
                x_min, y_min - 25, label,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=class_colors[class_name], alpha=0.5),
                fontsize=9, color='white', weight='bold'
            )

    # Update title to show both predictions and ground truth info
    gt_info = f", {len(ground_truth)} GT" if ground_truth else ""
    ax.set_title(f"{title}\n{len(detections)} predictions{gt_info} found", fontsize=14, weight='bold')
    ax.axis('off')

    # Add legend
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
            print(f"Warning: Failed to save visualization to {save_path}: {_e}", level=1)
    plt.show()

def create_results_summary(all_results: List[Dict]) -> Dict:
    """Create a summary of all inference results"""

    if not all_results:
        return {}

    total_detections = sum(len(result['detections']) for result in all_results)
    total_images = len(all_results)

    # Count detections per class
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
    """Main execution function for Qwen VLM inference and evaluation."""
    # Initialize model paths from command-line or environment
    _initialize_model_paths()
    
    # Initialize cache directories
    _initialize_cache_directories()
    
    # Initialize output directories
    output_root_dir, figures_dir, detections_dir = _initialize_output_directories()
    
    # Print configuration
    print("Configuration loaded:", level=2)
    print(f"   Model: {MODEL_PATH}", level=2)
    print(f"   Dataset: {WORKSPACE}/{PROJECT} v{VERSION}", level=2)
    print(f"   Max images: {MAX_IMAGES if MAX_IMAGES else 'All'}", level=2)
    print(f"   Viz confidence threshold: {CONFIDENCE_THRESHOLD}", level=2)
    print(f"   Eval confidence threshold: {EVAL_CONFIDENCE_THRESHOLD}", level=2)
    print(f"   Visualization: {ENABLE_VISUALIZATION}", level=2)
    print(f"   Max predictions/image: {MAX_PREDICTIONS_PER_IMAGE}", level=2)
    print(f"   Request top-K detections: {REQUEST_TOP_K_DETECTIONS}", level=2)
    print(f"   Max new tokens for detection: {DETECTION_MAX_NEW_TOKENS}", level=2)
    print(f"   Collapse classes to single '{COLLAPSED_CLASS_NAME}' for eval: {COLLAPSE_CLASSES_TO_WEAPON}", level=2)
    if HF_CACHE_DIR:
        print(f"   HF cache: {HF_CACHE_DIR}", level=1)
    if ADAPTER_PATH:
        print(f"   Adapter: {ADAPTER_PATH}", level=2)
    else:
        print("   Adapter: none", level=2)
    
    # Validate configuration
    if ROBOFLOW_API_KEY == "your_api_key_here" or ROBOFLOW_API_KEY == "your_roboflow_api_key_here":
        raise ValueError("Error: Please update ROBOFLOW_API_KEY with your actual API key!")

    if WORKSPACE == "your_workspace" or PROJECT == "your_project":
        raise ValueError("Error: Please update WORKSPACE and PROJECT with your actual values!")

    print("Configuration validated.", level=1)

    # Initialize dataset manager
    print("\nSetting up Roboflow dataset...", level=1)
    dataset = RoboflowDataset(ROBOFLOW_API_KEY)

    # Download dataset
    dataset_path = dataset.download_dataset(WORKSPACE, PROJECT, VERSION)
    class_names = dataset.class_names

    if not class_names:
        raise ValueError("Error: Could not extract class names from dataset!")

    print(f"Dataset classes: {class_names}", level=1)

    # Initialize Qwen VLM
    print("\nLoading Qwen VLM...", level=1)
    model = QwenVLMInference(MODEL_PATH, adapter_path=ADAPTER_PATH)

    print("\nModel loaded successfully. Ready for inference.", level=1)

    # Get test images with ground truth
    print("\nLoading test images with ground truth annotations...", level=1)
    image_gt_pairs = dataset.get_test_images_with_gt(MAX_IMAGES)

    if not image_gt_pairs:
        raise ValueError("Error: No images found in dataset!")

    print(f"Ready to process {len(image_gt_pairs)} images with ground truth for evaluation", level=1)

    # Initialize evaluator with original COCO file path for maximum efficiency
    print("Initializing COCO object detection evaluator...", level=1)
    print("Using original COCO annotations when available", level=2)
    evaluator = COCOObjectDetectionEvaluator(
        class_names,
        dataset.coco_annotation_path,
        alias_to_category_id=dataset.alias_to_category_id
    )

    # Run inference and evaluation on all images
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

            # Get image dimensions and the original COCO image id for evaluation
            image = Image.open(image_path)
            img_width, img_height = image.size
            basename = os.path.basename(image_path)
            # Prefer original COCO image id to align with ground truth file
            if hasattr(dataset, 'filename_to_image') and basename in dataset.filename_to_image:
                image_id = dataset.filename_to_image[basename]['id']
            else:
                image_id = i + 1  # fallback
            
            # Store image info for COCO evaluation
            image_info_list.append({
                "id": image_id,
                "width": img_width,
                "height": img_height,
                "file_name": os.path.basename(image_path)
            })

            # Run detection
            image_start = time.time()
            # For evaluation, accept all predictions (eval threshold = 0.0); cap to top-K later
            detections = model.detect_objects(image_path, class_names, EVAL_CONFIDENCE_THRESHOLD)
            
            # Sort by confidence and cap to top-K per image to align with COCO AR@100
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

            # Show detection details
            for j, detection in enumerate(detections[:3]):  # Show first 3
                print(f"   PRED {j+1}. {detection['class']} (confidence: {detection['confidence']:.3f})", level=2)

            if len(detections) > 3:
                print(f"   ... and {len(detections) - 3} more predictions", level=2)

            # Show ground truth details
            for j, gt in enumerate(gt_annotations[:3]):  # Show first 3
                print(f"   GT {j+1}. {gt['class']}", level=2)

            if len(gt_annotations) > 3:
                print(f"   ... and {len(gt_annotations) - 3} more ground truth", level=2)

            # Add image metadata to detections and ground truth for COCO evaluation
            for detection in detections:
                detection['image_id'] = image_id
                detection['image_width'] = img_width
                detection['image_height'] = img_height
                
            for gt in gt_annotations:
                gt['image_id'] = image_id
                gt['image_width'] = img_width
                gt['image_height'] = img_height

            # Store results
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

            # Accumulate for overall evaluation
            all_predictions.extend(detections)
            all_ground_truth.extend(gt_annotations)

            # Visualize results for first few images (use visualization confidence threshold)
            if ENABLE_VISUALIZATION and i < 3:
                print("Showing detection visualization with ground truth...", level=1)
                # Filter for visualization only
                viz_detections = [d for d in detections if float(d.get('confidence', 0.0)) >= CONFIDENCE_THRESHOLD]
                # Save overlay per model under detections dir
                overlay_filename = f"{i+1:04d}_" + os.path.splitext(os.path.basename(image_path))[0] + "_overlay.png"
                overlay_path = os.path.join(detections_dir, overlay_filename)
                visualize_detections(
                    image_path,
                    viz_detections,
                    gt_annotations,
                    f"Image {i+1}: {os.path.basename(image_path)}",
                    save_path=overlay_path,
                )

        except Exception as e:
            print(f"Error: Error processing {image_path}: {e}")
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

    # Calculate comprehensive evaluation metrics
    print("\nCalculating evaluation metrics (COCO)...", level=1)
    evaluation_results = evaluator.evaluate_dataset(all_predictions, all_ground_truth, image_info_list)

    # Generate basic summary
    summary = create_results_summary(all_results)

    print("\nCOCO EVALUATION RESULTS", level=0)
    print("=" * 70, level=0)

    # Main COCO evaluation metrics
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

    # Dataset statistics
    print(f"\nDATASET STATISTICS:", level=0)
    print(f"   Images processed: {summary['total_images']}")
    print(f"   Total predictions: {evaluation_results['total_predictions']}")
    print(f"   Total ground truth: {evaluation_results['total_ground_truth']}")
    print(f"   Avg predictions per image: {summary['avg_detections_per_image']:.1f}")
    print(f"   Average confidence: {summary['avg_confidence']:.3f}")

    # Per-class analysis
    print(f"\nPER-CLASS ANALYSIS:", level=2)
    pred_classes = evaluation_results.get('pred_class_counts', {})
    gt_classes = evaluation_results.get('gt_class_counts', {})

    for class_name in set(list(pred_classes.keys()) + list(gt_classes.keys())):
        pred_count = pred_classes.get(class_name, 0)
        gt_count = gt_classes.get(class_name, 0)
        print(f"   {class_name}: {pred_count} predictions, {gt_count} ground truth")

    # Visualization of results
    print("\nCreating evaluation visualizations...", level=2)

    # Create comprehensive results visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. mAP at different IoU thresholds
    map_scores = [
        evaluation_results.get('mAP50', 0),
        evaluation_results.get('mAP75', 0),
        evaluation_results.get('mAP95', 0)
    ]
    map_labels = ['mAP@0.50', 'mAP@0.75', 'mAP@0.95']

    axes[0, 0].bar(map_labels, map_scores, color=['green', 'orange', 'red'])
    axes[0, 0].set_title('mAP at Different IoU Thresholds', fontweight='bold')
    axes[0, 0].set_ylabel('mAP Score')
    axes[0, 0].set_ylim(0, 1)

    # Add value labels on bars
    for i, (label, score) in enumerate(zip(map_labels, map_scores)):
        axes[0, 0].text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')

    # 2. COCO Recall Metrics Comparison
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

    # Add value labels on bars
    for i, (label, score) in enumerate(zip(recall_labels, recall_metrics)):
        axes[0, 1].text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')

    # 3. Predictions vs Ground Truth comparison
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

    # 4. Per-image metrics distribution
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
    # Save evaluation overview figure per model
    overview_filename = f"evaluation_overview_{time.strftime('%Y%m%d_%H%M%S')}.png"
    overview_path = os.path.join(figures_dir, overview_filename)
    try:
        fig.savefig(overview_path, bbox_inches='tight', dpi=200)
        print(f"Saved evaluation overview: {overview_path}", level=1)
    except Exception as _e:
        print(f"Warning: Failed to save evaluation overview to {overview_path}: {_e}", level=1)
    plt.show()

    # Create a detailed metrics table
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

    # Save comprehensive results to JSON file
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

        # Also save a summary CSV for easy analysis
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

    print("\nEvaluation complete. Key results:", level=0)
    print(f"   • mAP@[0.5:0.95]: {evaluation_results.get('mAP_0.5_0.95', 0):.4f}", level=0)
    print(f"   • mAP@0.50: {evaluation_results.get('mAP50', 0):.4f}", level=0)
    print(f"   • mAP@0.75: {evaluation_results.get('mAP75', 0):.4f}", level=0)
    print(f"   • AR@10: {evaluation_results.get('recall@10', 0):.4f}", level=0)
    print(f"   • AR@100: {evaluation_results.get('recall@100', 0):.4f}", level=0)
    print("Use VERBOSITY=1 or 2 for more logs.", level=0)


if __name__ == "__main__":
    main()