import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from dotenv import load_dotenv
from roboflow import Roboflow
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenCaptioningSystem:
    def __init__(self, args):
        self.args = args
        self.prompt = self._load_prompt()
        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype(self.device)
        self.model, self.processor = self._load_qwen_model()
        self.dataset_location = self._download_dataset()
        self.generation_kwargs = self._build_generation_kwargs()

    def _load_prompt(self):
        try:
            with open(self.args.prompt_file, "r") as handle:
                return handle.read().strip()
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {self.args.prompt_file}")
            sys.exit(1)

    def _resolve_device(self):
        if self.args.device.lower() == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.args.device

    def _resolve_dtype(self, device):
        dtype_map = {
            "auto": torch.float16 if device.startswith("cuda") else torch.float32,
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        key = self.args.torch_dtype.lower()
        if key not in dtype_map:
            logging.error(f"Unsupported torch dtype: {self.args.torch_dtype}")
            sys.exit(1)
        dtype = dtype_map[key]
        if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            logging.warning("Falling back to float32 on CPU for numerical stability.")
            dtype = torch.float32
        return dtype

    def _load_qwen_model(self):
        model_id = self.args.model_id or DEFAULT_MODEL_ID
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=None,
                trust_remote_code=True,
            )
            model.to(self.device)
            model.eval()
            logging.info(f"Loaded Qwen model '{model_id}' on {self.device} with dtype {self.dtype}.")
            return model, processor
        except Exception as exc:
            logging.error(f"Failed to load Qwen model '{model_id}': {exc}")
            sys.exit(1)

    def _build_generation_kwargs(self):
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        return {
            "max_new_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "repetition_penalty": self.args.repetition_penalty,
            "pad_token_id": pad_id,
        }

    def _download_dataset(self):
        if not self.args.roboflow_api_key:
            logging.error("Roboflow API key is required. Set ROBOFLOW_API_KEY or pass --roboflow-api-key.")
            sys.exit(1)
        try:
            rf = Roboflow(api_key=self.args.roboflow_api_key)

            if hasattr(self.args, "workspace_id") and hasattr(self.args, "project_id"):
                workspace_id = self.args.workspace_id
                project_id = self.args.project_id
                version_num = getattr(self.args, "version_num", 1)
            else:
                url_parts = (self.args.dataset_url or "").rstrip("/").split("/")
                if len(url_parts) >= 3:
                    workspace_id, project_id, version_num = url_parts[-3], url_parts[-2], int(url_parts[-1])
                else:
                    workspace_id = "threat-detection-k7wmf"
                    project_id = "threat-detection-m8dvh"
                    version_num = 1
                    logging.info(f"Using default project: {workspace_id}/{project_id}/v{version_num}")

            project_obj = rf.workspace(workspace_id).project(project_id)
            dataset = project_obj.version(version_num).download("coco")
            logging.info(f"Dataset downloaded to: {dataset.location}")
            return dataset.location
        except Exception as exc:
            logging.error(f"Dataset download failed: {exc}")
            sys.exit(1)

    def run(self):
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        splits = [split.strip() for split in self.args.splits.split(",") if split.strip()]
        for split in splits:
            self._process_split(split)
        logging.info("--- All processing complete! ---")

    def _process_split(self, split_name):
        logging.info(f"--- Processing split: {split_name} ---")

        possible_dirs = [
            Path(self.dataset_location) / split_name / "images",
            Path(self.dataset_location) / split_name,
        ]

        image_dir = next((path for path in possible_dirs if path.exists()), None)
        if not image_dir:
            logging.warning(f"Image directory not found for split '{split_name}'.")
            return

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
        image_files = []
        for pattern in extensions:
            image_files.extend(image_dir.glob(pattern))
            image_files.extend(image_dir.glob(pattern.upper()))
        image_files = sorted(image_files)

        if not image_files:
            logging.warning(f"No images found in {image_dir}.")
            return

        output_file = Path(self.args.output_dir) / f"{split_name}_captions.json"
        coco_data = {"images": [], "annotations": []}
        processed_files = set()

        if output_file.exists():
            try:
                with open(output_file, "r") as handle:
                    coco_data = json.load(handle)
                processed_files = {item["file_name"] for item in coco_data.get("images", [])}
                logging.info(f"Resuming split '{split_name}' with {len(processed_files)} images already processed.")
            except (json.JSONDecodeError, KeyError) as exc:
                logging.warning(f"Failed to read existing output file: {exc}. Starting fresh.")
                coco_data = {"images": [], "annotations": []}
                processed_files = set()

        images_to_process = [path for path in image_files if path.name not in processed_files]
        if self.args.test_mode:
            images_to_process = images_to_process[: self.args.max_images_per_split]

        if not images_to_process:
            logging.info("No new images to process.")
            return

        logging.info(f"Processing {len(images_to_process)} images in split '{split_name}'.")

        failed = []
        image_id_counter = max([img["id"] for img in coco_data.get("images", [])] + [0]) + 1

        for index, image_path in enumerate(tqdm(images_to_process, desc=f"Captioning {split_name}")):
            caption = self._generate_caption(image_path)

            if caption:
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                    coco_data["images"].append(
                        {
                            "id": image_id_counter,
                            "file_name": image_path.name,
                            "width": width,
                            "height": height,
                        }
                    )
                    coco_data["annotations"].append(
                        {
                            "id": image_id_counter,
                            "image_id": image_id_counter,
                            "caption": caption,
                        }
                    )
                    image_id_counter += 1
                except Exception as exc:
                    logging.error(f"Failed to record caption for {image_path}: {exc}")
                    failed.append(str(image_path))
            else:
                failed.append(str(image_path))

            if ((index + 1) % 10 == 0) or ((index + 1) == len(images_to_process)):
                try:
                    with open(output_file, "w") as handle:
                        json.dump(coco_data, handle, indent=2)
                except Exception as exc:
                    logging.error(f"Failed to save progress to {output_file}: {exc}")

        if failed:
            failed_log = Path(self.args.output_dir) / f"{split_name}_failed.log"
            with open(failed_log, "a") as handle:
                handle.write("\n".join(failed) + "\n")
            logging.warning(f"{len(failed)} images failed. See {failed_log} for details.")

        logging.info(f"Split '{split_name}' complete: {len(coco_data['images'])} images captioned.")

    def _generate_caption(self, image_path, max_retries=3, retry_delay=2):
        for attempt in range(max_retries):
            try:
                with Image.open(image_path) as raw_img:
                    image = raw_img.convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ]

                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **self.generation_kwargs)

                prompt_length = inputs["input_ids"].shape[-1]
                generated_ids = output_ids[0][prompt_length:]
                caption = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
                caption = caption.strip('"')

                if not caption:
                    raise ValueError("Empty caption generated")

                return caption.replace("\n", " ")
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Retrying {image_path} due to error: {exc}. Waiting {wait_time}s.")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to caption {image_path} after {max_retries} attempts: {exc}")
                    return None


def main():
    parser = argparse.ArgumentParser(description="Qwen 2.5-VL based image captioning pipeline")

    parser.add_argument("--dataset-url", help="Optional Roboflow dataset URL (workspace/project/version)")
    # parser.add_argument("--workspace-id", default="ailecs-nmbrc", help="Roboflow workspace ID")
    # parser.add_argument("--project-id", default="40k-dataset-pvktf", help="Roboflow project ID")
    # parser.add_argument("--version-num", type=int, default=2, help="Dataset version number")
    parser.add_argument("--workspace-id", default="threat-detection-k7wmf", help="Roboflow workspace ID")
    parser.add_argument("--project-id", default="threat-detection-m8dvh", help="Roboflow project ID")
    parser.add_argument("--version-num", type=int, default=1, help="Dataset version number")
    parser.add_argument("--roboflow-api-key", default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow API key")

    parser.add_argument("--output-dir", default="./output", help="Directory to store generated captions")
    parser.add_argument("--prompt-file", default="prompt.md", help="File containing the captioning prompt")

    parser.add_argument("--model-id", default=os.getenv("QWEN_MODEL_ID", DEFAULT_MODEL_ID), help="Hugging Face model id for Qwen")
    parser.add_argument("--device", default="auto", help="Device to run inference on (auto|cpu|cuda|cuda:0|mps)")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype (auto|float16|float32|bfloat16)")

    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum tokens to generate per caption")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p nucleus sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty")

    parser.add_argument("--splits", default="valid,test,train", help="Comma-separated list of dataset splits to process")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode with limited images")
    parser.add_argument("--max-images-per-split", type=int, default=10, help="Max images per split in test mode")

    args = parser.parse_args()

    if not args.roboflow_api_key:
        logging.error("Roboflow API key is required. Set ROBOFLOW_API_KEY or pass --roboflow-api-key.")
        sys.exit(1)

    system = QwenCaptioningSystem(args)
    system.run()


if __name__ == "__main__":
    main()
