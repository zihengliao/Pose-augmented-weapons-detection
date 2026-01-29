import argparse
import os
import json
import time
import logging
import math
import sys
from pathlib import Path
from collections import deque

from roboflow import Roboflow
import google.generativeai as genai
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from dotenv import load_dotenv

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

MODEL_NAME = 'gemini-2.0-flash'
COST_PER_INPUT_TOKEN = 0.00000010 
COST_PER_OUTPUT_TOKEN = 0.00000040

# --- Main Application Class ---
class CaptioningSystem:
    def __init__(self, args):
        self.args = args
        self.prompt = self._load_prompt()
        self.model = self._configure_gemini()
        self.rate_limiter = self._configure_rate_limiter()
        self.dataset_location = self._download_dataset()

    def _load_prompt(self):
        try:
            with open(self.args.prompt_file, 'r') as f: 
                return f.read()
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {self.args.prompt_file}")
            sys.exit(1)

    def _configure_gemini(self):
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            logging.error("Gemini API key not found in environment variables.")
            sys.exit(1)
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(MODEL_NAME)
            logging.info(f"Successfully configured and using Gemini model: {MODEL_NAME}")
            return model
        except Exception as e:
            logging.error(f"Failed to configure Gemini API: {e}")
            sys.exit(1)

    def _configure_rate_limiter(self):
        return APIRateLimiter(self.args.requests_per_minute, self.args.daily_limit, self.args.max_cost_estimate)

    def _download_dataset(self):
        roboflow_key = os.getenv("ROBOFLOW_API_KEY")
        if not roboflow_key:
            logging.error("Roboflow API key not found in environment variables.")
            sys.exit(1)
        try:
            rf = Roboflow(api_key=roboflow_key)
            
            # Use your specific project structure
            if hasattr(self.args, 'workspace_id') and hasattr(self.args, 'project_id'):
                # Direct workspace/project specification
                workspace_id = self.args.workspace_id
                project_id = self.args.project_id
                version_num = getattr(self.args, 'version_num', 1)
            else:
                # Parse from dataset_url if provided in format: workspace/project/version
                url_parts = self.args.dataset_url.rstrip('/').split('/')
                if len(url_parts) >= 3:
                    workspace_id, project_id, version_num = url_parts[-3], url_parts[-2], int(url_parts[-1])
                else:
                    # Default to your specific project
                    workspace_id = "threat-detection-k7wmf"
                    project_id = "threat-detection-m8dvh"
                    version_num = 1
                    logging.info(f"Using default project: {workspace_id}/{project_id}/v{version_num}")
            
            project_obj = rf.workspace(workspace_id).project(project_id)
            dataset = project_obj.version(version_num).download("coco")
            logging.info(f"Dataset downloaded to: {dataset.location}")
            return dataset.location
        except Exception as e:
            logging.error(f"Dataset download failed: {e}")
            sys.exit(1)

    def run(self):
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        splits_to_process = [s.strip() for s in self.args.splits.split(',')]

        if not self.args.skip_prediction:
            if not self._run_cost_prediction(splits_to_process):
                logging.info("Processing cancelled by user.")
                return

        for split in splits_to_process:
            self._process_split(split)
        
        logging.info("--- All processing complete! ---")

    def _run_cost_prediction(self, splits):
        logging.info("--- Starting Cost Prediction Phase ---")
        predictor = CostPredictor(self.model, self.prompt, self.args.prediction_sample_size)
        all_images = self._get_all_image_paths(splits, check_resume=True)
        
        if self.args.target_images:
            images_for_prediction = all_images[:self.args.target_images]
        else:
            images_for_prediction = all_images

        if not images_for_prediction:
            logging.info("No new images found to process. Skipping cost prediction.")
            return True

        predictor.run_analysis(images_for_prediction)
        predictor.display_report(len(images_for_prediction))
        
        return predictor.get_user_confirmation(self.args.cost_threshold)

    def _get_all_image_paths(self, splits, check_resume=False):
        all_paths = []
        # Support multiple image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        for split in splits:
            # COCO format uses different directory structure
            # Check both: split/images and split (some COCO datasets put images directly in split folder)
            possible_dirs = [
                Path(self.dataset_location) / split / "images",  # Standard structure
                Path(self.dataset_location) / split,             # COCO might put images directly here
            ]
            
            image_dir = None
            for dir_path in possible_dirs:
                if dir_path.exists():
                    image_dir = dir_path
                    break
            
            if not image_dir:
                logging.warning(f"Image directory not found for split '{split}'. Tried: {possible_dirs}")
                continue
            
            logging.info(f"Using image directory: {image_dir}")
            
            image_files = []
            for ext in extensions:
                image_files.extend(image_dir.glob(ext))
                image_files.extend(image_dir.glob(ext.upper()))  # Case insensitive
            image_files = sorted(image_files)
            
            if check_resume:
                output_file = Path(self.args.output_dir) / f"{split}_captions.json"
                if output_file.exists():
                    try:
                        with open(output_file, 'r') as f: 
                            coco_data = json.load(f)
                        processed_files = {img['file_name'] for img in coco_data['images']}
                        image_files = [img for img in image_files if img.name not in processed_files]
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(f"Could not parse existing output file {output_file}: {e}")
            
            all_paths.extend(image_files)
        return all_paths

    def _process_split(self, split_name):
        logging.info(f"--- Processing split: {split_name} ---")
        
        # COCO format uses different directory structure
        # Check both: split/images and split (some COCO datasets put images directly in split folder)
        possible_dirs = [
            Path(self.dataset_location) / split_name / "images",  # Standard structure
            Path(self.dataset_location) / split_name,             # COCO might put images directly here
        ]
        
        image_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                image_dir = dir_path
                break
        
        if not image_dir:
            logging.warning(f"Image directory not found for split '{split_name}'. Tried: {possible_dirs}")
            return
            
        logging.info(f"Using image directory: {image_dir}")
        
        # Support multiple image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(ext))
            image_files.extend(image_dir.glob(ext.upper()))  # Case insensitive
        image_files = sorted(image_files)
        
        if not image_files:
            logging.warning(f"No images found in {image_dir}")
            return
        
        logging.info(f"Found {len(image_files)} images in {split_name}")
        
        output_file = Path(self.args.output_dir) / f"{split_name}_captions.json"
        coco_data = {"images": [], "annotations": []}
        processed_files = set()
        
        if output_file.exists():
            logging.info(f"Resuming from {output_file}")
            try:
                with open(output_file, 'r') as f: 
                    coco_data = json.load(f)
                processed_files = {img['file_name'] for img in coco_data['images']}
                logging.info(f"Already processed {len(processed_files)} images")
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Could not parse existing output file: {e}. Starting fresh.")
                coco_data = {"images": [], "annotations": []}

        images_to_process = [img for img in image_files if img.name not in processed_files]
        if self.args.test_mode:
            images_to_process = images_to_process[:self.args.max_images_per_split]

        if not images_to_process: 
            logging.info("No new images to process.")
            return

        logging.info(f"Processing {len(images_to_process)} new images")

        failed_captions = []
        image_id_counter = max([img['id'] for img in coco_data['images']] + [0]) + 1
        
        progress_bar = tqdm(images_to_process, desc=f"Captioning {split_name}")
        for i, img_path in enumerate(progress_bar):
            if not self.rate_limiter.wait_if_needed(self.args.delay_between_calls):
                logging.warning("Stopping due to rate limit or cost limit.")
                break

            caption, input_tokens, output_tokens = self._get_gemini_caption_with_tokens(img_path)
            self.rate_limiter.record_request(input_tokens or 0, output_tokens or 0)

            if caption:
                try:
                    with Image.open(img_path) as img: 
                        width, height = img.size
                    coco_data["images"].append({
                        "id": image_id_counter, 
                        "file_name": img_path.name, 
                        "width": width, 
                        "height": height
                    })
                    coco_data["annotations"].append({
                        "id": image_id_counter, 
                        "image_id": image_id_counter, 
                        "caption": caption
                    })
                    image_id_counter += 1
                except Exception as e:
                    logging.error(f"Corrupted image {img_path}: {e}")
                    failed_captions.append(str(img_path))
            else:
                failed_captions.append(str(img_path))

            # Save progress every 10 images OR if it's the last image
            if (i + 1) % 10 == 0 or (i + 1) == len(images_to_process):
                try:
                    with open(output_file, 'w') as f:
                        json.dump(coco_data, f, indent=2)
                    logging.info(f"Progress saved: {len(coco_data['images'])} images processed")
                except Exception as e:
                    logging.error(f"Failed to save progress: {e}")

            # Batch delay
            if (i + 1) % self.args.batch_size == 0 and self.args.batch_delay > 0:
                logging.info(f"Batch complete. Pausing for {self.args.batch_delay}s...")
                time.sleep(self.args.batch_delay)

        
        
        if failed_captions:
            failed_log = Path(self.args.output_dir) / f"{split_name}_failed.log"
            with open(failed_log, 'a') as f: 
                f.write("\n".join(failed_captions) + "\n")
            logging.warning(f"{len(failed_captions)} images failed. See {failed_log}")
        
        logging.info(f"Split {split_name} complete: {len(coco_data['images'])} images captioned")

    def _get_gemini_caption_with_tokens(self, image_path, max_retries=3, delay=5):
        """Get caption and return (caption, input_tokens, output_tokens)"""
        for attempt in range(max_retries):
            try:
                try:
                    with Image.open(image_path) as img:
                        img.verify() 
                except (UnidentifiedImageError, Image.DecompressionBombError, IOError) as e:
                    logging.warning(f"Skipping potential malicious or corrupt image {image_path}: {e}")
                    return None, None, None

                img = Image.open(image_path)
                response = self.model.generate_content([self.prompt, img])
                
                caption = response.text.strip().replace('\n', ' ')
                
                # Try to get actual token counts from response metadata
                input_tokens = output_tokens = None
                if hasattr(response, 'usage_metadata'):
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                
                # Fallback to estimation if not available
                if input_tokens is None or output_tokens is None:
                    try:
                        output_tokens = self.model.count_tokens(caption).total_tokens
                        # Estimate input tokens (prompt + image)
                        prompt_tokens = self.model.count_tokens(self.prompt).total_tokens
                        image_tokens = self._estimate_image_tokens(image_path)
                        input_tokens = prompt_tokens + image_tokens
                    except:
                        input_tokens = output_tokens = None
                
                return caption, input_tokens, output_tokens
                
            except Exception as e:
                if attempt < max_retries - 1: 
                    time.sleep(delay * (2 ** attempt))
                else: 
                    logging.error(f"Failed to caption {image_path} after {max_retries} attempts: {e}")
                    return None, None, None

    def _estimate_image_tokens(self, image_path):
        """Estimate image tokens using official Gemini formula"""
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if w == 0 or h == 0: 
                    return 258
                # Scale down to fit within 2048x2048
                if w > 2048 or h > 2048:
                    aspect_ratio = w / h
                    if aspect_ratio > 1:  # wider
                        w, h = 2048, int(2048 / aspect_ratio)
                    else:  # taller or square
                        w, h = int(2048 * aspect_ratio), 2048
                # Count 512x512 patches
                cols = math.ceil(w / 512)
                rows = math.ceil(h / 512)
                return (cols * rows * 128) + 258
        except Exception:
            return 258  # Default for corrupted images


class CostPredictor:
    def __init__(self, model, prompt, sample_size):
        self.model = model
        self.prompt_text = prompt
        self.sample_size = sample_size
        self.prompt_tokens = self._count_text_tokens(prompt)
        self.results = []
        self.projected_cost = 0.0  # Initialize to prevent AttributeError

    def _count_text_tokens(self, text): 
        try:
            return self.model.count_tokens(text).total_tokens
        except:
            # Rough estimate if API call fails
            return len(text.split()) * 1.3

    def _calculate_image_tokens(self, image_path):
        """Calculate image tokens using official Gemini 2.0 formula"""
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if w == 0 or h == 0: 
                    return 258
                # Scale down to fit within 2048x2048
                if w > 2048 or h > 2048:
                    aspect_ratio = w / h
                    if aspect_ratio > 1:  # wider
                        w, h = 2048, int(2048 / aspect_ratio)
                    else:  # taller or square
                        w, h = int(2048 * aspect_ratio), 2048
                # Count 512x512 patches
                cols = math.ceil(w / 512)
                rows = math.ceil(h / 512)
                return (cols * rows * 128) + 258
        except Exception: 
            return 258  # Default for corrupted images

    def _get_gemini_caption(self, image_path, max_retries=3, delay=5):
        """Get caption for cost prediction"""
        for attempt in range(max_retries):
            try:
                img = Image.open(image_path)
                response = self.model.generate_content([self.prompt_text, img])
                return response.text.strip().replace('\n', ' ')
            except Exception as e:
                if attempt < max_retries - 1: 
                    time.sleep(delay * (2 ** attempt))
                else: 
                    logging.error(f"Failed to caption {image_path} in cost prediction: {e}")
                    return None

    def run_analysis(self, all_image_paths):
        # Smart sampling: pick from beginning, middle, and end
        if len(all_image_paths) <= self.sample_size:
            sample_paths = all_image_paths
        else:
            if self.sample_size > 1:
                step = len(all_image_paths) // (self.sample_size - 1)
                sample_paths = [all_image_paths[i * step] for i in range(self.sample_size - 1)] + [all_image_paths[-1]]
            else:
                sample_paths = [all_image_paths[0]]
        
        logging.info(f"Analyzing a sample of {len(sample_paths)} images for cost prediction...")
        for img_path in tqdm(sample_paths, desc="Analyzing Samples"):
            start_time = time.time()
            image_tokens = self._calculate_image_tokens(img_path)
            input_tokens = self.prompt_tokens + image_tokens
            
            response_text = self._get_gemini_caption(img_path)
            if not response_text: 
                continue

            output_tokens = self._count_text_tokens(response_text)
            cost = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)
            self.results.append({
                'input': input_tokens, 
                'output': output_tokens, 
                'cost': cost, 
                'time': time.time() - start_time
            })

    def display_report(self, total_image_count):
        if not self.results: 
            logging.error("Cost analysis failed. No samples processed.")
            return
            
        avg_in = sum(r['input'] for r in self.results) / len(self.results)
        avg_out = sum(r['output'] for r in self.results) / len(self.results)
        avg_cost = sum(r['cost'] for r in self.results) / len(self.results)
        avg_time = sum(r['time'] for r in self.results) / len(self.results)

        # Calculate variance for confidence interval
        cost_variance = sum((r['cost'] - avg_cost) ** 2 for r in self.results) / len(self.results)
        cost_std = math.sqrt(cost_variance)

        total_cost = avg_cost * total_image_count
        total_time_sec = avg_time * total_image_count
        cost_range = cost_std * total_image_count
        
        logging.info("--- Cost Prediction Report ---")
        logging.info(f"Based on a sample of {len(self.results)} images:")
        logging.info(f"Average per image: {avg_in:,.0f} input tokens + {avg_out:,.0f} output tokens")
        logging.info(f"Average cost per image: ${avg_cost:.6f}")
        logging.info(f"Estimated total for {total_image_count:,} images: ${total_cost:,.2f} ± ${cost_range:.2f}")
        logging.info(f"Estimated completion time: {time.strftime('%Hh %Mm %Ss', time.gmtime(total_time_sec))}")
        logging.info("--------------------------------")
        self.projected_cost = total_cost

    def get_user_confirmation(self, threshold):
        if threshold is not None and self.projected_cost < threshold:
            logging.info(f"Projected cost (${self.projected_cost:.2f}) is below threshold of ${threshold:.2f}. Proceeding automatically.")
            return True
        try:
            response = input(f"Proceed with estimated cost of ${self.projected_cost:.2f}? (y/n): ").lower().strip()
            return response == 'y'
        except (EOFError, KeyboardInterrupt): 
            logging.info("User cancelled.")
            return False


class APIRateLimiter:
    def __init__(self, rpm, daily_limit, cost_limit):
        self.rpm = rpm or 60
        self.daily_limit = daily_limit
        self.cost_limit = cost_limit
        self.timestamps = deque()
        self.daily_calls = 0
        self.total_cost = 0.0

    def wait_if_needed(self, delay):
        # Check daily and cost limits
        if (self.daily_limit and self.daily_calls >= self.daily_limit):
            logging.warning(f"Daily limit of {self.daily_limit} calls reached.")
            return False
        if (self.cost_limit and self.total_cost >= self.cost_limit):
            logging.warning(f"Cost limit of ${self.cost_limit:.2f} reached.")
            return False
            
        # Check rate limit (requests per minute)
        now = time.time()
        # Remove timestamps older than 60 seconds
        while self.timestamps and self.timestamps[0] <= now - 60:
            self.timestamps.popleft()
            
        if len(self.timestamps) >= self.rpm:
            sleep_time = 60 - (now - self.timestamps[0])
            if sleep_time > 0:
                logging.info(f"RPM limit reached. Waiting {sleep_time:.2f}s...")
                time.sleep(sleep_time)
        
        time.sleep(delay)
        return True

    def record_request(self, in_tokens, out_tokens):
        self.timestamps.append(time.time())
        self.daily_calls += 1
        if in_tokens and out_tokens:
            self.total_cost += (in_tokens * COST_PER_INPUT_TOKEN) + (out_tokens * COST_PER_OUTPUT_TOKEN)


def main():
    parser = argparse.ArgumentParser(description="Advanced Gemini Image Captioning System")
    
    # Core arguments
    parser.add_argument("--dataset-url", help="Optional: Roboflow dataset URL (workspace/project/version). Defaults to 40K project.")
    parser.add_argument("--workspace-id", default="ailecs-nmbrc", help="Roboflow workspace ID")
    parser.add_argument("--project-id", default="40k-dataset-pvktf", help="Roboflow project ID") 
    parser.add_argument("--version-num", type=int, default=2, help="Dataset version number")
    parser.add_argument("--output-dir", default="./output", help="Output directory for caption files")
    parser.add_argument("--prompt-file", default="prompt.md", help="File containing captioning prompt")
    
    # Processing options
    parser.add_argument("--splits", default="valid,test,train", help="Comma-separated list of splits to process")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode with limited images")
    parser.add_argument("--max-images-per-split", type=int, default=10, help="Max images per split in test mode")
    
    # Rate limiting
    parser.add_argument("--requests-per-minute", type=int, default=300, help="Maximum API requests per minute")
    parser.add_argument("--daily-limit", type=int, help="Maximum API calls per day")
    parser.add_argument("--delay-between-calls", type=float, default=0.0, help="Delay between API calls (seconds)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of images to process before batch delay")
    parser.add_argument("--batch-delay", type=float, default=0.0, help="Delay between batches (seconds)")
    parser.add_argument("--max-cost-estimate", type=float, help="Maximum allowed cost estimate")
    
    # Cost prediction
    parser.add_argument("--prediction-sample-size", type=int, default=5, help="Number of images to sample for cost prediction")
    parser.add_argument("--cost-threshold", type=float, help="Auto-confirm if projected cost is under this amount")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip cost prediction phase")
    parser.add_argument("--target-images", type=int, help="Total target images for cost calculation (overrides actual count)")
    
    args = parser.parse_args()
    
    # Validation
    if not os.getenv("ROBOFLOW_API_KEY"):
        logging.error("Roboflow API key is required. Set ROBOFLOW_API_KEY environment variable.")
        sys.exit(1)
    if not os.getenv("GEMINI_API_KEY"):
        logging.error("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        sys.exit(1)
    
    system = CaptioningSystem(args)
    system.run()


if __name__ == "__main__":
    main()