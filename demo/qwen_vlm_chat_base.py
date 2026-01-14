# -*- coding: utf-8 -*-
import torch
from PIL import Image
import os
import sys


def load_qwen25vlm_model(model_path):
    """Load base Qwen2.5-VL model."""
    print("Loading base Qwen2.5-VL model...")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        print("? Loaded base Qwen2.5-VL model successfully")
        return model, processor, "qwen2_5vl_base"

    except Exception as e:
        print(f"Model load failed: {e}")
        return None, None, None


def make_chat_text(prompt: str) -> str:
    """Compose a single-turn chat in Qwen2.5-VL's expected format."""
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_inference_qwen2vl(model, processor, image_obj, prompt,
                          max_new_tokens=256,
                          do_sample=False,
                          temperature=1.0,
                          top_p=1.0,
                          top_k=0):
    """Run inference with Qwen2.5-VL."""
    from qwen_vl_utils import process_vision_info

    chat_text = make_chat_text(prompt)

    inputs = processor(
        text=[chat_text],
        images=[image_obj],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k))
    else:
        gen_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return response


def load_image_or_die(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    print(f"Image loaded: {img.size} ({os.path.basename(path)})")
    return img


def print_help():
    print(
        """
        Commands:
          /help            Show this help message
          /exit, /quit     Quit interactive mode
          /image <path>    Load a new image
          /greedy          Set decoding to greedy (deterministic)
          /sample          Set decoding to sampling (stochastic)
          /temp <float>    Set temperature for sampling
          /top_p <float>   Set top-p for sampling
          /top_k <int>     Set top-k for sampling
          /tokens <int>    Set max new tokens
        """
    )


def main():
    # ? Base Qwen2.5-VL model from Hugging Face
    MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
    IMAGE_PATH = "cop_school.png"

    model, processor, method = load_qwen25vlm_model(MODEL_PATH)
    if model is None:
        print("Failed to load model!")
        sys.exit(1)

    print(f"Model loaded successfully with method: {method}")

    # Load image
    try:
        image_obj = load_image_or_die(IMAGE_PATH)
    except Exception as e:
        print(f"Failed to load image: {e}")
        sys.exit(1)

    #prompt = "What do you see in this image? Describe any objects, people, and potential weapons or threats."
    prompt = "briefly describe the image"
    print(f"\n{'='*70}")
    print("QWEN2.5-VL BASE MODEL TEST")
    print(f"{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")

    try:
        response = run_inference_qwen2vl(model, processor, image_obj, prompt, do_sample=False)
        print("\nResponse:", response)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("Entering interactive mode. Type /help for commands.")
    print_help()

    # --- Interactive loop ---
    do_sample = False
    temperature = 1.0
    top_p = 1.0
    top_k = 0
    max_new_tokens = 256

    try:
        while True:
            try:
                user_in = input("\n>> ").strip()
            except EOFError:
                print("\nEOF received. Exiting.")
                break

            if not user_in:
                continue

            # Commands
            if user_in.lower() in ("/exit", "exit", "quit", "/quit"):
                print("Bye!")
                break
            if user_in.lower() in ("/help", "help"):
                print_help()
                continue
            if user_in.lower() == "/greedy":
                do_sample = False
                print("Decoding set to GREEDY.")
                continue
            if user_in.lower() == "/sample":
                do_sample = True
                print("Decoding set to SAMPLING.")
                continue
            if user_in.lower().startswith("/image "):
                path = user_in[7:].strip()
                try:
                    image_obj = load_image_or_die(path)
                except Exception as e:
                    print(f"Could not load image: {e}")
                continue
            if user_in.lower().startswith("/temp "):
                try:
                    temperature = float(user_in.split(maxsplit=1)[1])
                    print(f"temperature = {temperature}")
                except Exception:
                    print("Usage: /temp <float>")
                continue
            if user_in.lower().startswith("/top_p "):
                try:
                    top_p = float(user_in.split(maxsplit=1)[1])
                    print(f"top_p = {top_p}")
                except Exception:
                    print("Usage: /top_p <float>")
                continue
            if user_in.lower().startswith("/top_k "):
                try:
                    top_k = int(user_in.split(maxsplit=1)[1])
                    print(f"top_k = {top_k}")
                except Exception:
                    print("Usage: /top_k <int>")
                continue
            if user_in.lower().startswith("/tokens "):
                try:
                    max_new_tokens = int(user_in.split(maxsplit=1)[1])
                    print(f"max_new_tokens = {max_new_tokens}")
                except Exception:
                    print("Usage: /tokens <int>")
                continue

            # Otherwise treat as prompt
            try:
                resp = run_inference_qwen2vl(
                    model, processor, image_obj, user_in,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                print(f"\n[Assistant]\n{resp}")
            except Exception as e:
                print(f"Generation failed: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
