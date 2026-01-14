#!/usr/bin/env python3
"""Convert COCO-style caption annotations into conversation-style JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file '{path}' is empty")
    return text


def build_caption_records(
    captions_data: Dict,
    human_prompt: str,
    caption_prefix: str,
) -> Tuple[List[Dict], int, int]:
    """Return conversation records and counts of skipped items."""
    id_to_filename = {
        img["id"]: img["file_name"]
        for img in captions_data.get("images", [])
        if "id" in img and "file_name" in img
    }

    records: List[Dict] = []
    missing_image = 0
    missing_caption = 0

    for annotation in captions_data.get("annotations", []):
        image_id = annotation.get("image_id")
        caption = annotation.get("caption")

        if not caption:
            missing_caption += 1
            continue

        file_name = id_to_filename.get(image_id)
        if not file_name:
            missing_image += 1
            continue

        records.append(
            {
                "image": file_name,
                "conversations": [
                    {"from": "human", "value": human_prompt},
                    {"from": "gpt", "value": f"{caption_prefix}{caption.strip()}"},
                ],
            }
        )

    return records, missing_image, missing_caption


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "captions_json",
        type=Path,
        help="Input COCO-style captions JSON (with 'images' and 'annotations' keys)",
    )
    parser.add_argument(
        "prompt_path",
        type=Path,
        help="Path to prompt markdown file used for the human conversation turn",
    )
    parser.add_argument(
        "output_json",
        type=Path,
        help="Output path for the generated conversation JSON",
    )
    parser.add_argument(
        "--no-image-tag",
        action="store_true",
        help="Do not prepend '<image>' to the prompt contents",
    )
    parser.add_argument(
        "--caption-prefix",
        default="Caption: ",
        help="String to prepend to each caption in the assistant turn (default: 'Caption: ')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    captions = load_json(args.captions_json)
    prompt_text = load_prompt(args.prompt_path)

    if args.no_image_tag:
        human_prompt = prompt_text
    else:
        human_prompt = "<image>\n" + prompt_text

    records, missing_image, missing_caption = build_caption_records(
        captions,
        human_prompt,
        args.caption_prefix,
    )

    write_json(args.output_json, records)

    print(f"Wrote {len(records)} conversation items.")
    if missing_image:
        print(f"Skipped {missing_image} captions due to missing image entries.")
    if missing_caption:
        print(f"Skipped {missing_caption} annotations without caption text.")


if __name__ == "__main__":
    main()
