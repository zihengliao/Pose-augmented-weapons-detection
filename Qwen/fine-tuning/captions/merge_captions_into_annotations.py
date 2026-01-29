#!/usr/bin/env python3
"""Augment conversation-style annotations with captions.

The script reads a conversation JSON file (list of items containing an image
path and a conversation between human and model) and appends captions from a
separate captions JSON file. Captions are matched by image file name.

For each entry we:
* Update the human prompt to request a caption explicitly.
* Append a `Caption: ...` line to the assistant response.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

PROMPT_TEMPLATE = (
    "<image>\n"
    "Find the different objects in the image and follow the instructions below.\n\n"
    "# VLM Caption Prompt for Pose-Augmented Weapons Detection\n\n"
    "## Primary Prompt\n\n"
    "Generate a structured caption for this weapons detection image. **Keep the total caption under 20 words.**\n\n"
    "**Required Format:**\n"
    "\"Scene contains [hand count] hands in [detailed pose description] with [gun presence/absence and details]."
    " [Detection challenges or notable features].\"\n\n"
    "## Detection Categories to Consider\n\n"
    "**Hand Poses/Grips:**\n"
    "- Open palm, closed fist, pointing, gripping, extended, relaxed\n"
    "- Single-handed, two-handed, overlapping hands\n"
    "- Partial visibility, fully visible\n\n"
    "**Gun Types/Positions:**\n"
    "- Pistol, rifle, partially visible weapon\n"
    "- Held, holstered, on surface, in-hand\n"
    "- Horizontal, vertical, angled orientation\n\n"
    "**Detection Challenges:**\n"
    "- Occlusion (partial blocking)\n"
    "- Low lighting, shadows\n"
    "- Motion blur, unusual angle\n"
    "- Hand-weapon overlap\n"
    "- Background clutter\n\n"
    "## Example Outputs\n\n"
    "Good: \"Scene contains two hands gripping a pistol in vertical orientation with partial weapon occlusion by background objects.\"\n"
    "Good: \"Scene contains one open hand reaching toward holstered weapon with challenging lighting conditions throughout image.\"\n"
    "Good: \"Scene contains three hands in pointing gestures with rifle visible horizontally, hands overlapping weapon grip area.\"\n"
    "Good: \"Scene contains multiple hands in various poses with pistol weapon partially occluded by background objects creating detection difficulties.\"\n\n"
    "Too short: \"Two hands, gun present, occluded\"\n"
    "Too robotic: \"Hands: 2, Gun: pistol, Challenge: occlusion\""
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_caption_lookup(captions_data: Dict) -> Dict[str, str]:
    """Return mapping from image file name to a single caption string."""
    id_to_filename = {
        img["id"]: img["file_name"]
        for img in captions_data.get("images", [])
        if "id" in img and "file_name" in img
    }

    lookup: Dict[str, str] = {}
    for ann in captions_data.get("annotations", []):
        image_id = ann.get("image_id")
        caption = ann.get("caption")
        if image_id is None or caption is None:
            continue
        file_name = id_to_filename.get(image_id)
        if not file_name:
            continue
        cleaned = caption.strip()
        if not cleaned:
            continue
        lookup.setdefault(file_name, cleaned)
    return lookup


def update_conversations(
    records: List[Dict],
    caption_lookup: Dict[str, str],
) -> Tuple[int, int, int]:
    matched = 0
    already_captioned = 0
    missing = 0

    for item in records:
        file_name = item.get("image")
        conversations = item.get("conversations", [])
        if not conversations or not file_name:
            continue

        # Update human message if present
        human_msg = conversations[0]
        if human_msg.get("from") == "human":
            human_msg["value"] = PROMPT_TEMPLATE

        caption = caption_lookup.get(file_name)
        if not caption:
            missing += 1
            continue

        # Append caption to assistant message
        # Find first assistant reply
        assistant_msg = next(
            (msg for msg in conversations if msg.get("from") == "gpt"),
            None,
        )
        if assistant_msg is None:
            missing += 1
            continue

        value = assistant_msg.get("value", "")
        if "Caption:" in value:
            already_captioned += 1
            continue

        if value and not value.endswith("\n"):
            value += "\n"
        assistant_msg["value"] = f"{value}Caption: {caption}"
        matched += 1

    return matched, missing, already_captioned


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conversation_json", type=Path, help="Conversation JSON to update")
    parser.add_argument("captions_json", type=Path, help="Captions JSON file")
    parser.add_argument("output_json", type=Path, help="Output path for updated JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    conversations = load_json(args.conversation_json)
    captions = load_json(args.captions_json)
    caption_lookup = build_caption_lookup(captions)

    matched, missing, already_captioned = update_conversations(
        conversations,
        caption_lookup,
    )

    write_json(args.output_json, conversations)

    print(f"Added captions to {matched} conversation items.")
    if already_captioned:
        print(f"Skipped {already_captioned} items that already contained a caption.")
    if missing:
        print(f"Warning: {missing} images were missing captions or assistant replies.")


if __name__ == "__main__":
    main()
