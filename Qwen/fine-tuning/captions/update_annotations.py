
import json
import os
import sys

def convert_coco_to_qwen(input_file, output_file):
    with open(input_file, 'r') as f:
        coco_data = json.load(f)

    images = {image['id']: image for image in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    output_data = []
    for image_id, image_info in images.items():
        if image_id in annotations_by_image:
            annotations = annotations_by_image[image_id]
            
            # The image path in the output should be relative to the dataset directory
            # The script assumes the images are in the same directory as the annotation file.
            # Let's construct the path.
            # The annotation file is in a directory like 'test', 'train', or 'valid'.
            # The images are also in that same directory.
            # So the path should be relative to the parent of the annotation file directory.
            
            # Let's get the directory of the input file
            input_dir = os.path.dirname(input_file)
            
            # The image file name is in image_info['file_name']
            image_path = os.path.join(os.path.basename(input_dir), image_info['file_name'])


            answer = []
            for ann in annotations:
                category_name = categories[ann['category_id']]
                bbox = ann['bbox']
                answer.append(f"{category_name}: {bbox}")
            
            answer_string = "\n".join(answer)

            output_item = {
                "image": image_info['file_name'],
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nfind the different objects in the image"
                    },
                    {
                        "from": "gpt",
                        "value": answer_string
                    }
                ]
            }
            output_data.append(output_item)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_annotations.py <input_coco_file> <output_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_coco_to_qwen(input_file, output_file)
    print(f"Converted {input_file} to {output_file}")
