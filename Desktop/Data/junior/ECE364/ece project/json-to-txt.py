import os
import json

# Mapping Chinese characters to YOLO class IDs
char_to_id = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 11, "千": 12, "万": 13, "亿": 14
}

# Input directory containing .json files
json_dir = 'C:/Users/22806/Desktop/ECE364/training2/annotations'
# Output directory to save YOLO .txt labels
output_dir = 'C:/Users/22806/Desktop/ECE364/training2/val'
os.makedirs(output_dir, exist_ok=True)

# Image size (YOLO requires actual pixel values for normalization)
img_width = 1024  # change if different
img_height = 1024

# Loop through all JSON files
for filename in os.listdir(json_dir):
    if not filename.endswith('.json'):
        continue

    json_path = os.path.join(json_dir, filename)
    txt_path = os.path.join(output_dir, filename.replace('.json', '.txt'))

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    boxes = data.get("boxes", [])
    if not boxes:
        open(txt_path, 'w').close()  # write empty file if no boxes
        continue

    with open(txt_path, 'w', encoding='utf-8') as out:
        for box in boxes:
            char = box["char"]
            if char not in char_to_id:
                continue

            x_min = box["bbox"]["x_min"]
            y_min = box["bbox"]["y_min"]
            x_max = box["bbox"]["x_max"]
            y_max = box["bbox"]["y_max"]

            # Convert to YOLO format
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min

            out.write(f"{char_to_id[char]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

print(f"✅ All JSON files converted to YOLOv5 .txt format at:\n{output_dir}")
