import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Normalize bbox
def normalize_bbox(xyxy, img_w, img_h):
    return {
        "x_min": round(xyxy[0] / img_w, 6),
        "y_min": round(xyxy[1] / img_h, 6),
        "x_max": round(xyxy[2] / img_w, 6),
        "y_max": round(xyxy[3] / img_h, 6)
    }

# Compute IoU
def compute_iou(box, boxes):
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

# Weighted Box Fusion (WBF)
def weighted_box_fusion(boxes, scores, classes, iou_thresh=0.5, score_thresh=0.2, min_area=100):
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    fused = []
    used = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if used[i]:
            continue

        ref_box = boxes[i]
        ref_cls = classes[i]

        group_indices = [i]
        weights = [scores[i]]

        for j in range(i + 1, len(boxes)):
            if used[j] or classes[j] != ref_cls:
                continue
            iou = compute_iou(ref_box, boxes[j:j+1])[0]
            if iou >= iou_thresh:
                group_indices.append(j)
                weights.append(scores[j])

        group_boxes = boxes[group_indices]
        group_scores = scores[group_indices]
        weights = np.array(weights)
        total_weight = np.sum(weights)

        if total_weight == 0:
            continue

        fused_box = np.sum(group_boxes.T * weights, axis=1) / total_weight
        fused_score = np.average(group_scores, weights=weights)
        area = (fused_box[2] - fused_box[0]) * (fused_box[3] - fused_box[1])

        if fused_score >= score_thresh and area >= min_area:
            fused.append((fused_box, fused_score, ref_cls))

        used[group_indices] = True

    return fused

# Multi-model, multi-scale, TTA prediction
def ensemble_predict(model_paths, image_path, class_names, scales=[640, 1024], flip_modes=[None, "horizontal"], conf_thresh=0.3):
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    all_boxes, all_scores, all_classes = [], [], []

    for path in model_paths:
        model = YOLO(path)
        for sz in scales:
            for flip in flip_modes:
                img_rsz = cv2.resize(img, (sz, sz))
                if flip == "horizontal":
                    img_rsz = cv2.flip(img_rsz, 1)
                result = model.predict(img_rsz, imgsz=sz, conf=conf_thresh, verbose=False)[0]
                for box in result.boxes:
                    xyxy = box.xyxy.cpu().numpy().squeeze()
                    score = float(box.conf.item())
                    cls = int(box.cls.item())

                    if flip == "horizontal":
                        xyxy[[0, 2]] = sz - xyxy[[2, 0]]
                    # Resize back to original scale
                    scale_x, scale_y = W / sz, H / sz
                    xyxy[[0, 2]] *= scale_x
                    xyxy[[1, 3]] *= scale_y

                    all_boxes.append(xyxy)
                    all_scores.append(score)
                    all_classes.append(cls)

    # Apply weighted box fusion
    fused = weighted_box_fusion(all_boxes, all_scores, all_classes)
    output = []
    for box, score, cls_id in fused:
        norm_box = normalize_bbox(box, W, H)
        output.append({
            "char": class_names[int(cls_id)],
            "bbox": {k: float(v) for k, v in norm_box.items()},
            "score": float(round(score, 6))
        })
    return output

# Main function
def main():
    model_paths = [
        "C:/Users/22806/Desktop/ECE364/yolov8s/yolov8_char_box_seed/weights/best.pt",
        "C:/Users/22806/Desktop/ECE364/yolov8s/yolov8_char_box_seed3/weights/best.pt"
    ]
    source = "C:/Users/22806/Desktop/ECE364/test/images"
    class_names = ['零','一','二','三','四','五','六','七','八','九','十','百','千','万','亿']

    predictions = []
    for idx, fname in enumerate(sorted(os.listdir(source))):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(source, fname)
        print(f"Inferencing {fname} ...")
        objects = ensemble_predict(model_paths, img_path, class_names, conf_thresh=0.3)
        predictions.append({
            "Id": idx,
            "Predictions": json.dumps(objects, ensure_ascii=False)
        })

    df = pd.DataFrame(predictions)
    out_csv = "C:/Users/22806/Desktop/ECE364/yolov8s/prediction_ensemble_wbf_2s_seed(03).csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")

if __name__ == "__main__":
    main()
