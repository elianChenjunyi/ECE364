from ultralytics import YOLO
import random
import numpy as np
import torch


# Set all relevant seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    set_seed(42)  # Set seed before anything else
    # Load a pretrained YOLOv8s model.
    # The .pt file will be saved and reused from the specified local path.
    model = YOLO("C:/Users/22806/Desktop/ECE364/yolov8s/yolov8s.pt")

    # Start training the model
    model.train(
        data="C:/Users/22806/Desktop/ECE364/data.yaml",  # Path to dataset config file
        imgsz=1024,       # Input image size (higher helps small object detection)
        epochs=150,       # Number of training epochs
        batch=8,          # Batch size (adjusted for 4GB GPU)
        name="yolov8_char_box_seed",  # Folder name to save this run's results
        project="C:/Users/22806/Desktop/ECE364/yolov8s",  # Root folder for saving runs
        device=0,         # Set to GPU 0; use 'cpu' to train on CPU
        freeze=5,               # Freeze first 10 layers for fine-tuning
        patience=20,             # Early stopping after 20 rounds without improvement
        box=0.1,
        # -------- Augmentation & optimization hyperparameters --------
        lr0=0.003,            # Initial learning rate
        lrf=0.1,              # Final learning rate (multiplier)
        momentum=0.937,       # SGD momentum
        weight_decay=0.001,  # Weight decay (L2 regularization)
        warmup_epochs=3.0,    # Number of warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        warmup_bias_lr=0.1,   # Warmup learning rate for bias layers

        # Data augmentation
        hsv_h=0.015,          # Hue augmentation
        hsv_s=0.5,            # Saturation augmentation
        hsv_v=0.4,            # Value (brightness) augmentation
        translate=0.2,        # Translation (shift) augmentation
        scale=0.3,            # Scaling augmentation
        fliplr=0.4,           # Left-right flip probability
        mosaic=1.0,           # Mosaic augmentation probability
        mixup=0.1,         # MixUp augmentation (disabled here)

        # Reproducibility
        seed=42
    )

if __name__ == '__main__':
    main()
