import yaml
import os
from ultralytics import YOLO

# Loading custom hyperparameters
hyp_path = os.path.join(os.path.dirname(__file__), "custom_hyp.yaml")
with open(hyp_path) as f:
    hyp_dict = yaml.safe_load(f)

# Load YOLOv8 model
model = YOLO("runs/detect/train44_Transfer_Learning/weights/best.pt")

# Train the model
model.train(
    data="Datasets/Merged Dataset FINAL/data.yaml",
    epochs=200,
    imgsz= 416, #640,416
    batch=16,
    device="cuda",  # cuda for GPU
    amp=True, #False,
    workers=0,
    close_mosaic=10,
    patience=20,
    **hyp_dict
)
# === Notes ===
# - This training run uses:
#   - A custom model fine-tuned with transfer learning (best.pt)
#   - A modified YOLOv8 configuration (not shown here)
#   - A dynamic head architecture to improve detection of small/complex objects
#   - Custom hyperparameters from config/custom_hyp.yaml
#   - The "close_mosaic" argument disables mosaic augmentation after 10 epochs
# Transfer Learning:
# model = YOLO("C:/Users/jerem/Ultralytics/ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml")
# data="coco128.yaml",
