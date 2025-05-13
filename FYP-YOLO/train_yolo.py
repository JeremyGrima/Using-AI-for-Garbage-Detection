import yaml
from ultralytics import YOLO
import os

# Loading custom hyperparameters
hyp_path = os.path.join(os.path.dirname(__file__), "custom_hyp.yaml")
with open(hyp_path) as f:
    hyp_dict = yaml.safe_load(f)

# Choosing the Model
model = YOLO("yolo8s.pt")
# For this project, yolo8n, yolo8s, yolo11n, yolo11s, and a custom model in custom_model.py were tested

# 3. Training the Model
model.train(
    # Dataset Instructions:
    # To test this script with the local NGO images download the dataset from this link:
    # https://universe.roboflow.com/nadur-nadif-dataset-jeremy-grima/garbage-detection-using-ai
    # To test this script with the custom merged dataset as used in the project download the dataset from this link:
    # https://www.kaggle.com/datasets/jeremygrima/using-ai-for-garbage-detection-merged-dataset/data

    data="Datasets/Merged Dataset FINAL/data.yaml",
    epochs=200,
    imgsz=416,
    batch=16,
    device="cuda",
    amp=True,
    workers=0,
    patience=20,    # Implemented to allow for early stopping, useful since this wll run for a 100+ epochs
    **hyp_dict      # Uses the Custom Hyperparameters
)
