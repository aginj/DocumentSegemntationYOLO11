"""
YOLO11 Document Segmentation Training Script
Trains a YOLO11 segmentation model on your document dataset
"""

from ultralytics import YOLO
import torch
import os

# Check GPU availability
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Path to your data.yaml file
DATA_YAML = 'path/to/your/data.yaml'  # UPDATE THIS PATH

# Training parameters
EPOCHS = 100
IMGSZ = 640
BATCH_SIZE = 16  # Adjust based on GPU memory (8, 16, 32)
DEVICE = 0  # GPU device, use 'cpu' if no GPU available

print("=" * 50)
print("YOLO11 Document Segmentation Training")
print("=" * 50)

# Load a pretrained YOLO11 segmentation model
# Available sizes: nano(n), small(s), medium(m), large(l), extra-large(x)
# Using medium (m) - good balance between speed and accuracy
model = YOLO('yolo11m-seg.pt')

print(f"\nModel loaded: YOLO11m-seg")
print(f"Data YAML: {DATA_YAML}")
print(f"Epochs: {EPOCHS}")
print(f"Image Size: {IMGSZ}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Device: {DEVICE}\n")

# Train the model
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH_SIZE,
    device=DEVICE,
    patience=20,  # Early stopping patience
    save=True,
    # Optional parameters for better training:
    augment=True,
    mosaic=1.0,
    flipud=0.5,
    fliplr=0.5,
    degrees=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    perspective=0.0,
    # Learning rate scheduling
    lr0=0.01,
    lrf=0.01,
    # Class weights (useful if classes are imbalanced)
    # cls_pw=1.0,
    # obj_pw=1.0,
    # Validation
    val=True,
    split=0.8,
)

print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)

# The trained model is saved at: runs/segment/train/weights/best.pt
best_model_path = 'runs/segment/train/weights/best.pt'
print(f"\nBest model saved at: {best_model_path}")

# ==================== INFERENCE EXAMPLES ====================
print("\n" + "=" * 50)
print("Running Inference Examples")
print("=" * 50)

# Load the best trained model
trained_model = YOLO(best_model_path)

# Example 1: Predict on a single image
print("\n1. Predicting on single image...")
result = trained_model.predict(
    source='path/to/test/image.jpg',
    conf=0.5,
    save=True,
    project='runs/segment/predict',
)
print("Results saved in runs/segment/predict/")

# Example 2: Predict on multiple images
print("\n2. Predicting on multiple images...")
results = trained_model.predict(
    source='path/to/test/images/',
    conf=0.5,
    save=True,
    project='runs/segment/predict',
)

# Example 3: Batch prediction with visualization
print("\n3. Batch prediction with details...")
import cv2
import numpy as np

results = trained_model.predict(
    source='path/to/test/images/',
    conf=0.5,
    iou=0.45,
)

for idx, result in enumerate(results):
    # Get image
    im = result.orig_img
    
    # Get masks and boxes
    masks = result.masks
    boxes = result.boxes
    
    if masks is not None:
        print(f"\nImage {idx + 1}:")
        print(f"  - Number of detections: {len(boxes)}")
        print(f"  - Classes detected: {[result.names[int(box.cls)] for box in boxes]}")
        print(f"  - Confidence scores: {[f'{float(box.conf):.2f}' for box in boxes]}")
    
    # Plot and save
    im_annotated = result.plot()
    cv2.imwrite(f'result_{idx}.jpg', im_annotated)
    print(f"  - Annotated image saved: result_{idx}.jpg")

# ==================== VALIDATION ====================
print("\n" + "=" * 50)
print("Running Validation")
print("=" * 50)

metrics = trained_model.val()
print(f"\nValidation Metrics:")
print(f"  - mAP50: {metrics.seg.map50:.3f}")
print(f"  - mAP50-95: {metrics.seg.map:.3f}")

# ==================== MODEL EXPORT ====================
print("\n" + "=" * 50)
print("Exporting Model")
print("=" * 50)

# Export to different formats for deployment
export_formats = ['onnx', 'tflite', 'pt']  # pt, onnx, tflite, pb, tfjs, etc.

for fmt in export_formats:
    try:
        path = trained_model.export(format=fmt)
        print(f"✓ Exported to {fmt.upper()}: {path}")
    except Exception as e:
        print(f"✗ Failed to export to {fmt.upper()}: {str(e)}")

print("\n" + "=" * 50)
print("All Done!")
print("=" * 50)
