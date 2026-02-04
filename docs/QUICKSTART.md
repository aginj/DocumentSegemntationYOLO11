# YOLO11 Document Segmentation - Quick Start Guide

## Overview
This guide helps you train and use a YOLO11 segmentation model for document segmentation with your dataset.

Your dataset has:
- 4 classes: ['1', '2', '3', 'document']
- YOLO format with train/val/test splits
- Segmentation masks (polygon annotations)

---

## Step 1: Installation

```bash
# Install required packages
pip install ultralytics torch torchvision opencv-python numpy

# Or for CPU only (if no GPU):
pip install ultralytics opencv-python numpy
```

---

## Step 2: Prepare Your Dataset

Your `data.yaml` looks good! Make sure:

1. Extract your zip file to a directory
2. Your structure should be:
```
dataset/
├── train/
│   └── images/
│   └── labels/  (XML, JSON, or txt format)
├── valid/
│   └── images/
│   └── labels/
├── test/
│   └── images/
│   └── labels/
└── data.yaml
```

3. Update paths in `data.yaml` to match your directory structure:
```yaml
path: /absolute/path/to/dataset
train: train/images
val: valid/images
test: test/images
nc: 4
names: ['1', '2', '3', 'document']
```

---

## Step 3: Train the Model

### Basic Training (Recommended for beginners):

```python
from ultralytics import YOLO

# Load pretrained YOLO11m-seg model
model = YOLO('yolo11m-seg.pt')

# Train
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU device (use 'cpu' if no GPU)
)
```

### Or use the provided script:

```bash
python train_document_segmentation.py
```

**Before running, update:**
```python
DATA_YAML = 'path/to/your/data.yaml'  # Update this line
```

### Training Parameters to Adjust:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `epochs` | 50-200 | Number of training iterations |
| `batch` | 8, 16, 32 | Larger = faster but needs more GPU memory |
| `imgsz` | 640 (default) | Image size (640, 512, 1024 possible) |
| `device` | 0, 1, 2... or 'cpu' | GPU index or CPU |
| `patience` | 20 | Early stopping (stop if no improvement) |
| `lr0` | 0.01 | Initial learning rate |

### Model Sizes:

- `yolo11n-seg`: Nano (fastest, least accurate)
- `yolo11s-seg`: Small
- `yolo11m-seg`: Medium (recommended) ⭐
- `yolo11l-seg`: Large
- `yolo11x-seg`: Extra Large (slowest, most accurate)

---

## Step 4: Monitor Training

During training, you'll see:
- Loss values decreasing
- mAP (accuracy) increasing
- Validation metrics
- Training plots saved to `runs/segment/train/`

**Important files:**
- `runs/segment/train/weights/best.pt` - Best trained model
- `runs/segment/train/results.csv` - Training metrics
- `runs/segment/train/results.png` - Training plots

---

## Step 5: Run Inference

### Quick Prediction:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/segment/train/weights/best.pt')

# Predict on image
results = model.predict(source='image.jpg', conf=0.5)

# Predict on directory
results = model.predict(source='path/to/images/', conf=0.5)

# Predict on video
results = model.predict(source='video.mp4', conf=0.5)
```

### Or use the provided inference script:

```python
from inference_document_segmentation import DocumentSegmentationPredictor

predictor = DocumentSegmentationPredictor('runs/segment/train/weights/best.pt')

# Predict on single image
result = predictor.predict_image('test_image.jpg')

# Predict on batch
results = predictor.predict_batch('path/to/images/')

# Extract masks
masks_data = predictor.extract_masks('test_image.jpg')
```

---

## Step 6: Process Results

### Get Segmentation Masks:

```python
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('runs/segment/train/weights/best.pt')
results = model.predict('image.jpg', conf=0.5)

result = results[0]

# Get masks
if result.masks is not None:
    for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
        # Mask array
        mask_array = mask.data[0].cpu().numpy()
        
        # Class info
        class_id = int(box.cls)
        class_name = result.names[class_id]
        confidence = float(box.conf)
        
        # Bounding box
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        
        print(f"Object {i}: {class_name} ({confidence:.2f})")
        print(f"  Mask shape: {mask_array.shape}")
        print(f"  BBox: {bbox}")
```

### Visualize Results:

```python
import cv2

result = model.predict('image.jpg')[0]

# Plot with masks and boxes
im_annotated = result.plot()

# Display
cv2.imshow('Segmentation Results', im_annotated)
cv2.waitKey(0)

# Save
cv2.imwrite('result.jpg', im_annotated)
```

---

## Step 7: Evaluate Model

```python
# Validate on validation set
metrics = model.val()

print(f"mAP50: {metrics.seg.map50:.3f}")
print(f"mAP50-95: {metrics.seg.map:.3f}")

# Test on test set
metrics = model.val(data='test_config.yaml', split='test')
```

---

## Step 8: Export for Deployment

```python
# Export to different formats
model.export(format='onnx')   # ONNX format
model.export(format='tflite') # TensorFlow Lite (mobile)
model.export(format='pt')     # PyTorch (.pt file)
model.export(format='pb')     # TensorFlow saved_model
```

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size or image size:
```python
model.train(data='data.yaml', batch=8, imgsz=512)
```

### Issue: Poor accuracy
**Solution:**
1. Increase epochs (100, 200, 300)
2. Use larger model (yolo11l-seg, yolo11x-seg)
3. Check data quality
4. Increase training time

### Issue: Model not loading
**Solution:** Make sure you're using the correct path:
```python
model = YOLO('runs/segment/train/weights/best.pt')
# OR
model = YOLO('runs/segment/train/weights/last.pt')
```

### Issue: GPU not being used
**Solution:** 
```python
# Check GPU
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name

# Use GPU explicitly
model.train(device=0)  # GPU 0
model.train(device='0,1')  # Multiple GPUs
```

---

## Tips for Best Results

1. **Data Quality**: Ensure masks are properly annotated
2. **Augmentation**: The script includes augmentation by default
3. **Class Balance**: If classes are imbalanced, training might struggle
4. **Resolution**: Higher resolution (1024) = better accuracy but slower
5. **Batch Size**: Larger batches = more stable training but need more GPU memory
6. **Learning Rate**: Use default or tune lr0=0.01 for better results
7. **Early Stopping**: Enable patience to stop training if no improvement

---

## File Locations After Training

```
runs/
└── segment/
    ├── train/
    │   ├── weights/
    │   │   ├── best.pt          ← Use this for inference
    │   │   └── last.pt
    │   ├── results.csv
    │   ├── results.png
    │   └── confusion_matrix.png
    └── predict/
        └── exp/
            └── (prediction results)
```

---

## Performance Metrics

Training will display:
- **Loss**: Training error (should decrease)
- **mAP50**: Mean Average Precision at IOU 0.5
- **mAP50-95**: mAP averaged from 0.5 to 0.95 IOU
- **Precision & Recall**: Detection accuracy metrics

Aim for:
- **Good**: mAP50 > 0.7, mAP50-95 > 0.5
- **Excellent**: mAP50 > 0.85, mAP50-95 > 0.7

---

## Next Steps

1. ✅ Install packages
2. ✅ Prepare dataset (already done)
3. ✅ Train model
4. ✅ Test on images/videos
5. ✅ Deploy or integrate into application

---

## Resources

- Official Docs: https://docs.ultralytics.com/
- GitHub: https://github.com/ultralytics/ultralytics
- Segmentation Guide: https://docs.ultralytics.com/tasks/segment/
- Roboflow Integration: https://docs.ultralytics.com/integrations/roboflow/

---

## Support

- GitHub Issues: https://github.com/ultralytics/ultralytics/issues
- Discord: https://discord.com/invite/ultralytics
- Community Forum: https://community.ultralytics.com/
