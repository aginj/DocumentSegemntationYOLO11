"""
YOLO11 Document Segmentation Inference Script
Use this script to run predictions on images/videos with your trained model
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path

class DocumentSegmentationPredictor:
    """
    Wrapper class for document segmentation predictions
    """
    
    def __init__(self, model_path='runs/segment/train/weights/best.pt', conf_threshold=0.5):
        """
        Initialize the model
        
        Args:
            model_path: Path to trained YOLO model weights
            conf_threshold: Confidence threshold for predictions (0-1)
        """
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.class_names = self.model.names
        
        print(f"Model loaded: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Confidence threshold: {self.conf}")
    
    def predict_image(self, image_path, save_result=True, output_dir='predictions'):
        """
        Run segmentation on a single image
        
        Args:
            image_path: Path to input image
            save_result: Whether to save annotated image
            output_dir: Directory to save results
            
        Returns:
            results object with predictions
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            save=save_result,
            project=output_dir,
            exist_ok=True,
        )
        
        result = results[0]
        print(f"\nPredictions for: {image_path}")
        print(f"Detections: {len(result.boxes)}")
        
        if result.masks is not None:
            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                class_id = int(box.cls)
                class_name = self.class_names[class_id]
                confidence = float(box.conf)
                print(f"  [{i+1}] {class_name}: {confidence:.3f}")
        
        return result
    
    def predict_batch(self, image_dir, save_results=True, output_dir='predictions'):
        """
        Run segmentation on multiple images
        
        Args:
            image_dir: Directory containing images
            save_results: Whether to save annotated images
            output_dir: Directory to save results
            
        Returns:
            List of results objects
        """
        results = self.model.predict(
            source=image_dir,
            conf=self.conf,
            save=save_results,
            project=output_dir,
            exist_ok=True,
        )
        
        print(f"\nProcessed {len(results)} images")
        return results
    
    def predict_video(self, video_path, output_path='output_video.mp4'):
        """
        Run segmentation on video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
        """
        results = self.model.predict(
            source=video_path,
            conf=self.conf,
            save=True,
            project='runs/segment/predict_video',
        )
        
        print(f"Video prediction complete. Results saved.")
        return results
    
    def extract_masks(self, image_path):
        """
        Extract segmentation masks for each detected object
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with masks and metadata
        """
        result = self.model.predict(source=image_path, conf=self.conf)[0]
        image = cv2.imread(image_path)
        
        masks_data = {
            'original_image': image,
            'segmentation_masks': [],
            'bounding_boxes': [],
            'class_names': [],
            'confidences': [],
        }
        
        if result.masks is not None:
            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                # Get mask
                mask_array = mask.data[0].cpu().numpy().astype(np.uint8) * 255
                
                # Get box info
                class_id = int(box.cls)
                class_name = self.class_names[class_id]
                confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                
                masks_data['segmentation_masks'].append(mask_array)
                masks_data['bounding_boxes'].append(bbox)
                masks_data['class_names'].append(class_name)
                masks_data['confidences'].append(confidence)
        
        return masks_data
    
    def draw_custom_annotations(self, image_path, output_path='annotated.jpg'):
        """
        Draw custom annotations on image with detailed information
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
        """
        image = cv2.imread(image_path)
        result = self.model.predict(source=image_path, conf=self.conf)[0]
        
        # Draw on image
        h, w = image.shape[:2]
        
        if result.masks is not None:
            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                # Draw mask contour
                mask_array = mask.data[0].cpu().numpy().astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
                
                # Draw bounding box
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Draw label
                class_id = int(box.cls)
                class_name = self.class_names[class_id]
                confidence = float(box.conf)
                label = f"{class_name}: {confidence:.2f}"
                
                cv2.putText(image, label, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, image)
        print(f"Annotated image saved: {output_path}")
        return image


# ==================== USAGE EXAMPLES ====================
if __name__ == "__main__":
    # Initialize predictor with your trained model
    predictor = DocumentSegmentationPredictor(
        model_path='runs/segment/train/weights/best.pt',
        conf_threshold=0.5
    )
    
    # Example 1: Predict on single image
    print("\n" + "="*50)
    print("Example 1: Single Image Prediction")
    print("="*50)
    try:
        result = predictor.predict_image('path/to/test/image.jpg')
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Predict on batch of images
    print("\n" + "="*50)
    print("Example 2: Batch Prediction")
    print("="*50)
    try:
        results = predictor.predict_batch('path/to/test/images/')
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Extract and process masks
    print("\n" + "="*50)
    print("Example 3: Extract Segmentation Masks")
    print("="*50)
    try:
        masks_data = predictor.extract_masks('path/to/test/image.jpg')
        print(f"Extracted {len(masks_data['segmentation_masks'])} masks")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Custom annotations
    print("\n" + "="*50)
    print("Example 4: Custom Annotations")
    print("="*50)
    try:
        predictor.draw_custom_annotations('path/to/test/image.jpg', 'custom_annotated.jpg')
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 5: Predict on video
    print("\n" + "="*50)
    print("Example 5: Video Prediction")
    print("="*50)
    try:
        results = predictor.predict_video('path/to/test/video.mp4')
    except Exception as e:
        print(f"Error: {e}")
