import sys
import os
import logging
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_loader import safe_yolo_load

logger = logging.getLogger(__name__)

def test_yolo_model(model_path: str, video_path: str, output_path: str = "test_detections.jpg"):
    """Test YOLO model on first frame of video"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    logger.info(f"Loading model from: {model_path}")
    model = safe_yolo_load(model_path)
    
    logger.info(f"Loading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read video frame")
    
    logger.info(f"Frame shape: {frame.shape}")
    
    logger.info("Running detection with low confidence threshold...")
    results = model(frame, conf=0.01, verbose=False)
    
    logger.info(f"Number of results: {len(results)}")
    
    detection_count = 0
    for i, result in enumerate(results):
        logger.info(f"Result {i}:")
        boxes = result.boxes
        
        if boxes is not None:
            logger.info(f"  Number of boxes: {len(boxes)}")
            
            for j, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                logger.info(f"    Box {j}: class={cls}, conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2})")
                
                color = (0, 255, 0) if cls == 2 else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"cls:{cls} conf:{conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                detection_count += 1
        else:
            logger.info("  No boxes found")
    
    cv2.imwrite(output_path, frame)
    logger.info(f"Detection result saved to: {output_path}")
    logger.info(f"Total detections: {detection_count}")
    
    return detection_count

def main():
    parser = argparse.ArgumentParser(description='Test YOLO model on video')
    parser.add_argument('--model', type=str, 
                       default='D:/Projects2.0/Listai/models/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--video', type=str,
                       default='D:/Projects2.0/Listai/Assignment Materials/15sec_input_720p.mp4',
                       help='Path to input video')
    parser.add_argument('--output', type=str,
                       default='test_detections.jpg',
                       help='Output image path')
    
    args = parser.parse_args()
    
    try:
        test_yolo_model(args.model, args.video, args.output)
        logger.info("YOLO test completed successfully")
    except Exception as e:
        logger.error(f"YOLO test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()