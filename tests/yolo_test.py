import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from safe_loader import safe_yolo_load

def test_yolo_model(model_path, video_path, output_path="test_detections.jpg"):
    """Test YOLO model on first frame of video"""
    
    # Load model
    model = safe_yolo_load(model_path)
    
    # Load first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read video frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Run detection with very low confidence
    results = model(frame, conf=0.01, verbose=True)
    
    print(f"Number of results: {len(results)}")
    
    # Process results
    for i, result in enumerate(results):
        print(f"\nResult {i}:")
        boxes = result.boxes
        
        if boxes is not None:
            print(f"  Number of boxes: {len(boxes)}")
            
            # Draw all detections
            for j, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                print(f"    Box {j}: class={cls}, conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2})")
                
                # Draw bounding box
                color = (0, 255, 0) if cls == 0 else (255, 0, 0)  # Green for person, red for others
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"cls:{cls} conf:{conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            print("  No boxes found")
    
    # Save result
    cv2.imwrite(output_path, frame)
    print(f"\nDetection result saved to: {output_path}")

if __name__ == "__main__":
    model_path = "D:/Projects2.0/Listai/player-reid/models/best.pt"
    video_path = "D:/Projects2.0/Listai/Assignment Materials/15sec_input_720p.mp4"
    
    test_yolo_model(model_path, video_path)