import cv2
import numpy as np
import random
from typing import List, Dict, Tuple

class ResultVisualizer:
    def __init__(self):
        self.colors = self._generate_colors(50)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        self.text_thickness = 1
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        random.seed(42)
        colors = []
        for _ in range(num_colors):
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            colors.append(color)
        return colors
    
    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        return self.colors[track_id % len(self.colors)]
    
    def draw_tracks(self, frame: np.ndarray, tracked_players: List[Dict]) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for player in tracked_players:
            player_id = player['id']
            bbox = player['bbox']
            confidence = player['confidence']
            
            x1, y1, x2, y2 = bbox
            color = self._get_color(player_id)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.thickness)
            
            label = f"Player {player_id}: {confidence:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.text_thickness
            )
            
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - text_height - baseline),
                (x1 + text_width, label_y + baseline),
                color,
                -1
            )
            
            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y),
                self.font,
                self.font_scale,
                text_color,
                self.text_thickness
            )
            
            centroid = player['centroid']
            center_x, center_y = int(centroid[0]), int(centroid[1])
            cv2.circle(annotated_frame, (center_x, center_y), 4, color, -1)
        
        info_text = f"Players: {len(tracked_players)}"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            self.font,
            self.font_scale,
            (0, 255, 0),
            self.text_thickness
        )
        
        return annotated_frame
    
    def draw_detection(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                      label: str, confidence: float, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        annotated_frame = frame.copy()
        x1, y1, x2, y2 = bbox
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.thickness)
        
        full_label = f"{label}: {confidence:.2f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            full_label, self.font, self.font_scale, self.text_thickness
        )
        
        label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        
        cv2.rectangle(
            annotated_frame,
            (x1, label_y - text_height - baseline),
            (x1 + text_width, label_y + baseline),
            color,
            -1
        )
        
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(
            annotated_frame,
            full_label,
            (x1, label_y),
            self.font,
            self.font_scale,
            text_color,
            self.text_thickness
        )
        
        return annotated_frame
    
    def add_debug_info(self, frame: np.ndarray, text: str) -> None:
        cv2.putText(
            frame,
            text,
            (10, frame.shape[0] - 10),
            self.font,
            self.font_scale,
            (0, 255, 255),
            self.text_thickness
        )