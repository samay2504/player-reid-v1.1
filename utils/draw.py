import cv2
import numpy as np
import random
from typing import List, Dict, Tuple
from collections import defaultdict

class ResultVisualizer:
    def __init__(self):
        self.colors = self._generate_colors(50)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        self.text_thickness = 1
        
        # Smoothing buffers for stable visualization
        self.bbox_history = defaultdict(lambda: [])
        self.max_history = 5  # Number of frames to average for smoothing
        
        # Enhanced color scheme for better visibility
        self.stable_colors = [
            (0, 255, 0),    # Green - stable
            (255, 0, 0),    # Blue - medium stability
            (0, 0, 255),    # Red - low stability
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 0),    # Dark Green
            (128, 0, 0),    # Dark Blue
        ]
    
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
    
    def _get_color(self, track_id: int, stability: int = 0) -> Tuple[int, int, int]:
        """Get color based on track ID and stability"""
        if stability > 20:
            # Use stable colors for well-established tracks
            return self.stable_colors[track_id % len(self.stable_colors)]
        else:
            # Use random colors for new tracks
            return self.colors[track_id % len(self.colors)]
    
    def _smooth_bbox(self, track_id: int, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Apply temporal smoothing to bounding boxes"""
        self.bbox_history[track_id].append(bbox)
        
        # Keep only recent history
        if len(self.bbox_history[track_id]) > self.max_history:
            self.bbox_history[track_id].pop(0)
        
        # If we have enough history, apply smoothing
        if len(self.bbox_history[track_id]) >= 3:
            bboxes = np.array(self.bbox_history[track_id])
            # Use weighted average (more recent frames have higher weight)
            weights = np.linspace(0.5, 1.0, len(bboxes))
            weights = weights / np.sum(weights)
            
            smoothed_bbox = np.average(bboxes, axis=0, weights=weights)
            return tuple(map(int, smoothed_bbox))
        
        return bbox
    
    def draw_tracks(self, frame: np.ndarray, tracked_players: List[Dict]) -> np.ndarray:
        annotated_frame = frame.copy()
        
        # Sort players by stability for better visualization
        sorted_players = sorted(tracked_players, key=lambda x: x.get('stability', 0), reverse=True)
        
        for player in sorted_players:
            player_id = player['id']
            bbox = player['bbox']
            confidence = player['confidence']
            stability = player.get('stability', 0)
            
            # Apply smoothing to bounding box
            smoothed_bbox = self._smooth_bbox(player_id, bbox)
            x1, y1, x2, y2 = smoothed_bbox
            
            # Get color based on stability
            color = self._get_color(player_id, stability)
            
            # Draw bounding box with thickness based on stability
            box_thickness = min(3, max(1, stability // 10 + 1))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness)
            
            # Create label without stars, just Player {id}: {confidence}
            label = f"Player {player_id}: {confidence:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.text_thickness
            )
            
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            
            # Draw background for text
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - text_height - baseline),
                (x1 + text_width, label_y + baseline),
                color,
                -1
            )
            
            # Choose text color for good contrast
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
            
            # Draw centroid with size based on stability
            centroid = player['centroid']
            center_x, center_y = int(centroid[0]), int(centroid[1])
            circle_radius = min(6, max(3, stability // 5 + 2))
            cv2.circle(annotated_frame, (center_x, center_y), circle_radius, color, -1)
        
        # Enhanced info display
        stable_players = sum(1 for p in tracked_players if p.get('stability', 0) > 10)
        info_text = f"Players: {len(tracked_players)} (Stable: {stable_players})"
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