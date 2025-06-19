import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import sys
import os

# Add the parent directory to the path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_loader import safe_yolo_load
import types

logger = logging.getLogger(__name__)

# Fix for PyTorch 2.6 compatibility
def patch_ultralytics_torch_load():
    """Patch ultralytics to use weights_only=False for model loading"""
    try:
        from ultralytics.nn.tasks import torch_safe_load
        import functools
        
        # Create a patched version that uses weights_only=False
        def patched_torch_safe_load(file):
            return torch.load(file, map_location='cpu', weights_only=False)
        
        # Replace the function in the module
        import ultralytics.nn.tasks
        ultralytics.nn.tasks.torch_safe_load = patched_torch_safe_load
        
        # Add safe globals for common PyTorch modules
        torch.serialization.add_safe_globals([
            torch.nn.modules.container.Sequential,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.flatten.Flatten,
            torch.nn.modules.normalization.LayerNorm,
        ])
        
        logger.info("Successfully patched ultralytics for PyTorch 2.6 compatibility")
    except Exception as e:
        logger.warning(f"Failed to patch ultralytics: {e}")

class PlayerTracker:
    def __init__(self, model_path: str, conf_threshold: float = 0.3,  # Lowered confidence threshold
                 max_disappeared: int = 45,  # Increased disappearance threshold
                 max_distance: float = 150.0):  # Increased distance threshold
        # Apply PyTorch compatibility patch
        patch_ultralytics_torch_load()
        
        # Use safe loader for PyTorch 2.6 compatibility
        self.model = safe_yolo_load(model_path)
        self.conf_threshold = conf_threshold
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        self.next_id = 0
        self.players = {}
        self.disappeared = {}
        
        # Enhanced tracking state
        self.feature_memory = {}  # Store feature history for each player
        self.position_history = {}  # Store position history for each player
        self.inactive_players = {}  # Store data of players who left the frame
        self.feature_buffer_size = 30  # Number of recent features to keep
        
        self.frame_count = 0
        self.total_detections = 0
        self.total_tracks = 0
        
        # Track history for better re-identification
        self.track_history = {}
        
        logger.info(f"PlayerTracker initialized with model: {model_path}")
    
    def _extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(105)  # 96 color features + 9 gradient features
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(105)  # 96 color features + 9 gradient features
        
        # Extract color features in RGB space
        roi_resized = cv2.resize(roi, (64, 128))
        color_features = []
        
        for channel in cv2.split(roi_resized):
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
            hist = hist.flatten() / (np.sum(hist) + 1e-7)
            color_features.extend(hist)
        
        # Extract HOG features for shape information
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        
        # Simple HOG-like gradient features
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Compute histogram of gradients
        angles = np.rad2deg(ang)
        hist_grad = np.histogram(angles, bins=9, range=(0, 180), weights=mag)[0]
        hist_grad = hist_grad / (np.sum(hist_grad) + 1e-7)
        
        # Combine all features
        features = np.concatenate([color_features, hist_grad])
        
        return features / (np.linalg.norm(features) + 1e-7)
    
    def _get_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_players(self, frame: np.ndarray) -> List[Dict]:
        # Use a slightly lower confidence for initial detection to catch more potential players
        initial_conf = max(0.1, self.conf_threshold * 0.8)
        results = self.model(frame, conf=initial_conf, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Get all boxes and scores
                all_boxes = []
                all_scores = []
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Calculate dimensions and filtering criteria
                    height = y2 - y1
                    width = x2 - x1
                    area = height * width
                    aspect_ratio = height / max(width, 1)  # Avoid division by zero
                    
                    # Improved filtering criteria
                    is_player = cls == 2  # Soccer player class in our model
                    has_reasonable_size = area > 500  # Minimum area threshold
                    has_reasonable_aspect = 1.2 <= aspect_ratio <= 5.0  # More lenient aspect ratio
                    meets_confidence = conf >= self.conf_threshold
                    
                    if is_player and has_reasonable_size and has_reasonable_aspect and meets_confidence:
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(conf)
                
                # Apply NMS if there are detections
                if all_boxes:
                    all_boxes = np.array(all_boxes)
                    all_scores = np.array(all_scores)
                    
                    # Apply NMS with appropriate threshold
                    keep = cv2.dnn.NMSBoxes(
                        all_boxes.tolist(),
                        all_scores.tolist(),
                        self.conf_threshold,
                        0.4  # NMS threshold
                    )
                    
                    for idx in keep:
                        if isinstance(idx, np.ndarray):
                            idx = idx.item()
                        
                        bbox = tuple(all_boxes[idx])
                        conf = all_scores[idx]
                        
                        detection = {
                            'bbox': bbox,
                            'confidence': conf,
                            'centroid': self._get_centroid(bbox),
                            'features': self._extract_features(frame, bbox)
                        }
                        detections.append(detection)
        
        if not detections:
            logger.warning("No player detections found in frame")
        else:
            logger.debug(f"Found {len(detections)} player detections")
            
        self.total_detections += len(detections)
        return detections
    
    def _update_feature_memory(self, player_id: int, features: np.ndarray):
        """Update feature memory for a player"""
        if player_id not in self.feature_memory:
            self.feature_memory[player_id] = []
        
        self.feature_memory[player_id].append(features)
        if len(self.feature_memory[player_id]) > self.feature_buffer_size:
            self.feature_memory[player_id].pop(0)
    
    def _get_average_features(self, player_id: int) -> np.ndarray:
        """Get average features for a player from memory"""
        if player_id in self.feature_memory and self.feature_memory[player_id]:
            features = np.array(self.feature_memory[player_id])
            return np.mean(features, axis=0)
        return None
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors"""
        return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    
    def _match_with_inactive_players(self, detection: Dict) -> Optional[int]:
        """Try to match a detection with inactive players"""
        detection_features = detection['features']
        detection_centroid = np.array(detection['centroid'])
        best_match_id = None
        best_match_score = -1
        
        for player_id, player_data in self.inactive_players.items():
            # Calculate feature similarity
            stored_features = self._get_average_features(player_id)
            if stored_features is None:
                continue
            
            feature_similarity = self._calculate_feature_similarity(detection_features, stored_features)
            
            # Calculate position similarity based on last known position
            last_position = np.array(player_data['last_position'])
            position_distance = np.linalg.norm(detection_centroid - last_position)
            position_score = np.exp(-position_distance / self.max_distance)
            
            # Combined similarity score
            combined_score = 0.7 * feature_similarity + 0.3 * position_score
            
            if combined_score > 0.75 and combined_score > best_match_score:  # Increased threshold for more confident matching
                best_match_score = combined_score
                best_match_id = player_id
        
        return best_match_id
    
    def _handle_player_disappearance(self, player_id: int):
        """Store player data when they disappear from frame"""
        if player_id in self.players:
            player_data = self.players[player_id]
            self.inactive_players[player_id] = {
                'features': self._get_average_features(player_id),
                'last_position': player_data['centroid'],
                'last_seen': self.frame_count
            }
    
    def _associate_detections(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(self.players) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(self.players.keys()), []
        
        # First, try to match with currently tracked players
        player_ids = list(self.players.keys())
        detection_indices = list(range(len(detections)))
        
        cost_matrix = np.zeros((len(player_ids), len(detections)))
        
        for i, player_id in enumerate(player_ids):
            player_data = self.players[player_id]
            player_features = self._get_average_features(player_id)
            if player_features is None:
                player_features = player_data['features']
            
            # Get track history for this player
            track_hist = self.track_history.get(player_id, [])
            if track_hist:
                predicted_centroid = self._predict_next_position(track_hist)
            else:
                predicted_centroid = player_data['centroid']
            
            for j, detection in enumerate(detections):
                detection_centroid = detection['centroid']
                detection_features = detection['features']
                
                # Calculate spatial distance using predicted position
                distance = np.linalg.norm(np.array(predicted_centroid) - np.array(detection_centroid))
                
                # Calculate appearance similarity
                feature_similarity = self._calculate_feature_similarity(player_features, detection_features)
                
                # Combine distance and appearance with weighted sum
                distance_weight = 0.3
                appearance_weight = 0.7
                
                cost = (distance_weight * (distance / self.max_distance) + 
                       appearance_weight * (1 - feature_similarity))
                
                cost_matrix[i, j] = cost
        
        matched_indices = []
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 0.7:  # Stricter threshold for matching
                    matched_indices.append((player_ids[row], col))
        
        matched_player_ids = [match[0] for match in matched_indices]
        matched_detection_indices = [match[1] for match in matched_indices]
        
        # Try to match remaining detections with inactive players
        unmatched_detections = [i for i in detection_indices if i not in matched_detection_indices]
        additional_matches = []
        remaining_unmatched = []
        
        for det_idx in unmatched_detections:
            matched_id = self._match_with_inactive_players(detections[det_idx])
            if matched_id is not None:
                additional_matches.append((matched_id, det_idx))
                if matched_id in self.inactive_players:
                    del self.inactive_players[matched_id]
            else:
                remaining_unmatched.append(det_idx)
        
        # Combine all matches
        all_matches = matched_indices + additional_matches
        unmatched_players = [pid for pid in player_ids if pid not in matched_player_ids]
        
        return all_matches, unmatched_players, remaining_unmatched
    
    def _predict_next_position(self, track_history: List[Tuple[float, float]]) -> Tuple[float, float]:
        if len(track_history) < 2:
            return track_history[-1]
        
        # Use last two positions to predict next
        last_pos = np.array(track_history[-1])
        prev_pos = np.array(track_history[-2])
        velocity = last_pos - prev_pos
        
        return tuple(last_pos + velocity)
    
    def track_frame(self, frame: np.ndarray) -> List[Dict]:
        self.frame_count += 1
        detections = self._detect_players(frame)
        
        if len(detections) == 0:
            for player_id in list(self.disappeared.keys()):
                self.disappeared[player_id] += 1
                if self.disappeared[player_id] > self.max_disappeared:
                    self._handle_player_disappearance(player_id)
                    del self.players[player_id]
                    del self.disappeared[player_id]
                    if player_id in self.track_history:
                        del self.track_history[player_id]
            return []
        
        if len(self.players) == 0:
            for detection in detections:
                self.players[self.next_id] = {
                    'bbox': detection['bbox'],
                    'centroid': detection['centroid'],
                    'features': detection['features'],
                    'confidence': detection['confidence']
                }
                self._update_feature_memory(self.next_id, detection['features'])
                self.track_history[self.next_id] = [detection['centroid']]
                self.next_id += 1
                self.total_tracks += 1
        else:
            matches, unmatched_players, unmatched_detections = self._associate_detections(detections)
            
            for player_id, detection_idx in matches:
                detection = detections[detection_idx]
                self.players[player_id] = {
                    'bbox': detection['bbox'],
                    'centroid': detection['centroid'],
                    'features': detection['features'],
                    'confidence': detection['confidence']
                }
                
                self._update_feature_memory(player_id, detection['features'])
                
                # Update track history
                if player_id in self.track_history:
                    self.track_history[player_id].append(detection['centroid'])
                    if len(self.track_history[player_id]) > 30:
                        self.track_history[player_id] = self.track_history[player_id][-30:]
                else:
                    self.track_history[player_id] = [detection['centroid']]
                
                if player_id in self.disappeared:
                    del self.disappeared[player_id]
            
            for player_id in unmatched_players:
                if player_id not in self.disappeared:
                    self.disappeared[player_id] = 1
                else:
                    self.disappeared[player_id] += 1
                
                if self.disappeared[player_id] > self.max_disappeared:
                    self._handle_player_disappearance(player_id)
                    del self.players[player_id]
                    del self.disappeared[player_id]
                    if player_id in self.track_history:
                        del self.track_history[player_id]
            
            for detection_idx in unmatched_detections:
                detection = detections[detection_idx]
                self.players[self.next_id] = {
                    'bbox': detection['bbox'],
                    'centroid': detection['centroid'],
                    'features': detection['features'],
                    'confidence': detection['confidence']
                }
                self._update_feature_memory(self.next_id, detection['features'])
                self.track_history[self.next_id] = [detection['centroid']]
                self.next_id += 1
                self.total_tracks += 1
        
        result = []
        for player_id, player_data in self.players.items():
            result.append({
                'id': player_id,
                'bbox': player_data['bbox'],
                'centroid': player_data['centroid'],
                'confidence': player_data['confidence']
            })
        
        return result
    
    def get_statistics(self) -> Dict:
        return {
            'total_frames': self.frame_count,
            'total_detections': self.total_detections,
            'total_tracks': self.total_tracks,
            'active_tracks': len(self.players),
            'disappeared_tracks': len(self.disappeared)
        }