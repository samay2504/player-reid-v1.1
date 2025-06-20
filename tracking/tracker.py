import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
import torch
import sys
import os

# Add the parent directory to the path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_loader import safe_yolo_load

logger = logging.getLogger(__name__)

# Fix for PyTorch 2.6 compatibility
def patch_ultralytics_torch_load():
    """Patch ultralytics to use weights_only=False for model loading"""
    try:
        from ultralytics.nn.tasks import torch_safe_load
        import functools
        
        def patched_torch_safe_load(file):
            return torch.load(file, map_location='cpu', weights_only=False)
        
        import ultralytics.nn.tasks
        ultralytics.nn.tasks.torch_safe_load = patched_torch_safe_load
        
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

class EnhancedPlayerTracker:
    def __init__(self, model_path: str, conf_threshold: float = 0.35,
                 max_disappeared: int = 60,  # Increased for better re-identification
                 max_distance: float = 200.0,
                 reid_threshold: float = 0.65):  # Threshold for re-identification
        
        patch_ultralytics_torch_load()
        
        self.model = safe_yolo_load(model_path)
        self.conf_threshold = conf_threshold
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.reid_threshold = reid_threshold
        
        # Core tracking state
        self.next_id = 0
        self.active_players = {}  # Currently visible players
        self.disappeared_players = {}  # Players that disappeared with countdown
        self.player_gallery = {}  # Long-term storage of player features and data
        
        # Enhanced feature management
        self.feature_buffers = defaultdict(lambda: deque(maxlen=50))  # Larger buffer
        self.position_buffers = defaultdict(lambda: deque(maxlen=30))
        self.confidence_buffers = defaultdict(lambda: deque(maxlen=20))
        self.bbox_buffers = defaultdict(lambda: deque(maxlen=10))
        
        # Kalman filter-like motion prediction
        self.motion_models = {}
        
        # Stability tracking
        self.id_stability = defaultdict(int)  # How long each ID has been stable
        self.last_assignment = {}  # Last detection-to-ID assignment
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.total_reidentifications = 0
        self.id_switches = 0
        
        logger.info(f"EnhancedPlayerTracker initialized with model: {model_path}")
    
    def _extract_enhanced_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract comprehensive appearance features for robust matching"""
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(162)  # 162-dimensional feature vector
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(162)
        
        # Resize ROI to standard size for consistent feature extraction
        roi_resized = cv2.resize(roi, (64, 128))
        
        # 1. Enhanced Color Features (Multiple color spaces)
        color_features = []
        
        # RGB histogram
        for channel in cv2.split(roi_resized):
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
            hist = hist.flatten() / (np.sum(hist) + 1e-7)
            color_features.extend(hist)
        
        # HSV histogram for better color representation
        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        for i, channel in enumerate(cv2.split(hsv)):
            bins = 16 if i == 0 else 8  # More bins for hue
            hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
            hist = hist.flatten() / (np.sum(hist) + 1e-7)
            color_features.extend(hist)
        
        # 2. Enhanced Shape Features
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        
        # HOG-like gradient features
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Multiple orientation histograms
        angles = np.rad2deg(ang)
        hist_grad = np.histogram(angles, bins=18, range=(0, 180), weights=mag)[0]
        hist_grad = hist_grad / (np.sum(hist_grad) + 1e-7)
        
        # 3. Texture Features (Local Binary Pattern approximation)
        texture_features = []
        kernel_size = 3
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                shifted = np.roll(np.roll(gray, dx, axis=1), dy, axis=0)
                texture_features.append(np.mean(gray > shifted))
        
        # 4. Spatial layout features (divide into regions)
        spatial_features = []
        h, w = gray.shape
        for i in range(2):
            for j in range(2):
                region = gray[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                spatial_features.append(np.mean(region) / 255.0)
                spatial_features.append(np.std(region) / 255.0)
        
        # Combine all features (96 RGB + 32 HSV + 18 gradient + 8 texture + 8 spatial = 162)
        all_features = np.concatenate([
            color_features,      # RGB + HSV histograms
            hist_grad,          # Gradient orientation
            texture_features,   # Texture
            spatial_features    # Spatial layout
        ])
        
        # Normalize to unit vector
        return all_features / (np.linalg.norm(all_features) + 1e-7)
    
    def _update_motion_model(self, player_id: int, position: Tuple[float, float]):
        """Update motion model for better position prediction"""
        if player_id not in self.motion_models:
            self.motion_models[player_id] = {
                'positions': deque(maxlen=5),
                'velocities': deque(maxlen=3)
            }
        
        model = self.motion_models[player_id]
        model['positions'].append(position)
        
        if len(model['positions']) >= 2:
            vel = (
                model['positions'][-1][0] - model['positions'][-2][0],
                model['positions'][-1][1] - model['positions'][-2][1]
            )
            model['velocities'].append(vel)
    
    def _predict_position(self, player_id: int) -> Tuple[float, float]:
        """Predict next position using motion model"""
        if player_id not in self.motion_models:
            return (0, 0)
        
        model = self.motion_models[player_id]
        if not model['positions']:
            return (0, 0)
        
        last_pos = model['positions'][-1]
        
        if len(model['velocities']) == 0:
            return last_pos
        
        # Weighted average of recent velocities
        weights = np.array([0.5, 0.3, 0.2])[:len(model['velocities'])]
        weights = weights / np.sum(weights)
        
        avg_vel = np.average(list(model['velocities']), axis=0, weights=weights)
        predicted = (last_pos[0] + avg_vel[0], last_pos[1] + avg_vel[1])
        
        return predicted
    
    def _get_average_features(self, player_id: int) -> np.ndarray:
        """Get robust average features with outlier rejection"""
        if player_id not in self.feature_buffers or len(self.feature_buffers[player_id]) == 0:
            return None
        
        features = np.array(list(self.feature_buffers[player_id]))
        
        if len(features) == 1:
            return features[0]
        
        # Remove outliers using median-based filtering
        median_features = np.median(features, axis=0)
        distances = np.array([np.linalg.norm(f - median_features) for f in features])
        threshold = np.median(distances) + 2 * np.std(distances)
        
        valid_features = features[distances <= threshold]
        
        if len(valid_features) == 0:
            return median_features
        
        # Weighted average with more recent features having higher weight
        weights = np.linspace(0.5, 1.0, len(valid_features))
        weights = weights / np.sum(weights)
        
        return np.average(valid_features, axis=0, weights=weights)
    
    def _calculate_similarity_score(self, features1: np.ndarray, features2: np.ndarray,
                                   pos1: Tuple[float, float], pos2: Tuple[float, float],
                                   predicted_pos: Optional[Tuple[float, float]] = None) -> float:
        """Calculate comprehensive similarity score"""
        
        # 1. Appearance similarity (cosine similarity)
        appearance_sim = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-7
        )
        
        # 2. Position similarity
        position_dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
        position_sim = np.exp(-position_dist / self.max_distance)
        
        # 3. Motion consistency (if predicted position available)
        motion_sim = 1.0
        if predicted_pos is not None:
            motion_dist = np.linalg.norm(np.array(pos2) - np.array(predicted_pos))
            motion_sim = np.exp(-motion_dist / (self.max_distance * 0.5))
        
        # Weighted combination
        weights = [0.6, 0.25, 0.15]  # Appearance, position, motion
        combined_score = (weights[0] * appearance_sim + 
                         weights[1] * position_sim + 
                         weights[2] * motion_sim)
        
        return combined_score
    
    def _detect_players(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced player detection with better filtering"""
        results = self.model(frame, conf=max(0.1, self.conf_threshold * 0.7), verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                raw_detections = []
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Enhanced filtering
                    height = y2 - y1
                    width = x2 - x1
                    area = height * width
                    aspect_ratio = height / max(width, 1)
                    
                    # Strict quality filters
                    is_player = cls == 2  # Soccer player class
                    has_min_size = area > 800  # Increased minimum area
                    has_good_aspect = 1.5 <= aspect_ratio <= 4.0  # Tighter aspect ratio
                    has_confidence = conf >= self.conf_threshold
                    is_reasonable_size = height >= 40 and width >= 15  # Minimum dimensions
                    
                    if (is_player and has_min_size and has_good_aspect and 
                        has_confidence and is_reasonable_size):
                        raw_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'area': area
                        })
                
                # Apply sophisticated NMS
                if raw_detections:
                    # Sort by confidence
                    raw_detections.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # Custom NMS with area consideration
                    final_detections = []
                    for det in raw_detections:
                        bbox = det['bbox']
                        should_keep = True
                        
                        for kept_det in final_detections:
                            kept_bbox = kept_det['bbox']
                            iou = self._calculate_iou(bbox, kept_bbox)
                            
                            # More sophisticated NMS logic
                            if iou > 0.3:  # Lower IoU threshold for tighter filtering
                                # Keep the one with better confidence and area
                                if det['confidence'] <= kept_det['confidence']:
                                    should_keep = False
                                    break
                        
                        if should_keep:
                            final_detections.append(det)
                    
                    # Create final detection objects
                    for det in final_detections:
                        bbox = det['bbox']
                        detection = {
                            'bbox': bbox,
                            'confidence': det['confidence'],
                            'centroid': self._get_centroid(bbox),
                            'features': self._extract_enhanced_features(frame, bbox)
                        }
                        detections.append(detection)
        
        self.total_detections += len(detections)
        return detections
    
    def _get_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculate centroid with bottom-weighted bias for better tracking"""
        x1, y1, x2, y2 = bbox
        # Weight towards bottom of bounding box (more stable for people)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + 3 * y2) / 4.0  # Bottom-weighted
        return (center_x, center_y)
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Standard IoU calculation"""
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
    
    def _match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Enhanced detection-to-track association with multiple stages"""
        
        if len(self.active_players) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(self.active_players.keys()), []
        
        # Stage 1: Match with active players
        player_ids = list(self.active_players.keys())
        cost_matrix = np.full((len(player_ids), len(detections)), 1.0)
        
        for i, player_id in enumerate(player_ids):
            player_data = self.active_players[player_id]
            player_features = self._get_average_features(player_id)
            
            if player_features is None:
                continue
            
            # Get predicted position
            predicted_pos = self._predict_position(player_id)
            if predicted_pos == (0, 0):
                predicted_pos = player_data['centroid']
            
            for j, detection in enumerate(detections):
                # Calculate comprehensive similarity
                similarity = self._calculate_similarity_score(
                    player_features, detection['features'],
                    player_data['centroid'], detection['centroid'],
                    predicted_pos
                )
                
                # Convert similarity to cost (lower is better)
                cost_matrix[i, j] = 1.0 - similarity
        
        # Solve assignment problem
        matches = []
        if cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for row, col in zip(row_indices, col_indices):
                similarity = 1.0 - cost_matrix[row, col]
                # Stricter threshold for matching
                if similarity >= 0.7:  # Higher threshold
                    matches.append((player_ids[row], col))
        
        matched_player_ids = [match[0] for match in matches]
        matched_detection_indices = [match[1] for match in matches]
        unmatched_players = [pid for pid in player_ids if pid not in matched_player_ids]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detection_indices]
        
        # Stage 2: Try to match unmatched detections with disappeared players
        additional_matches = []
        remaining_unmatched = []
        
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            best_match_id = None
            best_similarity = 0.0
            
            # Check disappeared players
            for player_id in self.disappeared_players.keys():
                if player_id in self.player_gallery:
                    gallery_features = self._get_average_features(player_id)
                    if gallery_features is not None:
                        similarity = self._calculate_similarity_score(
                            gallery_features, detection['features'],
                            self.player_gallery[player_id]['last_centroid'],
                            detection['centroid']
                        )
                        
                        if similarity > best_similarity and similarity >= self.reid_threshold:
                            best_similarity = similarity
                            best_match_id = player_id
            
            if best_match_id is not None:
                additional_matches.append((best_match_id, det_idx))
                self.total_reidentifications += 1
                logger.info(f"Re-identified player {best_match_id} with similarity {best_similarity:.3f}")
            else:
                remaining_unmatched.append(det_idx)
        
        return matches + additional_matches, unmatched_players, remaining_unmatched
    
    def _update_player_data(self, player_id: int, detection: Dict):
        """Update all player data structures"""
        # Update active player data
        self.active_players[player_id] = {
            'bbox': detection['bbox'],
            'centroid': detection['centroid'],
            'confidence': detection['confidence'],
            'last_seen': self.frame_count
        }
        
        # Update feature and position buffers
        self.feature_buffers[player_id].append(detection['features'])
        self.position_buffers[player_id].append(detection['centroid'])
        self.confidence_buffers[player_id].append(detection['confidence'])
        self.bbox_buffers[player_id].append(detection['bbox'])
        
        # Update motion model
        self._update_motion_model(player_id, detection['centroid'])
        
        # Update player gallery (long-term storage)
        self.player_gallery[player_id] = {
            'last_centroid': detection['centroid'],
            'last_bbox': detection['bbox'],
            'last_confidence': detection['confidence'],
            'last_seen': self.frame_count,
            'total_detections': self.player_gallery.get(player_id, {}).get('total_detections', 0) + 1
        }
        
        # Update stability tracking
        self.id_stability[player_id] += 1
        
        # Remove from disappeared if it was there
        if player_id in self.disappeared_players:
            del self.disappeared_players[player_id]
    
    def track_frame(self, frame: np.ndarray) -> List[Dict]:
        """Main tracking function with enhanced consistency"""
        self.frame_count += 1
        detections = self._detect_players(frame)
        
        # Handle case with no detections
        if len(detections) == 0:
            # Move active players to disappeared
            for player_id in list(self.active_players.keys()):
                if player_id not in self.disappeared_players:
                    self.disappeared_players[player_id] = 1
                else:
                    self.disappeared_players[player_id] += 1
                
                # Remove from active
                del self.active_players[player_id]
            
            # Clean up disappeared players that have been gone too long
            for player_id in list(self.disappeared_players.keys()):
                if self.disappeared_players[player_id] > self.max_disappeared:
                    del self.disappeared_players[player_id]
                    logger.info(f"Player {player_id} permanently lost after {self.max_disappeared} frames")
            
            return []
        
        # Initialize tracking if no active players
        if len(self.active_players) == 0 and len(self.disappeared_players) == 0:
            # First frame or restart - assign new IDs to all detections
            for detection in detections:
                self._update_player_data(self.next_id, detection)
                logger.info(f"Initialized new player {self.next_id}")
                self.next_id += 1
        else:
            # Perform matching
            matches, unmatched_players, unmatched_detections = self._match_detections_to_tracks(detections)
            
            # Update matched players
            for player_id, detection_idx in matches:
                detection = detections[detection_idx]
                self._update_player_data(player_id, detection)
            
            # Handle unmatched players (move to disappeared)
            for player_id in unmatched_players:
                if player_id not in self.disappeared_players:
                    self.disappeared_players[player_id] = 1
                else:
                    self.disappeared_players[player_id] += 1
                
                # Remove from active
                if player_id in self.active_players:
                    del self.active_players[player_id]
            
            # Create new tracks for unmatched detections
            for detection_idx in unmatched_detections:
                detection = detections[detection_idx]
                self._update_player_data(self.next_id, detection)
                logger.info(f"Created new player track {self.next_id}")
                self.next_id += 1
            
            # Clean up disappeared players
            for player_id in list(self.disappeared_players.keys()):
                if self.disappeared_players[player_id] > self.max_disappeared:
                    del self.disappeared_players[player_id]
                    logger.info(f"Player {player_id} permanently lost")
        
        # Prepare results
        results = []
        for player_id, player_data in self.active_players.items():
            results.append({
                'id': player_id,
                'bbox': player_data['bbox'],
                'centroid': player_data['centroid'],
                'confidence': player_data['confidence'],
                'stability': self.id_stability[player_id]
            })
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get comprehensive tracking statistics"""
        return {
            'total_frames': self.frame_count,
            'total_detections': self.total_detections,
            'active_players': len(self.active_players),
            'disappeared_players': len(self.disappeared_players),
            'total_players_ever': self.next_id,
            'reidentifications': self.total_reidentifications,
            'id_switches': self.id_switches,
            'avg_detections_per_frame': self.total_detections / max(1, self.frame_count),
            'player_gallery_size': len(self.player_gallery)
        }