import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
from PIL import Image
import time
import timm
from torchvision import transforms

logger = logging.getLogger(__name__)

@dataclass
class VisualMemory:
    """Stores visual memory for a player"""
    player_id: int
    embeddings: List[np.ndarray]  # Multiple visual embeddings
    timestamps: List[float]       # When each embedding was captured
    confidence_scores: List[float] # Quality of each embedding
    distinctive_features: Dict[str, Any]  # Key visual characteristics
    last_seen: float
    memory_strength: float = 1.0  # How strong this memory is

    def get_representative_embedding(self) -> Optional[np.ndarray]:
        """Get a representative embedding for the player, weighted by confidence."""
        if not self.embeddings:
            return None
        
        embeddings = np.array(self.embeddings)
        confidences = np.array(self.confidence_scores)
        
        # Normalize confidences to use as weights
        weights = confidences / (np.sum(confidences) + 1e-7)
        
        # Compute weighted average
        representative_embedding = np.average(embeddings, axis=0, weights=weights)
        return representative_embedding / (np.linalg.norm(representative_embedding) + 1e-7)

class VisualMemorySystem:
    """
    Human-like visual memory system using EVA-02 for player recognition.
    Captures distinctive features and maintains consistent player identities.
    """
    
    def __init__(self, 
                 model_name: str = "eva02_base_patch14_224.mim_in22k",
                 memory_size: int = 50,
                 similarity_threshold: float = 0.85,
                 max_memory_age: float = 300.0):  # 5 minutes
        
        self.model_name = model_name
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.max_memory_age = max_memory_age
        
        # Initialize EVA-02 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_vision_model()
        
        # Get embedding dimension from model
        self.embedding_dim = self._get_embedding_dimension()
        
        # Visual memory storage
        self.player_memories: Dict[int, VisualMemory] = {}
        self.next_player_id = 0
        
        # Multi-scale feature extraction
        self.face_detector = self._load_face_detector()
        
        # Attention-based feature tracking
        self.attention_regions = defaultdict(list)
        
        logger.info(f"Visual Memory System initialized with {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _load_vision_model(self):
        """Load EVA-02 vision model for feature extraction"""
        try:
            logger.info(f"Loading vision model: {self.model_name}")
            
            # Load model using timm
            self.vision_model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            self.vision_model.to(self.device)
            self.vision_model.eval()
            
            # Create image processor for EVA-02
            self.image_processor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Vision model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise
    
    def _load_face_detector(self):
        """Load face detector for facial feature extraction"""
        try:
            # Use OpenCV's Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            return face_cascade
        except Exception as e:
            logger.warning(f"Face detector not available: {e}")
            return None
    
    def extract_visual_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
        """
        Extract high-quality visual embedding from player region
        Returns: (embedding, confidence_score)
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.embedding_dim), 0.0  # Dynamic embedding size
        
        # Extract player region
        player_roi = frame[y1:y2, x1:x2]
        
        # Convert to PIL and process
        try:
            player_pil = Image.fromarray(cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.warning(f"Failed to convert ROI to PIL image: {e}")
            return np.zeros(self.embedding_dim), 0.0
        
        try:
            # Process image for EVA-02
            input_tensor = self.image_processor(player_pil).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.vision_model(input_tensor)
                
                # Handle different output formats
                if hasattr(embedding, 'cpu'):
                    embedding = embedding.cpu().numpy()[0]  # Get the feature vector
                elif hasattr(embedding, 'numpy'):
                    embedding = embedding.numpy()[0]
                else:
                    # Fallback for unexpected output types
                    logger.warning(f"Unexpected embedding type: {type(embedding)}")
                    return np.zeros(self.embedding_dim), 0.0
            
            # Calculate confidence based on image quality
            confidence = self._calculate_image_quality(player_roi)
            
            return embedding, confidence
            
        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
            return np.zeros(self.embedding_dim), 0.0
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """Calculate image quality score for confidence weighting"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Normalize and combine
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize sharpness
        brightness_score = 1.0 - abs(brightness - 128) / 128  # Prefer medium brightness
        contrast_score = min(contrast / 50.0, 1.0)  # Normalize contrast
        
        # Weighted combination
        quality_score = 0.4 * sharpness_score + 0.3 * brightness_score + 0.3 * contrast_score
        return max(0.0, min(1.0, quality_score))
    
    def extract_distinctive_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Extract distinctive visual features that humans notice"""
        x1, y1, x2, y2 = bbox
        player_roi = frame[y1:y2, x1:x2]
        
        features = {}
        
        # 1. Color analysis
        features['dominant_colors'] = self._extract_dominant_colors(player_roi)
        
        # 2. Texture patterns
        features['texture_patterns'] = self._extract_texture_patterns(player_roi)
        
        # 3. Shape characteristics
        features['shape_features'] = self._extract_shape_features(player_roi)
        
        # 4. Facial features (if detectable)
        if self.face_detector is not None:
            features['facial_features'] = self._extract_facial_features(player_roi)
        
        # 5. Motion patterns (if available)
        features['motion_signature'] = self._extract_motion_signature(player_roi)
        
        return features
    
    def _extract_dominant_colors(self, image: np.ndarray) -> Dict[str, float]:
        """Extract dominant colors in different regions"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze different regions
        h, w = image.shape[:2]
        regions = {
            'upper': hsv[:h//3, :],
            'middle': hsv[h//3:2*h//3, :],
            'lower': hsv[2*h//3:, :]
        }
        
        dominant_colors = {}
        for region_name, region in regions.items():
            # Calculate mean hue and saturation
            mean_hue = np.mean(region[:, :, 0])
            mean_sat = np.mean(region[:, :, 1])
            mean_val = np.mean(region[:, :, 2])
            
            dominant_colors[f'{region_name}_hue'] = mean_hue
            dominant_colors[f'{region_name}_saturation'] = mean_sat
            dominant_colors[f'{region_name}_value'] = mean_val
        
        return dominant_colors
    
    def _extract_texture_patterns(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture patterns using Local Binary Patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple texture analysis using gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        return {
            'texture_strength': np.mean(magnitude),
            'texture_variance': np.var(magnitude),
            'edge_density': np.sum(magnitude > np.mean(magnitude)) / magnitude.size
        }
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract shape and contour features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            return {
                'contour_area': area,
                'contour_perimeter': perimeter,
                'circularity': 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0,
                'aspect_ratio': image.shape[1] / image.shape[0]
            }
        
        return {'contour_area': 0, 'contour_perimeter': 0, 'circularity': 0, 'aspect_ratio': 1.0}
    
    def _extract_facial_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract facial features if face is detected"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.face_detector is not None:
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                face_roi = gray[y:y+h, x:x+w]
                
                return {
                    'face_detected': True,
                    'face_size': w * h,
                    'face_position': (x, y),
                    'face_aspect_ratio': w / h if h > 0 else 1.0
                }
        
        return {'face_detected': False}
    
    def _extract_motion_signature(self, image: np.ndarray) -> Dict[str, float]:
        """Extract motion-related features (placeholder for temporal analysis)"""
        # This would be enhanced with temporal information
        return {
            'motion_stability': 1.0,  # Placeholder
            'motion_direction': 0.0   # Placeholder
        }
    
    def store_player_memory(self, player_id: int, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Store visual memory for a player"""
        current_time = time.time()
        
        # Extract visual embedding and confidence
        embedding, confidence = self.extract_visual_embedding(frame, bbox)
        
        # Extract distinctive features
        distinctive_features = self.extract_distinctive_features(frame, bbox)
        
        if player_id not in self.player_memories:
            # Create new memory
            self.player_memories[player_id] = VisualMemory(
                player_id=player_id,
                embeddings=[embedding],
                timestamps=[current_time],
                confidence_scores=[confidence],
                distinctive_features=distinctive_features,
                last_seen=current_time
            )
        else:
            # Update existing memory
            memory = self.player_memories[player_id]
            memory.embeddings.append(embedding)
            memory.timestamps.append(current_time)
            memory.confidence_scores.append(confidence)
            memory.last_seen = current_time
            
            # Update distinctive features (average over time)
            for key, value in distinctive_features.items():
                if key not in memory.distinctive_features:
                    memory.distinctive_features[key] = value
                else:
                    # Only average if both are numeric, else keep the latest value
                    old_value = memory.distinctive_features[key]
                    if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                        memory.distinctive_features[key] = (old_value + value) / 2
                    else:
                        memory.distinctive_features[key] = value
            
            # Maintain memory size
            if len(memory.embeddings) > self.memory_size:
                memory.embeddings.pop(0)
                memory.timestamps.pop(0)
                memory.confidence_scores.pop(0)
        
        logger.debug(f"Stored memory for player {player_id} (confidence: {confidence:.3f})")
    
    def recognize_player(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], threshold: float) -> Tuple[Optional[int], float]:
        """
        Recognize a player from visual memory using a representative embedding.
        Returns: (player_id, confidence_score) or (None, 0.0)
        """
        current_time = time.time()
        
        # Extract embedding for current detection
        embedding, confidence = self.extract_visual_embedding(frame, bbox)
        
        if confidence < 0.3:  # Low quality image, reject early
            return None, 0.0
        
        best_match_id = None
        best_match_score = 0.0
        
        # Compare against all stored memories
        for player_id, memory in self.player_memories.items():
            # Check if memory is too old
            if current_time - memory.last_seen > self.max_memory_age:
                continue

            # Get the representative embedding for the player
            representative_embedding = memory.get_representative_embedding()
            if representative_embedding is None:
                continue

            # Calculate cosine similarity with the representative embedding
            similarity = np.dot(embedding, representative_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(representative_embedding) + 1e-7
            )
            
            # The final score is a blend of similarity and the memory's strength
            final_score = similarity * memory.memory_strength

            if final_score > best_match_score and final_score >= threshold:
                best_match_score = final_score
                best_match_id = player_id
        
        return best_match_id, best_match_score
    
    def create_new_player_id(self) -> int:
        """Create a new player ID"""
        player_id = self.next_player_id
        self.next_player_id += 1
        return player_id
    
    def update_memory_strength(self, player_id: int, success: bool):
        """Update memory strength based on recognition success"""
        if player_id in self.player_memories:
            memory = self.player_memories[player_id]
            
            if success:
                # Strengthen memory on successful recognition
                memory.memory_strength = min(2.0, memory.memory_strength + 0.1)
            else:
                # Weaken memory on failed recognition
                memory.memory_strength = max(0.1, memory.memory_strength - 0.05)
    
    def cleanup_old_memories(self):
        """Remove old memories to prevent memory bloat"""
        current_time = time.time()
        to_remove = []
        
        for player_id, memory in self.player_memories.items():
            if current_time - memory.last_seen > self.max_memory_age:
                to_remove.append(player_id)
        
        for player_id in to_remove:
            del self.player_memories[player_id]
            logger.info(f"Removed old memory for player {player_id}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the visual memory system"""
        total_memories = len(self.player_memories)
        total_embeddings = sum(len(m.embeddings) for m in self.player_memories.values())
        
        return {
            'total_players': total_memories,
            'total_embeddings': total_embeddings,
            'avg_embeddings_per_player': total_embeddings / max(1, total_memories),
            'memory_size_limit': self.memory_size,
        }
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the model"""
        try:
            # Create a dummy input to get the output dimension
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                output = self.vision_model(dummy_input)
                
                # Handle different output formats
                if hasattr(output, 'shape') and len(output.shape) >= 2:
                    return output.shape[1]
                elif hasattr(output, 'size') and len(output.size()) >= 2:
                    return output.size(1)
                elif isinstance(output, torch.Tensor):
                    # If it's a tensor, try to get the last dimension
                    return output.shape[-1] if len(output.shape) > 0 else 768
                else:
                    # Fallback for unexpected output types
                    logger.warning(f"Unexpected model output type: {type(output)}")
                    return 768
                    
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension, using default 768: {e}")
            return 768 