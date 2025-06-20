import os
import sys
import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracking.tracker import EnhancedPlayerTracker
from utils.video_io import VideoProcessor
from utils.draw import ResultVisualizer

class TestEnhancedPlayerTracker(unittest.TestCase):
    
    def setUp(self):
        self.mock_model_path = "D:/Projects2.0/Listai/player-reid/models/best.pt"
        
    @patch('tracking.tracker.safe_yolo_load')
    def test_tracker_initialization(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = EnhancedPlayerTracker(self.mock_model_path)
        
        self.assertEqual(tracker.next_id, 0)
        self.assertEqual(len(tracker.active_players), 0)
        self.assertEqual(len(tracker.disappeared_players), 0)
        mock_safe_load.assert_called_once_with(self.mock_model_path)
    
    @patch('tracking.tracker.safe_yolo_load')
    def test_enhanced_feature_extraction(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = EnhancedPlayerTracker(self.mock_model_path)
        
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 300)
        
        features = tracker._extract_enhanced_features(test_frame, bbox)
        
        # Enhanced feature vector length is 162 (96 RGB + 32 HSV + 18 gradient + 8 texture + 8 spatial)
        self.assertEqual(len(features), 162)
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))
        
        # Test with invalid bbox
        invalid_bbox = (200, 200, 100, 100)
        features = tracker._extract_enhanced_features(test_frame, invalid_bbox)
        self.assertEqual(len(features), 162)
        self.assertTrue(np.all(features == 0))
    
    @patch('tracking.tracker.safe_yolo_load')
    def test_centroid_calculation(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = EnhancedPlayerTracker(self.mock_model_path)
        
        bbox = (100, 100, 200, 300)
        centroid = tracker._get_centroid(bbox)
        
        # Bottom-weighted centroid calculation
        expected_centroid = (150.0, 250.0)  # (x1+x2)/2, (y1+3*y2)/4
        self.assertEqual(centroid, expected_centroid)
    
    @patch('tracking.tracker.safe_yolo_load')
    def test_iou_calculation(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = EnhancedPlayerTracker(self.mock_model_path)
        
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        
        iou = tracker._calculate_iou(box1, box2)
        
        expected_iou = 2500 / 17500
        self.assertAlmostEqual(iou, expected_iou, places=5)
        
        box3 = (0, 0, 50, 50)
        box4 = (100, 100, 150, 150)
        
        iou_no_overlap = tracker._calculate_iou(box3, box4)
        self.assertEqual(iou_no_overlap, 0.0)
    
    @patch('tracking.tracker.safe_yolo_load')
    def test_motion_prediction(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = EnhancedPlayerTracker(self.mock_model_path)
        
        # Test with single position
        tracker._update_motion_model(1, (100, 100))
        predicted = tracker._predict_position(1)
        self.assertEqual(predicted, (100, 100))
        
        # Test with two positions
        tracker._update_motion_model(1, (110, 120))
        predicted = tracker._predict_position(1)
        self.assertEqual(predicted, (120, 140))
    
    @patch('tracking.tracker.safe_yolo_load')
    def test_track_frame_empty_detections(self, mock_safe_load):
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        mock_safe_load.return_value = mock_model

        tracker = EnhancedPlayerTracker(self.mock_model_path)
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Test with no detections
        with patch.object(tracker, '_detect_players', return_value=[]):
            result = tracker.track_frame(test_frame)
            self.assertEqual(len(result), 0)

        # Test with existing tracks that should disappear
        dummy_detection = {
            'bbox': (10, 20, 50, 80),
            'centroid': (30, 65),
            'features': np.ones(162),
            'confidence': 0.9
        }
        
        # First add a track
        with patch.object(tracker, '_detect_players', return_value=[dummy_detection]):
            result = tracker.track_frame(test_frame)
            self.assertEqual(len(result), 1)
        
        # Then let it disappear
        with patch.object(tracker, '_detect_players', return_value=[]):
            for _ in range(tracker.max_disappeared + 1):
                result = tracker.track_frame(test_frame)
            self.assertEqual(len(result), 0)

    @patch('tracking.tracker.safe_yolo_load')
    def test_track_frame_with_detections(self, mock_safe_load):
        mock_model = Mock()
        mock_result = Mock()
        mock_boxes = Mock()
        mock_box = Mock()
        mock_box.cls = [2]  # Class 2 for soccer player
        mock_box.conf = [0.9]
        mock_xyxy = Mock()
        mock_xyxy.cpu.return_value.numpy.return_value.astype.return_value = np.array([100, 100, 150, 250])
        mock_box.xyxy = [mock_xyxy]
        mock_boxes.__iter__ = lambda self: iter([mock_box])
        mock_result.boxes = mock_boxes

        mock_model.return_value = [mock_result]
        mock_safe_load.return_value = mock_model

        tracker = EnhancedPlayerTracker(self.mock_model_path)
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Test initial detection
        result = tracker.track_frame(test_frame)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        track = result[0]
        self.assertIn('id', track)
        self.assertIn('bbox', track)
        self.assertIn('centroid', track)
        self.assertIn('confidence', track)
        self.assertIn('stability', track)
        initial_id = track['id']

        # Test tracking consistency
        result = tracker.track_frame(test_frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], initial_id)

    @patch('tracking.tracker.safe_yolo_load')
    def test_similarity_score_calculation(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = EnhancedPlayerTracker(self.mock_model_path)
        
        features1 = np.random.rand(162)
        features2 = np.random.rand(162)
        pos1 = (100, 100)
        pos2 = (110, 110)
        
        score = tracker._calculate_similarity_score(features1, features2, pos1, pos2)
        
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)

    @patch('tracking.tracker.safe_yolo_load')
    def test_get_statistics(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = EnhancedPlayerTracker(self.mock_model_path)
        
        stats = tracker.get_statistics()
        
        expected_keys = [
            'total_frames', 'total_detections', 'active_players',
            'disappeared_players', 'total_players_ever', 'reidentifications',
            'id_switches', 'avg_detections_per_frame', 'player_gallery_size'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)

if __name__ == '__main__':
    unittest.main()
