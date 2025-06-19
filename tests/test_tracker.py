import os
import sys
import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracking.tracker import PlayerTracker
from utils.video_io import VideoProcessor
from utils.draw import ResultVisualizer

class TestPlayerTracker(unittest.TestCase):
    
    def setUp(self):
        self.mock_model_path = "D:/Projects2.0/Listai/player-reid/models/best.pt"
        
    @patch('tracking.tracker.safe_yolo_load')
    def test_tracker_initialization(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = PlayerTracker(self.mock_model_path)
        
        self.assertEqual(tracker.next_id, 0)
        self.assertEqual(len(tracker.players), 0)
        self.assertEqual(len(tracker.disappeared), 0)
        mock_safe_load.assert_called_once_with(self.mock_model_path)
    
    @patch('tracking.tracker.safe_yolo_load')
    def test_feature_extraction(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = PlayerTracker(self.mock_model_path)
        
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 300)  # Tall rectangle like a person
        
        features = tracker._extract_features(test_frame, bbox)
        
        # New feature vector length is 105 (96 color features + 9 gradient features)
        self.assertEqual(len(features), 105)
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))
        
        # Test with invalid bbox
        invalid_bbox = (200, 200, 100, 100)  # x2 < x1, y2 < y1
        features = tracker._extract_features(test_frame, invalid_bbox)
        self.assertEqual(len(features), 105)
        self.assertTrue(np.all(features == 0))
    
    @patch('tracking.tracker.safe_yolo_load')
    def test_centroid_calculation(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = PlayerTracker(self.mock_model_path)
        
        bbox = (100, 100, 200, 300)
        centroid = tracker._get_centroid(bbox)
        
        expected_centroid = (150.0, 200.0)
        self.assertEqual(centroid, expected_centroid)
    
    @patch('tracking.tracker.safe_yolo_load')
    def test_iou_calculation(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = PlayerTracker(self.mock_model_path)
        
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
    def test_predict_next_position(self, mock_safe_load):
        mock_model = Mock()
        mock_safe_load.return_value = mock_model
        
        tracker = PlayerTracker(self.mock_model_path)
        
        # Test with single position
        track_history = [(100, 100)]
        predicted = tracker._predict_next_position(track_history)
        self.assertEqual(predicted, (100, 100))
        
        # Test with two positions
        track_history = [(100, 100), (110, 120)]
        predicted = tracker._predict_next_position(track_history)
        self.assertEqual(predicted, (120, 140))
    
    @patch('tracking.tracker.safe_yolo_load')
    def test_track_frame_empty_detections(self, mock_safe_load):
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        mock_safe_load.return_value = mock_model

        tracker = PlayerTracker(self.mock_model_path)
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Test with no detections
        with patch.object(tracker, '_detect_players', return_value=[]):
            result = tracker.track_frame(test_frame)
            self.assertEqual(len(result), 0)

        # Test with existing tracks that should disappear
        dummy_detection = {
            'bbox': (10, 20, 50, 80),
            'centroid': (30, 50),
            'features': np.ones(105),
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
        mock_box.cls = [1]  # Class 1 for player
        mock_box.conf = [0.9]
        mock_xyxy = Mock()
        mock_xyxy.cpu.return_value.numpy.return_value.astype.return_value = np.array([100, 100, 150, 250])  # Tall box for person
        mock_box.xyxy = [mock_xyxy]
        mock_boxes.__iter__ = lambda self: iter([mock_box])
        mock_result.boxes = mock_boxes

        mock_model.return_value = [mock_result]
        mock_safe_load.return_value = mock_model

        tracker = PlayerTracker(self.mock_model_path)
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
        initial_id = track['id']

        # Test tracking consistency
        result = tracker.track_frame(test_frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], initial_id)

    @patch('tracking.tracker.safe_yolo_load')
    def test_aspect_ratio_filtering(self, mock_safe_load):
        mock_model = Mock()
        mock_result = Mock()
        mock_boxes = Mock()
        
        # Create two boxes: one with good aspect ratio, one with bad
        good_box = Mock()
        good_box.cls = [1]
        good_box.conf = [0.9]
        good_xyxy = Mock()
        good_xyxy.cpu.return_value.numpy.return_value.astype.return_value = np.array([100, 100, 150, 250])  # Good ratio
        good_box.xyxy = [good_xyxy]

        bad_box = Mock()
        bad_box.cls = [1]
        bad_box.conf = [0.9]
        bad_xyxy = Mock()
        bad_xyxy.cpu.return_value.numpy.return_value.astype.return_value = np.array([100, 100, 300, 150])  # Wide box
        bad_box.xyxy = [bad_xyxy]

        mock_boxes.__iter__ = lambda self: iter([good_box, bad_box])
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        mock_safe_load.return_value = mock_model

        tracker = PlayerTracker(self.mock_model_path)
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        result = tracker.track_frame(test_frame)
        self.assertEqual(len(result), 1)  # Only the good aspect ratio box should be detected

if __name__ == '__main__':
    unittest.main()
