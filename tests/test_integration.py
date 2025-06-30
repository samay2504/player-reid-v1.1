"""
Integration test script for Player Re-identification System
Tests the integration between tracker and visual memory system
"""

import os
import sys
import logging
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracking.tracker import EnhancedPlayerTracker
from visual_memory import VisualMemorySystem
from utils.draw import ResultVisualizer

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_visual_memory_initialization():
    """Test visual memory system initialization"""
    logger = setup_logging()
    logger.info("Testing Visual Memory System initialization...")
    
    try:
        vm = VisualMemorySystem()
        logger.info("✓ Visual Memory System initialized successfully")
        logger.info(f"  - Embedding dimension: {vm.embedding_dim}")
        logger.info(f"  - Memory size: {vm.memory_size}")
        return True
    except Exception as e:
        logger.error(f"✗ Visual Memory System initialization failed: {e}")
        return False

def test_tracker_initialization():
    """Test tracker initialization"""
    logger = setup_logging()
    logger.info("Testing Enhanced Player Tracker initialization...")
    
    try:
        # Use a mock model path for testing
        mock_model_path = "D:/Projects2.0/Listai/models/best.pt"
        
        # Test without visual memory
        tracker = EnhancedPlayerTracker(mock_model_path)
        logger.info("✓ Enhanced Player Tracker initialized successfully (without visual memory)")
        
        # Test with visual memory
        vm = VisualMemorySystem()
        tracker_with_vm = EnhancedPlayerTracker(mock_model_path, visual_memory=vm)
        logger.info("✓ Enhanced Player Tracker initialized successfully (with visual memory)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Enhanced Player Tracker initialization failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction from both systems"""
    logger = setup_logging()
    logger.info("Testing feature extraction...")
    
    try:
        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = (100, 100, 200, 300)
        
        # Test tracker feature extraction
        mock_model_path = "D:/Projects2.0/Listai/models/best.pt"
        tracker = EnhancedPlayerTracker(mock_model_path)
        tracker_features = tracker._extract_enhanced_features(test_frame, test_bbox)
        
        logger.info(f"✓ Tracker feature extraction: {len(tracker_features)} dimensions")
        
        # Test visual memory feature extraction
        vm = VisualMemorySystem()
        vm_embedding, vm_confidence = vm.extract_visual_embedding(test_frame, test_bbox)
        
        logger.info(f"✓ Visual Memory feature extraction: {len(vm_embedding)} dimensions, confidence: {vm_confidence:.3f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Feature extraction test failed: {e}")
        return False

def test_memory_operations():
    """Test visual memory operations"""
    logger = setup_logging()
    logger.info("Testing visual memory operations...")
    
    try:
        vm = VisualMemorySystem()
        
        # Test ID creation
        id1 = vm.create_new_player_id()
        id2 = vm.create_new_player_id()
        logger.info(f"✓ Player ID creation: {id1}, {id2}")
        
        # Test memory statistics
        stats = vm.get_memory_statistics()
        logger.info(f"✓ Memory statistics: {stats}")
        
        # Test memory storage (with dummy data)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = (100, 100, 200, 300)
        
        vm.store_player_memory(id1, test_frame, test_bbox)
        logger.info("✓ Memory storage test completed")
        
        # Test player recognition with a threshold
        vm_id, vm_score = vm.recognize_player(test_frame, test_bbox, threshold=0.75)
        logger.info(f"✓ Player recognition test: ID={vm_id}, Score={vm_score:.3f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Memory operations test failed: {e}")
        return False

def test_system_integration():
    """Test complete system integration"""
    logger = setup_logging()
    logger.info("Testing complete system integration...")
    
    try:
        # Initialize both systems
        vm = VisualMemorySystem()
        mock_model_path = "D:/Projects2.0/Listai/models/best.pt"
        tracker = EnhancedPlayerTracker(mock_model_path, visual_memory=vm)
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Test tracking with visual memory
        results = tracker.track_frame(test_frame)
        logger.info(f"✓ System integration test completed: {len(results)} results")
        
        return True
    except Exception as e:
        logger.error(f"✗ System integration test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    logger = setup_logging()
    logger.info("=== Player Re-identification System Integration Tests ===")
    
    tests = [
        ("Visual Memory Initialization", test_visual_memory_initialization),
        ("Tracker Initialization", test_tracker_initialization),
        ("Feature Extraction", test_feature_extraction),
        ("Memory Operations", test_memory_operations),
        ("System Integration", test_system_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"Test '{test_name}' failed!")
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! System integration is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 