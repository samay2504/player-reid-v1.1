import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add the parent directory to the path to import from sibling directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from tracking.tracker import EnhancedPlayerTracker
from utils.video_io import VideoProcessor

def test_consistent_ids():
    """Test that players maintain consistent IDs across frames"""
    
    # Initialize tracker with max 22 players
    model_path = "D:/Projects2.0/Listai/models/best.pt"
    tracker = EnhancedPlayerTracker(
        model_path=model_path,
        max_players=22,
        reid_threshold=0.6,
        max_disappeared=120
    )
    
    # Test video path
    video_path = "D:/Projects2.0/Listai/Assignment Materials/15sec_input_720p.mp4"
    
    if not os.path.exists(video_path):
        print(f"Test video not found: {video_path}")
        return
    
    print("Testing consistent ID management...")
    print(f"Max players: {tracker.max_players}")
    print(f"Re-ID threshold: {tracker.reid_threshold}")
    
    # Process first 100 frames
    video_processor = VideoProcessor(video_path)
    frames = video_processor.get_frames()[:100]  # First 100 frames
    
    player_id_history = {}  # Track ID consistency
    
    for frame_idx, frame in enumerate(frames):
        tracked_players = tracker.track_frame(frame)
        
        # Record player IDs
        for player in tracked_players:
            player_id = player['id']
            if player_id not in player_id_history:
                player_id_history[player_id] = {
                    'first_seen': frame_idx,
                    'last_seen': frame_idx,
                    'frames_seen': 1
                }
            else:
                player_id_history[player_id]['last_seen'] = frame_idx
                player_id_history[player_id]['frames_seen'] += 1
        
        if frame_idx % 20 == 0:
            print(f"Frame {frame_idx}: {len(tracked_players)} players tracked")
    
    # Analyze results
    print(f"\n=== Results ===")
    print(f"Total unique players tracked: {len(player_id_history)}")
    print(f"Max players allowed: {tracker.max_players}")
    
    stats = tracker.get_statistics()
    print(f"Re-identifications: {stats['reidentifications']}")
    print(f"Used IDs: {stats['used_ids']}")
    print(f"Registered players: {stats['total_players_registered']}")
    
    # Check for ID consistency
    consistent_players = 0
    for player_id, history in player_id_history.items():
        if history['frames_seen'] > 5:  # Player seen in multiple frames
            consistent_players += 1
    
    print(f"Consistent players (seen >5 frames): {consistent_players}")
    
    if len(player_id_history) <= tracker.max_players:
        print("✅ ID management working correctly - within max player limit")
    else:
        print("❌ Too many unique IDs created")
    
    if stats['reidentifications'] > 0:
        print("✅ Re-identification working - players being re-identified")
    else:
        print("⚠️  No re-identifications recorded")

if __name__ == "__main__":
    test_consistent_ids() 