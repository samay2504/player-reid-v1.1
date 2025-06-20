import os
import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import json

# Fix for PyTorch 2.6 compatibility - disable weights_only restriction
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced tracker
from tracking.tracker import EnhancedPlayerTracker
from utils.video_io import VideoProcessor
from utils.draw import ResultVisualizer

def setup_logging(log_level=logging.INFO):
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_player_reid.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def analyze_player_consistency(tracking_results, output_path=None):
    """Analyze player ID consistency throughout the video"""
    player_appearances = {}
    frame_count = 0
    
    for frame_results in tracking_results:
        frame_count += 1
        for player in frame_results:
            player_id = player['id']
            if player_id not in player_appearances:
                player_appearances[player_id] = {
                    'first_frame': frame_count,
                    'last_frame': frame_count,
                    'total_frames': 1,
                    'gaps': [],
                    'current_streak': 1,
                    'longest_streak': 1,
                    'last_seen': frame_count
                }
            else:
                data = player_appearances[player_id]
                gap = frame_count - data['last_seen'] - 1
                if gap > 0:
                    data['gaps'].append(gap)
                    data['current_streak'] = 1
                else:
                    data['current_streak'] += 1
                
                data['longest_streak'] = max(data['longest_streak'], data['current_streak'])
                data['last_frame'] = frame_count
                data['total_frames'] += 1
                data['last_seen'] = frame_count
    
    # Calculate consistency metrics
    analysis = {
        'total_frames': frame_count,
        'unique_players': len(player_appearances),
        'player_stats': {}
    }
    
    for player_id, data in player_appearances.items():
        duration = data['last_frame'] - data['first_frame'] + 1
        presence_ratio = data['total_frames'] / duration if duration > 0 else 0
        avg_gap = sum(data['gaps']) / len(data['gaps']) if data['gaps'] else 0
        
        analysis['player_stats'][player_id] = {
            'duration': duration,
            'presence_frames': data['total_frames'],
            'presence_ratio': presence_ratio,
            'gaps_count': len(data['gaps']),
            'average_gap': avg_gap,
            'longest_streak': data['longest_streak'],
            'first_appearance': data['first_frame'],
            'last_appearance': data['last_frame']
        }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description='Enhanced Player Re-identification System')
    parser.add_argument('--input', type=str, 
                       default='D:/Projects2.0/Listai/Assignment Materials/15sec_input_720p.mp4', 
                       help='Input video path')
    parser.add_argument('--output', type=str, 
                       default='D:/Projects2.0/Listai/player-reid/enhanced_output_tracked.mp4', 
                       help='Output video path')
    parser.add_argument('--model', type=str, 
                       default='D:/Projects2.0/Listai/models/best.pt', 
                       help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.35,
                       help='Confidence threshold (default: 0.35)')
    parser.add_argument('--max-disappeared', type=int, default=60,
                       help='Maximum frames before considering player lost (default: 60)')
    parser.add_argument('--max-distance', type=float, default=200.0,
                       help='Maximum distance for matching (default: 200.0)')
    parser.add_argument('--reid-threshold', type=float, default=0.65,
                       help='Re-identification similarity threshold (default: 0.65)')
    parser.add_argument('--show-debug', action='store_true',
                       help='Show debug information on frames')
    parser.add_argument('--analysis-output', type=str, 
                       default='D:/Projects2.0/Listai/player-reid/player_consistency_analysis.json',
                       help='Output path for player consistency analysis')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(log_level)
    
    # Validate input files
    if not os.path.exists(args.input):
        logger.error(f"Input video not found: {args.input}")
        return
    
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    try:
        logger.info("=== Enhanced Player Re-identification System ===")
        logger.info(f"Input video: {args.input}")
        logger.info(f"Output video: {args.output}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Configuration:")
        logger.info(f"  - Confidence threshold: {args.conf}")
        logger.info(f"  - Max disappeared frames: {args.max_disappeared}")
        logger.info(f"  - Max distance: {args.max_distance}")
        logger.info(f"  - Re-ID threshold: {args.reid_threshold}")
        
        # Initialize enhanced tracker with all parameters
        logger.info("Initializing enhanced player tracker...")
        tracker = EnhancedPlayerTracker(
            model_path=args.model,
            conf_threshold=args.conf,
            max_disappeared=args.max_disappeared,
            max_distance=args.max_distance,
            reid_threshold=args.reid_threshold
        )
        
        logger.info("Initializing video processor...")
        video_processor = VideoProcessor(args.input)
        
        logger.info("Initializing visualizer...")
        visualizer = ResultVisualizer()
        
        logger.info(f"Processing video: {args.input}")
        frames = video_processor.get_frames()
        fps = video_processor.get_fps()
        frame_size = video_processor.get_frame_size()
        
        logger.info(f"Video properties: {len(frames)} frames, {fps} FPS, {frame_size}")
        
        output_frames = []
        tracking_results = []  # Store all tracking results for analysis
        total_detections = 0
        frames_with_detections = 0
        
        # Process frames with enhanced tracking
        for frame_idx, frame in enumerate(tqdm(frames, desc="Processing frames")):
            tracked_players = tracker.track_frame(frame)
            tracking_results.append(tracked_players)
            
            if tracked_players:
                frames_with_detections += 1
                total_detections += len(tracked_players)
            
            annotated_frame = visualizer.draw_tracks(frame, tracked_players)
            
            # Add debug info if requested
            if args.show_debug:
                debug_text = f"Frame: {frame_idx+1}/{len(frames)} | Players: {len(tracked_players)}"
                visualizer.add_debug_info(annotated_frame, debug_text)
            
            output_frames.append(annotated_frame)
            
            # Log detailed information for debugging
            if log_level == logging.DEBUG and not tracked_players:
                logger.debug(f"No detections in frame {frame_idx+1}")
        
        # Calculate and log comprehensive statistics
        avg_detections = total_detections / len(frames) if frames else 0
        detection_rate = (frames_with_detections / len(frames)) * 100 if frames else 0
        
        logger.info("=== Processing Complete ===")
        logger.info(f"Detection Statistics:")
        logger.info(f"  - Total frames processed: {len(frames)}")
        logger.info(f"  - Frames with detections: {frames_with_detections} ({detection_rate:.1f}%)")
        logger.info(f"  - Total detections: {total_detections}")
        logger.info(f"  - Average detections per frame: {avg_detections:.2f}")
        
        # Get enhanced tracker statistics
        tracker_stats = tracker.get_statistics()
        logger.info(f"Enhanced Tracker Statistics:")
        logger.info(f"  - Active players: {tracker_stats['active_players']}")
        logger.info(f"  - Disappeared players: {tracker_stats['disappeared_players']}")
        logger.info(f"  - Total players ever tracked: {tracker_stats['total_players_ever']}")
        logger.info(f"  - Re-identifications: {tracker_stats['reidentifications']}")
        logger.info(f"  - ID switches: {tracker_stats['id_switches']}")
        logger.info(f"  - Player gallery size: {tracker_stats['player_gallery_size']}")
        
        # Perform player consistency analysis
        logger.info("Analyzing player consistency...")
        consistency_analysis = analyze_player_consistency(tracking_results, args.analysis_output)
        
        logger.info(f"Player Consistency Analysis:")
        logger.info(f"  - Unique players tracked: {consistency_analysis['unique_players']}")
        logger.info(f"  - Analysis saved to: {args.analysis_output}")
        
        # Save output video
        logger.info(f"Saving output video: {args.output}")
        video_processor.save_video(output_frames, args.output, fps)
        
        logger.info("=== Enhanced Player Re-identification Completed Successfully! ===")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()