import os
import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Fix for PyTorch 2.6 compatibility - disable weights_only restriction
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracking.tracker import PlayerTracker
from utils.video_io import VideoProcessor
from utils.draw import ResultVisualizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('player_reid.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Player Re-identification System')
    parser.add_argument('--input', type=str, default='D:/Projects2.0/Listai/Assignment Materials/15sec_input_720p.mp4', 
                       help='Input video path')
    parser.add_argument('--output', type=str, default='D:/Projects2.0/Listai/player-reid/output_tracked.mp4', 
                       help='Output video path')
    parser.add_argument('--model', type=str, default='D:/Projects2.0/Listai/player-reid/models/best.pt', 
                       help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.3,  # Lowered default confidence threshold
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--show-debug', action='store_true',
                       help='Show debug information on frames')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    if not os.path.exists(args.input):
        logger.error(f"Input video not found: {args.input}")
        return
    
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    try:
        logger.info("Initializing player tracker...")
        tracker = PlayerTracker(args.model, conf_threshold=args.conf)
        
        logger.info("Initializing video processor...")
        video_processor = VideoProcessor(args.input)
        
        logger.info("Initializing visualizer...")
        visualizer = ResultVisualizer()
        
        logger.info(f"Processing video: {args.input}")
        frames = video_processor.get_frames()
        fps = video_processor.get_fps()
        frame_size = video_processor.get_frame_size()
        
        output_frames = []
        total_detections = 0
        frames_with_detections = 0
        
        # Use tqdm for progress bar
        for frame_idx, frame in enumerate(tqdm(frames, desc="Processing frames")):
            tracked_players = tracker.track_frame(frame)
            
            if tracked_players:
                frames_with_detections += 1
                total_detections += len(tracked_players)
            
            annotated_frame = visualizer.draw_tracks(frame, tracked_players)
            
            # Add debug info if requested
            if args.show_debug:
                debug_text = f"Frame: {frame_idx+1}/{len(frames)} | Players: {len(tracked_players)}"
                visualizer.add_debug_info(annotated_frame, debug_text)
            
            output_frames.append(annotated_frame)
            
            # Log only if no detections (for debugging)
            if not tracked_players:
                logger.debug(f"No detections in frame {frame_idx+1}")
        
        # Log detection statistics
        avg_detections = total_detections / len(frames) if frames else 0
        detection_rate = (frames_with_detections / len(frames)) * 100 if frames else 0
        logger.info(f"Detection Statistics:")
        logger.info(f"- Average detections per frame: {avg_detections:.2f}")
        logger.info(f"- Frames with detections: {detection_rate:.1f}%")
        
        logger.info(f"Saving output video: {args.output}")
        video_processor.save_video(output_frames, args.output, fps)
        
        logger.info("Player re-identification completed successfully!")
        
        stats = tracker.get_statistics()
        logger.info(f"Tracking statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()