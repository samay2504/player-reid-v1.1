import cv2
import numpy as np
import logging
from typing import List, Tuple
from moviepy.editor import VideoFileClip
import tempfile
import os

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        try:
            self.video_clip = VideoFileClip(video_path)
            self.has_audio = self.video_clip.audio is not None
        except Exception as e:
            logger.warning(f"Could not load audio from video: {e}")
            self.has_audio = False
        
        logger.info(f"Video loaded: {video_path}")
        logger.info(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}, Frames: {self.frame_count}")
        if self.has_audio:
            logger.info("Audio track detected and loaded")
    
    def get_frames(self) -> List[np.ndarray]:
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame.copy())
        
        logger.info(f"Loaded {len(frames)} frames from video")
        return frames
    
    def get_frame_at(self, frame_number: int) -> np.ndarray:
        if frame_number < 0 or frame_number >= self.frame_count:
            raise ValueError(f"Frame number {frame_number} out of range [0, {self.frame_count})")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError(f"Cannot read frame {frame_number}")
        
        return frame
    
    def get_fps(self) -> int:
        return self.fps
    
    def get_frame_size(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def get_frame_count(self) -> int:
        return self.frame_count
    
    def save_video(self, frames: List[np.ndarray], output_path: str, fps: int = None) -> None:
        if not frames:
            raise ValueError("No frames to save")
        
        if fps is None:
            fps = self.fps
        
        height, width = frames[0].shape[:2]
        
        temp_output = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        if hasattr(self, 'has_audio') and self.has_audio:
            try:
                temp_clip = VideoFileClip(temp_output)
                final_clip = temp_clip.set_audio(self.video_clip.audio)
                final_clip.write_videofile(output_path, codec='libx264', 
                                        audio_codec='aac', fps=fps)
                
                temp_clip.close()
                final_clip.close()
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                
                logger.info(f"Video saved with audio: {output_path}")
            except Exception as e:
                logger.error(f"Failed to add audio to video, saving without audio: {e}")
                if os.path.exists(temp_output):
                    os.replace(temp_output, output_path)
        else:
            if os.path.exists(temp_output):
                os.replace(temp_output, output_path)
        
        logger.info(f"Video saved: {output_path} ({len(frames)} frames, {fps} FPS)")
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'video_clip'):
            self.video_clip.close()

def load_video(video_path: str) -> VideoProcessor:
    return VideoProcessor(video_path)

def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> None:
    if not frames:
        raise ValueError("No frames to save")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    logger.info(f"Video saved: {output_path} ({len(frames)} frames, {fps} FPS)")