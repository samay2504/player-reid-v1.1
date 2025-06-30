import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

class PoseExtractor:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def extract_keypoints(self, image):
        # image: np.ndarray (RGB)
        results = self.pose.process(image)
        if not results.pose_landmarks:
            return None
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.visibility])
        return np.array(keypoints) 