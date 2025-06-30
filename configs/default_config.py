"""
Default configuration for Player Re-identification System
"""

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'D:/Projects2.0/Listai/models/best.pt',
    'conf_threshold': 0.35,
    'max_disappeared': 60,
    'max_distance': 200.0,
    'reid_threshold': 0.65
}

# Visual Memory Configuration
VISUAL_MEMORY_CONFIG = {
    'model_name': 'eva02_base_patch14_224.mim_in22k',
    'memory_size': 50,
    'similarity_threshold': 0.85,
    'max_memory_age': 300.0  # 5 minutes
}

# File Paths
PATHS = {
    'input_video': 'D:/Projects2.0/Listai/Assignment Materials/15sec_input_720p.mp4',
    'output_video': 'outputs/enhanced_output_tracked.mp4',
    'analysis_output': 'outputs/player_consistency_analysis.json',
    'log_file': 'logs/enhanced_player_reid.log'
}

# Processing Configuration
PROCESSING_CONFIG = {
    'show_debug': False,
    'log_level': 'INFO'
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    'feature_dim': 162,  # Enhanced feature vector dimension
    'embedding_dim': 768,  # EVA-02 embedding dimension
    'roi_size': (64, 128),  # Standard ROI size for feature extraction
    'model_input_size': (224, 224)  # EVA-02 input size
} 