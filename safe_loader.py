import torch
import os
import logging
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential, ModuleList, Conv2d, BatchNorm2d, ReLU, SiLU, Upsample, Linear

logger = logging.getLogger(__name__)

# Disable weights_only restriction for PyTorch 2.6 compatibility
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# Dynamic safe globals - add classes as needed
SAFE_CLASSES = [
    DetectionModel, Sequential, ModuleList, Conv2d,
    BatchNorm2d, ReLU, SiLU, Upsample, Linear
]

def enable_safe_load():
    """Configure PyTorch for safe model loading with ultralytics compatibility"""
    torch.serialization.add_safe_globals(SAFE_CLASSES)
    
    try:
        import ultralytics.nn.modules
        for attr_name in dir(ultralytics.nn.modules):
            attr = getattr(ultralytics.nn.modules, attr_name)
            if isinstance(attr, type):
                torch.serialization.add_safe_globals([attr])
    except ImportError:
        logger.warning("ultralytics.nn.modules not available")
    except Exception as e:
        logger.warning(f"Error adding ultralytics classes: {e}")

    try:
        from ultralytics.nn.tasks import torch_safe_load
        import functools
        
        @functools.wraps(torch_safe_load)
        def patched_torch_safe_load(weight):
            ckpt = torch.load(weight, map_location='cpu')
            return ckpt, weight
            
        import ultralytics.nn.tasks
        ultralytics.nn.tasks.torch_safe_load = patched_torch_safe_load
        logger.debug("Successfully patched ultralytics torch_safe_load")
    except Exception as e:
        logger.warning(f"Could not patch torch_safe_load: {e}")

def safe_yolo_load(model_path: str):
    """Safely load a YOLO model with PyTorch compatibility"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    enable_safe_load()
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        logger.info(f"Successfully loaded YOLO model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        raise 