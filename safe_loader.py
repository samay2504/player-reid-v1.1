import torch
import os
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential, ModuleList, Conv2d, BatchNorm2d, ReLU, SiLU, Upsample, Linear

# Set environment variable to disable weights_only restriction
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# Dynamic safe globals - add classes as needed
SAFE_CLASSES = [
    DetectionModel, Sequential, ModuleList, Conv2d,
    BatchNorm2d, ReLU, SiLU, Upsample, Linear
]

def enable_safe_load():
    """Add safe globals and handle missing classes dynamically"""
    torch.serialization.add_safe_globals(SAFE_CLASSES)
    
    # Add any missing classes that might be needed
    try:
        # Try to add common ultralytics classes if they exist
        import ultralytics.nn.modules
        for attr_name in dir(ultralytics.nn.modules):
            attr = getattr(ultralytics.nn.modules, attr_name)
            if isinstance(attr, type):
                torch.serialization.add_safe_globals([attr])
    except:
        pass

    # Patch torch_safe_load in ultralytics
    try:
        from ultralytics.nn.tasks import torch_safe_load
        import functools
        
        @functools.wraps(torch_safe_load)
        def patched_torch_safe_load(weight):
            ckpt = torch.load(weight, map_location='cpu')
            return ckpt, weight
            
        import ultralytics.nn.tasks
        ultralytics.nn.tasks.torch_safe_load = patched_torch_safe_load
    except Exception as e:
        print(f"Warning: Could not patch torch_safe_load: {e}")

def safe_yolo_load(model_path: str):
    """Safely load a YOLO model with proper PyTorch compatibility"""
    enable_safe_load()
    from ultralytics import YOLO
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise 