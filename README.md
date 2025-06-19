# Player Re-identification System

A robust system for tracking and re-identifying soccer players in video feeds, maintaining consistent player identities even when they leave and re-enter the frame.

## Important Note About Model File

⚠️ The YOLO model file (`best.pt`, ~185MB) is not included in this repository due to size limitations. You can download it from:
[Google Drive Link - best.pt](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

After downloading, place the file in the `models/` directory:
```bash
mkdir -p models
mv path/to/downloaded/best.pt models/
```

## Features

- Real-time player detection and tracking
- Consistent player ID maintenance across frame exits/re-entries
- Feature-based player re-identification
- Motion prediction and track history
- Debug visualization options
- Comprehensive logging and statistics

## Directory Structure

```
player-reid/
├── models/                    # Model files
│   └── best.pt               # Primary YOLO model
│
├── tracking/                  # Core tracking logic
│   ├── __init__.py
│   └── tracker.py            # Main tracking implementation
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── draw.py              # Visualization utilities
│   └── video_io.py          # Video input/output handling
├── tests/                    # Test files
│   ├── test_tracker.py      # Tracker unit tests
│   └── yolo_test.py         # YOLO model testing
├── main.py                  # Main application entry
├── safe_loader.py           # Safe model loading utilities
└── requirements.txt         # Project dependencies
```

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- OpenCV with GPU support (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd player-reid
```

2. Create and activate a conda environment:
```bash
conda create -n player-reid python=3.11
conda activate player-reid
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the system with default settings:
```bash
python main.py
```

### Command Line Options

- `--input`: Input video path (default: Assignment Materials/15sec_input_720p.mp4)
- `--output`: Output video path (default: output_tracked.mp4)
- `--model`: YOLO model path (default: models/best.pt)
- `--conf`: Confidence threshold (default: 0.3)
- `--show-debug`: Enable debug visualization

Example with custom settings:
```bash
python main.py --input path/to/video.mp4 --output result.mp4 --conf 0.4 --show-debug
```

### Testing

Run the YOLO model test:
```bash
python tests/yolo_test.py
```

Run tracker unit tests:
```bash
python -m pytest tests/test_tracker.py
```

## Model Information

The system uses a custom-trained YOLO model optimized for soccer player detection:
- Input resolution: 640x640
- Class IDs: 2 (soccer player)
- Confidence threshold: 0.3 (default)

## Output

The system generates:
1. Annotated video with player tracking
2. Detailed logging in player_reid.log
3. Performance statistics
4. Debug visualization (if enabled)

## Logging

Logs are written to `player_reid.log` with the following information:
- Initialization status
- Frame processing details
- Detection statistics
- Error messages and warnings

## Performance Considerations

- GPU acceleration is recommended for real-time processing
- Adjust confidence threshold based on video quality
- Memory usage scales with number of tracked players
- Processing speed depends on video resolution and frame rate

## Known Limitations

- Best performance on well-lit scenes
- May struggle with dense player clusters
- Requires clear player visibility for reliable re-identification
- Processing speed depends on hardware capabilities

## Troubleshooting

1. Model loading issues:
   - Verify CUDA installation
   - Check PyTorch version compatibility
   - Ensure model file exists in correct location

2. Video processing issues:
   - Check video file format compatibility
   - Verify sufficient disk space
   - Monitor GPU memory usage

3. Performance issues:
   - Lower video resolution if needed
   - Adjust confidence threshold
   - Enable GPU acceleration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
