# Enhanced Player Re-identification System

A production-grade system for tracking and re-identifying soccer players in video feeds, maintaining consistent player identities even when they leave and re-enter the frame. This enhanced version includes advanced feature extraction, motion prediction, and comprehensive analysis capabilities.

## Important Note About Model File

⚠️ The YOLO model file (`best.pt`, ~185MB) is not included in this repository due to size limitations. You can download it from:
[Google Drive Link - best.pt](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

After downloading, place the file in the `models/` directory:
```bash
mkdir -p models
mv path/to/downloaded/best.pt models/
```

## Enhanced Features

- **Advanced Feature Extraction**: 162-dimensional feature vectors combining RGB/HSV color spaces, gradient features, texture analysis, and spatial layout
- **Motion Prediction**: Kalman filter-like motion models with weighted velocity averaging
- **Multi-stage Re-identification**: Two-stage matching with dedicated re-identification thresholds
- **Robust Tracking**: Outlier rejection, weighted feature averaging, and comprehensive state management
- **Player Consistency Analysis**: Detailed analysis of player appearance patterns and tracking stability
- **Comprehensive Logging**: Configurable logging levels with detailed performance metrics
- **Production-ready Configuration**: Extensive command-line options for fine-tuning

## Directory Structure

```
player-reid/
├── models/                    # Model files
│   └── best.pt               # Primary YOLO model
│
├── tracking/                  # Core tracking logic
│   ├── __init__.py
│   └── tracker.py            # Enhanced tracking implementation
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── draw.py              # Visualization utilities
│   └── video_io.py          # Video input/output handling
├── tests/                    # Test files
│   ├── test_tracker.py      # Enhanced tracker unit tests
│   └── yolo_test.py         # YOLO model testing
├── main.py                  # Enhanced main application
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

Run the enhanced system with default settings:
```bash
python main.py
```

### Advanced Command Line Options

- `--input`: Input video path (default: Assignment Materials/15sec_input_720p.mp4)
- `--output`: Output video path (default: enhanced_output_tracked.mp4)
- `--model`: YOLO model path (default: models/best.pt)
- `--conf`: Confidence threshold (default: 0.35)
- `--max-disappeared`: Max frames before player lost (default: 60)
- `--max-distance`: Max distance for matching (default: 200.0)
- `--reid-threshold`: Re-identification threshold (default: 0.65)
- `--analysis-output`: Player consistency analysis file
- `--log-level`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `--show-debug`: Enable debug visualization

### Example Usage Scenarios

**Basic tracking with custom confidence:**
```bash
python main.py --conf 0.4
```

**High-precision tracking with detailed analysis:**
```bash
python main.py --conf 0.5 --max-disappeared 80 --reid-threshold 0.7 --log-level DEBUG
```

**Custom output paths:**
```bash
python main.py --output my_result.mp4 --analysis-output my_analysis.json
```

### Testing

Run the enhanced tracker unit tests:
```bash
python -m pytest tests/test_tracker.py -v
```

Run YOLO model test:
```bash
python tests/yolo_test.py
```

## Enhanced Model Information

The system uses a custom-trained YOLO model optimized for soccer player detection:
- Input resolution: 640x640
- Class IDs: 2 (soccer player)
- Confidence threshold: 0.35 (default, enhanced)
- Feature dimensions: 162 (enhanced from 105)

## Output Files

The enhanced system generates:
1. **Annotated video** with player tracking and stability indicators
2. **Detailed logging** in `enhanced_player_reid.log`
3. **Comprehensive statistics** including re-identifications and ID switches
4. **Player consistency analysis** in JSON format
5. **Debug visualization** (if enabled)

## Enhanced Performance Metrics

The system provides detailed performance analysis:
- **Detection Statistics**: Total frames, detection rate, average detections
- **Tracking Statistics**: Active players, re-identifications, ID switches
- **Player Consistency**: Presence ratios, gaps analysis, longest streaks
- **Processing Performance**: Frame rate, memory usage, optimization metrics

## Production Features

### Robust Error Handling
- Comprehensive exception handling with detailed tracebacks
- Graceful degradation for model loading issues
- Input validation and file existence checks

### Performance Optimization
- Efficient numpy operations and vectorized computations
- Memory management with automatic cleanup
- GPU acceleration support

### Configuration Management
- Environment-based configuration
- Command-line parameter validation
- Default value optimization

## Known Limitations

- Best performance on well-lit scenes with clear player visibility
- May struggle with extreme occlusions or very dense player clusters
- Processing speed depends on hardware capabilities and video resolution
- Requires sufficient memory for feature buffers and player gallery

## Troubleshooting

### Common Issues

1. **Model Loading Issues**:
   - Verify CUDA installation and PyTorch compatibility
   - Check model file exists in correct location
   - Ensure sufficient GPU memory

2. **Performance Issues**:
   - Lower video resolution if needed
   - Adjust confidence and re-identification thresholds
   - Monitor GPU memory usage

3. **Tracking Quality**:
   - Increase `--reid-threshold` for stricter matching
   - Adjust `--max-disappeared` based on video characteristics
   - Use `--log-level DEBUG` for detailed analysis

### Performance Tuning

For optimal performance:
- Use GPU acceleration when available
- Adjust confidence threshold based on video quality
- Fine-tune re-identification threshold for your use case
- Monitor memory usage with large feature buffers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests for new features
4. Ensure code follows production standards
5. Submit a Pull Request with detailed description

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
