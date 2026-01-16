# Image Matting Tool

Professional background removal using AI-powered image matting.

## Features

- **One-Click Background Removal**: Simply load an image and click to remove the background
- **High Quality Results**: Uses MODNet for trimap-free portrait matting (MatteFormer coming in v0.2)
- **Multiple Output Options**: Transparent PNG, solid color backgrounds, or custom backgrounds
- **Quality Presets**: Standard (fast), High (balanced), Ultra (best quality)
- **Batch Processing**: Process entire folders of images at once
- **Offline Processing**: Everything runs locally - your images never leave your computer
- **Cross-Platform**: Works on Windows (Mac/Linux support planned)

## Installation

### From Release (Recommended)

1. Download the latest release from the [Releases](../../releases) page
2. Run the installer and follow the prompts
3. Launch from the desktop shortcut

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/image-matting-tool.git
cd image-matting-tool

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py
```

## Usage

### Single Image Processing

1. Click **Select Image** or drag an image onto the window
2. Choose your quality preset and background option
3. Click **Remove Background**
4. Click **Save Result** to export

### Batch Processing

1. Click **Batch Mode**
2. Select input and output folders
3. Configure processing options
4. Click **Start Processing**

## Requirements

### Minimum
- Windows 10/11 (64-bit)
- 4GB RAM
- 1GB free disk space

### Recommended
- 8GB RAM
- NVIDIA GPU with CUDA support for faster processing
- 1.5GB free disk space

## Technical Details

- **Models**: MODNet (trimap-free portrait matting)
- **Inference**: ONNX Runtime (GPU acceleration when available)
- **GUI**: PySimpleGUI
- **Packaging**: PyInstaller

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MODNet](https://github.com/ZHKKKe/MODNet) - Trimap-Free Portrait Matting
- [MatteFormer](https://github.com/webtoon/matteformer) - Transformer-Based Image Matting (planned)
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference

## Roadmap

- [x] Basic single-image processing
- [x] Quality presets
- [x] Background options
- [ ] Batch processing
- [ ] MatteFormer integration with auto-trimap
- [ ] Interactive refinement
- [ ] Mac/Linux support
