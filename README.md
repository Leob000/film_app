# FiLM Visual Reasoning with Interactive Web Interface

Enhanced implementation of Feature-wise Linear Modulation for visual question answering with modern tooling and custom datasets.

## Key Features & Contributions

- Interactive Streamlit Application: Complete web interface for real-time visual question answering with attention visualization
- Custom "Faklevr" Dataset: Lightweight 2D synthetic dataset enabling CPU-only training in 20-25 minutes vs. hours on GPU
- Attention Visualization: Interactive attention maps showing model focus areas for interpretability
- Cross-Platform Support: Windows/Mac/Linux compatibility with automatic device detection (CUDA/MPS/CPU)
- Model Analysis Tools: Statistical visualization of FiLM layer parameters (gamma/beta distributions)
- Optimized Training Pipeline: Streamlined scripts for both full CLEVR and custom dataset training

## Quick Start

### Launch Interactive Demo
```bash
pip install -r requirements.txt
streamlit run Hello.py
```

### Train Custom Model (20-25 min on a common CPU)
```bash
sh faklevr_scripts/faklevr_bundle.sh        # Generate dataset
sh scripts/train/film_faklevr_raw.sh         # Train model
```

### Use Pre-trained CLEVR Model
Download weights for full CLEVR dataset model:
```bash
wget "https://www.dropbox.com/scl/fi/1exvuj8mp0122c0faogte/best.pt?rlkey=huyzf4nhnr6p8jwsnyiy14nd0&st=odj3a2ns" -O data/best.pt
```

## Requirements
- Python 3.12
- Dependencies: `pip install -r requirements.txt`

# Advanced Features

## Interactive Streamlit Interface
- Real-time Question Answering: Ask questions about images and get instant responses
- Attention Visualization: See where the model focuses when answering questions
- Dual Dataset Support: Switch between full CLEVR and custom Faklevr datasets
- Parameter Analysis: Live visualization of FiLM layer gamma/beta distributions

## Custom Faklevr Dataset
Our lightweight alternative to CLEVR featuring:
- 2D Geometric Shapes: Rectangles, ellipses, triangles in red/green/blue
- Simplified Questions: Focus on counting and color/shape identification
- Fast Training: 20-25 minutes on CPU vs. hours for full CLEVR
- Raw Pixel Processing: No pre-trained CNN required

## Technical Improvements
- Device Agnostic: Automatic CUDA/MPS/CPU detection
- Cross-Platform: Windows, macOS, Linux support
- Error Handling: Robust input processing and model loading
- Modular Architecture: Clean separation of training, inference, and visualization

# Detailed Usage
## CLEVR Dataset
If you wish to run the models in the terminal and modify parameters, follow these instructions.

For each script, check the `.sh` and/or the `.py` associated file to modify parameters.
To download the data, run:
```bash
mkdir data
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O data/CLEVR_v1.0.zip
unzip data/CLEVR_v1.0.zip -d data
```

To preprocess the data from pngs to a h5 file for each train/val/test set, run the following code. The data will be the raw pixels, there are options to extract features with the option `--model resnet101` (1024x14x14 output), or to set a maximum number of X processed images `--max_images X` (check `extract_features.py`).
```bash
sh scripts/extract_features.sh
```

To preprocess the questions, execute this script:
```bash
sh scripts/preprocess_questions.sh
```

To train the model:
```bash
sh scripts/train/film.sh
```

To run the model (on `CLEVR_val_000017.png` by default):
```bash
sh scripts/run_model.sh
```

# Original Research & References
- Based on [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/pdf/1709.07871)
- Code forked and inspired by [Film](https://github.com/ethanjperez/film) and [Clever-iep](https://github.com/facebookresearch/clevr-iep)
- [Distill: Feature wise transformations](https://distill.pub/2018/feature-wise-transformations/)
