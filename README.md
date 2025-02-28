# Edge Detection Application

This Python application performs edge detection on images using vertical and horizontal filters. It demonstrates basic computer vision concepts using PyTorch and displays the results using Matplotlib. This tool is a simplified version of what happens in the first layer of many CNNs (convolutional neural networks), where the network learns these kernel values automatically instead of using predetermined values.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd python_projects
```

2. Install the required dependencies:
```bash
pip3 install torch torchvision pillow matplotlib requests
```

## Usage

### Running the Application

Navigate to the project directory and run:
```bash
python3 edge_checker.py
```

This will:
1. Load a sample chess image from the internet
2. Process it through edge detection filters
3. Display a 2x2 grid showing:
   - Original grayscale image
   - Vertical edge detection
   - Horizontal edge detection
   - Combined edge detection results

### Using Your Own Images

To use your own image, modify the `url` variable in `edge_checker.py`:
```python
url = 'your-image-url-here'
```

## How It Works

The application uses:
- Vertical and horizontal Sobel-like filters for edge detection
- PyTorch's convolution operations for applying filters
- Matplotlib for visualizing results

## File Structure

```
python_projects/
├── edge_checker.py   # Main application file
└── README.md        # This documentation
```

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are installed correctly
2. Check your internet connection (required for loading the sample image)
3. Verify Python and pip are correctly installed:
```bash
python3 --version
pip3 --version
```
