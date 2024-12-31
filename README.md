# U-Net Semantic Segmentation

A Python implementation of U-Net for semantic segmentation with GPU support and a user-friendly GUI.

## Features

- GPU-accelerated training with mixed precision
- Interactive GUI for image segmentation
- COCO dataset format support
- Real-time training progress visualization
- Support for images of any size
- TensorBoard integration for monitoring

## Requirements

- Python 3.7+
- NVIDIA GPU (optional, but recommended for training)
- CUDA Toolkit and cuDNN (for GPU support)

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd u-net
```

2. Setup the environment:
```bash
python setup.py --mode setup
```

3. Run the GUI application:
```bash
python setup.py --mode app
```

4. Train the model:
```bash
python setup.py --mode train --images_dir path/to/images --annotations_file path/to/annotations.json --epochs 50 --batch_size 4
```

## Directory Structure

```
u-net/
├── app.py                  # GUI application
├── setup.py               # Setup and runner script
├── requirements.txt       # Project dependencies
├── src/
│   ├── model/
│   │   ├── unet.py       # U-Net model implementation
│   │   └── train.py      # Training logic
│   └── inference/
│       └── predictor.py  # Inference code
├── weights/              # Saved model weights
├── logs/                 # Training logs
└── data/                 # Dataset directory
    ├── images/          # Training images
    └── annotations/     # COCO annotations
```

## Training

1. Prepare your dataset in COCO format
2. Place your images in the `data/images` directory
3. Place your annotations file in the `data/annotations` directory
4. Run training:
```bash
python setup.py --mode train \
    --images_dir data/images \
    --annotations_file data/annotations/instances.json \
    --epochs 50 \
    --batch_size 4
```

## Using the GUI

1. Start the application:
```bash
python setup.py --mode app
```

2. Use the "Load Image" button to select an image
3. Click "Segment Image" to perform segmentation
4. View the original image, segmentation mask, and overlay

## GPU Support

The implementation automatically detects and uses available GPUs. For optimal performance:

1. Install NVIDIA GPU drivers
2. Install CUDA Toolkit
3. Install cuDNN
4. Install tensorflow-gpu (handled by setup.py)

## Monitoring Training

Monitor training progress using TensorBoard:
```bash
tensorboard --logdir=./logs
```

## License

[Your License]

## Contributing

[Contribution guidelines]
