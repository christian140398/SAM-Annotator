# SAM Annotator

A graphical annotation tool for image segmentation using Meta's Segment Anything Model (SAM). Built with PySide6, this tool provides an interactive interface for creating high-quality segmentation masks with point-based prompts.

## Features

- **Interactive Segmentation**: Use SAM's powerful model to generate segmentation masks with simple point clicks
- **Multi-category Support**: Organize segments by labels
- **COCO Export**: Export annotations in COCO format for machine learning pipelines

## Installation

1. Clone this repository:
```bash
git clone https://github.com/christian140398/SAM-Annotator.git
cd SAM-Annotator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the SAM model checkpoint:
   - Download `sam_vit_b_01ec64.pth` from the [official SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
   - Place it in the `models/` folder

## Usage

1. Place your images in `input/images/` folder
2. Place your labels in `input/labels/` folder
   - **Important**: The system only processes images that have a corresponding XML file with bounding boxes
   - Each image must have a matching XML file with the same base filename (e.g., `image001.jpg` requires `image001.xml`)
   - The XML files should be in VOC format with bounding box annotations
3. Run the application:
```bash
python main.py
```

4. Use the interface to:
   - Navigate through images
   - Click to add positive (include) points (green)
   - Right-click to add negative (exclude) points (red)
   - Assign categories to segments
   - Save annotations

5. Annotated images and labels will be saved to `output/` folder

## Keyboard Shortcuts

- **S**: Save & next image
- **Scroll**: Zoom in/out
- **N**: Finalize current segment
- **U**: Undo last point
- **Q**: Quit application
- **1-4**: Select label category (first 4 categories)

## License

This project uses the Segment Anything Model by Meta. Please refer to their [license](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE) for SAM usage terms.


