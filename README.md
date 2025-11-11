# SAM Annotator

A graphical annotation tool for image segmentation using Meta's Segment Anything Model (SAM). Built with PySide6, this tool provides an interactive interface for creating high-quality segmentation masks with point-based prompts and brush-based editing.

## Features

- **Interactive Segmentation**: Use SAM's powerful model to generate segmentation masks with simple point clicks
- **Brush Tool**: Draw and erase segments directly on the image for precise mask editing
- **Multi-category Support**: Organize segments by customizable labels loaded from a configuration file
- **Dynamic Label System**: Labels are loaded from `label.txt` with automatically generated colors
- **Bounding Box Label Management**: Change original bounding box object labels using `bb_label.txt` - perfect for relabeling input annotations
- **Enhanced Undo System**: Undo brush strokes, point additions, and segment deletions with confirmation dialogs
- **Flexible Export Formats**: Export annotations in VOC (Pascal VOC XML) or COCO (JSON) format, configurable via `config.py`

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

4. Set up your labels:
   - **Segment Labels** (`label.txt`): Create `label.txt` in the project root, `label_example.txt` provides an example.
     - Edit `label.txt` to add your label categories for segmentation masks (one label per line)
     - Example `label.txt`:
       ```
       wheel
       door
       window
       ```
   - **Bounding Box Labels** (`bb_label.txt`) - Optional: Create `bb_label.txt` in the project root, `bb_label_example.txt` provides an example. 
     - Edit `bb_label.txt` to add label categories for bounding box objects (one label per line)
     - These labels appear in the segment panel dropdown for changing original bounding box labels
     - Example `bb_label.txt`:
       ```
       tesla
       bmw
       volvo
       ```
     - **Note**: If `bb_label.txt` is not provided, the original label names from input XML files will be used
   - **Note**: Both `label.txt` and `bb_label.txt` are in `.gitignore` and won't be committed to the repository. Use the example files as templates.

5. Configure export format (optional):
   - Edit `config.py` to set your preferred export format
   - Set `EXPORT_FORMAT = "voc"` for Pascal VOC XML format
   - Set `EXPORT_FORMAT = "coco"` for COCO JSON format
   - See the [Configuration](#configuration) section for more details

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
   - **Segment Tool (A)**: Click to add positive (include) points (green), right-click to add negative (exclude) points (red)
   - **Brush Tool (S)**: Left-click and drag to draw/add to segment, right-click and drag to erase/remove from segment
   - **Highlight Current Segment (H)**: Hold H to show a white outline around the current segment being created
   - **Change Bounding Box Labels**: In the segment panel, use the dropdown menus to change labels for original bounding box objects
     - Original bounding box labels are displayed at the top of the segment panel
     - Select a new label from the dropdown (labels from `bb_label.txt`)
     - The selected label will be saved in the output file instead of the original label
   - Assign categories to segments
   - Save annotations

5. Annotated images and labels will be saved to `output/` folder
   - Images are saved to `output/images/`
   - Annotations are saved to `output/labels/` in the format specified in `config.py`
   - VOC format: `.xml` files (Pascal VOC format)
   - COCO format: `.json` files (COCO format)

## Configuration

### Export Format Configuration

The application supports two export formats, configurable via `config.py`:

**File Location**: `config.py` in the project root directory

**Settings**:
- `EXPORT_FORMAT`: Set to `"voc"` or `"coco"` to choose the output format
- `COCO_CATEGORIES`: Optional list of category names for COCO export. If `None`, categories are automatically loaded from `label.txt`

**Example `config.py`**:
```python
# Export format: "voc" or "coco"
EXPORT_FORMAT = "voc"  # Change to "coco" to export in COCO format

# COCO export settings (only used when EXPORT_FORMAT = "coco")
COCO_CATEGORIES = None  # Set to None to auto-load from label.txt
```

**Export Formats**:
- **VOC Format** (Pascal VOC XML): 
  - Output: `.xml` files in `output/labels/`
  - Includes bounding boxes and polygon segmentations
  - Original bounding box objects are saved with their selected labels (from segment panel dropdown)
  - Compatible with many computer vision tools
- **COCO Format** (JSON):
  - Output: `.json` files in `output/labels/`
  - Includes RLE (Run-Length Encoded) segmentations and bounding boxes
  - Original bounding box objects are converted to rectangular masks and saved with their selected labels
  - Original labels are automatically added to the categories list if not already present
  - Standard format for many deep learning frameworks

### Label Configuration

The application uses two label configuration files:

#### Segment Labels (`label.txt`)

Defines the available label categories for segmentation masks created with SAM.

- **File Location**: `label.txt` in the project root directory
- **Format**: One label name per line (no empty lines or special characters)
- **Color Assignment**: Colors are automatically generated deterministically based on the label name, ensuring consistency across sessions
- **Example**: See `label_example.txt` for a sample label configuration
  ```
  wheel
  door
  window
  ```

#### Bounding Box Labels (`bb_label.txt`) - Optional

Defines the available label categories for changing original bounding box object labels. This is useful when you need to relabel input annotations (e.g., changing "UAV" to "tesla", "bmw", etc.).

- **File Location**: `bb_label.txt` in the project root directory
- **Format**: One label name per line (no empty lines or special characters)
- **Usage**: 
  - When an image is loaded, original bounding box objects from the input XML are displayed in the segment panel
  - Each object shows its original label name and a dropdown menu
  - Users can select a new label from `bb_label.txt` to replace the original label
  - The selected label is saved in the output file (both VOC and COCO formats)
  - If the original label name exists in `bb_label.txt`, it will be automatically selected as the default
- **Example**: See `bb_label_example.txt` for a sample label configuration
  ```
  tesla
  bmw
  volvo
  ```
- **Note**: If `bb_label.txt` is not provided, the original label names from input XML files will be used as-is

## Visualization Tool

The project includes a test script for visualizing segmentation annotations:

### `test/segmentation_test.py`

This script allows you to visualize polygon segmentations from annotation files (VOC XML or COCO JSON). It's useful for:
- Verifying annotation quality
- Debugging annotation issues
- Viewing segmentation overlays on images

**Usage**:
```bash
python test/segmentation_test.py <filename> [options]
```

**Important**: Provide only the filename without the extension. For example:
- If your files are `00004.jpg` and `00004.xml` (or `00004.json`), use: `python test/segmentation_test.py 00004`
- Do not include `.jpg`, `.xml`, `.json`, or any other file extension

## Keyboard Shortcuts

### Tools
- **A**: Segment tool (click to add points for segmentation)
- **S**: Brush tool (draw and erase segments directly)
- **Space**: Pan tool (hold and drag to move image)
- **F**: Fit to bounding box (zoom and center view on bounding box)

### Segmentation
#### Segment Tool (A)
- **Left Click**: Add positive point (include area) - green marker
- **Right Click**: Add negative point (exclude area) - red marker

#### Brush Tool (S)
- **Left Click + Drag**: Draw/add to segment
- **Right Click + Drag**: Erase/remove from segment

#### General
- **E**: Finalize current segment
- **H**: Highlight current segment (hold to show white outline around segment being created)
  - Only works when actively creating a segment (before finalizing)
  - Helps visualize the segment boundary for quality control
- **Z**: Undo last action (brush stroke, point addition, or segment deletion)
  - When undoing a finalized segment, a confirmation dialog will appear

### Navigation & Actions
- **Scroll**: Zoom in/out (scroll wheel)
- **Ctrl+S**: Save & next image
- **N**: Skip image (move to next image without saving)
- **Q**: Quit application

### Label Selection
- **1-N**: Select label category (where N is the number of labels in `label.txt`)
  - Key `1` selects the first label, `2` selects the second label, etc.
  - The keybind bar at the bottom displays all available labels with their corresponding numbers

## License

This project uses the Segment Anything Model by Meta. Please refer to their [license](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE) for SAM usage terms.


