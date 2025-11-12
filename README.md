# SAM Annotator

A graphical annotation tool for image segmentation using Meta's Segment Anything Model (SAM). Built with PySide6, this tool provides an interactive interface for creating high-quality segmentation masks with point-based prompts and brush-based editing.

## Features

- **Interactive Segmentation**: Use SAM's powerful model to generate segmentation masks with simple point clicks
- **Brush Tool**: Draw and erase segments directly on the image for precise mask editing
- **Bounding Box System**: Support for both pre-existing bounding boxes (from XML files) and interactive bounding box creation
  - Load images with existing bounding box annotations from XML files
  - Create new bounding boxes directly in the interface when working with images only
- **Multi-category Support**: Organize segments by customizable labels loaded from a configuration file
- **Dynamic Label System**: Labels are loaded from `label.txt` with automatically generated colors
- **Enhanced Undo System**: Undo brush strokes, point additions, and segment deletions with confirmation dialogs
- **Flexible Export Formats**: Export annotations in VOC (Pascal VOC XML) or COCO (JSON) format, configurable via `config.py`
- **Organized Output Structure**: Three separate output folders for images, segmentation labels, and bounding box labels

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
   - Create a `label.txt` file in the project root
   - Edit `label.txt` to add your label categories (one label per line), `label_example.txt` provides an example
   - Example `label.txt`:
     ```
     cat
     dog
     car
     ```
   - **Note**: `label.txt` is in `.gitignore` and won't be committed to the repository. 

5. Configure export format and bounding box settings (optional):
   - Edit `config.py` to set your preferred export format and bounding box behavior
   - Set `EXPORT_FORMAT = "voc"` for Pascal VOC XML format
   - Set `EXPORT_FORMAT = "coco"` for COCO JSON format
   - Set `BOUNDING_BOX_EXISTS = True` if you have XML files with bounding boxes in `input/labels/`
   - Set `BOUNDING_BOX_EXISTS = False` if you want to create bounding boxes interactively
   - Set `BB_LABEL` to specify the label name for bounding boxes when saved
   - See the [Configuration](#configuration) section for more details

## Usage

1. Place your images in `input/images/` folder

2. **Choose your workflow mode**:
   
   **Option A: With existing bounding boxes (XML files)**
   - Place your XML label files in `input/labels/` folder
   - Each image must have a matching XML file with the same base filename (e.g., `image001.jpg` requires `image001.xml`)
   - The XML files should be in VOC format with bounding box annotations
   - Set `BOUNDING_BOX_EXISTS = True` in `config.py`
   - The system will load the bounding boxes from the XML files and use them to constrain segmentation
   
   **Option B: Create bounding boxes interactively**
   - Only place images in `input/images/` folder (no XML files needed)
   - Set `BOUNDING_BOX_EXISTS = False` in `config.py`
   - Use the **Bounding Box Tool (B)** to draw bounding boxes directly on the image
   - This mode is ideal for starting annotation from scratch

3. Run the application:
```bash
python main.py
```

4. Use the interface to:
   - Navigate through images
   - **Bounding Box Tool (B)**: (Only available when `BOUNDING_BOX_EXISTS = False`) Click and drag to create or resize bounding boxes
   - **Segment Tool (A)**: Click to add positive (include) points (green), right-click to add negative (exclude) points (red)
   - **Brush Tool (S)**: Left-click and drag to draw/add to segment, right-click and drag to erase/remove from segment
   - **Highlight Current Segment (H)**: Hold H to show a white outline around the current segment being created
   - Assign categories to segments
   - Save annotations

5. Annotated files will be saved to `output/` folder with three separate subdirectories:
   - **`output/images/`**: Copies of the input images
   - **`output/segment_labels/`**: Segmentation annotations (polygon masks) in the format specified in `config.py`
     - VOC format: `.xml` files (Pascal VOC format)
     - COCO format: `.json` files (COCO format)
   - **`output/bb_labels/`**: Bounding box annotations (if a bounding box exists for the image)
     - VOC format: `.xml` files (Pascal VOC format)
     - COCO format: `.json` files (COCO format)

## Configuration

### Export Format Configuration

The application supports two export formats, configurable via `config.py`:

**File Location**: `config.py` in the project root directory

**Settings**:
- `EXPORT_FORMAT`: Set to `"voc"` or `"coco"` to choose the output format
- `COCO_CATEGORIES`: Optional list of category names for COCO export. If `None`, categories are automatically loaded from `label.txt`
- `BOUNDING_BOX_EXISTS`: Set to `True` if you have XML files with bounding boxes in `input/labels/`, or `False` to create bounding boxes interactively
- `BB_LABEL`: Label name for bounding boxes when saved to `output/bb_labels/` folder (e.g., `"drone"`, `"object"`)

**Example `config.py`**:
```python
# Export format: "voc" or "coco"
EXPORT_FORMAT = "voc"  # Change to "coco" to export in COCO format

# COCO export settings (only used when EXPORT_FORMAT = "coco")
COCO_CATEGORIES = None  # Set to None to auto-load from label.txt

# Bounding box configuration
BOUNDING_BOX_EXISTS = False  # Set to True if you have XML files, False to create interactively
BB_LABEL = "drone"  # Label name for bounding boxes in output files
```

**Export Formats**:
- **VOC Format** (Pascal VOC XML): 
  - Segmentation labels: `.xml` files in `output/segment_labels/`
  - Bounding box labels: `.xml` files in `output/bb_labels/`
  - Includes bounding boxes and polygon segmentations
  - Compatible with many computer vision tools
- **COCO Format** (JSON):
  - Segmentation labels: `.json` files in `output/segment_labels/`
  - Bounding box labels: `.json` files in `output/bb_labels/`
  - Includes RLE (Run-Length Encoded) segmentations and bounding boxes
  - Standard format for many deep learning frameworks

### Label Configuration

The application uses a simple text file (`label.txt`) to define the available label categories. Each line in the file represents one label category.

- **File Location**: `label.txt` in the project root directory
- **Format**: One label name per line (no empty lines or special characters)
- **Color Assignment**: Colors are automatically generated deterministically based on the label name, ensuring consistency across sessions
- **Example**: See `label_example.txt` for a sample label configuration

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
- **B**: Bounding Box tool (only available when `BOUNDING_BOX_EXISTS = False` - click and drag to create or resize bounding boxes)
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


