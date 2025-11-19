"""
Main window for SAM Annotator using PySide6
"""

import os
import uuid
import shutil
import xml.etree.ElementTree as ET
import hashlib
import re
from typing import Optional, List, Dict, Tuple
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QMessageBox,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
)
from PySide6.QtGui import QKeySequence, QShortcut, QKeyEvent
from PySide6.QtCore import Qt, QEvent, QThread, QObject, Signal
import cv2
import numpy as np
from frontend.theme import get_main_window_style, ITEM_BG
from frontend.components.topbar import TopBar
from frontend.components.toolbar import Toolbar
from frontend.components.imageview import ImageView
from frontend.components.segmentpanel import SegmentsPanel
from frontend.components.keybindbar import KeybindsBar
from segmentation.sam_model import SAMModel
from segmentation.voc_export import VOCExporter
from segmentation.coco_export import COCOExporter
import config


# Configuration
CHECKPOINT = r"models\sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
IMG_DIR = r"input\images"  # Images from input/images folder
LABEL_DIR = r"input\labels"  # Labels (XML files) from input/labels folder
OUTPUT_IMG_DIR = r"output\images"  # Output images folder
OUTPUT_LABEL_DIR = r"output\segment_labels"  # Output segment labels folder
OUTPUT_BB_LABEL_DIR = r"output\bb_labels"  # Output bounding box labels folder
CLIP_TO_XML_BOX = True
# Label file path (relative to project root)
LABEL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "label.txt"
)


def load_labels_from_file(label_file: str) -> List[str]:
    """
    Load labels from a text file, one label per line.

    Args:
        label_file: Path to the label file

    Returns:
        List of label names (stripped of whitespace)
    """
    labels = []
    if os.path.isfile(label_file):
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    label = line.strip()
                    if label:  # Skip empty lines
                        labels.append(label)
        except Exception as e:
            print(f"Warning: Could not load labels from {label_file}: {e}")
    else:
        print(f"Warning: Label file not found: {label_file}")

    return labels


def generate_random_color_hex(seed: Optional[str] = None) -> str:
    """
    Generate a deterministic color in hex format (#RRGGBB) based on seed.
    Colors are consistent across different images and sessions.

    Args:
        seed: Seed string for reproducible colors (e.g., label name)

    Returns:
        Hex color string (e.g., "#FF00AB")
    """
    if seed is None:
        seed = "default"

    # Use deterministic hash (MD5) to generate consistent colors
    hash_obj = hashlib.md5(seed.encode("utf-8"))
    hash_bytes = hash_obj.digest()

    # Generate bright, visible colors (avoid too dark colors)
    # Map hash bytes to color range 50-255
    r = 50 + (hash_bytes[0] % 206)  # 206 = 255 - 50 + 1
    g = 50 + (hash_bytes[1] % 206)
    b = 50 + (hash_bytes[2] % 206)

    return f"#{r:02X}{g:02X}{b:02X}"


def load_labels_with_colors(label_file: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load labels from file and generate random colors for each.

    Args:
        label_file: Path to the label file

    Returns:
        Tuple of (list of label names, dict mapping label name to hex color)
    """
    labels = load_labels_from_file(label_file)
    colors = {}

    for label in labels:
        # Use label name as seed for consistent color generation
        colors[label] = generate_random_color_hex(label)

    return labels, colors


def natural_sort_key(path: str) -> tuple:
    """
    Generate a sort key for natural/numeric sorting of file paths.
    This ensures that files like f1.jpg, f2.jpg, f10.jpg are sorted correctly.
    Handles any filename pattern by extracting and comparing numbers numerically.

    Args:
        path: File path to generate sort key for

    Returns:
        Tuple for sorting that compares numbers numerically
    """
    filename = os.path.basename(path)
    # Split filename into parts, converting numbers to integers for proper numeric comparison
    parts = []
    for part in re.split(r"(\d+)", filename):
        if part.isdigit():
            parts.append((0, int(part)))  # Numeric part - compare as integer
        else:
            parts.append((1, part.lower()))  # Text part - compare case-insensitively
    return tuple(parts)


class PreloadEmbeddingWorker(QObject):
    """Worker object for preloading next image embedding in background"""

    finished = Signal()
    embedding_complete = Signal()

    def __init__(self, preload_sam_model: SAMModel, image: np.ndarray):
        super().__init__()
        self.preload_sam_model = preload_sam_model  # Use separate preload model
        self.image = image.copy()  # Copy to avoid issues with thread safety

    def process(self):
        """Process the image with SAM model for preloading"""
        try:
            if self.preload_sam_model:
                self.preload_sam_model.set_image(self.image)
                print("Preload embedding completed for next image")
                self.embedding_complete.emit()
        except Exception as e:
            print(f"Error preloading SAM image: {str(e)}")
        finally:
            self.finished.emit()


def ensure_output_directories():
    """
    Create output directories if they don't exist and add .gitkeep files.
    This ensures the directories are tracked in git even when empty.
    """
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BB_LABEL_DIR, exist_ok=True)

    # Create .gitkeep files in all directories
    gitkeep_img = os.path.join(OUTPUT_IMG_DIR, ".gitkeep")
    gitkeep_label = os.path.join(OUTPUT_LABEL_DIR, ".gitkeep")
    gitkeep_bb_label = os.path.join(OUTPUT_BB_LABEL_DIR, ".gitkeep")

    # Only create if they don't exist to avoid unnecessary file writes
    if not os.path.exists(gitkeep_img):
        with open(gitkeep_img, "w"):
            pass  # Create empty file
    if not os.path.exists(gitkeep_label):
        with open(gitkeep_label, "w"):
            pass  # Create empty file
    if not os.path.exists(gitkeep_bb_label):
        with open(gitkeep_bb_label, "w"):
            pass  # Create empty file


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM Annotator")
        # Set minimum size to ensure window can be resized from all sides
        self.setMinimumSize(800, 600)
        # Window will open in windowed mode, geometry set as initial size
        self.setGeometry(100, 100, 1400, 900)
        # Ensure window is resizable (QMainWindow is resizable by default)

        # Set main window background color
        self.setStyleSheet(get_main_window_style())

        # Application state
        self.sam_model: Optional[SAMModel] = None
        self.preload_sam_model: Optional[SAMModel] = (
            None  # Separate SAM model for preloading
        )
        self.labels: List[Dict] = []
        self.current_label_id: Optional[str] = None
        self.image_paths: List[str] = []
        self.current_image_idx = 0
        self.current_image_path: Optional[str] = None
        self.current_xml_path: Optional[str] = None
        self.label_shortcuts: List[QShortcut] = []  # Store label shortcuts

        # Preloading state
        self.preloaded_image: Optional[np.ndarray] = None  # Preloaded next image data
        self.preloaded_image_idx: Optional[int] = None  # Index of preloaded image
        self.preload_thread: Optional[QThread] = None  # Thread for preloading embedding
        self.preload_worker: Optional[QObject] = None  # Worker for preloading
        self.preload_ready = False  # Flag to track if preload embedding is complete

        # Initialize SAM model
        self.init_sam_model()

        # Initialize labels
        self.init_labels()

        # Create output directories if they don't exist (with .gitkeep files)
        ensure_output_directories()

        # Load images
        self.load_image_list()

        # Create central widget
        central_widget = QWidget()
        central_widget.setStyleSheet(get_main_window_style())
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Topbar
        topbar_frame = QFrame()
        topbar_frame.setStyleSheet(f"background-color: {ITEM_BG};")
        topbar_layout = QVBoxLayout(topbar_frame)
        topbar_layout.setContentsMargins(0, 0, 0, 0)
        self.topbar = TopBar()
        topbar_layout.addWidget(self.topbar)
        main_layout.addWidget(topbar_frame, 0)

        # Set labels in topbar (labels are already initialized at this point)
        self.topbar.set_labels(self.labels)

        # Content area
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        # Toolbar
        toolbar_frame = QFrame()
        toolbar_frame.setStyleSheet(f"background-color: {ITEM_BG};")
        toolbar_layout = QVBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar = Toolbar()
        toolbar_layout.addWidget(self.toolbar)
        content_layout.addWidget(toolbar_frame, 0)

        # Image view
        content_frame = QFrame()
        content_frame.setStyleSheet(f"background-color: {ITEM_BG};")
        content_layout_inner = QVBoxLayout(content_frame)
        content_layout_inner.setContentsMargins(0, 0, 0, 0)

        self.image_view = ImageView()
        if self.sam_model:
            self.image_view.set_sam_model(self.sam_model)

        # Set label colors
        label_colors = {label["id"]: label["color"] for label in self.labels}
        self.image_view.set_label_colors(label_colors)

        # Set labels info for display
        self.image_view.set_labels(self.labels)

        # Set current label (must be done after image_view is created)
        if self.current_label_id:
            self.image_view.set_current_label(self.current_label_id)

        content_layout_inner.addWidget(self.image_view)
        content_layout.addWidget(content_frame, 1)

        # Segments panel
        self.segments_panel = SegmentsPanel()
        self.segments_panel.set_labels(self.labels)
        content_layout.addWidget(self.segments_panel, 0)

        main_layout.addLayout(content_layout, 1)

        # Initialize brush size indicator
        self.update_brush_size_indicator()

        # Keybinds bar
        keybinds_frame = QFrame()
        keybinds_frame.setStyleSheet(f"background-color: {ITEM_BG};")
        keybinds_layout = QVBoxLayout(keybinds_frame)
        keybinds_layout.setContentsMargins(0, 0, 0, 0)

        # Build keybinds list based on configuration
        keybinds = [
            {"key": "A", "label": "Segment tool"},
            {"key": "S", "label": "Brush tool"},
        ]
        # Add bounding box tool keybind if BOUNDING_BOX_EXISTS is False
        if not config.BOUNDING_BOX_EXISTS:
            keybinds.append({"key": "B", "label": "Bounding box tool"})
        keybinds.extend(
            [
                {"key": "Space", "label": "Pan tool"},
                {"key": "F", "label": "Fit to bounding box"},
                {"key": "E", "label": "Finalize segment"},
                {"key": "H", "label": "Highlight current segment"},
                {"key": "Z", "label": "Undo"},
                {"key": "Ctrl+S", "label": "Save & next image"},
                {"key": "N", "label": "Skip image"},
                {"key": "Scroll", "label": "Zoom"},
                {"key": "Q", "label": "Quit"},
            ]
        )

        self.keybinds_bar = KeybindsBar(keybinds=keybinds)
        # Set labels on keybind bar (labels are already initialized at this point)
        self.keybinds_bar.set_labels(self.labels)
        keybinds_layout.addWidget(self.keybinds_bar)
        main_layout.addWidget(keybinds_frame, 0)

        # Connect signals
        self.connect_signals()

        # Setup keyboard shortcuts
        self.setup_shortcuts()

        # Install event filter on application to capture space key events globally
        QApplication.instance().installEventFilter(self)

        # Show window in windowed mode
        self.show()

        # Load first image if available (do this after show() so widget is ready)
        if self.image_paths:
            self.load_current_image()
        else:
            print("No images found to load")

    def init_sam_model(self):
        """Initialize SAM model"""
        try:
            if os.path.isfile(CHECKPOINT):
                self.sam_model = SAMModel(CHECKPOINT, SAM_MODEL_TYPE)
                # Create a separate SAM model instance for preloading
                # This allows us to pre-embed the next image without affecting the current image
                self.preload_sam_model = SAMModel(CHECKPOINT, SAM_MODEL_TYPE)
            else:
                QMessageBox.warning(
                    self, "Warning", f"SAM checkpoint not found: {CHECKPOINT}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load SAM model: {str(e)}")

    def init_labels(self):
        """Initialize labels from label.txt file"""
        label_names, label_colors = load_labels_with_colors(LABEL_FILE)

        # If no labels found in file, don't use defaults - show error message instead
        if not label_names:
            print("No labels found in label.txt")
            return

        for i, name in enumerate(label_names):
            label_id = str(i + 1)
            color = label_colors.get(name, generate_random_color_hex(name))
            self.labels.append({"id": label_id, "name": name, "color": color})

        # Set first label as default
        if self.labels:
            self.current_label_id = self.labels[0]["id"]

    def load_image_list(self):
        """Load list of images from input/images directory in folder order"""
        if not os.path.isdir(IMG_DIR):
            # Create directory if it doesn't exist
            os.makedirs(IMG_DIR, exist_ok=True)
            print(f"No images directory found at {IMG_DIR}. Created directory.")
            return

        # Get all image files with their full paths
        image_files = []
        for filename in os.listdir(IMG_DIR):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(IMG_DIR, filename)
                image_files.append(full_path)

        # Sort using natural/numeric sorting to handle f1, f2, f10 correctly
        # This ensures numbers in filenames are compared numerically, not alphabetically
        # This will sort f1, f2, f3... f10, f11... f35 in the correct order
        self.image_paths = sorted(image_files, key=natural_sort_key)

        print(f"Found {len(self.image_paths)} image(s) in {IMG_DIR}")
        if self.image_paths:
            print(f"First image: {os.path.basename(self.image_paths[0])}")
            # Print all images for debugging
            print("Image order:")
            for i, path in enumerate(self.image_paths):
                print(f"  {i + 1}. {os.path.basename(path)}")

        # Find the first unprocessed image (skip images that already exist in output folder)
        self.find_first_unprocessed_image()

    def find_first_unprocessed_image(self):
        """
        Find the first image that hasn't been processed yet (doesn't exist in output folder).
        Sets current_image_idx to the first unprocessed image, or 0 if all are unprocessed.
        """
        if not self.image_paths:
            self.current_image_idx = 0
            return

        # Check if output directory exists
        if not os.path.isdir(OUTPUT_IMG_DIR):
            # No output directory means no images have been processed
            self.current_image_idx = 0
            print("No output directory found - starting from first image")
            return

        # Get list of already processed images (files in output/images folder)
        processed_images = set()
        if os.path.isdir(OUTPUT_IMG_DIR):
            for filename in os.listdir(OUTPUT_IMG_DIR):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    processed_images.add(filename.lower())

        # Find first image that hasn't been processed
        for idx, img_path in enumerate(self.image_paths):
            img_filename = os.path.basename(img_path)
            if img_filename.lower() not in processed_images:
                self.current_image_idx = idx
                print(
                    f"Found first unprocessed image: {img_filename} (index {idx + 1}/{len(self.image_paths)})"
                )
                if processed_images:
                    print(f"Skipped {len(processed_images)} already processed image(s)")
                return

        # All images have been processed
        self.current_image_idx = len(self.image_paths) - 1
        print(f"All {len(self.image_paths)} images have already been processed")

    def load_current_image(self):
        """Load the current image and corresponding label from input/labels folder"""
        if not self.image_paths or self.current_image_idx >= len(self.image_paths):
            print("No images to load or index out of range")
            return

        img_path = self.image_paths[self.current_image_idx]
        print(f"Loading image: {img_path}")

        # Find corresponding XML file in input/labels folder
        xml_path = None
        if config.BOUNDING_BOX_EXISTS and CLIP_TO_XML_BOX:
            # Get base name without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            # Look for XML in input/labels folder with same base name
            xml_path = os.path.join(LABEL_DIR, base_name + ".xml")

            # Create label directory if it doesn't exist
            if not os.path.isdir(LABEL_DIR):
                os.makedirs(LABEL_DIR, exist_ok=True)

            if not os.path.isfile(xml_path):
                print(f"No label file found at {xml_path}")
                xml_path = None
            else:
                print(f"Found label file: {xml_path}")
        elif not config.BOUNDING_BOX_EXISTS:
            print("BOUNDING_BOX_EXISTS is False - skipping XML file lookup")

        try:
            # Check if we have a preloaded embedding ready for this image
            # The embedding might have been started right before navigation
            use_preloaded = (
                self.preloaded_image_idx == self.current_image_idx
                and self.preloaded_image is not None
                and self.preload_ready
            )

            if use_preloaded:
                print(
                    f"Using preloaded embedding for image {self.current_image_idx + 1} - skipping embedding step"
                )
            else:
                # If embedding is in progress, wait for it (but with timeout)
                if (
                    self.preloaded_image_idx == self.current_image_idx
                    and self.preloaded_image is not None
                    and self.preload_thread is not None
                    and self.preload_thread.isRunning()
                ):
                    print(
                        f"Waiting for preload embedding to complete for image {self.current_image_idx + 1}..."
                    )
                    # Wait up to 5 seconds for embedding to complete (embedding can take time)
                    if self.preload_thread.wait(5000):
                        # Embedding completed, use it
                        if self.preload_ready:
                            use_preloaded = True
                            print(
                                f"Preload embedding completed, using it for image {self.current_image_idx + 1}"
                            )
                    else:
                        print("Preload embedding timed out, will embed normally")
                        # Cancel the preload thread since it timed out
                        self.cancel_preload()

            # If using preloaded embedding, swap the SAM models to use the preloaded one
            # This avoids recomputing the embedding - we just swap which model ImageView uses
            if (
                use_preloaded
                and self.preload_sam_model is not None
                and self.preloaded_image is not None
            ):
                print("Swapping to preloaded SAM model (embedding already computed)...")

                # Verify the preload model has the correct image by checking dimensions
                preload_img = self.preload_sam_model.get_current_image()
                if preload_img is not None:
                    preload_h, preload_w = preload_img.shape[:2]
                    preloaded_h, preloaded_w = self.preloaded_image.shape[:2]
                    if preload_h == preloaded_h and preload_w == preloaded_w:
                        # Dimensions match - safe to swap
                        # Swap the models: preload_sam_model (has embedding) becomes main, main becomes preload
                        temp_model = self.sam_model
                        self.sam_model = self.preload_sam_model
                        self.preload_sam_model = temp_model
                        # Update ImageView to use the swapped model
                        self.image_view.set_sam_model(self.sam_model)
                        print(
                            "Swapped to preloaded SAM model - no embedding computation needed!"
                        )
                    else:
                        print(
                            f"Warning: Preload model has wrong image dimensions ({preload_h}x{preload_w} vs {preloaded_h}x{preloaded_w}), will recompute"
                        )
                        # Don't swap - just embed normally
                        use_preloaded = False
                else:
                    print("Warning: Preload model has no image, will recompute")
                    use_preloaded = False

            # Clear preload state BEFORE loading image (if it was for this image)
            # This prevents on_sam_embedding_complete from trying to preload the current image
            preload_was_for_current = self.preloaded_image_idx == self.current_image_idx
            if preload_was_for_current:
                # Clear the preload state before emitting sam_embedding_complete
                self.preloaded_image = None
                self.preloaded_image_idx = None
                self.preload_ready = False
                # Cancel thread without waiting (let it finish in background)
                if self.preload_thread is not None:
                    try:
                        if self.preload_thread.isRunning():
                            self.preload_thread.quit()
                    except RuntimeError:
                        pass

            # Preload next image data BEFORE loading current image
            # This ensures that when sam_embedding_complete signal is emitted,
            # the next image is already ready to be embedded
            self.preload_next_image()

            # Load image, skipping embedding if preloaded and ready
            self.image_view.load_image(img_path, xml_path, skip_embedding=use_preloaded)

            # After loading, verify the SAM model has the correct image embedded
            # This is especially important after swapping models
            if use_preloaded and self.sam_model is not None:
                model_img = self.sam_model.get_current_image()
                if model_img is not None:
                    model_h, model_w = model_img.shape[:2]
                    # Get the actual image dimensions from the loaded image
                    actual_img = cv2.imread(img_path)
                    if actual_img is not None:
                        actual_h, actual_w = actual_img.shape[:2]
                        if model_h != actual_h or model_w != actual_w:
                            print(
                                f"Error: Model has wrong image after swap ({model_h}x{model_w} vs {actual_h}x{actual_w}), forcing re-embedding"
                            )
                            # Force re-embedding by calling load_image again without skip_embedding
                            self.image_view.load_image(
                                img_path, xml_path, skip_embedding=False
                            )
                            use_preloaded = (
                                False  # Mark as not using preloaded to avoid issues
                            )

            print(f"Successfully loaded image: {os.path.basename(img_path)}")

            # Store current paths for saving
            self.current_image_path = img_path
            self.current_xml_path = xml_path

            # Reset segment ID mapping for new image
            self._segment_id_map = {}

            # Update topbar
            self.topbar.set_image_name(os.path.basename(img_path))

            # Update window title
            self.setWindowTitle(
                f"SAM Annotator - {os.path.basename(img_path)} "
                f"({self.current_image_idx + 1}/{len(self.image_paths)})"
            )

            # Update segments panel
            self.update_segments_panel()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def connect_signals(self):
        """Connect component signals"""
        # Toolbar -> ImageView
        self.toolbar.tool_changed.connect(self.on_tool_changed)

        # ImageView -> SegmentsPanel
        self.image_view.segment_finalized.connect(self.on_segment_finalized)
        self.image_view.mask_updated.connect(self.on_mask_updated)
        self.image_view.sam_embedding_complete.connect(self.on_sam_embedding_complete)

        # SegmentsPanel -> ImageView
        self.segments_panel.segment_selected.connect(self.on_segment_selected)
        self.segments_panel.segment_deleted.connect(self.on_segment_deleted)
        self.segments_panel.visibility_toggled.connect(
            self.on_segment_visibility_toggled
        )
        self.segments_panel.label_updated.connect(self.on_segment_label_updated)
        self.segments_panel.segment_hovered.connect(self.on_segment_hovered)

        # Topbar -> MainWindow
        self.topbar.label_selected.connect(self.on_label_selected)
        self.topbar.label_added.connect(self.on_label_added)
        self.topbar.label_color_changed.connect(self.on_label_color_changed)

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Clear existing label shortcuts
        for shortcut in self.label_shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self.label_shortcuts.clear()

        # Label selection (1-N for all labels)
        for i, label in enumerate(self.labels):
            shortcut = QShortcut(QKeySequence(str(i + 1)), self)
            shortcut.activated.connect(lambda lid=label["id"]: self.select_label(lid))
            self.label_shortcuts.append(shortcut)

        # E - Finalize segment
        shortcut_e = QShortcut(QKeySequence("E"), self)
        shortcut_e.activated.connect(self.finalize_segment)

        # Z - Undo
        shortcut_z = QShortcut(QKeySequence("Z"), self)
        shortcut_z.activated.connect(self.undo_action)

        # A - Segment tool
        shortcut_a_tool = QShortcut(QKeySequence("A"), self)
        shortcut_a_tool.activated.connect(lambda: self.select_tool("segment"))

        # S - Brush tool
        shortcut_s_tool = QShortcut(QKeySequence("S"), self)
        shortcut_s_tool.activated.connect(lambda: self.select_tool("brush"))

        # F - Fit to bounding box
        shortcut_f = QShortcut(QKeySequence("F"), self)
        shortcut_f.activated.connect(lambda: self.select_tool("fit_bbox"))

        # B - Bounding box tool (only when BOUNDING_BOX_EXISTS is False)
        if not config.BOUNDING_BOX_EXISTS:
            shortcut_b = QShortcut(QKeySequence("B"), self)
            shortcut_b.activated.connect(lambda: self.select_tool("bbox"))

        # Ctrl+S - Save and next image
        shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut_save.activated.connect(self.save_and_next_image)

        # N - Skip image (move to next without saving)
        shortcut_skip = QShortcut(QKeySequence("N"), self)
        shortcut_skip.activated.connect(self.skip_image)

        # Q - Quit
        shortcut_q = QShortcut(QKeySequence("Q"), self)
        shortcut_q.activated.connect(self.close)

        # Arrow Up - Increase brush size
        shortcut_up = QShortcut(QKeySequence(Qt.Key_Up), self)
        shortcut_up.activated.connect(self.increase_brush_size)

        # Arrow Down - Decrease brush size
        shortcut_down = QShortcut(QKeySequence(Qt.Key_Down), self)
        shortcut_down.activated.connect(self.decrease_brush_size)

    def select_tool(self, tool_id: str):
        """Select a tool by ID (can be called from keyboard shortcuts)"""
        # fit_bbox is an action, not a tool - it doesn't change active tool state
        if tool_id == "fit_bbox":
            # Just emit the signal without changing active tool
            self.toolbar.tool_changed.emit(tool_id)
        else:
            # Set active tool in toolbar (this will emit signal and update image view)
            self.toolbar.set_active_tool(tool_id)
            self.toolbar.tool_changed.emit(tool_id)

    def on_tool_changed(self, tool_id: str):
        """Handle tool change from toolbar"""
        if tool_id == "fit_bbox":
            # This is an action, not a tool - fit view to bounding box
            self.image_view.fit_to_bounding_box()
        else:
            # Regular tool selection
            self.image_view.active_tool = tool_id
            self.image_view.update_cursor()
            # Update brush size indicator
            self.update_brush_size_indicator()

    def update_brush_size_indicator(self):
        """Update the brush size indicator in the toolbar"""
        if self.image_view:
            brush_size = self.image_view.get_brush_size()
            self.toolbar.update_brush_size(brush_size)

    def increase_brush_size(self):
        """Increase brush size by 1"""
        if self.image_view:
            current_size = self.image_view.get_brush_size()
            self.image_view.set_brush_size(current_size + 1)
            self.update_brush_size_indicator()

    def decrease_brush_size(self):
        """Decrease brush size by 1"""
        if self.image_view:
            current_size = self.image_view.get_brush_size()
            self.image_view.set_brush_size(current_size - 1)
            self.update_brush_size_indicator()

    def on_label_selected(self, label_id: str):
        """Handle label selection from topbar"""
        # Check if there's an active segment before changing label
        if self.image_view.has_active_segment():
            # Show prompt dialog
            reply = self._show_label_change_dialog()

            if reply == "finalize":
                # Finalize segment
                self.finalize_segment()
                self.current_label_id = label_id
                self.image_view.set_current_label(label_id)
            elif reply == "delete":
                # Delete segment (clear current segment)
                self.image_view.clear_current_segment()
                self.current_label_id = label_id
                self.image_view.set_current_label(label_id)
            # If "cancel", do nothing (don't change label)
        else:
            # No active segment, just change label
            self.current_label_id = label_id
            self.image_view.set_current_label(label_id)

    def _show_label_change_dialog(self) -> str:
        """Show dialog asking what to do with active segment when changing label

        Returns:
            "finalize", "delete", or "cancel"
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Change Label")
        dialog.setModal(True)
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(16)

        # Message label
        message_label = QLabel(
            "You are about to change labels without finalizing the current segment."
        )
        message_label.setWordWrap(True)
        message_label.setStyleSheet("font-size: 14px; color: #ffffff; padding: 8px;")
        layout.addWidget(message_label)

        # Buttons
        button_box = QDialogButtonBox()

        finalize_button = button_box.addButton(
            "Finalize segment", QDialogButtonBox.AcceptRole
        )
        delete_button = button_box.addButton(
            "Delete segment", QDialogButtonBox.AcceptRole
        )
        cancel_button = button_box.addButton("Cancel", QDialogButtonBox.RejectRole)

        # Style buttons
        button_box.setStyleSheet(f"""
            QPushButton {{
                background-color: {ITEM_BG};
                border: 1px solid #2b303b;
                color: #ffffff;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #2b303b;
            }}
            QPushButton:default {{
                background-color: #3b82f6;
            }}
        """)

        layout.addWidget(button_box)

        # Style dialog
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {ITEM_BG};
                color: #ffffff;
            }}
            QLabel {{
                color: #ffffff;
                background-color: transparent;
            }}
        """)

        # Connect buttons
        result = {"action": "cancel"}

        def on_finalize():
            result["action"] = "finalize"
            dialog.accept()

        def on_delete():
            result["action"] = "delete"
            dialog.accept()

        finalize_button.clicked.connect(on_finalize)
        delete_button.clicked.connect(on_delete)
        cancel_button.clicked.connect(dialog.reject)

        # Show dialog
        dialog.exec()
        return result["action"]

    def on_label_added(self, label_data: dict):
        """Handle new label added"""
        self.labels.append(label_data)
        self.segments_panel.set_labels(self.labels)

        # Update keybinds bar with new labels
        self.keybinds_bar.set_labels(self.labels)

        # Update topbar with new labels
        self.topbar.set_labels(self.labels)

        # Update image view colors and labels info
        label_colors = {label["id"]: label["color"] for label in self.labels}
        self.image_view.set_label_colors(label_colors)
        self.image_view.set_labels(self.labels)

        # Recreate shortcuts to include new label
        self.setup_shortcuts()

    def on_label_color_changed(self, label_id: str, new_color: str):
        """Handle label color change from settings dialog"""
        # Update the label color in self.labels
        for label in self.labels:
            if label["id"] == label_id:
                label["color"] = new_color
                break

        # Update all components with new colors
        label_colors = {label["id"]: label["color"] for label in self.labels}
        self.image_view.set_label_colors(
            label_colors
        )  # This will invalidate cache and update display
        self.image_view.set_labels(self.labels)

        # Update segments panel
        self.segments_panel.set_labels(self.labels)

        # Update topbar (to refresh its internal state)
        self.topbar.set_labels(self.labels)

    def select_label(self, label_id: str):
        """Select a label by ID"""
        # Check if there's an active segment before changing label
        if self.image_view.has_active_segment():
            # Show prompt dialog
            reply = self._show_label_change_dialog()

            if reply == "finalize":
                # Finalize segment
                self.finalize_segment()
                self.current_label_id = label_id
                self.image_view.set_current_label(label_id)
            elif reply == "delete":
                # Delete segment (clear current segment)
                self.image_view.clear_current_segment()
                self.current_label_id = label_id
                self.image_view.set_current_label(label_id)
            # If "cancel", do nothing (don't change label)
        else:
            # No active segment, just change label
            self.current_label_id = label_id
            self.image_view.set_current_label(label_id)

    def finalize_segment(self):
        """Finalize the current segment"""
        self.image_view.finalize_current_segment()

    def undo_action(self):
        """Undo last action (point or segment)"""
        if not self.image_view.undo_last_point():
            # Check if there are finalized segments to delete
            if len(self.image_view.finalized_masks) > 0:
                # Show confirmation dialog
                reply = QMessageBox.question(
                    self,
                    "Delete Segment",
                    "You are about to delete last finished segment, will you proceed?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )

                if reply == QMessageBox.Yes:
                    self.image_view.undo_last_segment()
                    self.update_segments_panel()
            else:
                # No segments to delete, just update panel
                self.update_segments_panel()
        else:
            # Successfully undid a point/brush stroke, update panel
            self.update_segments_panel()

    def on_segment_finalized(self, _mask, _label_id: str):
        """Handle segment finalized signal"""
        self.update_segments_panel()

    def on_mask_updated(self, _mask):
        """Handle mask update signal"""
        # Could update UI if needed

    def on_sam_embedding_complete(self):
        """Handle SAM embedding complete signal - start preloading next image embedding"""
        # Start embedding the preloaded next image if available
        # We use a separate SAM model (preload_sam_model) so it doesn't interfere with current image
        # This allows background embedding without breaking tools
        # Only start if we have a preloaded image that is NOT the current image
        if (
            self.preloaded_image is not None
            and self.preloaded_image_idx is not None
            and self.preloaded_image_idx != self.current_image_idx
        ):
            # Start immediately - no delay needed since we use separate SAM model
            self.start_preload_embedding()

    def preload_next_image(self):
        """Preload the next image data into memory"""
        # Cancel any existing preload
        self.cancel_preload()

        # Check if there's a next image
        next_idx = self.current_image_idx + 1
        if next_idx >= len(self.image_paths):
            # No next image to preload - clear any stale preload state
            self.preloaded_image = None
            self.preloaded_image_idx = None
            self.preload_ready = False
            return

        # Load next image into memory
        next_img_path = self.image_paths[next_idx]
        try:
            img = cv2.imread(next_img_path)
            if img is not None:
                self.preloaded_image = img
                self.preloaded_image_idx = next_idx
                self.preload_ready = False
                print(f"Preloaded image data: {os.path.basename(next_img_path)}")
            else:
                print(f"Failed to preload image: {next_img_path}")
        except Exception as e:
            print(f"Error preloading image: {str(e)}")

    def start_preload_embedding(self):
        """Start embedding the preloaded next image in background thread"""
        if self.preloaded_image is None or self.preload_sam_model is None:
            return

        # Cancel any existing preload thread (without waiting to avoid blocking)
        if self.preload_thread is not None:
            try:
                if self.preload_thread.isRunning():
                    self.preload_thread.quit()
                    # Don't wait - let it finish in background
            except RuntimeError:
                pass

        # Create new thread and worker for preloading
        # Use separate preload_sam_model so it doesn't interfere with current image
        self.preload_thread = QThread()
        self.preload_worker = PreloadEmbeddingWorker(
            self.preload_sam_model, self.preloaded_image
        )
        self.preload_worker.moveToThread(self.preload_thread)

        # Connect signals - but don't connect to any ImageView signals that might trigger UI updates
        self.preload_thread.started.connect(self.preload_worker.process)
        self.preload_worker.embedding_complete.connect(
            self._on_preload_embedding_complete
        )
        self.preload_worker.finished.connect(self.preload_thread.quit)
        self.preload_worker.finished.connect(self.preload_worker.deleteLater)
        self.preload_thread.finished.connect(self._cleanup_preload_thread)

        # Start preloading in background (silently, no UI updates)
        self.preload_thread.start()
        print(
            f"Started preloading embedding for image {self.preloaded_image_idx + 1} (silent, using separate SAM model)"
        )

    def _on_preload_embedding_complete(self):
        """Called when preload embedding is complete"""
        self.preload_ready = True
        if self.preloaded_image_idx is not None:
            print(f"Preload embedding ready for image {self.preloaded_image_idx + 1}")
        else:
            print("Preload embedding ready")

    def _cleanup_preload_thread(self):
        """Clean up preload thread resources"""
        try:
            if self.preload_thread is not None:
                self.preload_thread.deleteLater()
                self.preload_thread = None
        except RuntimeError:
            pass

    def cancel_preload(self):
        """Cancel any ongoing preload operations"""
        if self.preload_thread is not None:
            try:
                if self.preload_thread.isRunning():
                    self.preload_thread.quit()
                    # Don't wait - let it finish in background to avoid blocking
            except RuntimeError:
                pass
            # Cleanup will happen when thread finishes via _cleanup_preload_thread

    def on_segment_selected(self, segment_id: str):
        """Handle segment selection from panel"""
        # TODO: Highlight selected segment in image view

    def on_segment_hovered(self, segment_id: str, is_hovered: bool):
        """Handle segment hover from panel"""
        # Find segment index from ID mapping
        segment_idx = None
        if hasattr(self, "_segment_id_map"):
            for idx, seg_id in self._segment_id_map.items():
                if seg_id == segment_id:
                    segment_idx = idx
                    break

        # Set hovered segment index (None if not hovering)
        self.image_view.set_hovered_segment_index(segment_idx if is_hovered else None)

    def on_segment_deleted(self, segment_id: str):
        """Handle segment deletion from panel"""
        # Find segment index from ID mapping
        segment_idx = None
        if hasattr(self, "_segment_id_map"):
            for idx, seg_id in self._segment_id_map.items():
                if seg_id == segment_id:
                    segment_idx = idx
                    break

        if segment_idx is not None:
            # Remove segment
            self.image_view.finalized_masks.pop(segment_idx)
            self.image_view.finalized_labels.pop(segment_idx)

            # Update ID mapping (shift indices)
            new_map = {}
            for old_idx, seg_id in self._segment_id_map.items():
                if old_idx < segment_idx:
                    new_map[old_idx] = seg_id
                elif old_idx > segment_idx:
                    new_map[old_idx - 1] = seg_id
            self._segment_id_map = new_map

        self.update_segments_panel()
        # Invalidate overlay cache because segment was deleted
        self.image_view.overlay_cache_valid = False
        self.image_view.update_display()
        self.image_view.update()

    def on_segment_visibility_toggled(self, segment_id: str):
        """Handle segment visibility toggle"""
        # TODO: Implement visibility toggle

    def on_segment_label_updated(self, segment_id: str, label_id: str):
        """Handle segment label update from panel"""
        # Find segment index from ID mapping
        segment_idx = None
        if hasattr(self, "_segment_id_map"):
            for idx, seg_id in self._segment_id_map.items():
                if seg_id == segment_id:
                    segment_idx = idx
                    break

        if segment_idx is not None and segment_idx < len(
            self.image_view.finalized_labels
        ):
            self.image_view.finalized_labels[segment_idx] = label_id

        # Invalidate overlay cache because label color changed
        self.image_view.overlay_cache_valid = False
        self.image_view.update_display()
        self.image_view.update()

    def update_segments_panel(self):
        """Update segments panel with current segments"""
        segments = self.image_view.get_segments()

        # Convert to panel format
        # Store mapping from index to segment ID for later reference
        if not hasattr(self, "_segment_id_map"):
            self._segment_id_map = {}

        segment_dicts = []
        for i, (mask, label_id) in enumerate(segments):
            # Reuse ID if exists, otherwise create new
            if i not in self._segment_id_map:
                self._segment_id_map[i] = f"seg_{i}_{uuid.uuid4().hex[:8]}"
            segment_id = self._segment_id_map[i]

            segment_dicts.append(
                {
                    "id": segment_id,
                    "labelId": label_id,
                    "area": int(mask.sum()),
                    "visible": True,
                }
            )

        # Clean up old IDs
        self._segment_id_map = {
            i: self._segment_id_map[i]
            for i in range(len(segments))
            if i in self._segment_id_map
        }

        self.segments_panel.set_segments(segment_dicts)

    def _load_input_objects(self, xml_path: Optional[str]) -> List[Dict]:
        """
        Load all objects with their bounding boxes from input XML file

        Args:
            xml_path: Path to input XML file, or None

        Returns:
            List of dicts with keys: 'name', 'bbox' (xmin, ymin, xmax, ymax), 'truncated', 'difficult'
        """
        input_objects = []
        if xml_path and os.path.isfile(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                for obj in root.findall("object"):
                    name_elem = obj.find("name")
                    name = name_elem.text if name_elem is not None else "unknown"

                    bndbox_elem = obj.find("bndbox")
                    if bndbox_elem is not None:
                        try:
                            xmin = int(float(bndbox_elem.find("xmin").text))
                            ymin = int(float(bndbox_elem.find("ymin").text))
                            xmax = int(float(bndbox_elem.find("xmax").text))
                            ymax = int(float(bndbox_elem.find("ymax").text))

                            truncated_elem = obj.find("truncated")
                            truncated = (
                                truncated_elem.text
                                if truncated_elem is not None
                                else "0"
                            )

                            difficult_elem = obj.find("difficult")
                            difficult = (
                                difficult_elem.text
                                if difficult_elem is not None
                                else "0"
                            )

                            input_objects.append(
                                {
                                    "name": name,
                                    "bbox": (xmin, ymin, xmax, ymax),
                                    "truncated": truncated,
                                    "difficult": difficult,
                                }
                            )
                        except (ValueError, AttributeError):
                            continue
            except Exception as e:
                print(f"Warning: Could not load input XML objects: {e}")

        return input_objects

    def save_voc_xml(self, xml_path: str, image_path: str, segments: List[tuple]):
        """
        Save segments as VOC XML annotation file

        Args:
            xml_path: Path to save XML file
            image_path: Path to image file
            segments: List of (mask, label_id) tuples
        """
        # Get image dimensions (or use from image_view if available)
        image_shape = None
        if (
            hasattr(self.image_view, "base_image")
            and self.image_view.base_image is not None
        ):
            h, w = self.image_view.base_image.shape[:2]
            image_shape = (h, w)

        # Use VOC exporter
        exporter = VOCExporter()
        exporter.export(
            xml_path=xml_path,
            image_path=image_path,
            segments=segments,
            labels=self.labels,
            input_xml_path=self.current_xml_path,
            image_shape=image_shape,
        )

        return True

    def save_coco_json(self, json_path: str, image_path: str, segments: List[tuple]):
        """
        Save segments as COCO JSON annotation file

        Args:
            json_path: Path to save JSON file
            image_path: Path to image file
            segments: List of (mask, label_id) tuples
        """
        # Get image dimensions
        if (
            hasattr(self.image_view, "base_image")
            and self.image_view.base_image is not None
        ):
            h, w = self.image_view.base_image.shape[:2]
        else:
            img = cv2.imread(image_path)
            if img is None:
                return False
            h, w = img.shape[:2]

        # Get categories from config or labels
        if config.COCO_CATEGORIES is not None:
            categories = config.COCO_CATEGORIES
        else:
            # Auto-load from labels
            categories = [label["name"] for label in self.labels]

        if not categories:
            categories = ["UAV"]  # Default fallback

        # Create COCO exporter
        exporter = COCOExporter(categories)

        # Add image (use base name as image_id for per-image files)
        image_id = 1  # For per-image COCO files, we use 1
        exporter.add_image(
            image_id=image_id,
            file_path=image_path,
            width=w,
            height=h,
            output_dir=os.path.dirname(json_path),
        )

        # Add annotations
        for mask, label_id in segments:
            # Get label name
            label_name = categories[0] if categories else "UAV"  # Default
            for label in self.labels:
                if label["id"] == label_id:
                    label_name = label["name"]
                    break

            # Only add if category is in the categories list
            if label_name in categories:
                exporter.add_annotation(
                    image_id=image_id, mask=mask, category_name=label_name
                )

        # Export JSON file
        exporter.export(json_path)

        return True

    def save_voc_bbox(
        self,
        xml_path: str,
        image_path: str,
        bbox: Tuple[int, int, int, int],
        label_name: str,
    ):
        """
        Save bounding box as VOC XML annotation file

        Args:
            xml_path: Path to save XML file
            image_path: Path to image file
            bbox: Tuple (xmin, ymin, xmax, ymax)
            label_name: Label name for the bounding box
        """
        # Get image dimensions
        if (
            hasattr(self.image_view, "base_image")
            and self.image_view.base_image is not None
        ):
            h, w = self.image_view.base_image.shape[:2]
        else:
            img = cv2.imread(image_path)
            if img is None:
                return False
            h, w = img.shape[:2]

        xmin, ymin, xmax, ymax = bbox

        # Create XML root
        root = ET.Element("annotation")

        # Folder
        folder_elem = ET.SubElement(root, "folder")
        folder_elem.text = "images"

        # Filename
        filename_elem = ET.SubElement(root, "filename")
        filename_elem.text = os.path.basename(image_path)

        # Path
        path_elem = ET.SubElement(root, "path")
        path_elem.text = image_path

        # Source
        source_elem = ET.SubElement(root, "source")
        database_elem = ET.SubElement(source_elem, "database")
        database_elem.text = "SAM Annotator"

        # Size
        size_elem = ET.SubElement(root, "size")
        width_elem = ET.SubElement(size_elem, "width")
        width_elem.text = str(w)
        height_elem = ET.SubElement(size_elem, "height")
        height_elem.text = str(h)
        depth_elem = ET.SubElement(size_elem, "depth")
        depth_elem.text = "3"

        # Segmented
        segmented_elem = ET.SubElement(root, "segmented")
        segmented_elem.text = "0"

        # Object with bounding box
        obj_elem = ET.SubElement(root, "object")

        name_elem = ET.SubElement(obj_elem, "name")
        name_elem.text = label_name

        pose_elem = ET.SubElement(obj_elem, "pose")
        pose_elem.text = "Unspecified"

        truncated_elem = ET.SubElement(obj_elem, "truncated")
        truncated_elem.text = "0"

        difficult_elem = ET.SubElement(obj_elem, "difficult")
        difficult_elem.text = "0"

        # Bounding box
        bndbox_elem = ET.SubElement(obj_elem, "bndbox")
        xmin_elem = ET.SubElement(bndbox_elem, "xmin")
        xmin_elem.text = str(max(0, min(w - 1, xmin)))
        ymin_elem = ET.SubElement(bndbox_elem, "ymin")
        ymin_elem.text = str(max(0, min(h - 1, ymin)))
        xmax_elem = ET.SubElement(bndbox_elem, "xmax")
        xmax_elem.text = str(max(0, min(w - 1, xmax)))
        ymax_elem = ET.SubElement(bndbox_elem, "ymax")
        ymax_elem.text = str(max(0, min(h - 1, ymax)))

        # Create output directory if needed
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)

        # Write XML file with proper formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        print(f"✅ Saved VOC bounding box to {xml_path}")
        return True

    def save_coco_bbox(
        self,
        json_path: str,
        image_path: str,
        bbox: Tuple[int, int, int, int],
        label_name: str,
    ):
        """
        Save bounding box as COCO JSON annotation file

        Args:
            json_path: Path to save JSON file
            image_path: Path to image file
            bbox: Tuple (xmin, ymin, xmax, ymax)
            label_name: Label name for the bounding box
        """
        # Get image dimensions
        if (
            hasattr(self.image_view, "base_image")
            and self.image_view.base_image is not None
        ):
            h, w = self.image_view.base_image.shape[:2]
        else:
            img = cv2.imread(image_path)
            if img is None:
                return False
            h, w = img.shape[:2]

        xmin, ymin, xmax, ymax = bbox

        # Calculate COCO bbox format: [x, y, width, height] (top-left corner + size)
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        coco_bbox = [xmin, ymin, bbox_width, bbox_height]

        # Create COCO exporter with just the bounding box label
        categories = [label_name]
        exporter = COCOExporter(categories)

        # Add image
        image_id = 1  # For per-image COCO files, we use 1
        exporter.add_image(
            image_id=image_id,
            file_path=image_path,
            width=w,
            height=h,
            output_dir=os.path.dirname(json_path),
        )

        # Add bounding box annotation (without segmentation mask)
        # We need to manually add the annotation since add_annotation expects a mask
        category_id = 1  # First category
        area = float(bbox_width * bbox_height)

        exporter.annotations_json.append(
            {
                "id": exporter.ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [],  # No segmentation for bounding box only
                "area": area,
                "bbox": coco_bbox,
                "iscrowd": 0,
            }
        )
        exporter.ann_id += 1

        # Export JSON file
        exporter.export(json_path)

        return True

    def skip_image(self):
        """Skip current image and move to next without creating output files"""
        if self.current_image_path is None:
            QMessageBox.warning(self, "Warning", "No image loaded to skip")
            return

        # Move to next image without saving
        if self.current_image_idx < len(self.image_paths) - 1:
            next_idx = self.current_image_idx + 1

            # Only cancel preload if it's for a different image
            # If it's for the image we're about to load, keep it running and wait for it
            if (
                self.preloaded_image_idx is not None
                and self.preloaded_image_idx != next_idx
            ):
                # Cancel preload for wrong image
                self.cancel_preload()
                # Process events to ensure thread cleanup completes
                if QApplication.instance() is not None:
                    QApplication.instance().processEvents()

            self.current_image_idx += 1
            self.load_current_image()
        else:
            QMessageBox.information(
                self, "Info", "Reached last image. No files were created."
            )

    def save_and_next_image(self):
        """Save image and label to output folders, then move to next image"""
        if self.current_image_path is None:
            QMessageBox.warning(self, "Warning", "No image loaded to save")
            return

        # Check if there's an active segment being drawn
        if self.image_view.has_active_segment():
            # Show dialog with options
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Active Segment")
            msg_box.setText(
                "You have an active segment being drawn. What would you like to do?"
            )
            msg_box.setIcon(QMessageBox.Question)

            # Add custom buttons
            finalize_btn = msg_box.addButton("Finalize Segment", QMessageBox.AcceptRole)
            delete_btn = msg_box.addButton(
                "Delete Segment", QMessageBox.DestructiveRole
            )
            cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)

            # Set default button
            msg_box.setDefaultButton(cancel_btn)

            # Show dialog and get result
            msg_box.exec()
            clicked_button = msg_box.clickedButton()

            if clicked_button == cancel_btn:
                # User cancelled, don't save
                return
            elif clicked_button == finalize_btn:
                # Finalize the segment
                self.finalize_segment()
            elif clicked_button == delete_btn:
                # Delete/clear the current segment
                self.image_view.clear_current_segment()
                self.update_segments_panel()
        else:
            # No active segment, just finalize (which will do nothing if no segment)
            self.finalize_segment()

        # Process events to show UI updates before blocking operations
        if QApplication.instance() is not None:
            QApplication.instance().processEvents()

        try:
            # Create output directories (with .gitkeep files)
            ensure_output_directories()

            # Process events after directory creation
            if QApplication.instance() is not None:
                QApplication.instance().processEvents()

            # Get base filename
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]

            # Copy image to output/images
            output_img_path = os.path.join(
                OUTPUT_IMG_DIR, os.path.basename(self.current_image_path)
            )
            shutil.copy2(self.current_image_path, output_img_path)

            # Process events after image copy
            if QApplication.instance() is not None:
                QApplication.instance().processEvents()

            # Get all segments
            segments = self.image_view.get_segments()

            # Save annotations based on configured format
            if config.EXPORT_FORMAT.lower() == "coco":
                # COCO format: Save as JSON file
                output_json_path = os.path.join(OUTPUT_LABEL_DIR, base_name + ".json")
                self.save_coco_json(output_json_path, self.current_image_path, segments)
            else:
                # VOC format: Save as XML file (default)
                output_xml_path = os.path.join(OUTPUT_LABEL_DIR, base_name + ".xml")
                self.save_voc_xml(output_xml_path, self.current_image_path, segments)

            # Save bounding box if it exists
            if self.image_view.bounding_box is not None:
                bbox = self.image_view.bounding_box
                bb_label_name = config.BB_LABEL

                if config.EXPORT_FORMAT.lower() == "coco":
                    # COCO format: Save as JSON file
                    output_bb_json_path = os.path.join(
                        OUTPUT_BB_LABEL_DIR, base_name + ".json"
                    )
                    self.save_coco_bbox(
                        output_bb_json_path,
                        self.current_image_path,
                        bbox,
                        bb_label_name,
                    )
                else:
                    # VOC format: Save as XML file (default)
                    output_bb_xml_path = os.path.join(
                        OUTPUT_BB_LABEL_DIR, base_name + ".xml"
                    )
                    self.save_voc_bbox(
                        output_bb_xml_path, self.current_image_path, bbox, bb_label_name
                    )

            # Process events before loading next image
            if QApplication.instance() is not None:
                QApplication.instance().processEvents()

            # Move to next image
            if self.current_image_idx < len(self.image_paths) - 1:
                next_idx = self.current_image_idx + 1

                # Only cancel preload if it's for a different image
                # If it's for the image we're about to load, keep it running and wait for it
                if (
                    self.preloaded_image_idx is not None
                    and self.preloaded_image_idx != next_idx
                ):
                    # Cancel preload for wrong image
                    self.cancel_preload()
                    # Process events to ensure thread cleanup completes
                    if QApplication.instance() is not None:
                        QApplication.instance().processEvents()

                self.current_image_idx += 1
                self.load_current_image()
            else:
                QMessageBox.information(
                    self, "Info", "Reached last image. Files saved successfully."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save files: {str(e)}")

    def eventFilter(self, obj, event):
        """Event filter to capture space key and H key events globally"""
        # Process key events for space key and H key
        if isinstance(event, QKeyEvent) and (
            event.key() == Qt.Key_Space or event.key() == Qt.Key_H
        ):
            # Forward space key and H key events to image view regardless of focus
            # Skip auto-repeat events to avoid interference
            if hasattr(self, "image_view") and self.image_view:
                if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                    # Forward the key press event to image view
                    if hasattr(self.image_view, "keyPressEvent"):
                        self.image_view.keyPressEvent(event)
                        # Don't consume the event, let it propagate if needed
                elif (
                    event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat()
                ):
                    # Forward key release event to image view
                    if hasattr(self.image_view, "keyReleaseEvent"):
                        self.image_view.keyReleaseEvent(event)

        # Let all events pass through normally (don't consume them)
        return False
