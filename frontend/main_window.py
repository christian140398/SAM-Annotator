"""
Main window for SAM Annotator using PySide6
"""
import os
import glob
import uuid
import shutil
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox
from PySide6.QtGui import QKeySequence, QShortcut
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from frontend.theme import get_main_window_style, ITEM_BG
from frontend.components.topbar import TopBar
from frontend.components.toolbar import Toolbar
from frontend.components.imageview import ImageView
from frontend.components.segmentpanel import SegmentsPanel
from frontend.components.keybindbar import KeybindsBar
from segmentation.sam_model import SAMModel


# Configuration
CHECKPOINT = r"models\sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
IMG_DIR = r"input\images"  # Images from input/images folder
LABEL_DIR = r"input\labels"  # Labels (XML files) from input/labels folder
OUTPUT_IMG_DIR = r"output\images"  # Output images folder
OUTPUT_LABEL_DIR = r"output\labels"  # Output labels folder
CLIP_TO_XML_BOX = True

# Default categories/labels
DEFAULT_CATEGORIES = ["body", "rotor", "camera", "other"]
DEFAULT_COLORS = {
    "body": "#00FF00",
    "rotor": "#FFC800",
    "camera": "#00C8FF",
    "other": "#C800FF"
}


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM Annotator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set main window background color
        self.setStyleSheet(get_main_window_style())
        
        # Application state
        self.sam_model: Optional[SAMModel] = None
        self.labels: List[Dict] = []
        self.current_label_id: Optional[str] = None
        self.image_paths: List[str] = []
        self.current_image_idx = 0
        self.current_image_path: Optional[str] = None
        self.current_xml_path: Optional[str] = None
        
        # Initialize SAM model
        self.init_sam_model()
        
        # Initialize labels
        self.init_labels()
        
        # Create output directories if they don't exist
        os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
        os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
        
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
        
        # Keybinds bar
        keybinds_frame = QFrame()
        keybinds_frame.setStyleSheet(f"background-color: {ITEM_BG};")
        keybinds_layout = QVBoxLayout(keybinds_frame)
        keybinds_layout.setContentsMargins(0, 0, 0, 0)
        self.keybinds_bar = KeybindsBar()
        # Set labels on keybind bar (labels are already initialized at this point)
        self.keybinds_bar.set_labels(self.labels)
        keybinds_layout.addWidget(self.keybinds_bar)
        main_layout.addWidget(keybinds_frame, 0)
        
        # Connect signals
        self.connect_signals()
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
        
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
            else:
                QMessageBox.warning(self, "Warning", f"SAM checkpoint not found: {CHECKPOINT}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load SAM model: {str(e)}")
    
    def init_labels(self):
        """Initialize default labels"""
        for i, name in enumerate(DEFAULT_CATEGORIES):
            label_id = str(i + 1)
            color = DEFAULT_COLORS.get(name, "#FFFFFF")
            self.labels.append({
                "id": label_id,
                "name": name,
                "color": color
            })
        
        # Set first label as default
        if self.labels:
            self.current_label_id = self.labels[0]["id"]
    
    def load_image_list(self):
        """Load list of images from input/images directory"""
        if not os.path.isdir(IMG_DIR):
            # Create directory if it doesn't exist
            os.makedirs(IMG_DIR, exist_ok=True)
            print(f"No images directory found at {IMG_DIR}. Created directory.")
            return
        
        self.image_paths = sorted([
            p for p in glob.glob(os.path.join(IMG_DIR, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        
        print(f"Found {len(self.image_paths)} image(s) in {IMG_DIR}")
        if self.image_paths:
            print(f"First image: {os.path.basename(self.image_paths[0])}")
    
    def load_current_image(self):
        """Load the current image and corresponding label from input/labels folder"""
        if not self.image_paths or self.current_image_idx >= len(self.image_paths):
            print("No images to load or index out of range")
            return
        
        img_path = self.image_paths[self.current_image_idx]
        print(f"Loading image: {img_path}")
        
        # Find corresponding XML file in input/labels folder
        xml_path = None
        if CLIP_TO_XML_BOX:
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
        
        try:
            self.image_view.load_image(img_path, xml_path)
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
        
        # SegmentsPanel -> ImageView
        self.segments_panel.segment_selected.connect(self.on_segment_selected)
        self.segments_panel.segment_deleted.connect(self.on_segment_deleted)
        self.segments_panel.visibility_toggled.connect(self.on_segment_visibility_toggled)
        self.segments_panel.label_updated.connect(self.on_segment_label_updated)
        
        # Topbar -> MainWindow
        self.topbar.label_selected.connect(self.on_label_selected)
        self.topbar.label_added.connect(self.on_label_added)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Label selection (1-4)
        for i, label in enumerate(self.labels[:4]):
            shortcut = QShortcut(QKeySequence(str(i + 1)), self)
            shortcut.activated.connect(lambda lid=label["id"]: self.select_label(lid))
        
        # N - Next segment (finalize current)
        shortcut_n = QShortcut(QKeySequence("N"), self)
        shortcut_n.activated.connect(self.finalize_segment)
        
        # U - Undo
        shortcut_u = QShortcut(QKeySequence("U"), self)
        shortcut_u.activated.connect(self.undo_action)
        
        # S - Save and next image
        shortcut_s = QShortcut(QKeySequence("S"), self)
        shortcut_s.activated.connect(self.save_and_next_image)
        
        # Q - Quit
        shortcut_q = QShortcut(QKeySequence("Q"), self)
        shortcut_q.activated.connect(self.close)
    
    def on_tool_changed(self, tool_id: str):
        """Handle tool change from toolbar"""
        self.image_view.active_tool = tool_id
        self.image_view.update_cursor()
    
    def on_label_selected(self, label_id: str):
        """Handle label selection from topbar"""
        self.current_label_id = label_id
        self.image_view.set_current_label(label_id)
    
    def on_label_added(self, label_data: dict):
        """Handle new label added"""
        self.labels.append(label_data)
        self.segments_panel.set_labels(self.labels)
        
        # Update keybinds bar with new labels
        self.keybinds_bar.set_labels(self.labels)
        
        # Update image view colors and labels info
        label_colors = {label["id"]: label["color"] for label in self.labels}
        self.image_view.set_label_colors(label_colors)
        self.image_view.set_labels(self.labels)
    
    def select_label(self, label_id: str):
        """Select a label by ID"""
        # Finalize current segment if exists
        self.finalize_segment()
        self.on_label_selected(label_id)
    
    def finalize_segment(self):
        """Finalize the current segment"""
        self.image_view.finalize_current_segment()
    
    def undo_action(self):
        """Undo last action (point or segment)"""
        if not self.image_view.undo_last_point():
            self.image_view.undo_last_segment()
        self.update_segments_panel()
    
    def on_segment_finalized(self, _mask, _label_id: str):
        """Handle segment finalized signal"""
        self.update_segments_panel()
    
    def on_mask_updated(self, _mask):
        """Handle mask update signal"""
        # Could update UI if needed
    
    def on_segment_selected(self, segment_id: str):
        """Handle segment selection from panel"""
        # TODO: Highlight selected segment in image view
    
    def on_segment_deleted(self, segment_id: str):
        """Handle segment deletion from panel"""
        # Find segment index from ID mapping
        segment_idx = None
        if hasattr(self, '_segment_id_map'):
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
        self.image_view.update_display()
        self.image_view.update()
    
    def on_segment_visibility_toggled(self, segment_id: str):
        """Handle segment visibility toggle"""
        # TODO: Implement visibility toggle
    
    def on_segment_label_updated(self, segment_id: str, label_id: str):
        """Handle segment label update from panel"""
        # Find segment index from ID mapping
        segment_idx = None
        if hasattr(self, '_segment_id_map'):
            for idx, seg_id in self._segment_id_map.items():
                if seg_id == segment_id:
                    segment_idx = idx
                    break
        
        if segment_idx is not None and segment_idx < len(self.image_view.finalized_labels):
            self.image_view.finalized_labels[segment_idx] = label_id
        
        self.image_view.update_display()
        self.image_view.update()
    
    def update_segments_panel(self):
        """Update segments panel with current segments"""
        segments = self.image_view.get_segments()
        
        # Convert to panel format
        # Store mapping from index to segment ID for later reference
        if not hasattr(self, '_segment_id_map'):
            self._segment_id_map = {}
        
        segment_dicts = []
        for i, (mask, label_id) in enumerate(segments):
            # Reuse ID if exists, otherwise create new
            if i not in self._segment_id_map:
                self._segment_id_map[i] = f"seg_{i}_{uuid.uuid4().hex[:8]}"
            segment_id = self._segment_id_map[i]
            
            segment_dicts.append({
                "id": segment_id,
                "labelId": label_id,
                "area": int(mask.sum()),
                "visible": True
            })
        
        # Clean up old IDs
        self._segment_id_map = {i: self._segment_id_map[i] for i in range(len(segments)) if i in self._segment_id_map}
        
        self.segments_panel.set_segments(segment_dicts)
    
    def save_voc_xml(self, xml_path: str, image_path: str, segments: List[tuple]):
        """
        Save segments as VOC XML annotation file
        
        Args:
            xml_path: Path to save XML file
            image_path: Path to image file
            segments: List of (mask, label_id) tuples
        """
        # Load image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        
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
        segmented_elem.text = "1" if segments else "0"
        
        # Add objects for each segment
        for mask, label_id in segments:
            # Get label name
            label_name = "UAV"  # Default
            for label in self.labels:
                if label["id"] == label_id:
                    label_name = label["name"]
                    break
            
            # Calculate bounding box from mask
            from segmentation.sam_utils import mask_to_rle
            rle = mask_to_rle(mask)
            bbox = mask_utils.toBbox({
                "size": [h, w],
                "counts": rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]
            }).tolist()
            
            x, y, bbox_w, bbox_h = bbox
            xmin = int(x)
            ymin = int(y)
            xmax = int(x + bbox_w)
            ymax = int(y + bbox_h)
            
            # Create object element
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
            xmin_elem.text = str(max(0, xmin))
            ymin_elem = ET.SubElement(bndbox_elem, "ymin")
            ymin_elem.text = str(max(0, ymin))
            xmax_elem = ET.SubElement(bndbox_elem, "xmax")
            xmax_elem.text = str(min(w - 1, xmax))
            ymax_elem = ET.SubElement(bndbox_elem, "ymax")
            ymax_elem.text = str(min(h - 1, ymax))
            
            # Add segmentation polygon
            # Convert mask to polygon coordinates
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create segmentation element (VOC format can include polygon segmentation)
            segmentation_elem = ET.SubElement(obj_elem, "segmentation")
            
            # Add polygon for each contour (usually one, but handle multiple)
            for contour in contours:
                # Simplify contour if too many points (reduce to reasonable number)
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Create polygon element
                polygon_elem = ET.SubElement(segmentation_elem, "polygon")
                
                # Add points as x1,y1 x2,y2 ... format
                points = []
                for point in approx:
                    x, y = point[0]
                    points.append(f"{x},{y}")
                
                polygon_elem.text = " ".join(points)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)
        
        # Write XML file with proper formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        
        return True
    
    def save_and_next_image(self):
        """Save image and label to output folders, then move to next image"""
        if self.current_image_path is None:
            QMessageBox.warning(self, "Warning", "No image loaded to save")
            return
        
        # Finalize current segment
        self.finalize_segment()
        
        try:
            # Create output directories
            os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
            os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
            
            # Get base filename
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            
            # Copy image to output/images
            output_img_path = os.path.join(OUTPUT_IMG_DIR, os.path.basename(self.current_image_path))
            shutil.copy2(self.current_image_path, output_img_path)
            
            # Get all segments
            segments = self.image_view.get_segments()
            
            # Save XML to output/labels
            output_xml_path = os.path.join(OUTPUT_LABEL_DIR, base_name + ".xml")
            self.save_voc_xml(output_xml_path, self.current_image_path, segments)
            
            # Move to next image
            if self.current_image_idx < len(self.image_paths) - 1:
                self.current_image_idx += 1
                self.load_current_image()
            else:
                QMessageBox.information(self, "Info", "Reached last image. Files saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save files: {str(e)}")
