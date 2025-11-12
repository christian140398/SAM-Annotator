"""
ImageView component for SAM Annotator
Image display/view area component with SAM segmentation support
"""

from typing import Optional, List, Tuple
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QProgressBar,
    QApplication,
)
from PySide6.QtCore import Qt, Signal, QPoint, QThread, QObject, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QWheelEvent, QKeyEvent
from frontend.theme import CANVAS_BG, ITEM_BORDER, TEXT_COLOR, ITEM_BG
from segmentation.sam_model import SAMModel


class SAMImageProcessor(QObject):
    """Worker object for processing images with SAM in background thread"""

    finished = Signal()
    processing_complete = Signal()

    def __init__(self, sam_model: SAMModel, image: np.ndarray):
        super().__init__()
        self.sam_model = sam_model
        self.image = image.copy()  # Copy to avoid issues with thread safety

    def process(self):
        """Process the image with SAM model"""
        try:
            if self.sam_model:
                self.sam_model.set_image(self.image)
                print("SAM image processing completed")
                self.processing_complete.emit()
        except Exception as e:
            print(f"Error processing SAM image: {str(e)}")
        finally:
            self.finished.emit()


class ImageView(QWidget):
    """Image view widget for displaying images with SAM segmentation"""

    # Signals
    mask_updated = Signal(object)  # Emits when current mask is updated
    segment_finalized = Signal(
        object, str
    )  # Emits (mask, label_id) when segment is finalized
    point_added = Signal()  # Emits when a point is added
    sam_embedding_complete = Signal()  # Emits when SAM embedding is complete

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageView")
        self.setMinimumHeight(400)
        self.setCursor(Qt.CrossCursor)
        # Enable keyboard focus to receive space key events
        self.setFocusPolicy(Qt.StrongFocus)

        # Image state
        self.base_image: Optional[np.ndarray] = None  # BGR format
        self.display_image: Optional[QPixmap] = None
        self.base_scale = 1.0  # Base scale to fit widget (never upscales)
        self.zoom_scale = 1.0  # Additional zoom scale (can be > 1.0)
        self.display_scale = 1.0  # Combined scale = base_scale * zoom_scale
        self.image_offset_x = 0  # Centering offset (for base scale)
        self.image_offset_y = 0  # Centering offset (for base scale)
        self.pan_offset_x = 0  # Pan offset (for dragging)
        self.pan_offset_y = 0  # Pan offset (for dragging)

        # Pan/zoom state
        self.is_panning = False
        self.last_pan_pos: Optional[QPoint] = None
        self.space_pressed = (
            False  # Track if space key is held down for temporary pan mode
        )
        self.h_pressed = (
            False  # Track if H key is held down for highlighting current segment
        )

        # SAM model
        self.sam_model: Optional[SAMModel] = None
        self.sam_thread: Optional[QThread] = None
        self.sam_worker: Optional[SAMImageProcessor] = None
        self.sam_ready = False  # Flag to track if SAM processing is complete

        # Segmentation state
        self.current_points: List[
            Tuple[Tuple[int, int], bool]
        ] = []  # List of ((x, y), is_positive)
        self.current_mask: Optional[np.ndarray] = None
        self.finalized_masks: List[np.ndarray] = []
        self.finalized_labels: List[str] = []  # Label IDs
        self.current_label_id: Optional[str] = None
        self.bounding_box: Optional[Tuple[int, int, int, int]] = (
            None  # (xmin, ymin, xmax, ymax)
        )
        self.hovered_segment_index: Optional[int] = None  # Index of hovered segment

        # Undo history for current segment (stores mask snapshots)
        self.mask_history: List[np.ndarray] = []  # History of mask states for undo
        self.points_history: List[
            List[Tuple[Tuple[int, int], bool]]
        ] = []  # History of points for undo

        # Label color palette (label_id -> BGR color tuple)
        self.label_colors: dict = {}
        # Label info (label_id -> dict with 'name', 'color')
        self.label_info: dict = {}

        # Tool state
        self.active_tool = "segment"  # "segment", "brush", "pan", or "bbox"

        # Brush state
        self.is_brushing = False
        self.brush_mode = "draw"  # "draw" or "erase"
        self.brush_size = 10  # Brush radius in pixels (image coordinates)
        self.last_brush_pos: Optional[Tuple[int, int]] = (
            None  # Last brush position in image coordinates
        )

        # Bounding box drawing state
        self.is_drawing_bbox = False
        self.bbox_start_pos: Optional[Tuple[int, int]] = (
            None  # Start position in widget coordinates
        )
        self.bbox_current_pos: Optional[Tuple[int, int]] = (
            None  # Current position in widget coordinates
        )
        self.temp_bbox: Optional[Tuple[int, int, int, int]] = (
            None  # Temporary bbox being drawn (xmin, ymin, xmax, ymax) in image coordinates
        )

        # Bounding box edge resizing state
        self.is_resizing_bbox_edge = False
        self.bbox_resize_edge: Optional[str] = None  # "top", "bottom", "left", "right"
        self.bbox_resize_start_pos: Optional[Tuple[int, int]] = (
            None  # Start position in image coordinates
        )
        self.bbox_resize_original_bbox: Optional[Tuple[int, int, int, int]] = (
            None  # Original bbox before resizing
        )
        self.EDGE_DETECTION_THRESHOLD = (
            10  # Pixels (in image coordinates) to detect edge clicks
        )

        # Create label indicator widget
        self.label_indicator = None

        # Create loading indicator widget
        self.loading_indicator = None

        self.apply_styles()
        self.update_cursor()
        self.setup_label_indicator()
        self.setup_loading_indicator()

    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(f"""
            QWidget#ImageView {{
                background-color: {CANVAS_BG};
            }}
        """)

    def setup_label_indicator(self):
        """Setup the label indicator widget in the top right corner"""
        # Create container widget for absolute positioning
        self.label_indicator = QWidget(self)
        self.label_indicator.setObjectName("LabelIndicator")

        # Layout for the indicator
        indicator_layout = QHBoxLayout(self.label_indicator)
        indicator_layout.setContentsMargins(8, 4, 8, 4)
        indicator_layout.setSpacing(6)

        # Color dot
        self.indicator_color_dot = QLabel()
        self.indicator_color_dot.setFixedSize(12, 12)
        self.indicator_color_dot.setStyleSheet(f"""
            background-color: #ffffff;
            border-radius: 6px;
            border: 1px solid {ITEM_BORDER};
        """)
        indicator_layout.addWidget(self.indicator_color_dot)

        # Label name
        self.indicator_label_text = QLabel("No label")
        self.indicator_label_text.setStyleSheet(f"""
            color: {TEXT_COLOR};
            font-size: 12px;
            background-color: transparent;
        """)
        indicator_layout.addWidget(self.indicator_label_text)

        # Style the indicator container
        self.label_indicator.setStyleSheet(f"""
            QWidget#LabelIndicator {{
                background-color: rgba(29, 34, 42, 0.85);
                border: 1px solid {ITEM_BORDER};
                border-radius: 6px;
            }}
        """)

        # Initially hide if no label
        self.update_label_indicator()

    def update_label_indicator(self):
        """Update the label indicator display"""
        if self.label_indicator is None:
            return

        if self.current_label_id and self.current_label_id in self.label_info:
            label = self.label_info[self.current_label_id]
            label_name = label.get("name", "Unknown")
            label_color = label.get("color", "#ffffff")

            # Update color dot
            self.indicator_color_dot.setStyleSheet(f"""
                background-color: {label_color};
                border-radius: 6px;
                border: 1px solid {ITEM_BORDER};
            """)

            # Update text
            self.indicator_label_text.setText(label_name)
            self.label_indicator.show()
        else:
            # Hide if no label selected
            self.indicator_color_dot.setStyleSheet(f"""
                background-color: #ffffff;
                border-radius: 6px;
                border: 1px solid {ITEM_BORDER};
            """)
            self.indicator_label_text.setText("No label")
            # Don't hide, just show "No label"
            self.label_indicator.show()

    def setup_loading_indicator(self):
        """Setup the loading indicator widget that shows during SAM processing"""
        # Create container widget for absolute positioning
        self.loading_indicator = QWidget(self)
        self.loading_indicator.setObjectName("LoadingIndicator")

        # Layout for the indicator
        indicator_layout = QVBoxLayout(self.loading_indicator)
        indicator_layout.setContentsMargins(20, 16, 20, 16)
        indicator_layout.setSpacing(12)
        indicator_layout.setAlignment(Qt.AlignCenter)

        # Create progress bar (indeterminate/spinning)
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 0)  # Indeterminate mode
        self.loading_progress.setFixedWidth(200)
        self.loading_progress.setFixedHeight(4)
        self.loading_progress.setTextVisible(False)
        indicator_layout.addWidget(self.loading_progress)

        # Label text
        self.loading_label = QLabel("Embedding image...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet(f"""
            color: {TEXT_COLOR};
            font-size: 13px;
            background-color: transparent;
        """)
        indicator_layout.addWidget(self.loading_label)

        # Style the indicator container
        self.loading_indicator.setStyleSheet(f"""
            QWidget#LoadingIndicator {{
                background-color: rgba(29, 34, 42, 0.92);
                border: 1px solid {ITEM_BORDER};
                border-radius: 8px;
            }}
            QProgressBar {{
                border: none;
                background-color: {ITEM_BG};
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {TEXT_COLOR};
                border-radius: 2px;
            }}
        """)

        # Initially hide
        self.loading_indicator.hide()

    def update_loading_indicator(self):
        """Update loading indicator visibility based on SAM ready state"""
        if self.loading_indicator is None:
            return

        # Only show loading indicator if:
        # 1. SAM is not ready
        # 2. Base image is loaded
        # 3. We have an active SAM thread (meaning we're actively embedding the CURRENT image)
        # This prevents showing the indicator during preloading of the next image
        is_actively_embedding = (
            self.sam_thread is not None
            and self.sam_thread.isRunning()
            and self.sam_worker is not None
        )

        if not self.sam_ready and self.base_image is not None and is_actively_embedding:
            # Show loading indicator (only for current image embedding)
            self.loading_indicator.show()
            # Center it in the widget
            indicator_width = 240
            indicator_height = 80
            x = (self.width() - indicator_width) // 2
            y = (self.height() - indicator_height) // 2
            self.loading_indicator.setGeometry(x, y, indicator_width, indicator_height)
        else:
            # Hide loading indicator
            self.loading_indicator.hide()

    def resizeEvent(self, _event):
        """Handle resize event"""
        self.update_display()
        # Update label indicator position
        if self.label_indicator:
            # Position in top right corner with some margin
            margin = 12
            indicator_width = self.label_indicator.sizeHint().width()
            indicator_height = self.label_indicator.sizeHint().height()
            x = self.width() - indicator_width - margin
            y = margin
            self.label_indicator.setGeometry(x, y, indicator_width, indicator_height)

        # Update loading indicator position
        self.update_loading_indicator()

        super().resizeEvent(_event)

    def update_cursor(
        self, widget_x: Optional[int] = None, widget_y: Optional[int] = None
    ):
        """Update cursor based on active tool and SAM ready state"""
        # If SAM is not ready, show "not allowed" cursor for segmentation and brush
        if not self.sam_ready and (
            self.active_tool == "segment" or self.active_tool == "brush"
        ):
            self.setCursor(Qt.ForbiddenCursor)
        elif self.active_tool == "pan":
            self.setCursor(Qt.OpenHandCursor)
        elif self.active_tool == "bbox":
            # Check if mouse is over a bounding box edge for resizing
            if (
                widget_x is not None
                and widget_y is not None
                and self.bounding_box is not None
            ):
                img_coords = self.widget_to_image_coords(widget_x, widget_y)
                if img_coords:
                    edge = self._detect_bbox_edge(img_coords[0], img_coords[1])
                    if edge:
                        # Show resize cursor based on edge
                        if edge in ["top", "bottom"]:
                            self.setCursor(Qt.SizeVerCursor)  # Vertical resize
                        else:  # left or right
                            self.setCursor(Qt.SizeHorCursor)  # Horizontal resize
                    else:
                        self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.CrossCursor)
        elif self.active_tool == "brush":
            # For brush tool, we'll use a circle cursor when drawing/erasing
            # For now use crosshair cursor (could be customized later)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def set_sam_model(self, sam_model: SAMModel):
        """Set the SAM model instance"""
        self.sam_model = sam_model

    def set_label_colors(self, label_colors: dict):
        """
        Set label color mapping

        Args:
            label_colors: Dict mapping label_id to hex color string (e.g., "#00FF00")
        """
        self.label_colors = {}
        for label_id, hex_color in label_colors.items():
            # Convert hex to BGR tuple for OpenCV
            rgb = QColor(hex_color).getRgb()[:3]
            self.label_colors[label_id] = (rgb[2], rgb[1], rgb[0])  # RGB to BGR

    def set_labels(self, labels: List[dict]):
        """
        Set label information for display

        Args:
            labels: List of label dicts with 'id', 'name', and 'color' keys
        """
        self.label_info = {label["id"]: label for label in labels}
        self.update_label_indicator()

    def load_image(
        self,
        image_path: str,
        xml_path: Optional[str] = None,
        skip_embedding: bool = False,
    ):
        """
        Load and display an image

        Args:
            image_path: Path to image file
            xml_path: Optional path to VOC XML file for bounding box
            skip_embedding: If True, skip SAM embedding (embedding already done)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.base_image = img.copy()

        # Load bounding box if provided
        if xml_path:
            from segmentation.sam_utils import load_voc_box

            self.bounding_box = load_voc_box(xml_path)
        else:
            self.bounding_box = None

        # Reset segmentation state
        self.current_points = []
        self.current_mask = None
        self.finalized_masks = []
        self.finalized_labels = []
        self.hovered_segment_index = None
        self.mask_history = []  # Clear undo history
        self.points_history = []  # Clear points history

        # Reset bounding box drawing state
        self.is_drawing_bbox = False
        self.bbox_start_pos = None
        self.bbox_current_pos = None
        self.temp_bbox = None

        # Reset bounding box edge resizing state
        self.is_resizing_bbox_edge = False
        self.bbox_resize_edge = None
        self.bbox_resize_start_pos = None
        self.bbox_resize_original_bbox = None

        # Reset pan/zoom
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.zoom_scale = 1.0

        # Show image immediately (before SAM processing)
        self.update_display()
        self.update()

        # Fit to bounding box if it exists (do this after update_display so widget size is known)
        if self.bounding_box is not None:
            # Use QTimer.singleShot to ensure widget is sized properly before fitting
            QTimer.singleShot(100, self.fit_to_bounding_box)

        # Show loading indicator
        self.update_loading_indicator()

        # Process SAM in background thread (non-blocking) or skip if already embedded
        if skip_embedding:
            # Embedding already done, just mark as ready
            self.sam_ready = True
            self.update_loading_indicator()
            self.update_cursor()
            print("Skipped embedding - using preloaded embedding")
            # Emit signal so MainWindow can start preloading next image
            self.sam_embedding_complete.emit()
        elif self.sam_model:
            self.sam_ready = False  # Reset SAM ready flag
            self._process_sam_image_async(img)
        else:
            # No SAM model, but still show as ready
            self.sam_ready = True
            self.update_loading_indicator()
            self.update_cursor()

    def _process_sam_image_async(self, img):
        """Process image with SAM model in background thread"""
        # Cancel any existing processing
        if self.sam_thread is not None:
            try:
                if self.sam_thread.isRunning():
                    self.sam_thread.quit()
                    self.sam_thread.wait(1000)  # Wait up to 1 second
            except RuntimeError:
                # Thread already deleted, ignore
                pass

        # Create new thread and worker
        self.sam_thread = QThread()
        self.sam_worker = SAMImageProcessor(self.sam_model, img)
        self.sam_worker.moveToThread(self.sam_thread)

        # Connect signals
        self.sam_thread.started.connect(self.sam_worker.process)
        self.sam_worker.processing_complete.connect(self._on_sam_ready)
        self.sam_worker.finished.connect(self.sam_thread.quit)
        self.sam_worker.finished.connect(self.sam_worker.deleteLater)
        self.sam_thread.finished.connect(self._cleanup_thread)

        # Start processing in background
        self.sam_thread.start()

    def _on_sam_ready(self):
        """Called when SAM processing is complete"""
        self.sam_ready = True
        print("SAM is ready for segmentation")
        # Hide loading indicator
        self.update_loading_indicator()
        # Update cursor now that SAM is ready
        self.update_cursor()
        # Emit signal to notify that embedding is complete
        self.sam_embedding_complete.emit()

    def _cleanup_thread(self):
        """Clean up thread resources"""
        try:
            if self.sam_thread is not None:
                self.sam_thread.deleteLater()
                self.sam_thread = None
        except RuntimeError:
            # Thread already deleted, ignore
            pass

    def set_current_label(self, label_id: Optional[str]):
        """Set the current label for new segments"""
        self.current_label_id = label_id
        self.update_label_indicator()

    def set_hovered_segment_index(self, segment_index: Optional[int]):
        """Set the hovered segment index for highlighting"""
        self.hovered_segment_index = segment_index
        self.update_display()
        self.update()

    def widget_to_image_coords(
        self, widget_x: int, widget_y: int
    ) -> Optional[Tuple[int, int]]:
        """
        Convert widget coordinates to image coordinates

        Returns:
            (x, y) in image coordinates, or None if outside image bounds
        """
        if self.base_image is None:
            return None

        img_h, img_w = self.base_image.shape[:2]

        # Convert widget coords to image coords accounting for scaling, centering, and pan
        total_offset_x = self.image_offset_x + self.pan_offset_x
        total_offset_y = self.image_offset_y + self.pan_offset_y
        img_x = int((widget_x - total_offset_x) / self.display_scale)
        img_y = int((widget_y - total_offset_y) / self.display_scale)

        # Clamp to image bounds
        img_x = max(0, min(img_w - 1, img_x))
        img_y = max(0, min(img_h - 1, img_y))

        return img_x, img_y

    def _detect_bbox_edge(self, img_x: int, img_y: int) -> Optional[str]:
        """
        Detect which edge of the bounding box (if any) is near the given image coordinates.

        Args:
            img_x: X coordinate in image space
            img_y: Y coordinate in image space

        Returns:
            Edge name ("top", "bottom", "left", "right") or None if not near any edge
        """
        if self.bounding_box is None:
            return None

        xmin, ymin, xmax, ymax = self.bounding_box
        threshold = (
            self.EDGE_DETECTION_THRESHOLD / self.display_scale
        )  # Adjust for zoom

        # Check top edge
        if abs(img_y - ymin) <= threshold and xmin <= img_x <= xmax:
            return "top"

        # Check bottom edge
        if abs(img_y - ymax) <= threshold and xmin <= img_x <= xmax:
            return "bottom"

        # Check left edge
        if abs(img_x - xmin) <= threshold and ymin <= img_y <= ymax:
            return "left"

        # Check right edge
        if abs(img_x - xmax) <= threshold and ymin <= img_y <= ymax:
            return "right"

        return None

    def _initialize_mask_for_brush(self):
        """Initialize an empty mask for brush drawing if one doesn't exist"""
        if self.base_image is None or self.current_label_id is None:
            return

        # Create an empty mask matching image dimensions
        h, w = self.base_image.shape[:2]
        self.current_mask = np.zeros((h, w), dtype=bool)

    def _save_mask_to_history(self):
        """Save current mask state and points to history for undo"""
        if self.current_mask is not None:
            self.mask_history.append(self.current_mask.copy())
            # Also save points state
            self.points_history.append(self.current_points.copy())
            # Limit history size to prevent memory issues (keep last 50 states)
            if len(self.mask_history) > 50:
                self.mask_history.pop(0)
                self.points_history.pop(0)

    def _apply_brush_stroke(self, img_coords: Tuple[int, int], mode: str):
        """
        Apply brush stroke at a single point

        Args:
            img_coords: (x, y) in image coordinates
            mode: "draw" or "erase"
        """
        if self.current_mask is None or self.base_image is None:
            return

        # History is saved in mousePressEvent before starting brush stroke
        # so we don't save it here to avoid duplicate saves

        img_x, img_y = img_coords
        h, w = self.base_image.shape[:2]

        # Clamp coordinates to image bounds
        img_x = max(0, min(w - 1, img_x))
        img_y = max(0, min(h - 1, img_y))

        # Calculate brush size in image coordinates (adjusted for current zoom)
        # Brush size should appear consistent regardless of zoom level
        brush_radius = (
            max(1, int(self.brush_size / self.display_scale))
            if self.display_scale > 0
            else self.brush_size
        )

        # Create a temporary mask for the brush stroke
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(temp_mask, (img_x, img_y), brush_radius, 255, -1)
        temp_mask = temp_mask.astype(bool)

        # Apply to current mask
        if mode == "draw":
            self.current_mask = self.current_mask | temp_mask
        elif mode == "erase":
            self.current_mask = self.current_mask & ~temp_mask

        # Clip to bounding box if present
        if self.bounding_box is not None:
            from segmentation.sam_utils import apply_clip_to_box

            self.current_mask = apply_clip_to_box(
                self.current_mask, self.bounding_box, h, w
            )

        # Update display
        self.update_display()
        self.update()

    def _apply_brush_line(
        self, start_coords: Tuple[int, int], end_coords: Tuple[int, int], mode: str
    ):
        """
        Apply brush stroke along a line between two points

        Args:
            start_coords: (x, y) start position in image coordinates
            end_coords: (x, y) end position in image coordinates
            mode: "draw" or "erase"
        """
        if self.current_mask is None or self.base_image is None:
            return

        # Note: History is saved in _apply_brush_stroke on first stroke, so we don't save again here

        start_x, start_y = start_coords
        end_x, end_y = end_coords
        h, w = self.base_image.shape[:2]

        # Clamp coordinates to image bounds
        start_x = max(0, min(w - 1, start_x))
        start_y = max(0, min(h - 1, start_y))
        end_x = max(0, min(w - 1, end_x))
        end_y = max(0, min(h - 1, end_y))

        # Calculate brush size in image coordinates
        brush_radius = (
            max(1, int(self.brush_size / self.display_scale))
            if self.display_scale > 0
            else self.brush_size
        )

        # Create a temporary mask for the brush line
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        thickness = (
            brush_radius * 2 if brush_radius > 1 else -1
        )  # -1 means filled circle
        cv2.line(temp_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
        # Also draw circles at start and end for smoother strokes
        cv2.circle(temp_mask, (start_x, start_y), brush_radius, 255, -1)
        cv2.circle(temp_mask, (end_x, end_y), brush_radius, 255, -1)
        temp_mask = temp_mask.astype(bool)

        # Apply to current mask
        if mode == "draw":
            self.current_mask = self.current_mask | temp_mask
        elif mode == "erase":
            self.current_mask = self.current_mask & ~temp_mask

        # Clip to bounding box if present
        if self.bounding_box is not None:
            from segmentation.sam_utils import apply_clip_to_box

            self.current_mask = apply_clip_to_box(
                self.current_mask, self.bounding_box, h, w
            )

        # Update display
        self.update_display()
        self.update()

    def add_point(self, widget_x: int, widget_y: int, is_positive: bool):
        """
        Add a point for segmentation

        Args:
            widget_x: X coordinate in widget space
            widget_y: Y coordinate in widget space
            is_positive: True for positive point (include), False for negative (exclude)
        """
        if self.base_image is None or self.sam_model is None or not self.sam_ready:
            return

        # Convert to image coordinates
        img_coords = self.widget_to_image_coords(widget_x, widget_y)
        if img_coords is None:
            return

        # Save current mask state to history before adding point
        self._save_mask_to_history()

        img_x, img_y = img_coords

        # Add point
        self.current_points.append(((img_x, img_y), is_positive))

        # Update mask prediction
        self.update_mask_from_points()
        self.point_added.emit()

    def update_mask_from_points(self):
        """Update the mask prediction based on current points"""
        if self.base_image is None or self.sam_model is None:
            self.current_mask = None
            return

        # Check if SAM is ready (image processing complete)
        if not self.sam_ready:
            print("SAM is not ready yet, please wait...")
            return

        # Separate positive and negative points
        positive_points = [(x, y) for (x, y), is_pos in self.current_points if is_pos]
        negative_points = [
            (x, y) for (x, y), is_pos in self.current_points if not is_pos
        ]

        # Need at least one positive point
        if len(positive_points) == 0:
            self.current_mask = None
            self.update_display()
            self.update()
            return

        # Create points and labels arrays
        all_points = positive_points + negative_points
        labels = [1] * len(positive_points) + [0] * len(negative_points)

        # Predict mask
        try:
            mask = self.sam_model.predict_mask(all_points, labels, self.bounding_box)
            # Validate mask dimensions match current image
            if mask is not None and self.base_image is not None:
                img_h, img_w = self.base_image.shape[:2]
                mask_h, mask_w = mask.shape[:2]
                if mask_h != img_h or mask_w != img_w:
                    print(
                        f"Warning: Mask dimensions ({mask_h}, {mask_w}) don't match image ({img_h}, {img_w}), skipping"
                    )
                    self.current_mask = None
                    return
            self.current_mask = mask
        except RuntimeError as e:
            if "image must be set" in str(e):
                print("SAM image not set yet, please wait...")
                self.sam_ready = False
            else:
                raise
            self.current_mask = None
            return

        # Emit signal
        self.mask_updated.emit(mask)

        # Update display
        self.update_display()
        self.update()

    def finalize_current_segment(self) -> bool:
        """
        Finalize the current segment being built

        Returns:
            True if a segment was finalized, False otherwise
        """
        # Need either a mask or points to finalize
        # Brush tool creates masks directly without points
        # Segment tool creates masks from points
        if self.current_mask is None and len(self.current_points) == 0:
            return False

        if self.current_label_id is None:
            return False

        # If we have points but no mask yet, try to generate mask from points
        if self.current_mask is None and len(self.current_points) > 0:
            self.update_mask_from_points()
            if self.current_mask is None:
                return False

        # Need a valid mask to finalize
        if self.current_mask is None:
            return False

        # Validate mask has some content (not empty)
        if self.current_mask.sum() == 0:
            return False

        # Validate mask dimensions match current image before finalizing
        if self.base_image is not None:
            img_h, img_w = self.base_image.shape[:2]
            mask_h, mask_w = self.current_mask.shape[:2]
            if mask_h != img_h or mask_w != img_w:
                print(
                    f"Warning: Cannot finalize mask with dimensions ({mask_h}, {mask_w}) != image ({img_h}, {img_w})"
                )
                return False

        # Add to finalized masks
        self.finalized_masks.append(self.current_mask.copy())
        self.finalized_labels.append(self.current_label_id)

        # Clear current state
        self.current_points = []
        self.current_mask = None
        self.mask_history = []  # Clear undo history when finalizing
        self.points_history = []  # Clear points history when finalizing

        # Emit signal
        self.segment_finalized.emit(self.finalized_masks[-1], self.current_label_id)

        # Update display
        self.update_display()
        self.update()

        return True

    def undo_last_point(self) -> bool:
        """
        Undo last action (point, brush stroke, or erase)

        Returns:
            True if an action was undone, False otherwise
        """
        # First, try to restore from mask history (for brush strokes and point additions)
        if self.mask_history and self.points_history:
            # Restore previous mask state
            self.current_mask = self.mask_history.pop()
            # Restore previous points state
            self.current_points = self.points_history.pop()

            # Update display
            self.update_display()
            self.update()
            # Emit signal
            if self.current_mask is not None:
                self.mask_updated.emit(self.current_mask)
            return True

        # If no mask history, try undoing last point
        if self.current_points:
            # Remove last point and regenerate mask
            self.current_points.pop()
            self.update_mask_from_points()
            return True

        return False

    def undo_last_segment(self) -> bool:
        """
        Undo last finalized segment

        Returns:
            True if segment was removed, False otherwise
        """
        if self.finalized_masks:
            self.finalized_masks.pop()
            self.finalized_labels.pop()
            self.update_display()
            self.update()
            return True
        return False

    def draw_overlay(self, img: np.ndarray) -> np.ndarray:
        """
        Draw segmentation overlays on image

        Args:
            img: Base image (BGR format)

        Returns:
            Image with overlays
        """
        overlay = img.copy()

        # Draw finalized masks
        for idx, (mask, label_id) in enumerate(
            zip(self.finalized_masks, self.finalized_labels)
        ):
            # Validate mask dimensions match current image
            if mask is not None and self.base_image is not None:
                img_h, img_w = self.base_image.shape[:2]
                mask_h, mask_w = mask.shape[:2]
                if mask_h != img_h or mask_w != img_w:
                    print(
                        f"Warning: Skipping mask {idx} with dimensions ({mask_h}, {mask_w}) != image ({img_h}, {img_w})"
                    )
                    continue

            color = self.label_colors.get(label_id, (255, 0, 0))

            # Highlight hovered segment with brighter color and border
            if idx == self.hovered_segment_index:
                # Brighter overlay for hovered segment
                overlay[mask] = (
                    0.4 * overlay[mask] + 0.6 * np.array(color, dtype=np.uint8)
                ).astype(np.uint8)

                # Draw border around hovered segment
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                highlight_color = (255, 255, 255)  # White border for hover
                cv2.drawContours(overlay, contours, -1, highlight_color, 1)
            else:
                # Normal overlay for non-hovered segments
                overlay[mask] = (
                    0.6 * overlay[mask] + 0.4 * np.array(color, dtype=np.uint8)
                ).astype(np.uint8)

            # Label text is no longer shown on image view (shown in segment panel instead)

        # Draw current mask being built
        if self.current_mask is not None:
            # Validate mask dimensions match current image
            if self.base_image is not None:
                img_h, img_w = self.base_image.shape[:2]
                mask_h, mask_w = self.current_mask.shape[:2]
                if mask_h == img_h and mask_w == img_w:
                    color = self.label_colors.get(self.current_label_id, (255, 0, 0))
                    overlay[self.current_mask] = (
                        0.5 * overlay[self.current_mask]
                        + 0.5 * np.array(color, dtype=np.uint8)
                    ).astype(np.uint8)

                    # Draw white outline if H key is pressed (highlight current segment)
                    if self.h_pressed:
                        mask_uint8 = (self.current_mask * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(
                            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        highlight_color = (255, 255, 255)  # White border for highlight
                        cv2.drawContours(
                            overlay, contours, -1, highlight_color, 1
                        )  # Same thickness as hover outline
                else:
                    # Clear invalid mask
                    self.current_mask = None

        # Draw points
        for (x, y), is_positive in self.current_points:
            if is_positive:
                # Positive: green
                cv2.circle(overlay, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.circle(overlay, (int(x), int(y)), 4, (0, 200, 0), 1)
            else:
                # Negative: red
                cv2.circle(overlay, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.circle(overlay, (int(x), int(y)), 4, (0, 0, 200), 1)

        # Draw bounding box if present (dotted line)
        if self.bounding_box is not None:
            xmin, ymin, xmax, ymax = self.bounding_box
            color = (0, 0, 255)  # Red in BGR
            thickness = 1
            dash_length = 10
            gap_length = 5

            # Draw top edge
            x = xmin
            while x < xmax:
                end_x = min(x + dash_length, xmax)
                cv2.line(overlay, (x, ymin), (end_x, ymin), color, thickness)
                x += dash_length + gap_length

            # Draw bottom edge
            x = xmin
            while x < xmax:
                end_x = min(x + dash_length, xmax)
                cv2.line(overlay, (x, ymax), (end_x, ymax), color, thickness)
                x += dash_length + gap_length

            # Draw left edge
            y = ymin
            while y < ymax:
                end_y = min(y + dash_length, ymax)
                cv2.line(overlay, (xmin, y), (xmin, end_y), color, thickness)
                y += dash_length + gap_length

            # Draw right edge
            y = ymin
            while y < ymax:
                end_y = min(y + dash_length, ymax)
                cv2.line(overlay, (xmax, y), (xmax, end_y), color, thickness)
                y += dash_length + gap_length

        # Draw temporary bounding box being drawn
        if self.temp_bbox is not None:
            xmin, ymin, xmax, ymax = self.temp_bbox
            color = (0, 255, 0)  # Green in BGR for temporary bbox
            thickness = 2
            # Draw rectangle outline
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, thickness)

        return overlay

    def update_display(self):
        """Update the display image with overlays"""
        if self.base_image is None:
            return

        # Draw overlays
        display_img = self.draw_overlay(self.base_image)

        # Convert BGR to RGB for QImage
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        # Create QImage
        h, w = rgb_img.shape[:2]
        q_image = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)

        # Calculate base scale to fit widget (never upscales from this)
        widget_w = self.width()
        widget_h = self.height()

        if widget_w > 1 and widget_h > 1:
            scale_w = widget_w / w
            scale_h = widget_h / h
            self.base_scale = min(scale_w, scale_h, 1.0)  # Don't upscale initially

            # Combined scale (base * zoom)
            self.display_scale = self.base_scale * self.zoom_scale

            new_w = int(w * self.display_scale)
            new_h = int(h * self.display_scale)

            # Create scaled pixmap
            self.display_image = QPixmap.fromImage(q_image).scaled(
                new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            self.display_image = QPixmap.fromImage(q_image)
            self.base_scale = 1.0
            self.display_scale = self.base_scale * self.zoom_scale

    def paintEvent(self, _event):
        """Paint the image with overlays"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(CANVAS_BG))

        if self.display_image is not None:
            # Calculate base centering offset based on base scale image size
            # This is where the image would be centered if at base_scale
            if self.base_image is not None:
                h, w = self.base_image.shape[:2]
                base_display_w = int(w * self.base_scale)
                base_display_h = int(h * self.base_scale)
            else:
                base_display_w = self.display_image.width()
                base_display_h = self.display_image.height()

            widget_w = self.width()
            widget_h = self.height()

            # Base centering offset (centers image at base_scale)
            self.image_offset_x = (widget_w - base_display_w) // 2
            self.image_offset_y = (widget_h - base_display_h) // 2

            # When zoomed, we need to adjust centering because the image is larger
            # The difference between zoomed size and base size needs to be centered
            if self.zoom_scale > 1.0:
                zoom_diff_w = (self.display_image.width() - base_display_w) // 2
                zoom_diff_h = (self.display_image.height() - base_display_h) // 2
                self.image_offset_x -= zoom_diff_w
                self.image_offset_y -= zoom_diff_h

            # Apply pan offset
            draw_x = self.image_offset_x + self.pan_offset_x
            draw_y = self.image_offset_y + self.pan_offset_y

            # Draw image
            painter.drawPixmap(draw_x, draw_y, self.display_image)

        # Update label indicator position (in case widget was resized)
        if self.label_indicator:
            margin = 12
            indicator_width = self.label_indicator.sizeHint().width()
            indicator_height = self.label_indicator.sizeHint().height()
            x = self.width() - indicator_width - margin
            y = margin
            self.label_indicator.setGeometry(x, y, indicator_width, indicator_height)

        # Update loading indicator position
        self.update_loading_indicator()

    def showEvent(self, event):
        """Handle show event to position label indicator"""
        super().showEvent(event)
        if self.label_indicator:
            # Position in top right corner with some margin
            margin = 12
            indicator_width = self.label_indicator.sizeHint().width()
            indicator_height = self.label_indicator.sizeHint().height()
            x = self.width() - indicator_width - margin
            y = margin
            self.label_indicator.setGeometry(x, y, indicator_width, indicator_height)

        # Update loading indicator
        self.update_loading_indicator()

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        # Ensure widget has keyboard focus when clicked
        if not self.hasFocus():
            self.setFocus()

        # Check for space-pan mode first (temporary pan when space is held)
        if self.space_pressed and event.button() == Qt.LeftButton:
            # Start panning when space is held (temporary pan mode)
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif self.active_tool == "pan" and event.button() == Qt.LeftButton:
            # Start panning (always allow panning)
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif (
            self.active_tool == "brush"
            and event.button() == Qt.LeftButton
            and not self.space_pressed
        ):
            # Left click: start drawing with brush (only if SAM is ready and space is not held)
            if self.sam_ready:
                if self.current_mask is not None or self.current_label_id is not None:
                    # Ensure we have a mask to draw on
                    if self.current_mask is None:
                        self._initialize_mask_for_brush()
                    if self.current_mask is not None:
                        # Save history BEFORE starting brush stroke
                        if not self.is_brushing:
                            self._save_mask_to_history()
                        self.is_brushing = True
                        self.brush_mode = "draw"
                        img_coords = self.widget_to_image_coords(event.x(), event.y())
                        if img_coords:
                            self.last_brush_pos = img_coords
                            self._apply_brush_stroke(img_coords, "draw")
        elif (
            self.active_tool == "brush"
            and event.button() == Qt.RightButton
            and not self.space_pressed
        ):
            # Right click: start erasing with brush (only if SAM is ready and space is not held)
            if self.sam_ready:
                if self.current_mask is not None:
                    # Save history BEFORE starting brush stroke
                    if not self.is_brushing:
                        self._save_mask_to_history()
                    self.is_brushing = True
                    self.brush_mode = "erase"
                    img_coords = self.widget_to_image_coords(event.x(), event.y())
                    if img_coords:
                        self.last_brush_pos = img_coords
                        self._apply_brush_stroke(img_coords, "erase")
        elif (
            event.button() == Qt.LeftButton
            and self.active_tool == "segment"
            and not self.space_pressed
        ):
            # Left click: positive point (only if SAM is ready and space is not held)
            if self.sam_ready:
                self.add_point(event.x(), event.y(), True)
        elif (
            event.button() == Qt.RightButton
            and self.active_tool == "segment"
            and not self.space_pressed
        ):
            # Right click: negative point (only if SAM is ready and space is not held)
            if self.sam_ready:
                self.add_point(event.x(), event.y(), False)
        elif (
            event.button() == Qt.LeftButton
            and self.active_tool == "bbox"
            and not self.space_pressed
        ):
            # Left click: start drawing bounding box
            if self.base_image is not None:
                self.is_drawing_bbox = True
                self.bbox_start_pos = (event.x(), event.y())
                self.bbox_current_pos = (event.x(), event.y())
                self.temp_bbox = None
        elif (
            event.button() == Qt.RightButton
            and self.active_tool == "bbox"
            and not self.space_pressed
        ):
            # Right click: check if clicking on bounding box edge to resize
            if self.base_image is not None and self.bounding_box is not None:
                img_coords = self.widget_to_image_coords(event.x(), event.y())
                if img_coords:
                    edge = self._detect_bbox_edge(img_coords[0], img_coords[1])
                    if edge:
                        # Start resizing this edge
                        self.is_resizing_bbox_edge = True
                        self.bbox_resize_edge = edge
                        self.bbox_resize_start_pos = img_coords
                        self.bbox_resize_original_bbox = self.bounding_box
                        print(f"Resizing bounding box {edge} edge")
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        # Check if space is pressed and we should start panning mid-drag
        buttons = QApplication.instance().mouseButtons()
        if (
            self.space_pressed
            and (buttons & Qt.LeftButton)
            and not self.is_panning
            and not self.is_brushing
        ):
            # Space was pressed during a drag, switch to pan mode
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

        if self.is_panning and self.last_pan_pos is not None:
            # Calculate pan delta
            delta_x = event.pos().x() - self.last_pan_pos.x()
            delta_y = event.pos().y() - self.last_pan_pos.y()

            # Update pan offset
            self.pan_offset_x += delta_x
            self.pan_offset_y += delta_y

            # Update last position
            self.last_pan_pos = event.pos()

            # Redraw
            self.update()
        elif (
            self.is_brushing and self.active_tool == "brush" and not self.space_pressed
        ):
            # Continue brush stroke while moving (only if space is not held)
            img_coords = self.widget_to_image_coords(event.x(), event.y())
            if img_coords:
                # Draw line from last position to current position for smooth strokes
                if self.last_brush_pos:
                    self._apply_brush_line(
                        self.last_brush_pos, img_coords, self.brush_mode
                    )
                else:
                    self._apply_brush_stroke(img_coords, self.brush_mode)
                self.last_brush_pos = img_coords
        elif self.space_pressed and not self.is_panning:
            # If space is pressed but we're not panning yet, stop any ongoing brush strokes
            if self.is_brushing:
                self.is_brushing = False
                self.last_brush_pos = None
        elif (
            self.is_drawing_bbox
            and self.active_tool == "bbox"
            and not self.space_pressed
        ):
            # Update bounding box drawing
            self.bbox_current_pos = (event.x(), event.y())
            # Convert widget coordinates to image coordinates
            if self.bbox_start_pos is not None:
                start_img = self.widget_to_image_coords(
                    self.bbox_start_pos[0], self.bbox_start_pos[1]
                )
                current_img = self.widget_to_image_coords(event.x(), event.y())
                if start_img is not None and current_img is not None:
                    x1, y1 = start_img
                    x2, y2 = current_img
                    # Calculate bounding box (xmin, ymin, xmax, ymax)
                    xmin = min(x1, x2)
                    ymin = min(y1, y2)
                    xmax = max(x1, x2)
                    ymax = max(y1, y2)
                    self.temp_bbox = (xmin, ymin, xmax, ymax)
                    # Update display to show temporary bbox
                    self.update_display()
                    self.update()
        elif (
            self.is_resizing_bbox_edge
            and self.active_tool == "bbox"
            and not self.space_pressed
        ):
            # Update bounding box edge resizing
            current_img = self.widget_to_image_coords(event.x(), event.y())
            if (
                current_img is not None
                and self.bounding_box is not None
                and self.bbox_resize_edge is not None
            ):
                x, y = current_img
                xmin, ymin, xmax, ymax = self.bounding_box

                # Resize based on which edge is being dragged
                # Constrain movement along the edge's axis
                if self.bbox_resize_edge == "top":
                    # Top edge: only change ymin, keep xmin/xmax the same
                    ymin = max(0, min(y, ymax - 1))  # Ensure ymin < ymax
                elif self.bbox_resize_edge == "bottom":
                    # Bottom edge: only change ymax, keep xmin/xmax the same
                    ymax = min(
                        self.base_image.shape[0] - 1, max(y, ymin + 1)
                    )  # Ensure ymax > ymin
                elif self.bbox_resize_edge == "left":
                    # Left edge: only change xmin, keep ymin/ymax the same
                    xmin = max(0, min(x, xmax - 1))  # Ensure xmin < xmax
                elif self.bbox_resize_edge == "right":
                    # Right edge: only change xmax, keep ymin/ymax the same
                    xmax = min(
                        self.base_image.shape[1] - 1, max(x, xmin + 1)
                    )  # Ensure xmax > xmin

                # Update bounding box
                self.bounding_box = (xmin, ymin, xmax, ymax)
                # Update display
                self.update_display()
                self.update()
        else:
            # Update cursor when hovering over bbox edges (for bbox tool)
            if self.active_tool == "bbox" and not self.space_pressed:
                self.update_cursor(event.x(), event.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.is_panning and event.button() == Qt.LeftButton:
            # Stop panning only if mouse button is released
            # Check if space is still pressed - if so, keep ready for next click
            self.is_panning = False
            self.last_pan_pos = None
            # If space is still pressed, keep pan cursor ready for next click
            if self.space_pressed:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.update_cursor()
        elif (
            self.is_brushing
            and (event.button() == Qt.LeftButton or event.button() == Qt.RightButton)
            and not self.space_pressed
        ):
            # Stop brushing
            self.is_brushing = False
            self.last_brush_pos = None
            # Emit mask updated signal
            if self.current_mask is not None:
                self.mask_updated.emit(self.current_mask)
        elif (
            event.button() == Qt.LeftButton
            and self.is_drawing_bbox
            and self.active_tool == "bbox"
        ):
            # Finish drawing bounding box
            if self.temp_bbox is not None:
                # Set the bounding box
                self.bounding_box = self.temp_bbox
                print(f"Bounding box set: {self.bounding_box}")
            # Reset drawing state
            self.is_drawing_bbox = False
            self.bbox_start_pos = None
            self.bbox_current_pos = None
            self.temp_bbox = None
            # Update display
            self.update_display()
            self.update()
        elif (
            event.button() == Qt.RightButton
            and self.is_resizing_bbox_edge
            and self.active_tool == "bbox"
        ):
            # Finish resizing bounding box edge
            if self.bounding_box is not None:
                print(f"Bounding box resized: {self.bounding_box}")
            # Reset resizing state
            self.is_resizing_bbox_edge = False
            self.bbox_resize_edge = None
            self.bbox_resize_start_pos = None
            self.bbox_resize_original_bbox = None
            # Update cursor
            self.update_cursor()
            # Update display
            self.update_display()
            self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming"""
        if self.base_image is None:
            return

        # Get mouse position in widget coordinates
        # Try position() first (Qt 6.2+), fallback to pos() for older versions
        try:
            mouse_pos = event.position()
            widget_x = mouse_pos.x()
            widget_y = mouse_pos.y()
        except AttributeError:
            # Fallback for older PySide6 versions
            mouse_pos = event.pos()
            widget_x = mouse_pos.x()
            widget_y = mouse_pos.y()

        # Get zoom factor (typically 1.15 for smooth zooming)
        zoom_factor = 1.15
        angle_delta = event.angleDelta().y()

        if angle_delta > 0:
            # Zoom in
            new_zoom = self.zoom_scale * zoom_factor
            # Limit max zoom (e.g., 5x)
            if new_zoom <= 5.0:
                self.zoom_in_at_position(widget_x, widget_y, zoom_factor)
        elif angle_delta < 0:
            # Zoom out
            new_zoom = self.zoom_scale / zoom_factor
            # Limit min zoom (can't go below base_scale)
            if new_zoom >= self.base_scale:
                self.zoom_in_at_position(widget_x, widget_y, 1.0 / zoom_factor)

        event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            # Only process initial key press, not auto-repeat
            self.space_pressed = True
            # Cancel any ongoing brush strokes when space is pressed
            if self.is_brushing:
                self.is_brushing = False
                self.last_brush_pos = None

            # If space is pressed and mouse button might be down, check if we should start panning
            # Check if left mouse button is currently pressed
            buttons = QApplication.instance().mouseButtons()
            if buttons & Qt.LeftButton and not self.is_panning:
                # Mouse is already down, start panning now
                cursor_pos = self.mapFromGlobal(QApplication.instance().cursor().pos())
                self.is_panning = True
                self.last_pan_pos = cursor_pos
                self.setCursor(Qt.ClosedHandCursor)
            elif not self.is_panning:
                # Mouse not down, just show open hand cursor
                self.setCursor(Qt.OpenHandCursor)
        elif event.key() == Qt.Key_H and not event.isAutoRepeat():
            # Only process initial key press, not auto-repeat
            self.h_pressed = True
            # Regenerate display image with highlight and trigger redraw
            self.update_display()
            self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events"""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            # Only process actual key release, not auto-repeat release
            self.space_pressed = False
            # Only stop panning if mouse button is also released
            # Check if left mouse button is still pressed
            buttons = QApplication.instance().mouseButtons()
            if not (buttons & Qt.LeftButton):
                # Mouse is released, stop panning
                if self.is_panning:
                    self.is_panning = False
                    self.last_pan_pos = None
                # Restore cursor to active tool cursor
                self.update_cursor()
            # If mouse is still down when space is released, panning continues until mouse is released
            # This is handled in mouseReleaseEvent
        elif event.key() == Qt.Key_H and not event.isAutoRepeat():
            # Only process actual key release, not auto-repeat release
            self.h_pressed = False
            # Regenerate display image without highlight and trigger redraw
            self.update_display()
            self.update()
        super().keyReleaseEvent(event)

    def zoom_in_at_position(self, widget_x: float, widget_y: float, zoom_factor: float):
        """
        Zoom in/out while keeping the point under the mouse cursor fixed

        Args:
            widget_x: X position in widget coordinates
            widget_y: Y position in widget coordinates
            zoom_factor: Factor to zoom by (> 1.0 zooms in, < 1.0 zooms out)
        """
        if self.base_image is None or self.display_image is None:
            return

        # Get image coordinates before zoom (this is what we want to keep under cursor)
        img_coords = self.widget_to_image_coords(int(widget_x), int(widget_y))
        if img_coords is None:
            return

        img_x, img_y = img_coords

        # Calculate current total offset (centering + pan) - need to recalc image_offset first
        # We need the current image_offset which depends on zoom state
        h, w = self.base_image.shape[:2]
        base_display_w = int(w * self.base_scale)
        base_display_h = int(h * self.base_scale)
        widget_w = self.width()
        widget_h = self.height()
        current_image_offset_x = (widget_w - base_display_w) // 2
        current_image_offset_y = (widget_h - base_display_h) // 2

        # Adjust for current zoom
        if self.zoom_scale > 1.0:
            current_zoom_diff_w = (self.display_image.width() - base_display_w) // 2
            current_zoom_diff_h = (self.display_image.height() - base_display_h) // 2
            current_image_offset_x -= current_zoom_diff_w
            current_image_offset_y -= current_zoom_diff_h

        # Update zoom scale
        self.zoom_scale *= zoom_factor
        self.display_scale = self.base_scale * self.zoom_scale

        # Recalculate display image with new zoom
        self.update_display()

        # Recalculate new image offset (will be adjusted in paintEvent, but we need it now)
        new_base_display_w = int(w * self.base_scale)
        new_base_display_h = int(h * self.base_scale)
        new_image_offset_x = (widget_w - new_base_display_w) // 2
        new_image_offset_y = (widget_h - new_base_display_h) // 2

        # Adjust for new zoom
        if self.zoom_scale > 1.0:
            new_zoom_diff_w = (self.display_image.width() - new_base_display_w) // 2
            new_zoom_diff_h = (self.display_image.height() - new_base_display_h) // 2
            new_image_offset_x -= new_zoom_diff_w
            new_image_offset_y -= new_zoom_diff_h

        # Calculate where the image point should be after zoom
        # We want: img_coord = (widget_x - new_total_offset) / new_scale
        # So: new_total_offset = widget_x - img_coord * new_scale
        new_total_offset_x = widget_x - img_x * self.display_scale
        new_total_offset_y = widget_y - img_y * self.display_scale

        # Calculate new pan offset to achieve this
        self.pan_offset_x = new_total_offset_x - new_image_offset_x
        self.pan_offset_y = new_total_offset_y - new_image_offset_y

        # Update display
        self.update()

    def get_segments(self) -> List[Tuple[np.ndarray, str]]:
        """
        Get all finalized segments

        Returns:
            List of (mask, label_id) tuples
        """
        return list(zip(self.finalized_masks, self.finalized_labels))

    def clear_segments(self):
        """Clear all segments"""
        self.finalized_masks = []
        self.finalized_labels = []
        self.current_points = []
        self.current_mask = None
        self.update_display()
        self.update()

    def fit_to_bounding_box(self):
        """
        Fit the view to show the bounding box with some padding
        """
        if self.bounding_box is None or self.base_image is None:
            return

        # Get widget dimensions
        widget_w = self.width()
        widget_h = self.height()

        if widget_w <= 1 or widget_h <= 1:
            # Widget not sized yet, try again later
            QTimer.singleShot(100, self.fit_to_bounding_box)
            return

        # Get bounding box coordinates
        xmin, ymin, xmax, ymax = self.bounding_box

        # Get image dimensions
        img_h, img_w = self.base_image.shape[:2]

        # Calculate bounding box size
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin

        # Add padding (10% on each side)
        padding_factor = 0.05
        padded_w = bbox_w * (1 + 2 * padding_factor)
        padded_h = bbox_h * (1 + 2 * padding_factor)

        # Calculate the scale needed to fit the bounding box in the widget
        # We need to account for base_scale (which fits image to widget without upscaling)
        # First calculate base_scale to fit image
        base_scale_w = widget_w / img_w
        base_scale_h = widget_h / img_h
        self.base_scale = min(base_scale_w, base_scale_h, 1.0)

        # Now calculate what zoom_scale would make the padded bbox fit
        # After base_scale, the bbox would be: bbox_w * base_scale, bbox_h * base_scale
        # We want: padded_w * base_scale * zoom_scale <= widget_w * margin
        # And: padded_h * base_scale * zoom_scale <= widget_h * margin
        # Where margin is like 0.9 to leave some space

        margin = 0.9  # Use 90% of widget space
        zoom_scale_w = (
            (widget_w * margin) / (padded_w * self.base_scale) if padded_w > 0 else 1.0
        )
        zoom_scale_h = (
            (widget_h * margin) / (padded_h * self.base_scale) if padded_h > 0 else 1.0
        )
        self.zoom_scale = min(zoom_scale_w, zoom_scale_h)

        # Don't zoom out (zoom_scale < 1.0 means zooming out beyond base fit)
        # We want to zoom in to fit bbox, so minimum is 1.0 (no additional zoom)
        if self.zoom_scale < 1.0:
            self.zoom_scale = 1.0

        # Limit max zoom
        if self.zoom_scale > 5.0:
            self.zoom_scale = 5.0

        # Calculate display scale
        self.display_scale = self.base_scale * self.zoom_scale

        # Calculate the center of the bounding box in image coordinates
        bbox_center_x = (xmin + xmax) / 2.0
        bbox_center_y = (ymin + ymax) / 2.0

        # Calculate where the bbox center should be in widget coordinates to center it
        # We want: widget_center = bbox_center_img * display_scale + image_offset + pan_offset
        # So: pan_offset = widget_center - bbox_center_img * display_scale - image_offset

        # First update display to get correct image offset calculations
        self.update_display()

        # Recalculate image offset (same as in paintEvent)
        base_display_w = int(img_w * self.base_scale)
        base_display_h = int(img_h * self.base_scale)
        self.image_offset_x = (widget_w - base_display_w) // 2
        self.image_offset_y = (widget_h - base_display_h) // 2

        # Adjust for zoom
        if self.zoom_scale > 1.0:
            zoom_diff_w = (self.display_image.width() - base_display_w) // 2
            zoom_diff_h = (self.display_image.height() - base_display_h) // 2
            self.image_offset_x -= zoom_diff_w
            self.image_offset_y -= zoom_diff_h

        # Calculate pan offset to center bounding box
        widget_center_x = widget_w / 2.0
        widget_center_y = widget_h / 2.0

        # Position where bbox center would be without pan
        bbox_center_in_widget_x = (
            bbox_center_x * self.display_scale + self.image_offset_x
        )
        bbox_center_in_widget_y = (
            bbox_center_y * self.display_scale + self.image_offset_y
        )

        # Calculate pan offset to center it
        self.pan_offset_x = widget_center_x - bbox_center_in_widget_x
        self.pan_offset_y = widget_center_y - bbox_center_in_widget_y

        # Update display
        self.update_display()
        self.update()

    def __del__(self):
        """Cleanup thread when widget is destroyed"""
        try:
            if self.sam_thread is not None and self.sam_thread.isRunning():
                self.sam_thread.quit()
                self.sam_thread.wait(1000)  # Wait up to 1 second
        except RuntimeError:
            # Thread already deleted, ignore
            pass
