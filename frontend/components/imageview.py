"""
ImageView component for SAM Annotator
Image display/view area component with SAM segmentation support
"""
from typing import Optional, List, Tuple
import numpy as np
import cv2
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, Signal, QPoint, QThread, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QWheelEvent
from frontend.theme import CANVAS_BG, ITEM_BG, ITEM_BORDER, TEXT_COLOR, TOPBAR_TEXT_MUTED
from segmentation.sam_model import SAMModel


class ImageView(QWidget):
    """Image view widget for displaying images with SAM segmentation"""
    
    # Signals
    mask_updated = Signal(object)  # Emits when current mask is updated
    segment_finalized = Signal(object, str)  # Emits (mask, label_id) when segment is finalized
    point_added = Signal()  # Emits when a point is added
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageView")
        self.setMinimumHeight(400)
        self.setCursor(Qt.CrossCursor)
        
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
        
        # SAM model
        self.sam_model: Optional[SAMModel] = None
        
        # Segmentation state
        self.current_points: List[Tuple[Tuple[int, int], bool]] = []  # List of ((x, y), is_positive)
        self.current_mask: Optional[np.ndarray] = None
        self.finalized_masks: List[np.ndarray] = []
        self.finalized_labels: List[str] = []  # Label IDs
        self.current_label_id: Optional[str] = None
        self.bounding_box: Optional[Tuple[int, int, int, int]] = None  # (xmin, ymin, xmax, ymax)
        
        # Label color palette (label_id -> BGR color tuple)
        self.label_colors: dict = {}
        # Label info (label_id -> dict with 'name', 'color')
        self.label_info: dict = {}
        
        # Tool state
        self.active_tool = "segment"  # "segment" or "pan"
        
        # Create label indicator widget
        self.label_indicator = None
        
        self.apply_styles()
        self.update_cursor()
        self.setup_label_indicator()
    
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
            label_name = label.get('name', 'Unknown')
            label_color = label.get('color', '#ffffff')
            
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
        super().resizeEvent(_event)
    
    def update_cursor(self):
        """Update cursor based on active tool"""
        if self.active_tool == "pan":
            self.setCursor(Qt.OpenHandCursor)
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
        self.label_info = {label['id']: label for label in labels}
        self.update_label_indicator()
    
    def load_image(self, image_path: str, xml_path: Optional[str] = None):
        """
        Load and display an image
        
        Args:
            image_path: Path to image file
            xml_path: Optional path to VOC XML file for bounding box
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
        
        # Reset pan/zoom
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.zoom_scale = 1.0
        
        # Show image immediately (before SAM processing)
        self.update_display()
        self.update()
        
        # Process SAM in background (non-blocking)
        # Use QTimer to defer SAM processing until after UI update
        if self.sam_model:
            QTimer.singleShot(0, lambda: self._process_sam_image(img))
    
    def _process_sam_image(self, img):
        """Process image with SAM model (called asynchronously)"""
        try:
            if self.sam_model:
                # This is the expensive operation - now runs after UI updates
                self.sam_model.set_image(img)
                print("SAM image processing completed")
        except Exception as e:
            print(f"Error processing SAM image: {str(e)}")
    
    def set_current_label(self, label_id: Optional[str]):
        """Set the current label for new segments"""
        self.current_label_id = label_id
        self.update_label_indicator()
    
    def widget_to_image_coords(self, widget_x: int, widget_y: int) -> Optional[Tuple[int, int]]:
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
    
    def add_point(self, widget_x: int, widget_y: int, is_positive: bool):
        """
        Add a point for segmentation
        
        Args:
            widget_x: X coordinate in widget space
            widget_y: Y coordinate in widget space
            is_positive: True for positive point (include), False for negative (exclude)
        """
        if self.base_image is None or self.sam_model is None:
            return
        
        # Convert to image coordinates
        img_coords = self.widget_to_image_coords(widget_x, widget_y)
        if img_coords is None:
            return
        
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
        
        # Separate positive and negative points
        positive_points = [(x, y) for (x, y), is_pos in self.current_points if is_pos]
        negative_points = [(x, y) for (x, y), is_pos in self.current_points if not is_pos]
        
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
        mask = self.sam_model.predict_mask(all_points, labels, self.bounding_box)
        self.current_mask = mask
        
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
        if self.current_mask is None or len(self.current_points) == 0:
            return False
        
        if self.current_label_id is None:
            return False
        
        # Add to finalized masks
        self.finalized_masks.append(self.current_mask.copy())
        self.finalized_labels.append(self.current_label_id)
        
        # Clear current state
        self.current_points = []
        self.current_mask = None
        
        # Emit signal
        self.segment_finalized.emit(self.finalized_masks[-1], self.current_label_id)
        
        # Update display
        self.update_display()
        self.update()
        
        return True
    
    def undo_last_point(self) -> bool:
        """
        Undo last point
        
        Returns:
            True if point was removed, False otherwise
        """
        if self.current_points:
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
        for mask, label_id in zip(self.finalized_masks, self.finalized_labels):
            color = self.label_colors.get(label_id, (255, 0, 0))
            overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array(color, dtype=np.uint8)).astype(np.uint8)
        
        # Draw current mask being built
        if self.current_mask is not None:
            color = self.label_colors.get(self.current_label_id, (255, 0, 0))
            overlay[self.current_mask] = (0.5 * overlay[self.current_mask] + 0.5 * np.array(color, dtype=np.uint8)).astype(np.uint8)
        
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
        
        # Draw bounding box if present
        if self.bounding_box is not None:
            xmin, ymin, xmax, ymax = self.bounding_box
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        
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
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if self.active_tool == "pan" and event.button() == Qt.LeftButton:
            # Start panning
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton and self.active_tool == "segment":
            # Left click: positive point
            self.add_point(event.x(), event.y(), True)
        elif event.button() == Qt.RightButton and self.active_tool == "segment":
            # Right click: negative point
            self.add_point(event.x(), event.y(), False)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
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
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.is_panning and event.button() == Qt.LeftButton:
            # Stop panning
            self.is_panning = False
            self.last_pan_pos = None
            self.update_cursor()
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
        
        # Current total offset
        old_total_offset_x = current_image_offset_x + self.pan_offset_x
        old_total_offset_y = current_image_offset_y + self.pan_offset_y
        old_scale = self.display_scale
        
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
