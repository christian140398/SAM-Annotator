"""
ImageView component for SAM Annotator
Image display/view area component with SAM segmentation support
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QKeyEvent,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from frontend.theme import CANVAS_BG, ITEM_BG, ITEM_BORDER, TEXT_COLOR
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
        self.actual_display_scale = (
            1.0  # Actual scale used after capping (for coordinate calculations)
        )
        self.image_offset_x = 0  # Centering offset (for base scale)
        self.image_offset_y = 0  # Centering offset (for base scale)
        self.pan_offset_x = 0  # Pan offset (for dragging)
        self.pan_offset_y = 0  # Pan offset (for dragging)
        self.viewport_offset_x = 0  # Offset when using viewport rendering
        self.viewport_offset_y = 0  # Offset when using viewport rendering

        # Performance optimization: cache overlay images
        self.cached_overlay_image: Optional[np.ndarray] = (
            None  # Cached full overlay (BGR)
        )
        self.overlay_cache_valid = False  # Whether full overlay cache is up to date
        self.cached_base_overlay: Optional[np.ndarray] = (
            None  # Cached base overlay (finalized segments only)
        )
        self.base_overlay_valid = False  # Whether base overlay cache is up to date
        self.base_overlay_dirty = True  # Whether base overlay needs rebuilding
        self.dynamic_overlay_dirty = True  # Whether dynamic overlay needs rebuilding
        self.last_zoom_scale = 1.0  # Track zoom changes for cache invalidation

        # Zoom throttling: delay display updates during rapid zoom
        self.zoom_update_timer = QTimer(self)
        self.zoom_update_timer.setSingleShot(True)
        self.zoom_update_timer.timeout.connect(self._delayed_zoom_update)
        self.pending_zoom_update = False
        self.pending_zoom_widget_x = 0.0
        self.pending_zoom_widget_y = 0.0
        self.pending_zoom_img_x = 0
        self.pending_zoom_img_y = 0
        self.pending_zoom_factor = 1.0

        # Performance: track if we're at very high zoom for additional optimizations
        self.is_very_high_zoom = False  # True when zoom_scale > 5.0

        # Drawing performance: track if actively drawing to use faster updates
        self.is_actively_drawing = False  # True when brush is active or segmenting

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
        self.sam_embedding_in_progress = False  # Flag to prevent concurrent embeddings
        self.sam_cleanup_in_progress = False  # Flag to track cleanup state
        self.sam_embedding_cancelled = (
            False  # Flag to prevent signals from cancelled embeddings
        )

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

        # Performance: cache contours for finalized segments to avoid recalculating on hover
        self.segment_contours: Dict[
            int, List
        ] = {}  # Cache of contours for each segment index

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
        self.brush_size = (
            10  # Brush size (side length for squares) in pixels (image coordinates)
        )
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

        # Show loading indicator if:
        # 1. SAM is not ready
        # 2. Base image is loaded
        # 3. We're actively embedding (either thread is running OR embedding is in progress)
        # This ensures the indicator shows whenever we're embedding the current image
        is_actively_embedding = (
            self.sam_embedding_in_progress  # Embedding has started
            or (
                self.sam_thread is not None
                and self.sam_thread.isRunning()
                and self.sam_worker is not None
            )  # Or thread is actively running
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
        # If SAM is not ready, show "not allowed" cursor for segmentation, brush, and bbox
        if not self.sam_ready and (
            self.active_tool == "segment"
            or self.active_tool == "brush"
            or self.active_tool == "bbox"
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

        # Invalidate base overlay because label colors changed (affects finalized segments)
        self.invalidate_base_overlay()
        self.update_display()
        self.update()

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
        self.actual_display_scale = 1.0
        self.is_very_high_zoom = False
        self.viewport_offset_x = 0
        self.viewport_offset_y = 0

        # Invalidate all overlays when loading new image
        self.invalidate_base_overlay()
        self.dynamic_overlay_dirty = True
        self.last_zoom_scale = 1.0
        # Clear contour cache for old segments
        self.segment_contours.clear()

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
        # Prevent concurrent embeddings
        if self.sam_embedding_in_progress:
            print("SAM embedding already in progress, skipping concurrent request")
            return

        # Prevent starting if cleanup is in progress
        if self.sam_cleanup_in_progress:
            print("Thread cleanup in progress, skipping embedding start")
            return

        # Set embedding flag and clear cancelled flag
        self.sam_embedding_in_progress = True
        self.sam_embedding_cancelled = False

        # Cancel any existing processing (non-blocking - don't wait)
        if self.sam_thread is not None:
            self.cancel_sam_embedding()
            # Don't wait - just proceed. Cleanup will happen asynchronously
            # Process events once to allow cancellation to start
            if QApplication.instance() is not None:
                QApplication.instance().processEvents()

        try:
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

            # Update loading indicator to show it's starting
            self.update_loading_indicator()

            # Start processing in background
            self.sam_thread.start()
        except Exception as e:
            print(f"Error starting SAM embedding: {str(e)}")
            # Reset flag on error
            self.sam_embedding_in_progress = False
            # Clean up on error
            try:
                if self.sam_thread is not None:
                    self._cleanup_thread()
            except Exception:
                pass

    def _on_sam_ready(self):
        """Called when SAM processing is complete"""
        # Don't emit signal if embedding was cancelled
        if self.sam_embedding_cancelled:
            return

        self.sam_ready = True
        print("SAM is ready for segmentation")
        # Hide loading indicator
        self.update_loading_indicator()
        # Update cursor now that SAM is ready
        self.update_cursor()
        # Emit signal to notify that embedding is complete
        self.sam_embedding_complete.emit()

    def cancel_sam_embedding(self):
        """Cancel any ongoing SAM embedding operations (non-blocking)"""
        if self.sam_thread is not None:
            try:
                if self.sam_thread.isRunning():
                    # Disconnect all signals to prevent callbacks on stale thread
                    # Check if objects are valid before disconnecting
                    if self.sam_worker is not None:
                        try:
                            # Only try to disconnect if worker still exists
                            if hasattr(self.sam_worker, "processing_complete"):
                                self.sam_worker.processing_complete.disconnect()
                            if hasattr(self.sam_worker, "finished"):
                                self.sam_worker.finished.disconnect()
                        except (RuntimeError, TypeError, AttributeError):
                            pass  # Signals already disconnected or object deleted

                    try:
                        # Only try to disconnect if thread still exists
                        if hasattr(self.sam_thread, "started"):
                            self.sam_thread.started.disconnect()
                        if hasattr(self.sam_thread, "finished"):
                            self.sam_thread.finished.disconnect()
                    except (RuntimeError, TypeError, AttributeError):
                        pass  # Signals already disconnected

                    # Quit the thread immediately (non-blocking - don't wait)
                    self.sam_thread.quit()
                    # Reconnect finished signal to ensure cleanup happens asynchronously
                    try:
                        # Try to connect - if already connected, this will be a no-op
                        self.sam_thread.finished.connect(self._cleanup_thread)
                    except (RuntimeError, TypeError):
                        pass
                    # Don't wait - cleanup will happen via finished signal

                # Clean up worker if it exists
                if self.sam_worker is not None:
                    try:
                        self.sam_worker.deleteLater()
                    except RuntimeError:
                        pass
                    self.sam_worker = None

                # Reset embedding flag and mark as cancelled
                self.sam_embedding_in_progress = False
                self.sam_embedding_cancelled = True
            except RuntimeError:
                pass
            except Exception as e:
                print(f"Error cancelling SAM embedding: {str(e)}")
                # Ensure cleanup happens even on error
                try:
                    self._cleanup_thread()
                except Exception:
                    pass

    def _cleanup_thread(self):
        """Clean up thread resources"""
        # Prevent concurrent cleanup
        if self.sam_cleanup_in_progress:
            return

        self.sam_cleanup_in_progress = True
        try:
            if self.sam_thread is not None:
                thread_to_clean = self.sam_thread
                self.sam_thread = None  # Clear reference immediately to prevent reuse

                # Ensure thread is not running before deletion
                try:
                    if thread_to_clean.isRunning():
                        # Thread is still running, wait longer for it to finish
                        if not thread_to_clean.wait(500):
                            # Force termination if it doesn't finish
                            try:
                                thread_to_clean.terminate()
                                if not thread_to_clean.wait(300):
                                    # Still running after terminate, don't delete yet
                                    # Schedule cleanup to try again later
                                    print(
                                        "Warning: Thread still running after terminate, will retry cleanup"
                                    )
                                    # Reconnect finished signal to retry cleanup
                                    try:
                                        thread_to_clean.finished.connect(
                                            self._cleanup_thread
                                        )
                                    except Exception:
                                        pass
                                    # Don't delete yet - let it finish naturally
                                    # Restore reference so it doesn't get garbage collected
                                    self.sam_thread = thread_to_clean
                                    self.sam_cleanup_in_progress = False
                                    return
                            except RuntimeError:
                                pass
                except RuntimeError:
                    pass  # Thread already deleted or in invalid state

                # Disconnect all signals before deletion
                try:
                    if self.sam_worker is not None:
                        try:
                            # Only try to disconnect if worker still exists and has the signal
                            if hasattr(self.sam_worker, "processing_complete"):
                                self.sam_worker.processing_complete.disconnect()
                            if hasattr(self.sam_worker, "finished"):
                                self.sam_worker.finished.disconnect()
                        except (RuntimeError, TypeError, AttributeError):
                            pass
                except RuntimeError:
                    pass

                try:
                    # Only try to disconnect if thread still exists and has the signals
                    if hasattr(thread_to_clean, "started"):
                        thread_to_clean.started.disconnect()
                    if hasattr(thread_to_clean, "finished"):
                        thread_to_clean.finished.disconnect()
                except (RuntimeError, TypeError, AttributeError):
                    pass

                # Only delete if thread is definitely not running
                try:
                    if not thread_to_clean.isRunning():
                        thread_to_clean.deleteLater()
                    else:
                        # Thread is still running, don't delete - let it finish naturally
                        # The finished signal will trigger cleanup again
                        print("Warning: Thread still running, deferring deletion")
                        # Keep reference temporarily and let finished signal handle it
                        self.sam_thread = thread_to_clean
                        try:
                            thread_to_clean.finished.connect(self._cleanup_thread)
                        except Exception:
                            pass
                        self.sam_cleanup_in_progress = False
                        return
                except RuntimeError:
                    pass

            # Clear worker reference
            if self.sam_worker is not None:
                try:
                    self.sam_worker.deleteLater()
                except RuntimeError:
                    pass
                self.sam_worker = None
        except Exception as e:
            print(f"Error in thread cleanup: {str(e)}")
        finally:
            self.sam_cleanup_in_progress = False
            self.sam_embedding_in_progress = False
            self.sam_embedding_cancelled = False

    def set_current_label(self, label_id: Optional[str]):
        """Set the current label for new segments"""
        self.current_label_id = label_id
        self.update_label_indicator()

    def set_hovered_segment_index(self, segment_index: Optional[int]):
        """Set the hovered segment index for highlighting"""
        self.hovered_segment_index = segment_index
        # Invalidate only dynamic overlay because hover affects dynamic rendering
        self.invalidate_dynamic_overlay()
        self.update_display()
        self.update()

    def invalidate_base_overlay(self):
        """Invalidate base overlay cache (called when segments are added/removed/modified)"""
        self.base_overlay_dirty = True
        self.base_overlay_valid = False
        self.cached_base_overlay = None
        # Also invalidate full overlay since it depends on base
        self.overlay_cache_valid = False
        self.cached_overlay_image = None

    def invalidate_dynamic_overlay(self):
        """Invalidate dynamic overlay cache (called when hover, current mask, points, or bbox changes)"""
        self.dynamic_overlay_dirty = True
        # Only invalidate full overlay, base overlay can stay cached
        self.overlay_cache_valid = False
        self.cached_overlay_image = None

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
        # Use actual_display_scale if it's been set (when display size was capped), otherwise use display_scale
        scale_to_use = getattr(self, "actual_display_scale", self.display_scale)
        if scale_to_use <= 0:
            scale_to_use = self.display_scale

        # Account for viewport offset if using viewport rendering
        # When viewport rendering is active, the display_image is cropped and drawn centered
        viewport_offset_x = getattr(self, "viewport_offset_x", 0)
        viewport_offset_y = getattr(self, "viewport_offset_y", 0)

        if viewport_offset_x != 0 or viewport_offset_y != 0:
            # Viewport rendering is active
            # The viewport image is a cropped portion of the full image, scaled and drawn centered
            # The viewport represents full image region [viewport_offset_x, viewport_offset_y] to [x2, y2]
            #
            # To convert widget coords to full image coords:
            # 1. Convert widget coords to viewport image coords (accounting for draw position and scale)
            # 2. Add viewport_offset to get full image coords
            #
            # The viewport image is drawn at image_offset_x/y (centered)
            # The actual scale is determined by the actual QPixmap size vs the viewport image size

            # Use the actual scale that was applied to the viewport image
            # This should match the scale used in update_display
            viewport_scale = getattr(self, "actual_display_scale", self.display_scale)
            if viewport_scale <= 0:
                viewport_scale = self.display_scale

            # The viewport image is drawn centered at image_offset_x/y
            # Widget coordinates relative to the draw position
            rel_x = widget_x - self.image_offset_x
            rel_y = widget_y - self.image_offset_y

            # Convert to viewport image coordinates (in image space)
            # The viewport image was scaled from its original size using viewport_scale
            # So: viewport_img_coord = widget_rel_coord / viewport_scale
            viewport_img_x = rel_x / viewport_scale if viewport_scale > 0 else 0
            viewport_img_y = rel_y / viewport_scale if viewport_scale > 0 else 0

            # Convert to full image coordinates by adding the viewport offset
            img_x = int(viewport_img_x + viewport_offset_x)
            img_y = int(viewport_img_y + viewport_offset_y)
        else:
            # No viewport rendering - standard conversion
            # Calculate the actual draw position (matching paintEvent logic)
            # Base centering offset
            h, w = self.base_image.shape[:2]
            base_display_w = int(w * self.base_scale)
            base_display_h = int(h * self.base_scale)

            # Start with base image offset
            image_offset_x = self.image_offset_x
            image_offset_y = self.image_offset_y

            # Adjust for zoom (matching paintEvent)
            if self.zoom_scale > 1.0 and self.display_image is not None:
                zoom_diff_w = (self.display_image.width() - base_display_w) // 2
                zoom_diff_h = (self.display_image.height() - base_display_h) // 2
                image_offset_x -= zoom_diff_w
                image_offset_y -= zoom_diff_h

            # Add pan offset (matching paintEvent)
            total_offset_x = image_offset_x + self.pan_offset_x
            total_offset_y = image_offset_y + self.pan_offset_y

            img_x = int((widget_x - total_offset_x) / scale_to_use)
            img_y = int((widget_y - total_offset_y) / scale_to_use)

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
        # Use actual_display_scale if available (when display was capped), otherwise display_scale
        scale_to_use = getattr(self, "actual_display_scale", self.display_scale)
        if scale_to_use <= 0:
            scale_to_use = self.display_scale
        threshold = self.EDGE_DETECTION_THRESHOLD / scale_to_use  # Adjust for zoom

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
        # Use actual_display_scale if available (when display was capped), otherwise display_scale
        scale_to_use = getattr(self, "actual_display_scale", self.display_scale)
        if scale_to_use <= 0:
            scale_to_use = self.display_scale
        # For squares: brush_size represents side length, so radius = (size - 1) / 2
        # This allows brush_size=1 to create a 1x1 pixel square (radius=0)
        scaled_size = (
            int(self.brush_size / scale_to_use) if scale_to_use > 0 else self.brush_size
        )
        brush_radius = max(0, (scaled_size - 1) // 2) if scaled_size > 0 else 0
        # Ensure brush_radius is valid (at least 0 for 1-pixel square, and not too large)
        brush_radius = max(0, min(brush_radius, 1000))  # Cap at reasonable maximum

        # Create a temporary mask for the brush stroke
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        # Draw a square instead of a circle
        # When brush_radius is 0, this creates a 1x1 pixel square
        top_left = (max(0, img_x - brush_radius), max(0, img_y - brush_radius))
        bottom_right = (
            min(w - 1, img_x + brush_radius),
            min(h - 1, img_y + brush_radius),
        )
        cv2.rectangle(temp_mask, top_left, bottom_right, 255, -1)
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

        # Update display (invalidate dynamic overlay when current mask changes)
        self.invalidate_dynamic_overlay()
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
        # Use actual_display_scale if available (when display was capped), otherwise display_scale
        scale_to_use = getattr(self, "actual_display_scale", self.display_scale)
        if scale_to_use <= 0:
            scale_to_use = self.display_scale
        # For squares: brush_size represents side length, so radius = (size - 1) / 2
        # This allows brush_size=1 to create a 1x1 pixel square (radius=0)
        scaled_size = (
            int(self.brush_size / scale_to_use) if scale_to_use > 0 else self.brush_size
        )
        brush_radius = max(0, (scaled_size - 1) // 2) if scaled_size > 0 else 0
        # Ensure brush_radius is valid (at least 0 for 1-pixel square, and not too large)
        brush_radius = max(0, min(brush_radius, 1000))  # Cap at reasonable maximum

        # Create a temporary mask for the brush line
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        # Calculate thickness, ensuring it's within OpenCV's valid range
        # OpenCV MAX_THICKNESS is typically 255, and thickness must be > 0 for cv2.line
        if brush_radius <= 1:
            thickness = 1  # Minimum valid thickness for cv2.line
        else:
            thickness = min(brush_radius * 2, 255)  # Cap at OpenCV's MAX_THICKNESS
            thickness = max(1, int(thickness))  # Ensure at least 1 and is integer

        cv2.line(temp_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
        # Also draw squares at start and end for smoother strokes
        # Start point square
        start_top_left = (
            max(0, start_x - brush_radius),
            max(0, start_y - brush_radius),
        )
        start_bottom_right = (
            min(w - 1, start_x + brush_radius),
            min(h - 1, start_y + brush_radius),
        )
        cv2.rectangle(temp_mask, start_top_left, start_bottom_right, 255, -1)
        # End point square
        end_top_left = (max(0, end_x - brush_radius), max(0, end_y - brush_radius))
        end_bottom_right = (
            min(w - 1, end_x + brush_radius),
            min(h - 1, end_y + brush_radius),
        )
        cv2.rectangle(temp_mask, end_top_left, end_bottom_right, 255, -1)
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

        # Update display (invalidate dynamic overlay when current mask changes)
        self.invalidate_dynamic_overlay()
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

        # Mark as actively drawing for performance optimization
        self.is_actively_drawing = True

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
            # Invalidate dynamic overlay because current mask was cleared
            self.invalidate_dynamic_overlay()
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

        # Update display (invalidate dynamic overlay when current mask changes)
        # Note: is_actively_drawing is already set by add_point, so fast scaling will be used
        self.invalidate_dynamic_overlay()
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
        segment_index = len(self.finalized_masks)
        self.finalized_masks.append(self.current_mask.copy())
        self.finalized_labels.append(self.current_label_id)

        # Cache contours for this new segment
        self._get_segment_contours(segment_index, self.current_mask)

        # Clear current state
        self.current_points = []
        self.current_mask = None
        self.mask_history = []  # Clear undo history when finalizing
        self.points_history = []  # Clear points history when finalizing
        self.is_actively_drawing = False  # No longer actively drawing

        # Emit signal
        self.segment_finalized.emit(self.finalized_masks[-1], self.current_label_id)

        # Update display (invalidate base overlay when segment is finalized)
        self.invalidate_base_overlay()
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

            # Update display (invalidate cache when mask changes)
            self.overlay_cache_valid = False
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
            segment_index = len(self.finalized_masks) - 1
            self.finalized_masks.pop()
            self.finalized_labels.pop()
            # Remove from contour cache
            if segment_index in self.segment_contours:
                del self.segment_contours[segment_index]
            # Rebuild contour cache indices for remaining segments
            # (indices shift when we remove one)
            if segment_index > 0:
                # Rebuild cache with correct indices
                new_contours = {}
                for i, mask in enumerate(self.finalized_masks):
                    new_contours[i] = self._get_segment_contours(i, mask)
                self.segment_contours = new_contours
            # Invalidate base overlay because segment was removed
            self.invalidate_base_overlay()
            self.update_display()
            self.update()
            return True
        return False

    def _get_segment_contours(self, segment_index: int, mask: np.ndarray) -> List:
        """
        Get contours for a segment, using cache if available

        Args:
            segment_index: Index of the segment
            mask: The mask array for the segment

        Returns:
            List of contours
        """
        if segment_index not in self.segment_contours:
            # Calculate and cache contours
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            self.segment_contours[segment_index] = contours
        return self.segment_contours[segment_index]

    def _draw_base_overlay(self, img: np.ndarray) -> np.ndarray:
        """
        Draw base overlay with all finalized segments (without hover effects)

        Args:
            img: Base image (BGR format)

        Returns:
            Image with base overlays (finalized segments only)
        """
        overlay = img.copy()

        # Draw finalized masks (normal overlay, no hover effects)
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

            # Normal overlay for all segments (hover effects handled in dynamic overlay)
            # Use efficient NumPy operations for blending
            mask_bool = mask.astype(bool)
            if mask_bool.any():
                color_array = np.array(color, dtype=np.uint8)
                # Blend: 0.6 * overlay + 0.4 * color
                overlay[mask_bool] = (
                    0.6 * overlay[mask_bool].astype(np.float32)
                    + 0.4 * color_array.astype(np.float32)
                ).astype(np.uint8)

        return overlay

    def _draw_dynamic_overlay(self, base_overlay: np.ndarray) -> np.ndarray:
        """
        Draw dynamic overlay elements: hover effects, current mask, points, bounding boxes

        Args:
            base_overlay: Base overlay image with finalized segments

        Returns:
            Image with dynamic overlays added
        """
        overlay = base_overlay.copy()

        # Draw hover effects for hovered segment
        if self.hovered_segment_index is not None and self.hovered_segment_index < len(
            self.finalized_masks
        ):
            idx = self.hovered_segment_index
            mask = self.finalized_masks[idx]
            label_id = self.finalized_labels[idx]

            # Validate mask dimensions
            if mask is not None and self.base_image is not None:
                img_h, img_w = self.base_image.shape[:2]
                mask_h, mask_w = mask.shape[:2]
                if mask_h == img_h and mask_w == img_w:
                    color = self.label_colors.get(label_id, (255, 0, 0))

                    # Brighter overlay for hovered segment
                    mask_bool = mask.astype(bool)
                    if mask_bool.any():
                        color_array = np.array(color, dtype=np.uint8)
                        # Blend: 0.4 * overlay + 0.6 * color (brighter)
                        overlay[mask_bool] = (
                            0.4 * overlay[mask_bool].astype(np.float32)
                            + 0.6 * color_array.astype(np.float32)
                        ).astype(np.uint8)

                    # Draw border around hovered segment using cached contours
                    contours = self._get_segment_contours(idx, mask)
                    highlight_color = (255, 255, 255)  # White border for hover
                    cv2.drawContours(overlay, contours, -1, highlight_color, 1)

        # Draw current mask being built
        if self.current_mask is not None:
            # Validate mask dimensions match current image
            if self.base_image is not None:
                img_h, img_w = self.base_image.shape[:2]
                mask_h, mask_w = self.current_mask.shape[:2]
                if mask_h == img_h and mask_w == img_w:
                    color = self.label_colors.get(self.current_label_id, (255, 0, 0))
                    mask_bool = self.current_mask.astype(bool)
                    if mask_bool.any():
                        color_array = np.array(color, dtype=np.uint8)
                        # Blend: 0.5 * overlay + 0.5 * color
                        overlay[mask_bool] = (
                            0.5 * overlay[mask_bool].astype(np.float32)
                            + 0.5 * color_array.astype(np.float32)
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
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 200, 0), 1)
            else:
                # Negative: red
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 0, 255), -1)
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 0, 200), 1)

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
            thickness = 1  # Match the red bounding box thickness
            # Draw rectangle outline
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, thickness)

        return overlay

    def draw_overlay(self, img: np.ndarray) -> np.ndarray:
        """
        Draw full overlay by composing base and dynamic overlays

        Args:
            img: Base image (BGR format)

        Returns:
            Image with all overlays
        """
        # Draw base overlay (finalized segments)
        base = self._draw_base_overlay(img)
        # Draw dynamic overlay (hover, current mask, points, bbox)
        return self._draw_dynamic_overlay(base)

    def update_display(self, force_rebuild_overlay: bool = False):
        """
        Update the display image with overlays

        Args:
            force_rebuild_overlay: If True, force rebuild of overlay cache even if valid
        """
        if self.base_image is None:
            return

        # Check if we need to rebuild overlay cache
        # Rebuild if: cache invalid, force rebuild, or zoom changed significantly
        # At very high zoom, be more aggressive about cache invalidation threshold
        # Also, don't rebuild overlay during active zoom (when timer is running) - wait until zoom stops
        zoom_threshold = 0.3 if self.zoom_scale > 5.0 else 0.15
        zoom_changed = abs(self.zoom_scale - self.last_zoom_scale) > zoom_threshold

        # Track very high zoom state for performance optimizations
        self.is_very_high_zoom = self.zoom_scale > 5.0

        # Skip overlay rebuild if zoom is actively happening (timer running) - this reduces stutter
        # Only rebuild if zoom has stopped or changed significantly
        is_zooming = (
            self.zoom_update_timer.isActive()
            if hasattr(self, "zoom_update_timer")
            else False
        )

        # Rebuild base overlay if dirty or zoom changed significantly
        if (
            self.base_overlay_dirty
            or not self.base_overlay_valid
            or force_rebuild_overlay
            or (zoom_changed and not is_zooming)
        ):
            # Rebuild base overlay (finalized segments only)
            self.cached_base_overlay = self._draw_base_overlay(self.base_image)
            self.base_overlay_valid = True
            self.base_overlay_dirty = False
            # Base overlay changed, so full overlay is also invalid
            self.overlay_cache_valid = False

        # Rebuild dynamic overlay if dirty or base changed
        if (
            self.dynamic_overlay_dirty
            or not self.overlay_cache_valid
            or force_rebuild_overlay
            or (zoom_changed and not is_zooming)
        ):
            # Compose base + dynamic overlays
            if self.cached_base_overlay is not None:
                self.cached_overlay_image = self._draw_dynamic_overlay(
                    self.cached_base_overlay
                )
            else:
                # Fallback: rebuild everything if base overlay is missing
                self.cached_overlay_image = self.draw_overlay(self.base_image)
            self.overlay_cache_valid = True
            self.dynamic_overlay_dirty = False
            self.last_zoom_scale = self.zoom_scale

        display_img = self.cached_overlay_image

        # Calculate base scale to fit widget (never upscales from this)
        widget_w = self.width()
        widget_h = self.height()
        h, w = display_img.shape[:2]

        if widget_w > 1 and widget_h > 1:
            scale_w = widget_w / w
            scale_h = widget_h / h
            self.base_scale = min(scale_w, scale_h, 1.0)  # Don't upscale initially

            # Combined scale (base * zoom)
            self.display_scale = self.base_scale * self.zoom_scale

            # Calculate base scale offsets first (needed for viewport calculation)
            # This matches the calculation in paintEvent
            base_display_w = int(w * self.base_scale)
            base_display_h = int(h * self.base_scale)
            base_image_offset_x = (widget_w - base_display_w) // 2
            base_image_offset_y = (widget_h - base_display_h) // 2

            # Performance optimization: at very high zoom, use viewport-based rendering
            # Only render the visible portion of the image to dramatically improve performance
            # This requires calculating viewport based on current pan/zoom state
            viewport = None
            if self.zoom_scale > 10.0:  # Only use viewport rendering at very high zoom
                # Calculate viewport based on what should be visible
                # Use the FULL image's base offsets for calculation (not viewport image offsets)
                # This ensures the viewport calculation is consistent
                full_image_base_offset_x = (widget_w - base_display_w) // 2
                full_image_base_offset_y = (widget_h - base_display_h) // 2

                # Estimate zoomed full image size for offset calculation
                estimated_zoom_w = int(w * self.display_scale)
                estimated_zoom_h = int(h * self.display_scale)

                # Adjust for zoom (estimate for full image)
                full_image_offset_x = full_image_base_offset_x
                full_image_offset_y = full_image_base_offset_y
                if self.zoom_scale > 1.0:
                    zoom_diff_w = (estimated_zoom_w - base_display_w) // 2
                    zoom_diff_h = (estimated_zoom_h - base_display_h) // 2
                    full_image_offset_x -= zoom_diff_w
                    full_image_offset_y -= zoom_diff_h

                # Calculate what portion of FULL image is visible
                # Use full image offsets + pan_offset to determine visible area
                scale_to_use = self.display_scale
                center_widget_x = widget_w / 2.0
                center_widget_y = widget_h / 2.0

                # Calculate where widget center maps to in FULL image coordinates
                # This uses the full image's offsets + pan_offset
                total_full_offset_x = full_image_offset_x + self.pan_offset_x
                total_full_offset_y = full_image_offset_y + self.pan_offset_y
                center_img_x = (
                    (center_widget_x - total_full_offset_x) / scale_to_use
                    if scale_to_use > 0
                    else w / 2
                )
                center_img_y = (
                    (center_widget_y - total_full_offset_y) / scale_to_use
                    if scale_to_use > 0
                    else h / 2
                )

                # Calculate viewport around this center
                visible_w = widget_w / scale_to_use if scale_to_use > 0 else w
                visible_h = widget_h / scale_to_use if scale_to_use > 0 else h

                img_x1 = int(center_img_x - visible_w / 2)
                img_y1 = int(center_img_y - visible_h / 2)
                img_x2 = int(center_img_x + visible_w / 2)
                img_y2 = int(center_img_y + visible_h / 2)

                # Add padding to avoid edge artifacts
                padding = 100  # pixels in image space
                img_x1 = max(0, img_x1 - padding)
                img_y1 = max(0, img_y1 - padding)
                img_x2 = min(w, img_x2 + padding)
                img_y2 = min(h, img_y2 + padding)

                # Only use viewport if it's significantly smaller than full image
                viewport_area = (img_x2 - img_x1) * (img_y2 - img_y1)
                full_area = w * h
                if (
                    viewport_area < full_area * 0.5
                    and img_x2 > img_x1
                    and img_y2 > img_y1
                ):
                    # Only use viewport if it's less than 50% of image and valid
                    viewport = (img_x1, img_y1, img_x2, img_y2)

            if viewport is not None:
                # Crop to visible viewport
                x1, y1, x2, y2 = viewport
                if x2 > x1 and y2 > y1:
                    # Store original dimensions before cropping (for reference)
                    original_h, original_w = display_img.shape[:2]
                    display_img = display_img[y1:y2, x1:x2].copy()
                    h, w = display_img.shape[:2]
                    # Store viewport offset for coordinate calculations
                    self.viewport_offset_x = x1
                    self.viewport_offset_y = y1
                    # DON'T recalculate base_scale - keep it based on the original full image
                    # The viewport is just a crop for rendering, coordinates stay in full image space
                    # We'll calculate image_offset_x/y after we know the actual displayed size (after capping)
                    # Store a flag to recalculate offsets after scaling
                    self._viewport_needs_offset_recalc = True
                else:
                    viewport = None

            if viewport is None:
                # No viewport cropping
                self.viewport_offset_x = 0
                self.viewport_offset_y = 0
                # Set image offsets for non-viewport case
                self.image_offset_x = base_image_offset_x
                self.image_offset_y = base_image_offset_y

            new_w = int(w * self.display_scale)
            new_h = int(h * self.display_scale)

            # Performance optimization: limit maximum display size to prevent extreme memory usage
            # When zoomed in very close, cap the display size to improve performance
            # This prevents creating extremely large images that cause lag
            # Use adaptive max size based on zoom level - allow more zoom at higher levels
            # At very high zoom (>10x), allow larger display sizes for detailed inspection
            if self.zoom_scale > 10.0:
                MAX_DISPLAY_SIZE = 20000  # Larger limit for very high zoom
            elif self.zoom_scale > 5.0:
                MAX_DISPLAY_SIZE = 15000  # Medium limit for high zoom
            else:
                MAX_DISPLAY_SIZE = 10000  # Standard limit for normal zoom

            if new_w > MAX_DISPLAY_SIZE or new_h > MAX_DISPLAY_SIZE:
                # Scale down to max size while maintaining aspect ratio
                scale_factor = min(MAX_DISPLAY_SIZE / new_w, MAX_DISPLAY_SIZE / new_h)
                new_w = int(new_w * scale_factor)
                new_h = int(new_h * scale_factor)
                # Store the actual scale used for coordinate calculations
                # This is the scale that was actually applied after capping
                self.actual_display_scale = min(new_w / w, new_h / h)
            else:
                # No capping needed, use the full display_scale
                self.actual_display_scale = self.display_scale

            # If viewport rendering is active, we'll recalculate image_offset_x/y after QPixmap is created
            # This is because the actual QPixmap size may differ from new_w/new_h due to aspect ratio
            viewport_needs_offset_recalc = getattr(
                self, "_viewport_needs_offset_recalc", False
            )

            # Convert BGR to RGB for QImage
            rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)

            # Performance optimization: use faster scaling for very large images
            # When zoomed in a lot, the image is very large, so use FastTransformation
            # This is much faster than SmoothTransformation for large images
            # Also use fast transformation when actively drawing to reduce stutter
            # Lower thresholds for better performance at high zoom levels
            # At very high zoom, always use fast transformation
            # Also use fast transformation during active zoom (when timer is running)
            is_zooming = (
                self.zoom_update_timer.isActive()
                if hasattr(self, "zoom_update_timer")
                else False
            )
            if (
                new_w > 2000
                or new_h > 2000
                or self.zoom_scale > 2.0
                or self.is_actively_drawing
                or self.is_very_high_zoom
                or is_zooming  # Use fast transformation during active zoom
            ):
                # For very large images, high zoom, or during active drawing, use FastTransformation
                # The quality difference is minimal when zoomed in or during drawing
                self.display_image = QPixmap.fromImage(q_image).scaled(
                    new_w, new_h, Qt.KeepAspectRatio, Qt.FastTransformation
                )
            else:
                # Use smooth transformation for normal zoom levels when not drawing
                self.display_image = QPixmap.fromImage(q_image).scaled(
                    new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )

            # If viewport rendering is active, recalculate image_offset_x/y using actual QPixmap size
            # The actual size may differ from new_w/new_h due to aspect ratio preservation
            if viewport_needs_offset_recalc and self.display_image is not None:
                # The viewport image is displayed at the actual QPixmap size
                actual_pixmap_w = self.display_image.width()
                actual_pixmap_h = self.display_image.height()
                # Center it in the widget
                self.image_offset_x = (widget_w - actual_pixmap_w) // 2
                self.image_offset_y = (widget_h - actual_pixmap_h) // 2
                self._viewport_needs_offset_recalc = False
        else:
            rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)
            self.display_image = QPixmap.fromImage(q_image)
            self.base_scale = 1.0
            self.display_scale = self.base_scale * self.zoom_scale

    def paintEvent(self, _event):
        """Paint the image with overlays"""
        painter = QPainter(self)
        # Disable antialiasing at very high zoom for better performance
        # The quality difference is minimal when zoomed in very close
        if not self.is_very_high_zoom:
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

            # Base centering offset (centers image at base_scale)
            # Use the offsets calculated in update_display
            image_offset_x = self.image_offset_x
            image_offset_y = self.image_offset_y

            # Check if viewport rendering is active
            viewport_offset_x = getattr(self, "viewport_offset_x", 0)
            viewport_offset_y = getattr(self, "viewport_offset_y", 0)

            if viewport_offset_x == 0 and viewport_offset_y == 0:
                # No viewport rendering - standard zoom adjustment
                if self.zoom_scale > 1.0:
                    zoom_diff_w = (self.display_image.width() - base_display_w) // 2
                    zoom_diff_h = (self.display_image.height() - base_display_h) // 2
                    image_offset_x -= zoom_diff_w
                    image_offset_y -= zoom_diff_h
            else:
                # Viewport rendering is active
                # The image_offset_x/y from update_display already centers the viewport image
                # No additional zoom adjustment needed - the viewport image is already at the correct size
                pass

            # Apply pan offset
            viewport_offset_x = getattr(self, "viewport_offset_x", 0)
            viewport_offset_y = getattr(self, "viewport_offset_y", 0)

            if viewport_offset_x != 0 or viewport_offset_y != 0:
                # Viewport rendering is active
                # The viewport already represents what's visible based on pan_offset
                # So the viewport image should be centered (no pan_offset applied)
                # pan_offset is incorporated into the viewport calculation in update_display
                draw_x = image_offset_x
                draw_y = image_offset_y
            else:
                # No viewport rendering - standard pan offset
                draw_x = image_offset_x + self.pan_offset_x
                draw_y = image_offset_y + self.pan_offset_y

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
                        self.is_actively_drawing = True  # Mark as actively drawing
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
                    self.is_actively_drawing = True  # Mark as actively drawing
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
            # Left click: start drawing bounding box (only if SAM is ready)
            if self.sam_ready and self.base_image is not None:
                self.is_drawing_bbox = True
                self.bbox_start_pos = (event.x(), event.y())
                self.bbox_current_pos = (event.x(), event.y())
                self.temp_bbox = None
        elif (
            event.button() == Qt.RightButton
            and self.active_tool == "bbox"
            and not self.space_pressed
        ):
            # Right click: check if clicking on bounding box edge to resize (only if SAM is ready)
            if (
                self.sam_ready
                and self.base_image is not None
                and self.bounding_box is not None
            ):
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

            # When viewport rendering is active, we need to recalculate the viewport
            # based on the new pan offset. Otherwise, just redraw.
            viewport_offset_x = getattr(self, "viewport_offset_x", 0)
            viewport_offset_y = getattr(self, "viewport_offset_y", 0)
            is_viewport_active = viewport_offset_x != 0 or viewport_offset_y != 0
            will_use_viewport = self.zoom_scale > 10.0

            if is_viewport_active or will_use_viewport:
                # Viewport rendering is active - need to recalculate viewport
                # Don't rebuild overlay cache (it's still valid, just need new viewport)
                self.update_display()

            # Always trigger repaint to show the new pan position
            self.update()
        elif (
            self.is_brushing and self.active_tool == "brush" and not self.space_pressed
        ):
            # Continue brush stroke while moving (only if space is not held)
            self.is_actively_drawing = True  # Mark as actively drawing for performance
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
                    # Invalidate dynamic overlay so temp_bbox is drawn
                    self.invalidate_dynamic_overlay()
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
                # Update display (invalidate dynamic overlay when bbox changes)
                self.invalidate_dynamic_overlay()
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
            self.is_actively_drawing = False  # No longer actively drawing
            self.last_brush_pos = None
            # Emit mask updated signal
            if self.current_mask is not None:
                self.mask_updated.emit(self.current_mask)
            # Do a final high-quality update now that drawing stopped
            self.invalidate_dynamic_overlay()
            self.update_display()
            self.update()
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
            # Update display (invalidate cache when bbox changes)
            self.overlay_cache_valid = False
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
            # Update display (invalidate cache when bbox changes)
            self.overlay_cache_valid = False
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
            # Check if we're already at max zoom - if so, ignore the event
            MAX_ZOOM = 200.0
            if self.zoom_scale >= MAX_ZOOM:
                # Already at max zoom, don't process zoom in events
                event.accept()
                return

            new_zoom = self.zoom_scale * zoom_factor
            # Limit max zoom (200x for very detailed inspection)
            if new_zoom <= MAX_ZOOM:
                self.zoom_in_at_position(
                    widget_x, widget_y, zoom_factor, immediate=False
                )
            else:
                # Would exceed max zoom, clamp to max and zoom to that limit
                # Calculate the factor needed to reach exactly MAX_ZOOM
                factor_to_max = MAX_ZOOM / self.zoom_scale
                if factor_to_max > 1.0:  # Only if we're not already at max
                    self.zoom_in_at_position(
                        widget_x, widget_y, factor_to_max, immediate=False
                    )
        elif angle_delta < 0:
            # Zoom out
            new_zoom = self.zoom_scale / zoom_factor
            # Allow zooming out further (minimum 0.1x zoom_scale)
            if new_zoom >= 0.1:
                self.zoom_in_at_position(
                    widget_x, widget_y, 1.0 / zoom_factor, immediate=False
                )

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
                cursor_pos = self.mapFromGlobal(QCursor.pos())
                self.is_panning = True
                self.last_pan_pos = cursor_pos
                self.setCursor(Qt.ClosedHandCursor)
            elif not self.is_panning:
                # Mouse not down, just show open hand cursor
                self.setCursor(Qt.OpenHandCursor)
        elif event.key() == Qt.Key_H and not event.isAutoRepeat():
            # Only process initial key press, not auto-repeat
            self.h_pressed = True
            # Invalidate dynamic overlay because H highlight affects current mask rendering
            self.invalidate_dynamic_overlay()
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
            # Invalidate cache because H highlight affects overlay rendering
            self.overlay_cache_valid = False
            # Regenerate display image without highlight and trigger redraw
            self.update_display()
            self.update()
        super().keyReleaseEvent(event)

    def zoom_in_at_position(
        self,
        widget_x: float,
        widget_y: float,
        zoom_factor: float,
        immediate: bool = True,
    ):
        """
        Zoom in/out while keeping the point under the mouse cursor fixed

        Args:
            widget_x: X position in widget coordinates
            widget_y: Y position in widget coordinates
            zoom_factor: Factor to zoom by (> 1.0 zooms in, < 1.0 zooms out)
            immediate: If True, update display immediately. If False, use throttled update.
        """
        if self.base_image is None:
            return

        # Store zoom parameters for delayed update if needed
        if not immediate:
            # Get image coordinates BEFORE updating zoom scale
            # This is the point we want to keep under the cursor
            img_coords = self.widget_to_image_coords(int(widget_x), int(widget_y))
            if img_coords is None:
                return

            img_x, img_y = img_coords

            # Store parameters for delayed update
            self.pending_zoom_widget_x = widget_x
            self.pending_zoom_widget_y = widget_y
            self.pending_zoom_img_x = img_x
            self.pending_zoom_img_y = img_y
            self.pending_zoom_factor = zoom_factor
            self.pending_zoom_update = True

            # Update zoom scale immediately for smooth feel
            # Clamp zoom scale to valid range immediately to prevent accumulation
            MAX_ZOOM = 200.0
            MIN_ZOOM = 0.1
            self.zoom_scale *= zoom_factor
            # Clamp immediately to prevent going over limits
            if self.zoom_scale > MAX_ZOOM:
                self.zoom_scale = MAX_ZOOM
            elif self.zoom_scale < MIN_ZOOM:
                self.zoom_scale = MIN_ZOOM
            self.display_scale = self.base_scale * self.zoom_scale

            # Calculate new pan offset to keep the same image point under the cursor
            # IMPORTANT: Calculate pan offset BEFORE update_display() so the viewport uses correct pan offset
            # The key insight: before zoom, widget_x maps to img_x via:
            #   img_x = (widget_x - total_offset_x_old) / old_scale
            #   where total_offset_x_old = image_offset_x_old (with zoom adj) + pan_offset_x_old
            # After zoom, we want the same img_x to map to widget_x:
            #   img_x = (widget_x - total_offset_x_new) / new_scale
            #   where total_offset_x_new = image_offset_x_new (with zoom adj) + pan_offset_x_new
            # Solving: total_offset_x_new = widget_x - img_x * new_scale
            # So: pan_offset_x_new = total_offset_x_new - image_offset_x_new (with zoom adj)

            # Check if viewport rendering will be active
            will_use_viewport = self.zoom_scale > 10.0

            if will_use_viewport:
                # Viewport rendering will be active
                # Pan offset affects which part of the image is in the viewport
                # The viewport center is: center_img = (widget_center - full_image_offset - pan_offset) / display_scale
                # We want img_x to appear at widget_x
                # So: img_x = center_img + (widget_x - widget_center) / display_scale
                # Therefore: center_img = img_x - (widget_x - widget_center) / display_scale
                # And: pan_offset = widget_center - full_image_offset - center_img * display_scale

                h, w = self.base_image.shape[:2]
                base_display_w = int(w * self.base_scale)
                base_display_h = int(h * self.base_scale)
                widget_w = self.width()
                widget_h = self.height()

                # Calculate full image offset (matching update_display viewport calculation)
                full_image_base_offset_x = (widget_w - base_display_w) // 2
                full_image_base_offset_y = (widget_h - base_display_h) // 2

                # Estimate zoomed full image size for offset calculation
                estimated_zoom_w = int(w * self.display_scale)
                estimated_zoom_h = int(h * self.display_scale)

                # Adjust for zoom
                full_image_offset_x = full_image_base_offset_x
                full_image_offset_y = full_image_base_offset_y
                if self.zoom_scale > 1.0:
                    zoom_diff_w = (estimated_zoom_w - base_display_w) // 2
                    zoom_diff_h = (estimated_zoom_h - base_display_h) // 2
                    full_image_offset_x -= zoom_diff_w
                    full_image_offset_y -= zoom_diff_h

                # Calculate desired viewport center so that img_x appears at widget_x
                widget_center_x = widget_w / 2.0
                widget_center_y = widget_h / 2.0
                desired_viewport_center_x = (
                    img_x - (widget_x - widget_center_x) / self.display_scale
                    if self.display_scale > 0
                    else img_x
                )
                desired_viewport_center_y = (
                    img_y - (widget_y - widget_center_y) / self.display_scale
                    if self.display_scale > 0
                    else img_y
                )

                # Calculate pan offset to achieve this viewport center
                self.pan_offset_x = (
                    widget_center_x
                    - full_image_offset_x
                    - desired_viewport_center_x * self.display_scale
                )
                self.pan_offset_y = (
                    widget_center_y
                    - full_image_offset_y
                    - desired_viewport_center_y * self.display_scale
                )
            else:
                # No viewport rendering - standard calculation
                # Estimate the new image offset and pan offset
                h, w = self.base_image.shape[:2]
                base_display_w = int(w * self.base_scale)
                base_display_h = int(h * self.base_scale)
                widget_w = self.width()
                widget_h = self.height()

                # Calculate base image offset
                base_image_offset_x = (widget_w - base_display_w) // 2
                base_image_offset_y = (widget_h - base_display_h) // 2

                # Estimate zoomed image size
                estimated_zoom_w = int(w * self.display_scale)
                estimated_zoom_h = int(h * self.display_scale)

                # Estimate new image offset (with zoom adjustment)
                new_image_offset_x = base_image_offset_x
                new_image_offset_y = base_image_offset_y
                if self.zoom_scale > 1.0:
                    zoom_diff_w = (estimated_zoom_w - base_display_w) // 2
                    zoom_diff_h = (estimated_zoom_h - base_display_h) // 2
                    new_image_offset_x -= zoom_diff_w
                    new_image_offset_y -= zoom_diff_h

                # Use display_scale for initial calculation (will refine after update_display if capped)
                scale_to_use = self.display_scale

                # Calculate total offset needed to position img_x at widget_x
                total_offset_x_needed = widget_x - img_x * scale_to_use
                total_offset_y_needed = widget_y - img_y * scale_to_use

                # Calculate pan offset to achieve this
                self.pan_offset_x = total_offset_x_needed - new_image_offset_x
                self.pan_offset_y = total_offset_y_needed - new_image_offset_y

            # Now update display with the correct pan offset
            self.update_display()

            # Refine pan offset if display was capped (actual_display_scale differs from display_scale)
            scale_to_use = getattr(self, "actual_display_scale", self.display_scale)
            if (
                scale_to_use != self.display_scale
                and scale_to_use > 0
                and not will_use_viewport
            ):
                # Scale was capped, recalculate pan offset with actual scale
                h, w = self.base_image.shape[:2]
                base_display_w = int(w * self.base_scale)
                base_display_h = int(h * self.base_scale)
                widget_w = self.width()
                widget_h = self.height()

                if self.display_image is not None:
                    actual_display_w = self.display_image.width()
                    actual_display_h = self.display_image.height()
                    base_image_offset_x = (widget_w - base_display_w) // 2
                    base_image_offset_y = (widget_h - base_display_h) // 2

                    new_image_offset_x = base_image_offset_x
                    new_image_offset_y = base_image_offset_y
                    if self.zoom_scale > 1.0:
                        zoom_diff_w = (actual_display_w - base_display_w) // 2
                        zoom_diff_h = (actual_display_h - base_display_h) // 2
                        new_image_offset_x -= zoom_diff_w
                        new_image_offset_y -= zoom_diff_h

                    # Recalculate pan offset with actual scale
                    total_offset_x_needed = widget_x - img_x * scale_to_use
                    total_offset_y_needed = widget_y - img_y * scale_to_use
                    self.pan_offset_x = total_offset_x_needed - new_image_offset_x
                    self.pan_offset_y = total_offset_y_needed - new_image_offset_y

            # Restart timer to delay expensive display update (for quality improvement)
            # This batches rapid zoom events together
            # Use longer delay at very high zoom levels for better performance
            # More aggressive delays to reduce stutter
            if self.zoom_scale > 10.0:
                delay_ms = 150  # Very high zoom - longer delay
            elif self.zoom_scale > 5.0:
                delay_ms = 120  # High zoom - medium delay
            else:
                delay_ms = 80  # Normal zoom - shorter delay
            self.zoom_update_timer.stop()
            self.zoom_update_timer.start(delay_ms)  # Delay - update after zoom stops

            # Update display with correct positioning (but skip expensive overlay rebuild)
            # This gives immediate visual feedback without full quality update
            self.update()
            return

        if self.display_image is None:
            return

        # Get image coordinates before zoom (this is what we want to keep under cursor)
        img_coords = self.widget_to_image_coords(int(widget_x), int(widget_y))
        if img_coords is None:
            return

        img_x, img_y = img_coords

        # Update zoom scale
        # Clamp zoom scale to valid range immediately to prevent accumulation
        MAX_ZOOM = 200.0
        MIN_ZOOM = 0.1
        self.zoom_scale *= zoom_factor
        # Clamp immediately to prevent going over limits
        if self.zoom_scale > MAX_ZOOM:
            self.zoom_scale = MAX_ZOOM
        elif self.zoom_scale < MIN_ZOOM:
            self.zoom_scale = MIN_ZOOM
        self.display_scale = self.base_scale * self.zoom_scale

        # Calculate pan offset BEFORE update_display() so viewport uses correct pan offset
        will_use_viewport = self.zoom_scale > 10.0

        if will_use_viewport:
            # Viewport rendering will be active
            h, w = self.base_image.shape[:2]
            base_display_w = int(w * self.base_scale)
            base_display_h = int(h * self.base_scale)
            widget_w = self.width()
            widget_h = self.height()

            # Calculate full image offset (matching update_display viewport calculation)
            full_image_base_offset_x = (widget_w - base_display_w) // 2
            full_image_base_offset_y = (widget_h - base_display_h) // 2

            # Estimate zoomed full image size for offset calculation
            estimated_zoom_w = int(w * self.display_scale)
            estimated_zoom_h = int(h * self.display_scale)

            # Adjust for zoom
            full_image_offset_x = full_image_base_offset_x
            full_image_offset_y = full_image_base_offset_y
            if self.zoom_scale > 1.0:
                zoom_diff_w = (estimated_zoom_w - base_display_w) // 2
                zoom_diff_h = (estimated_zoom_h - base_display_h) // 2
                full_image_offset_x -= zoom_diff_w
                full_image_offset_y -= zoom_diff_h

            # Calculate desired viewport center so that img_x appears at widget_x
            widget_center_x = widget_w / 2.0
            widget_center_y = widget_h / 2.0
            desired_viewport_center_x = (
                img_x - (widget_x - widget_center_x) / self.display_scale
                if self.display_scale > 0
                else img_x
            )
            desired_viewport_center_y = (
                img_y - (widget_y - widget_center_y) / self.display_scale
                if self.display_scale > 0
                else img_y
            )

            # Calculate pan offset to achieve this viewport center
            self.pan_offset_x = (
                widget_center_x
                - full_image_offset_x
                - desired_viewport_center_x * self.display_scale
            )
            self.pan_offset_y = (
                widget_center_y
                - full_image_offset_y
                - desired_viewport_center_y * self.display_scale
            )
        else:
            # No viewport rendering - standard calculation
            # Estimate the new image offset and pan offset
            h, w = self.base_image.shape[:2]
            base_display_w = int(w * self.base_scale)
            base_display_h = int(h * self.base_scale)
            widget_w = self.width()
            widget_h = self.height()

            # Calculate base image offset
            base_image_offset_x = (widget_w - base_display_w) // 2
            base_image_offset_y = (widget_h - base_display_h) // 2

            # Estimate zoomed image size
            estimated_zoom_w = int(w * self.display_scale)
            estimated_zoom_h = int(h * self.display_scale)

            # Estimate new image offset (with zoom adjustment)
            new_image_offset_x = base_image_offset_x
            new_image_offset_y = base_image_offset_y
            if self.zoom_scale > 1.0:
                zoom_diff_w = (estimated_zoom_w - base_display_w) // 2
                zoom_diff_h = (estimated_zoom_h - base_display_h) // 2
                new_image_offset_x -= zoom_diff_w
                new_image_offset_y -= zoom_diff_h

            # Use display_scale for initial calculation (will refine after update_display if capped)
            scale_to_use = self.display_scale

            # Calculate total offset needed to position img_x at widget_x
            total_offset_x_needed = widget_x - img_x * scale_to_use
            total_offset_y_needed = widget_y - img_y * scale_to_use

            # Calculate pan offset to achieve this
            self.pan_offset_x = total_offset_x_needed - new_image_offset_x
            self.pan_offset_y = total_offset_y_needed - new_image_offset_y

        # Recalculate display image with new zoom (don't rebuild overlay, just recalc viewport)
        # Overlay cache stays valid, only viewport/scaling changes
        self.update_display()

        # Refine pan offset if display was capped (actual_display_scale differs from display_scale)
        scale_to_use = getattr(self, "actual_display_scale", self.display_scale)
        if (
            scale_to_use != self.display_scale
            and scale_to_use > 0
            and not will_use_viewport
        ):
            # Scale was capped, recalculate pan offset with actual scale
            h, w = self.base_image.shape[:2]
            base_display_w = int(w * self.base_scale)
            base_display_h = int(h * self.base_scale)
            widget_w = self.width()
            widget_h = self.height()

            if self.display_image is not None:
                actual_display_w = self.display_image.width()
                actual_display_h = self.display_image.height()
                base_image_offset_x = (widget_w - base_display_w) // 2
                base_image_offset_y = (widget_h - base_display_h) // 2

                new_image_offset_x = base_image_offset_x
                new_image_offset_y = base_image_offset_y
                if self.zoom_scale > 1.0:
                    zoom_diff_w = (actual_display_w - base_display_w) // 2
                    zoom_diff_h = (actual_display_h - base_display_h) // 2
                    new_image_offset_x -= zoom_diff_w
                    new_image_offset_y -= zoom_diff_h

                # Recalculate pan offset with actual scale
                total_offset_x_needed = widget_x - img_x * scale_to_use
                total_offset_y_needed = widget_y - img_y * scale_to_use
                self.pan_offset_x = total_offset_x_needed - new_image_offset_x
                self.pan_offset_y = total_offset_y_needed - new_image_offset_y

        # Update display
        self.update()

    def _delayed_zoom_update(self):
        """Perform the delayed zoom update after zoom has stopped"""
        if not self.pending_zoom_update:
            return

        # Get stored zoom parameters (image coordinates were captured before zoom)
        widget_x = getattr(self, "pending_zoom_widget_x", 0)
        widget_y = getattr(self, "pending_zoom_widget_y", 0)
        img_x = getattr(self, "pending_zoom_img_x", 0)
        img_y = getattr(self, "pending_zoom_img_y", 0)

        # Reset pending flag
        self.pending_zoom_update = False

        # Now do the full update with proper scaling and overlay rebuild
        # Force overlay rebuild now that zoom has stopped for best quality
        self.update_display(force_rebuild_overlay=True)

        # Recalculate pan offset to keep point under cursor
        # Use the stored image coordinates (captured before zoom changed)
        if self.display_image is not None and self.base_image is not None:
            # Check if viewport rendering is active
            viewport_offset_x = getattr(self, "viewport_offset_x", 0)
            viewport_offset_y = getattr(self, "viewport_offset_y", 0)
            is_viewport_active = viewport_offset_x != 0 or viewport_offset_y != 0
            will_use_viewport = self.zoom_scale > 10.0

            if is_viewport_active or will_use_viewport:
                # Viewport rendering is active (or will be)
                h, w = self.base_image.shape[:2]
                base_display_w = int(w * self.base_scale)
                base_display_h = int(h * self.base_scale)
                widget_w = self.width()
                widget_h = self.height()

                # Calculate full image offset (matching update_display viewport calculation)
                full_image_base_offset_x = (widget_w - base_display_w) // 2
                full_image_base_offset_y = (widget_h - base_display_h) // 2

                # Estimate zoomed full image size for offset calculation
                estimated_zoom_w = int(w * self.display_scale)
                estimated_zoom_h = int(h * self.display_scale)

                # Adjust for zoom
                full_image_offset_x = full_image_base_offset_x
                full_image_offset_y = full_image_base_offset_y
                if self.zoom_scale > 1.0:
                    zoom_diff_w = (estimated_zoom_w - base_display_w) // 2
                    zoom_diff_h = (estimated_zoom_h - base_display_h) // 2
                    full_image_offset_x -= zoom_diff_w
                    full_image_offset_y -= zoom_diff_h

                # Calculate desired viewport center so that img_x appears at widget_x
                widget_center_x = widget_w / 2.0
                widget_center_y = widget_h / 2.0
                desired_viewport_center_x = (
                    img_x - (widget_x - widget_center_x) / self.display_scale
                    if self.display_scale > 0
                    else img_x
                )
                desired_viewport_center_y = (
                    img_y - (widget_y - widget_center_y) / self.display_scale
                    if self.display_scale > 0
                    else img_y
                )

                # Calculate pan offset to achieve this viewport center
                self.pan_offset_x = (
                    widget_center_x
                    - full_image_offset_x
                    - desired_viewport_center_x * self.display_scale
                )
                self.pan_offset_y = (
                    widget_center_y
                    - full_image_offset_y
                    - desired_viewport_center_y * self.display_scale
                )
            else:
                # No viewport rendering - standard calculation
                h, w = self.base_image.shape[:2]
                base_display_w = int(w * self.base_scale)
                base_display_h = int(h * self.base_scale)
                widget_w = self.width()
                widget_h = self.height()

                # Calculate new image offset (base centering)
                new_image_offset_x = self.image_offset_x
                new_image_offset_y = self.image_offset_y

                # Adjust for zoom (matching paintEvent logic)
                if self.zoom_scale > 1.0:
                    new_zoom_diff_w = (self.display_image.width() - base_display_w) // 2
                    new_zoom_diff_h = (
                        self.display_image.height() - base_display_h
                    ) // 2
                    new_image_offset_x -= new_zoom_diff_w
                    new_image_offset_y -= new_zoom_diff_h

                # Use actual_display_scale if available (when display was capped), otherwise display_scale
                scale_to_use = getattr(self, "actual_display_scale", self.display_scale)
                if scale_to_use <= 0:
                    scale_to_use = self.display_scale

                # Calculate total offset needed to position img_x at widget_x
                total_offset_x_needed = widget_x - img_x * scale_to_use
                total_offset_y_needed = widget_y - img_y * scale_to_use

                # Calculate pan offset to achieve this
                self.pan_offset_x = total_offset_x_needed - new_image_offset_x
                self.pan_offset_y = total_offset_y_needed - new_image_offset_y

        # Update display
        self.update()

    def get_segments(self) -> List[Tuple[np.ndarray, str]]:
        """
        Get all finalized segments

        Returns:
            List of (mask, label_id) tuples
        """
        return list(zip(self.finalized_masks, self.finalized_labels))

    def get_brush_size(self) -> int:
        """
        Get the current brush size

        Returns:
            Brush size in pixels (image coordinates)
        """
        return self.brush_size

    def set_brush_size(self, size: int):
        """
        Set the brush size

        Args:
            size: Brush size in pixels (image coordinates), will be clamped to valid range
        """
        # Clamp brush size to reasonable range (1-100)
        self.brush_size = max(1, min(100, size))

    def clear_segments(self):
        """Clear all segments"""
        self.finalized_masks = []
        self.finalized_labels = []
        self.current_points = []
        self.current_mask = None
        self.update_display()
        self.update()

    def has_active_segment(self) -> bool:
        """
        Check if there's an active segment being drawn (has points or mask)

        Returns:
            True if there's an active segment, False otherwise
        """
        return (self.current_mask is not None and self.current_mask.sum() > 0) or len(
            self.current_points
        ) > 0

    def clear_current_segment(self):
        """Clear the current segment being drawn (without finalizing)"""
        self.current_points = []
        self.current_mask = None
        self.mask_history = []
        self.points_history = []
        # Invalidate dynamic overlay because we're removing the current segment
        self.invalidate_dynamic_overlay()
        self.update_display()
        self.update()

    def start_editing_segment(self, segment_index: int) -> bool:
        """
        Start editing a finalized segment by loading it as the current mask

        Args:
            segment_index: Index of the segment in finalized_masks to edit

        Returns:
            True if successfully started editing, False otherwise
        """
        if segment_index < 0 or segment_index >= len(self.finalized_masks):
            return False

        # Get the mask and label
        mask = self.finalized_masks[segment_index]
        label_id = self.finalized_labels[segment_index]

        # Remove from finalized lists
        self.finalized_masks.pop(segment_index)
        self.finalized_labels.pop(segment_index)

        # Remove from contour cache
        if segment_index in self.segment_contours:
            del self.segment_contours[segment_index]
        # Rebuild contour cache indices for remaining segments (indices shift when we remove one)
        if segment_index < len(self.finalized_masks):
            # Rebuild cache with correct indices
            new_contours = {}
            for i, m in enumerate(self.finalized_masks):
                new_contours[i] = self._get_segment_contours(i, m)
            self.segment_contours = new_contours

        # Load as current mask for editing
        self.current_mask = mask.copy()
        self.current_label_id = label_id
        self.current_points = []  # Clear points since we're editing the mask directly

        # Clear history since we're starting fresh
        self.mask_history = []
        self.points_history = []

        # Update label indicator to show the correct label
        self.update_label_indicator()

        # Invalidate base overlay because segment was removed from finalized list
        # Also invalidate dynamic overlay because current mask changed
        self.invalidate_base_overlay()
        self.invalidate_dynamic_overlay()
        self.update_display()
        self.update()

        return True

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
        if self.zoom_scale > 200.0:
            self.zoom_scale = 200.0

        # Calculate display scale
        self.display_scale = self.base_scale * self.zoom_scale

        # Calculate the center of the bounding box in image coordinates
        bbox_center_x = (xmin + xmax) / 2.0
        bbox_center_y = (ymin + ymax) / 2.0

        # Calculate pan offset BEFORE update_display() so viewport uses correct pan offset
        # Check if viewport rendering will be active
        will_use_viewport = self.zoom_scale > 10.0

        if will_use_viewport:
            # Viewport rendering will be active
            # Calculate full image offset (matching update_display viewport calculation)
            base_display_w = int(img_w * self.base_scale)
            base_display_h = int(img_h * self.base_scale)
            full_image_base_offset_x = (widget_w - base_display_w) // 2
            full_image_base_offset_y = (widget_h - base_display_h) // 2

            # Estimate zoomed full image size for offset calculation
            estimated_zoom_w = int(img_w * self.display_scale)
            estimated_zoom_h = int(img_h * self.display_scale)

            # Adjust for zoom
            full_image_offset_x = full_image_base_offset_x
            full_image_offset_y = full_image_base_offset_y
            if self.zoom_scale > 1.0:
                zoom_diff_w = (estimated_zoom_w - base_display_w) // 2
                zoom_diff_h = (estimated_zoom_h - base_display_h) // 2
                full_image_offset_x -= zoom_diff_w
                full_image_offset_y -= zoom_diff_h

            # Calculate desired viewport center so that bbox_center appears at widget center
            widget_center_x = widget_w / 2.0
            widget_center_y = widget_h / 2.0
            desired_viewport_center_x = bbox_center_x
            desired_viewport_center_y = bbox_center_y

            # Calculate pan offset to achieve this viewport center
            self.pan_offset_x = (
                widget_center_x
                - full_image_offset_x
                - desired_viewport_center_x * self.display_scale
            )
            self.pan_offset_y = (
                widget_center_y
                - full_image_offset_y
                - desired_viewport_center_y * self.display_scale
            )
        else:
            # No viewport rendering - standard calculation
            # Estimate the new image offset and pan offset
            base_display_w = int(img_w * self.base_scale)
            base_display_h = int(img_h * self.base_scale)
            base_image_offset_x = (widget_w - base_display_w) // 2
            base_image_offset_y = (widget_h - base_display_h) // 2

            # Estimate zoomed image size
            estimated_zoom_w = int(img_w * self.display_scale)
            estimated_zoom_h = int(img_h * self.display_scale)

            # Estimate new image offset (with zoom adjustment)
            new_image_offset_x = base_image_offset_x
            new_image_offset_y = base_image_offset_y
            if self.zoom_scale > 1.0:
                zoom_diff_w = (estimated_zoom_w - base_display_w) // 2
                zoom_diff_h = (estimated_zoom_h - base_display_h) // 2
                new_image_offset_x -= zoom_diff_w
                new_image_offset_y -= zoom_diff_h

            # Use display_scale for initial calculation (will refine after update_display if capped)
            scale_to_use = self.display_scale

            # Calculate total offset needed to position bbox_center at widget center
            widget_center_x = widget_w / 2.0
            widget_center_y = widget_h / 2.0
            total_offset_x_needed = widget_center_x - bbox_center_x * scale_to_use
            total_offset_y_needed = widget_center_y - bbox_center_y * scale_to_use

            # Calculate pan offset to achieve this
            self.pan_offset_x = total_offset_x_needed - new_image_offset_x
            self.pan_offset_y = total_offset_y_needed - new_image_offset_y

        # Now update display with the correct pan offset
        # Invalidate dynamic overlay to ensure fresh display (viewport may have changed)
        self.invalidate_dynamic_overlay()
        self.update_display()

        # Refine pan offset if display was capped (actual_display_scale differs from display_scale)
        scale_to_use = getattr(self, "actual_display_scale", self.display_scale)
        if (
            scale_to_use != self.display_scale
            and scale_to_use > 0
            and not will_use_viewport
        ):
            # Scale was capped, recalculate pan offset with actual scale
            base_display_w = int(img_w * self.base_scale)
            base_display_h = int(img_h * self.base_scale)

            if self.display_image is not None:
                actual_display_w = self.display_image.width()
                actual_display_h = self.display_image.height()
                base_image_offset_x = (widget_w - base_display_w) // 2
                base_image_offset_y = (widget_h - base_display_h) // 2

                new_image_offset_x = base_image_offset_x
                new_image_offset_y = base_image_offset_y
                if self.zoom_scale > 1.0:
                    zoom_diff_w = (actual_display_w - base_display_w) // 2
                    zoom_diff_h = (actual_display_h - base_display_h) // 2
                    new_image_offset_x -= zoom_diff_w
                    new_image_offset_y -= zoom_diff_h

                # Recalculate pan offset with actual scale
                widget_center_x = widget_w / 2.0
                widget_center_y = widget_h / 2.0
                total_offset_x_needed = widget_center_x - bbox_center_x * scale_to_use
                total_offset_y_needed = widget_center_y - bbox_center_y * scale_to_use
                self.pan_offset_x = total_offset_x_needed - new_image_offset_x
                self.pan_offset_y = total_offset_y_needed - new_image_offset_y
                # Update display again with refined pan offset
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
