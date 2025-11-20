"""
SegmentsPanel component for SAM Annotator
Replicates the functionality of the React SegmentsPanel component
"""

import os
import re
from typing import Optional, List, Dict, Any
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, QByteArray, QSize
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer
from frontend.theme import (
    ITEM_BG,
    ITEM_BORDER,
    TEXT_COLOR,
    BACKGROUND_MAIN,
    PRIMARY_COLOR,
)

# Get the directory of this file to resolve icon paths
_ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")


def create_colored_svg_icon(svg_path: str, color: str = "#3b82f6") -> QIcon:
    """Load SVG file and create a colored version as QIcon"""
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

        # Replace stroke colors with specified color
        # Match stroke:#... in CSS styles (e.g., stroke:#020202;)
        svg_content = re.sub(
            r"stroke:#[0-9a-fA-F]{3,6}", f"stroke:{color}", svg_content
        )
        # Match stroke="..." attributes
        svg_content = re.sub(r'stroke="[^"]*"', f'stroke="{color}"', svg_content)

        # Replace fill colors with specified color (except for fill="none" or fill:none)
        # Match fill:#... in CSS styles (but not fill:none)
        svg_content = re.sub(
            r"fill:#([0-9a-fA-F]{3,6})(?!\s*none)", f"fill:{color}", svg_content
        )
        # Match fill="..." attributes (but not fill="none")
        svg_content = re.sub(r'fill="(?!none)[^"]*"', f'fill="{color}"', svg_content)

        # Also replace any color: attributes in CSS
        svg_content = re.sub(r"color:#[0-9a-fA-F]{3,6}", f"color:{color}", svg_content)

        # Create QIcon from modified SVG content using QSvgRenderer
        svg_bytes = QByteArray(svg_content.encode("utf-8"))
        renderer = QSvgRenderer(svg_bytes)
        if renderer.isValid():
            pixmap = QPixmap(16, 16)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            icon = QIcon(pixmap)
            return icon
        else:
            # Fallback to loading SVG as-is
            return QIcon(svg_path)
    except Exception:
        # Fallback to loading SVG as-is
        return QIcon(svg_path)


class SegmentItem(QFrame):
    """Individual segment item widget"""

    def __init__(
        self,
        segment: Dict[str, Any],
        _label: Optional[Dict[str, Any]],
        all_labels: List[Dict[str, Any]],
        is_selected: bool,
        on_click_callback=None,
        on_hover_callback=None,
        on_edit_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self.segment = segment
        self.is_selected = is_selected
        self.all_labels = all_labels
        self.segment_id = segment.get("id")  # Store segment ID for click handling
        self.on_click_callback = on_click_callback
        self.on_hover_callback = on_hover_callback
        self.on_edit_callback = on_edit_callback

        # Enable mouse tracking for hover detection
        self.setMouseTracking(True)

        self.setup_ui()
        self.update_style()

    def setup_ui(self):
        """Set up the segment item UI"""
        self.setObjectName("SegmentItem")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Get label info
        label_name = "Unlabeled"
        label_color = "#ffffff"  # Default white color
        if self.segment.get("labelId"):
            label = next(
                (
                    label_item
                    for label_item in self.all_labels
                    if label_item.get("id") == self.segment.get("labelId")
                ),
                None,
            )
            if label:
                label_name = label.get("name", "Unlabeled")
                label_color = label.get("color", "#ffffff")

        # Color circle indicator
        color_circle = QLabel()
        color_circle.setFixedSize(12, 12)
        color_circle.setStyleSheet(f"""
            background-color: {label_color};
            border-radius: 6px;
            border: 1px solid {ITEM_BORDER};
        """)
        layout.addWidget(color_circle)

        # Label name text (no background box)
        self.name_label = QLabel(label_name)
        self.name_label.setStyleSheet(f"""
            font-size: 14px; 
            color: {TEXT_COLOR};
            background-color: transparent;
            border: none;
            padding: 0px;
        """)
        layout.addWidget(self.name_label)
        layout.addStretch()

        # Edit button
        edit_icon_path = os.path.join(_ICON_DIR, "edit-3-svgrepo-com.svg")
        edit_icon = (
            create_colored_svg_icon(edit_icon_path, "#3b82f6")
            if os.path.exists(edit_icon_path)
            else None
        )

        self.edit_button = QPushButton()
        self.edit_button.setFixedSize(28, 28)
        self.edit_button.setToolTip("Edit segment")
        if edit_icon:
            self.edit_button.setIcon(edit_icon)
            self.edit_button.setIconSize(QSize(16, 16))
        else:
            # Fallback to emoji if icon file not found
            self.edit_button.setText("✏️")
        self.edit_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 14px;
                color: #3b82f6;
            }
            QPushButton:hover {
                background-color: #2b303b;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.edit_button)

        # Delete button
        self.delete_button = QPushButton("🗑")
        self.delete_button.setFixedSize(28, 28)
        self.delete_button.setToolTip("Delete segment")
        self.delete_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 14px;
                color: #ef4444;
            }
            QPushButton:hover {
                background-color: #2b303b;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.delete_button)

    def enterEvent(self, event):
        """Handle mouse enter event (hover start)"""
        if self.on_hover_callback and self.segment_id:
            self.on_hover_callback(self.segment_id, True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave event (hover end)"""
        if self.on_hover_callback and self.segment_id:
            self.on_hover_callback(self.segment_id, False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle clicks on the segment item"""
        if event.button() == Qt.LeftButton:
            # Don't propagate if clicking on buttons
            child = self.childAt(event.pos())
            if child and isinstance(child, QPushButton):
                return

            # Call the click callback if provided
            if self.on_click_callback and self.segment_id:
                self.on_click_callback(self.segment_id)
        super().mousePressEvent(event)

    def update_style(self):
        """Update the item style based on selection state"""
        if self.is_selected:
            style = f"""
                QFrame#SegmentItem {{
                    background-color: rgba(59, 130, 246, 0.1);
                    border: 1px solid {PRIMARY_COLOR};
                    border-radius: 8px;
                }}
            """
        else:
            style = f"""
                QFrame#SegmentItem {{
                    background-color: {ITEM_BG};
                    border: 1px solid {ITEM_BORDER};
                    border-radius: 8px;
                }}
                QFrame#SegmentItem:hover {{
                    background-color: {BACKGROUND_MAIN};
                }}
            """
        self.setStyleSheet(style)


class SegmentsPanel(QWidget):
    """Segments panel widget matching the React SegmentsPanel component"""

    # Signals
    segment_selected = Signal(
        str
    )  # Emits segment_id when segment is selected (or empty string to deselect)
    segment_deleted = Signal(str)  # Emits segment_id when segment is deleted
    visibility_toggled = Signal(str)  # Emits segment_id when visibility is toggled
    label_updated = Signal(
        str, str
    )  # Emits (segment_id, label_id) when label is updated
    segment_hovered = Signal(
        str, bool
    )  # Emits (segment_id, is_hovered) when segment is hovered
    segment_edit_requested = Signal(str)  # Emits segment_id when edit button is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)  # w-80 = 320px
        self.setObjectName("SegmentsPanel")

        # Internal state
        self.segments: List[Dict[str, Any]] = []
        self.labels: List[Dict[str, Any]] = []
        self.selected_segment_id: Optional[str] = None
        self.search_query = ""

        # Apply styles
        self.apply_styles()

        self.setup_ui()

    def setup_ui(self):
        """Set up the UI layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header section
        header_widget = QWidget()
        header_widget.setObjectName("SegmentsPanelHeader")
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(16, 16, 16, 16)
        header_layout.setSpacing(12)

        # Title
        self.title_label = QLabel("Segments (0)")
        self.title_label.setStyleSheet(
            "font-size: 14px; font-weight: 600; color: #ffffff;"
        )
        header_layout.addWidget(self.title_label)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search segments...")
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {BACKGROUND_MAIN};
                border: 1px solid {ITEM_BORDER};
                color: {TEXT_COLOR};
                padding: 6px;
                border-radius: 4px;
                font-size: 14px;
            }}
        """)
        self.search_input.textChanged.connect(self.on_search_changed)
        header_layout.addWidget(self.search_input)

        main_layout.addWidget(header_widget)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"color: {ITEM_BORDER};")
        divider.setFixedHeight(1)
        main_layout.addWidget(divider)

        # Scroll area for segments
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        # Scroll content widget
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("SegmentsPanelContent")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(8, 8, 8, 8)
        self.scroll_layout.setSpacing(4)
        self.scroll_layout.addStretch()

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, 1)

    def apply_styles(self):
        """Apply styles to the panel"""
        self.setStyleSheet(f"""
            QWidget#SegmentsPanel {{
                background-color: {ITEM_BG};
                border-left: 1px solid {ITEM_BORDER};
            }}
            QWidget#SegmentsPanelHeader {{
                background-color: {ITEM_BG};
                border-bottom: 1px solid {ITEM_BORDER};
            }}
            QWidget#SegmentsPanelContent {{
                background-color: {ITEM_BG};
            }}
        """)

    def on_search_changed(self, text: str):
        """Handle search query change"""
        self.search_query = text
        self.update_segments_display()

    def set_segments(self, segments: List[Dict[str, Any]]):
        """Set the segments list"""
        self.segments = segments
        self.update_title()
        self.update_segments_display()

    def set_labels(self, labels: List[Dict[str, Any]]):
        """Set the labels list"""
        self.labels = labels
        self.update_segments_display()

    def set_selected_segment(self, segment_id: Optional[str]):
        """Set the selected segment"""
        self.selected_segment_id = segment_id
        self.update_segments_display()

    def update_title(self):
        """Update the title with segment count"""
        count = len(self.segments)
        self.title_label.setText(f"Segments ({count})")

    def get_label_for_segment(self, segment: Dict[str, Any]):
        """Get the label for a segment"""
        label_id = segment.get("labelId")
        if not label_id:
            return None
        return next(
            (
                label_item
                for label_item in self.labels
                if label_item.get("id") == label_id
            ),
            None,
        )

    def filter_segments(self):
        """Filter segments based on search query"""
        if not self.search_query:
            return self.segments

        query = self.search_query.lower()
        filtered = []
        for segment in self.segments:
            label = self.get_label_for_segment(segment)
            label_name = label.get("name", "").lower() if label else ""
            segment_id = segment.get("id", "").lower()

            if query in label_name or query in segment_id:
                filtered.append(segment)

        return filtered

    def update_segments_display(self):
        """Update the displayed segments"""
        # Clear existing items
        while self.scroll_layout.count() > 1:  # Keep the stretch
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get filtered segments
        filtered_segments = self.filter_segments()

        if not filtered_segments:
            # Show empty state
            empty_label = QLabel("No segments yet")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("""
                color: #9ca3af;
                font-size: 14px;
                padding: 32px;
            """)
            self.scroll_layout.insertWidget(0, empty_label)
            return

        # Add segment items
        for segment in filtered_segments:
            label = self.get_label_for_segment(segment)
            is_selected = self.selected_segment_id == segment.get("id")

            item = SegmentItem(
                segment,
                label,
                self.labels,
                is_selected,
                on_click_callback=self.on_segment_item_clicked,
                on_hover_callback=self.on_segment_item_hovered,
                on_edit_callback=self.on_edit,
            )

            # Connect button signals
            seg_id = segment.get("id")
            item.edit_button.clicked.connect(
                lambda checked, sid=seg_id: self.on_edit(sid)
            )
            item.delete_button.clicked.connect(
                lambda checked, sid=seg_id: self.on_delete(sid)
            )

            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, item)

    def on_segment_item_clicked(self, segment_id: str):
        """Handle segment item click"""
        if self.selected_segment_id == segment_id:
            self.selected_segment_id = None
            self.segment_selected.emit("")
        else:
            self.selected_segment_id = segment_id
            self.segment_selected.emit(segment_id)
        self.update_segments_display()

    def on_segment_item_hovered(self, segment_id: str, is_hovered: bool):
        """Handle segment item hover"""
        self.segment_hovered.emit(segment_id, is_hovered)

    def on_label_changed(self, index: int, segment_id: str, item: SegmentItem):
        """Handle label combo box change"""
        label_id = item.label_combo.itemData(index)
        if label_id:
            self.label_updated.emit(segment_id, label_id)

    def on_toggle_visibility(self, segment_id: str):
        """Handle visibility toggle"""
        self.visibility_toggled.emit(segment_id)

    def on_edit(self, segment_id: str):
        """Handle segment edit request"""
        self.segment_edit_requested.emit(segment_id)

    def on_delete(self, segment_id: str):
        """Handle segment deletion"""
        self.segment_deleted.emit(segment_id)
