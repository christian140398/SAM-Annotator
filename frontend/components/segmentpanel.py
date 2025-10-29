"""
SegmentsPanel component for SAM Annotator
Replicates the functionality of the React SegmentsPanel component
"""
from typing import Optional, List, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QStyledItemDelegate
from frontend.theme import (
    ITEM_BG, ITEM_BORDER, TEXT_COLOR,
    BACKGROUND_MAIN, PRIMARY_COLOR
)


class LabelComboDelegate(QStyledItemDelegate):
    """Custom delegate for rendering label items with color dots in combo box"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels = []
    
    def set_labels(self, labels: List[Dict[str, Any]]):
        """Set the labels for rendering"""
        self.labels = labels
    
    def paint(self, painter, option, index):
        """Paint the item with color dot"""
        if index.row() >= len(self.labels):
            super().paint(painter, option, index)
            return
        
        label = self.labels[index.row()]
        
        # Draw selection background
        if option.state & option.State_Selected:
            painter.fillRect(option.rect, QColor(ITEM_BORDER))
        else:
            painter.fillRect(option.rect, QColor(ITEM_BG))
        
        # Draw color dot
        dot_size = 12
        dot_x = option.rect.left() + 8
        dot_y = option.rect.center().y() - dot_size // 2
        
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(label.get('color', '#ffffff'))
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(dot_x, dot_y, dot_size, dot_size)
        
        # Draw text
        text_x = dot_x + dot_size + 8
        text_y = option.rect.center().y() + 5
        painter.setPen(QColor(TEXT_COLOR))
        painter.drawText(text_x, text_y, label.get('name', ''))
    
    def sizeHint(self, _option, _index):
        """Return size hint for the item"""
        return QSize(200, 24)


class SegmentItem(QFrame):
    """Individual segment item widget"""
    def __init__(self, segment: Dict[str, Any], _label: Optional[Dict[str, Any]], 
                 all_labels: List[Dict[str, Any]], is_selected: bool, 
                 on_click_callback=None, parent=None):
        super().__init__(parent)
        self.segment = segment
        self.is_selected = is_selected
        self.all_labels = all_labels
        self.segment_id = segment.get('id')  # Store segment ID for click handling
        self.on_click_callback = on_click_callback
        
        self.setup_ui()
        self.update_style()
    
    def setup_ui(self):
        """Set up the segment item UI"""
        self.setObjectName("SegmentItem")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Top row: Label color dot, ID, and action buttons
        top_layout = QHBoxLayout()
        
        # Left side: Label dot and ID
        left_layout = QHBoxLayout()
        left_layout.setSpacing(8)
        
        if self.segment.get('labelId'):
            label = next((l for l in self.all_labels if l.get('id') == self.segment.get('labelId')), None)
            if label:
                # Color dot
                dot_label = QLabel()
                dot_label.setFixedSize(12, 12)
                dot_label.setStyleSheet(f"""
                    background-color: {label.get('color', '#ffffff')};
                    border-radius: 6px;
                """)
                left_layout.addWidget(dot_label)
        
        # Segment ID
        segment_id = self.segment.get('id', '')[:8]  # First 8 chars
        id_label = QLabel(f"ID: {segment_id}")
        id_label.setObjectName("muted")
        id_label.setStyleSheet("font-size: 11px; color: #9ca3af;")
        left_layout.addWidget(id_label)
        left_layout.addStretch()
        
        top_layout.addLayout(left_layout)
        
        # Right side: Action buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(4)
        
        # Visibility toggle button
        self.visibility_button = QPushButton("👁" if self.segment.get('visible', True) else "🚫")
        self.visibility_button.setFixedSize(28, 28)
        self.visibility_button.setToolTip("Toggle visibility")
        self.visibility_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2b303b;
                border-radius: 4px;
            }
        """)
        buttons_layout.addWidget(self.visibility_button)
        
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
        buttons_layout.addWidget(self.delete_button)
        
        top_layout.addLayout(buttons_layout)
        
        layout.addLayout(top_layout)
        
        # Label selector
        self.label_combo = QComboBox()
        self.label_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {ITEM_BG};
                border: 1px solid {ITEM_BORDER};
                color: {TEXT_COLOR};
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                height: 24px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        """)
        
        # Set custom delegate
        combo_delegate = LabelComboDelegate(self.label_combo)
        combo_delegate.set_labels(self.all_labels)
        self.label_combo.setItemDelegate(combo_delegate)
        
        # Add labels to combo
        self.label_combo.addItem("Assign label", None)
        for label in self.all_labels:
            self.label_combo.addItem(label.get('name', ''), label.get('id'))
        
        # Select current label
        current_label_id = self.segment.get('labelId')
        if current_label_id:
            for i in range(self.label_combo.count()):
                if self.label_combo.itemData(i) == current_label_id:
                    self.label_combo.setCurrentIndex(i)
                    break
        
        layout.addWidget(self.label_combo)
        
        # Area information
        area = self.segment.get('area', 0)
        area_label = QLabel(f"Area: {area:,} px²")
        area_label.setObjectName("muted")
        area_label.setStyleSheet("font-size: 11px; color: #9ca3af;")
        layout.addWidget(area_label)
        
        layout.addStretch()
        
        # Enable mouse tracking for click handling
        self.setMouseTracking(True)
    
    def mousePressEvent(self, event):
        """Handle clicks on the segment item"""
        if event.button() == Qt.LeftButton:
            # Don't propagate if clicking on buttons or combo box
            child = self.childAt(event.pos())
            if child and (isinstance(child, QPushButton) or isinstance(child, QComboBox)):
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
    segment_selected = Signal(str)  # Emits segment_id when segment is selected (or empty string to deselect)
    segment_deleted = Signal(str)  # Emits segment_id when segment is deleted
    visibility_toggled = Signal(str)  # Emits segment_id when visibility is toggled
    label_updated = Signal(str, str)  # Emits (segment_id, label_id) when label is updated
    
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
        self.title_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #ffffff;")
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
        label_id = segment.get('labelId')
        if not label_id:
            return None
        return next((l for l in self.labels if l.get('id') == label_id), None)
    
    def filter_segments(self):
        """Filter segments based on search query"""
        if not self.search_query:
            return self.segments
        
        query = self.search_query.lower()
        filtered = []
        for segment in self.segments:
            label = self.get_label_for_segment(segment)
            label_name = label.get('name', '').lower() if label else ''
            segment_id = segment.get('id', '').lower()
            
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
            is_selected = self.selected_segment_id == segment.get('id')
            
            item = SegmentItem(segment, label, self.labels, is_selected, 
                             on_click_callback=self.on_segment_item_clicked)
            
            # Connect signals - use default parameters to capture loop variables correctly
            seg_id = segment.get('id')
            item.label_combo.currentIndexChanged.connect(
                lambda idx, sid=seg_id, itm=item: self.on_label_changed(idx, sid, itm)
            )
            item.visibility_button.clicked.connect(
                lambda checked, sid=seg_id: self.on_toggle_visibility(sid)
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
    
    def on_label_changed(self, index: int, segment_id: str, item: SegmentItem):
        """Handle label combo box change"""
        label_id = item.label_combo.itemData(index)
        if label_id:
            self.label_updated.emit(segment_id, label_id)
    
    def on_toggle_visibility(self, segment_id: str):
        """Handle visibility toggle"""
        self.visibility_toggled.emit(segment_id)
    
    def on_delete(self, segment_id: str):
        """Handle segment deletion"""
        self.segment_deleted.emit(segment_id)

