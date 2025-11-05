"""
KeybindsBar component for SAM Annotator
Displays keyboard shortcuts in a bottom bar
"""
from typing import List, Dict
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from frontend.theme import (
    ITEM_BORDER, TEXT_COLOR, TOPBAR_TEXT_MUTED
)


class KbdWidget(QLabel):
    """Widget to display a keyboard key (like <kbd> in HTML)"""
    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self.key = key
        self.setText(key)  # Set the text to display
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(18)  # Small height for compact display
        self.update_style()
    
    def update_style(self):
        """Update the keyboard key style"""
        # Calculate width based on key length (minimum width for short keys)
        min_width = 24
        width_padding = 8  # Padding per side
        estimated_width = max(min_width, len(self.key) * 8 + width_padding * 2)
        
        self.setFixedWidth(estimated_width)
        
        self.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(31, 41, 55, 0.8);
                border: 1px solid {ITEM_BORDER};
                border-radius: 3px;
                color: {TEXT_COLOR};
                font-size: 11px;
                font-weight: 500;
                padding: 2px 6px;
            }}
        """)


class KeybindItem(QWidget):
    """A single keybind item (key + label)"""
    def __init__(self, key: str, label: str, parent=None):
        super().__init__(parent)
        self.setup_ui(key, label)
    
    def setup_ui(self, key: str, label: str):
        """Set up the keybind item UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)  # gap-1.5 equivalent (6px)
        
        # Keyboard key widget
        kbd_widget = KbdWidget(key)
        layout.addWidget(kbd_widget)
        
        # Label text
        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"""
            color: {TOPBAR_TEXT_MUTED};
            font-size: 11px;
            background-color: transparent;
        """)
        layout.addWidget(label_widget)
        
        layout.addStretch()


class KeybindsBar(QWidget):
    """Keybinds bar widget displayed at the bottom"""
    
    # Default keybinds based on React component and actual implementation
    DEFAULT_KEYBINDS = [
        {"key": "A", "label": "Segment tool"},
        {"key": "S", "label": "Brush tool"},
        {"key": "Space", "label": "Pan tool"},
        {"key": "F", "label": "Fit to bounding box"},
        {"key": "E", "label": "Finalize segment"},
        {"key": "Z", "label": "Undo"},
        {"key": "q", "label": "Quit"},
        {"key": "Ctrl+S", "label": "Save & next image"},
        {"key": "M", "label": "Skip image"},
        {"key": "Scroll", "label": "Zoom"},
    ]
    
    def __init__(self, parent=None, keybinds: List[Dict[str, str]] = None):
        super().__init__(parent)
        self.setObjectName("KeybindsBar")
        
        # Use provided keybinds or defaults
        self.keybinds = keybinds if keybinds is not None else self.DEFAULT_KEYBINDS.copy()
        self.labels: List[Dict[str, str]] = []  # List of label dicts with 'id', 'name', 'color'
        
        # Apply styles
        self.apply_styles()
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 8, 16, 8)  # Add vertical padding
        main_layout.setSpacing(8)  # Space between keybinds and labels rows
        
        # Keybinds row
        keybinds_layout = QHBoxLayout()
        keybinds_layout.setContentsMargins(0, 0, 0, 0)
        keybinds_layout.setSpacing(24)  # gap-6 equivalent (24px)
        keybinds_layout.setAlignment(Qt.AlignCenter)
        
        # Add keybind items
        for bind in self.keybinds:
            item = KeybindItem(bind["key"], bind["label"])
            keybinds_layout.addWidget(item)
        
        keybinds_layout.addStretch()
        main_layout.addLayout(keybinds_layout)
        
        # Labels row
        self.labels_layout = QHBoxLayout()
        self.labels_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_layout.setSpacing(16)  # Space between label items
        self.labels_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(self.labels_layout)
        
        # Update labels display
        self.update_labels_display()
        
        # Adjust height dynamically
        self.adjust_height()
    
    def update_labels_display(self):
        """Update the labels display with numbers (keybinds)"""
        # Clear existing label items
        while self.labels_layout.count():
            item = self.labels_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # If no labels, show error message
        if not self.labels:
            error_label = QLabel("No labels found. Create label.txt in project root with one label per line")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet(f"""
                color: #ef4444;
                font-size: 11px;
                background-color: transparent;
                padding: 4px 8px;
            """)
            self.labels_layout.addWidget(error_label)
            self.labels_layout.addStretch()
            self.adjust_height()
            return
        
        # Add labels with their keybind numbers (all labels)
        for i, label in enumerate(self.labels):
            keybind_key = str(i + 1)
            label_name = label.get('name', '')
            # Create a KeybindItem to match the style of other keybinds
            item = KeybindItem(keybind_key, label_name)
            self.labels_layout.addWidget(item)
        
        self.labels_layout.addStretch()
        self.adjust_height()
    
    def adjust_height(self):
        """Adjust the height based on content"""
        # Base height for keybinds row (~28px) + labels row (~20px) + padding (16px)
        base_height = 64
        # Add extra height if labels exist
        if self.labels:
            base_height += 4  # Extra spacing for labels
        self.setFixedHeight(base_height)
    
    def apply_styles(self):
        """Apply styles to the keybinds bar"""
        # Use semi-transparent background with border-top
        # Note: PySide6 doesn't support backdrop-blur directly,
        # but we can use a semi-transparent background for similar effect
        self.setStyleSheet(f"""
            QWidget#KeybindsBar {{
                background-color: rgba(29, 34, 42, 0.5);
                border-top: 1px solid {ITEM_BORDER};
            }}
        """)
    
    def set_keybinds(self, keybinds: List[Dict[str, str]]):
        """Update the keybinds displayed"""
        self.keybinds = keybinds
        
        # Clear and rebuild the entire UI
        main_layout = self.layout()
        if main_layout:
            # Clear all items
            while main_layout.count():
                item = main_layout.takeAt(0)
                if item.layout():
                    # Clear layout items
                    layout = item.layout()
                    while layout.count():
                        sub_item = layout.takeAt(0)
                        if sub_item.widget():
                            sub_item.widget().deleteLater()
                    # Delete the layout itself
                    item.layout().deleteLater()
                elif item.widget():
                    item.widget().deleteLater()
        
        # Rebuild UI
        self.setup_ui()
    
    def set_labels(self, labels: List[Dict[str, str]]):
        """Update the labels displayed"""
        self.labels = labels
        self.update_labels_display()

