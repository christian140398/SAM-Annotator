"""
Toolbar component for SAM Annotator
Replicates the functionality of the React Toolbar component
"""
from typing import Literal, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from frontend.theme import (
    ITEM_BG, ITEM_BORDER, TEXT_COLOR, TOPBAR_TEXT_MUTED,
    PRIMARY_COLOR, PRIMARY_HOVER, PRIMARY_PRESSED
)

# Tool type definition
Tool = Literal["select", "point", "box", "brush", "erase", "pan", "segment"]

# Tool definitions with icons (using Unicode/emoji symbols)
TOOLS = [
    {"id": "pan", "icon": "✋", "label": "Pan", "shortcut": "H"},
    {"id": "segment", "icon": "✂", "label": "Segment", "shortcut": "S"},
]


class ToolButton(QPushButton):
    """Custom button for toolbar tools with active state"""
    def __init__(self, tool_id: str, icon: str, label: str, shortcut: str, parent=None):
        super().__init__(parent)
        self.tool_id = tool_id
        self.label = label
        self.shortcut = shortcut
        self._is_active = False
        
        # Set button properties
        self.setText(icon)
        self.setFixedSize(48, 48)  # w-12 h-12 = 48px
        self.setToolTip(f"{label}\n{shortcut}")
        
        # Set initial style
        self.update_style()
    
    def set_active(self, active: bool):
        """Set the active state of the button"""
        self._is_active = active
        self.update_style()
    
    def update_style(self):
        """Update button style based on active state"""
        if self._is_active:
            # Active state: primary color background
            style = f"""
                QPushButton {{
                    background-color: {PRIMARY_COLOR};
                    color: #ffffff;
                    border: 1px solid {PRIMARY_COLOR};
                    border-radius: 4px;
                    font-size: 20px;
                }}
                QPushButton:hover {{
                    background-color: {PRIMARY_HOVER};
                }}
                QPushButton:pressed {{
                    background-color: {PRIMARY_PRESSED};
                }}
            """
        else:
            # Inactive state: transparent with hover
            style = f"""
                QPushButton {{
                    background-color: transparent;
                    color: {TOPBAR_TEXT_MUTED};
                    border: none;
                    border-radius: 4px;
                    font-size: 20px;
                }}
                QPushButton:hover {{
                    background-color: {ITEM_BG};
                    color: {TEXT_COLOR};
                }}
                QPushButton:pressed {{
                    background-color: {ITEM_BORDER};
                }}
            """
        self.setStyleSheet(style)


class Toolbar(QWidget):
    """Vertical toolbar widget matching the React Toolbar component"""
    
    # Signal emitted when tool changes
    tool_changed = Signal(str)  # Emits tool_id when tool is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(64)  # w-16 = 64px
        self.setObjectName("ToolbarWidget")
        
        # Current active tool
        self.active_tool: Optional[Tool] = None
        
        # Store tool buttons
        self.tool_buttons: dict[str, ToolButton] = {}
        
        # Ensure the widget itself has the background color using palette
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(ITEM_BG))
        self.setPalette(palette)
        
        # Apply styles
        self.apply_styles()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 16)  # py-4 = 16px vertical padding
        layout.setSpacing(8)  # gap-2 = 8px spacing
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        
        # Create tool buttons
        for tool in TOOLS:
            button = ToolButton(
                tool["id"],
                tool["icon"],
                tool["label"],
                tool["shortcut"],
                self
            )
            button.clicked.connect(lambda checked, tid=tool["id"]: self.on_tool_clicked(tid))
            self.tool_buttons[tool["id"]] = button
            layout.addWidget(button)
        
        # Add stretch to push buttons to top
        layout.addStretch()
    
    def apply_styles(self):
        """Apply styles to the toolbar"""
        self.setStyleSheet(f"""
            QWidget#ToolbarWidget {{
                background-color: {ITEM_BG};
                border: none;
            }}
            QWidget#ToolbarWidget QWidget {{
                background-color: {ITEM_BG};
            }}
        """)
    
    def on_tool_clicked(self, tool_id: str):
        """Handle tool button click"""
        self.set_active_tool(tool_id)
        self.tool_changed.emit(tool_id)
    
    def set_active_tool(self, tool_id: Optional[Tool]):
        """Set the active tool"""
        # Deactivate all buttons
        for button in self.tool_buttons.values():
            button.set_active(False)
        
        # Activate selected tool
        if tool_id and tool_id in self.tool_buttons:
            self.tool_buttons[tool_id].set_active(True)
            self.active_tool = tool_id
        else:
            self.active_tool = None

