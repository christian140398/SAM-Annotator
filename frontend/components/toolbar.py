"""
Toolbar component for SAM Annotator
Replicates the functionality of the React Toolbar component
"""
import os
import re
from typing import Literal, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal, QSize, QByteArray
from PySide6.QtGui import QColor, QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer
from frontend.theme import (
    ITEM_BG, ITEM_BORDER, TEXT_COLOR, TOPBAR_TEXT_MUTED,
    PRIMARY_COLOR, PRIMARY_HOVER, PRIMARY_PRESSED
)
import config

# Tool type definition
Tool = Literal["select", "point", "box", "brush", "erase", "pan", "segment", "bbox"]

# Get the directory of this file to resolve icon paths
_ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")


def create_white_svg_icon(svg_path: str) -> QIcon:
    """Load SVG file and create a white-colored version as QIcon"""
    try:
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Replace stroke colors with white
        # Match stroke:#... in CSS styles (e.g., stroke:#020202;)
        svg_content = re.sub(r'stroke:#[0-9a-fA-F]{3,6}', 'stroke:#ffffff', svg_content)
        # Match stroke="..." attributes
        svg_content = re.sub(r'stroke="[^"]*"', 'stroke="#ffffff"', svg_content)
        
        # Replace fill colors with white (except for fill="none" or fill:none)
        # Match fill:#... in CSS styles (but not fill:none)
        svg_content = re.sub(r'fill:#([0-9a-fA-F]{3,6})(?!\s*none)', 'fill:#ffffff', svg_content)
        # Match fill="..." attributes (but not fill="none")
        svg_content = re.sub(r'fill="(?!none)[^"]*"', 'fill="#ffffff"', svg_content)
        
        # Also replace any color: attributes in CSS
        svg_content = re.sub(r'color:#[0-9a-fA-F]{3,6}', 'color:#ffffff', svg_content)
        
        # Create QIcon from modified SVG content using QSvgRenderer
        svg_bytes = QByteArray(svg_content.encode('utf-8'))
        renderer = QSvgRenderer(svg_bytes)
        if renderer.isValid():
            pixmap = QPixmap(24, 24)
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

# Base tool definitions with icons (SVG file paths or emoji fallback)
BASE_TOOLS = [
    {"id": "segment", "icon": os.path.join(_ICON_DIR, "line-segments-fill-svgrepo-com.svg"), "label": "Segment", "shortcut": "A"},
    {"id": "brush", "icon": os.path.join(_ICON_DIR, "draw-svgrepo-com.svg"), "label": "Brush", "shortcut": "S"},
    {"id": "pan", "icon": os.path.join(_ICON_DIR, "pan-cursor-svgrepo-com.svg"), "label": "Pan", "shortcut": "Space"},
    {"id": "fit_bbox", "icon": os.path.join(_ICON_DIR, "fit-to-screen-svgrepo-com.svg"), "label": "Fit to Bounding Box", "shortcut": "F"},
]

# Bounding box tool (only when BOUNDING_BOX_EXISTS is False)
BBOX_TOOL = {"id": "bbox", "icon": os.path.join(_ICON_DIR, "resize-svgrepo-com.svg"), "label": "Bounding Box", "shortcut": "B"}

def get_tools():
    """Get tools list based on configuration"""
    tools = BASE_TOOLS.copy()
    # Add bounding box tool if BOUNDING_BOX_EXISTS is False
    if not config.BOUNDING_BOX_EXISTS:
        # Insert bbox tool after brush tool
        tools.insert(2, BBOX_TOOL)
    return tools


class ToolButton(QPushButton):
    """Custom button for toolbar tools with active state"""
    def __init__(self, tool_id: str, icon: str, label: str, shortcut: str, parent=None):
        super().__init__(parent)
        self.tool_id = tool_id
        self.label = label
        self.shortcut = shortcut
        self._is_active = False
        self._icon_path = icon if os.path.isfile(icon) else None
        
        # Set button properties
        if self._icon_path:
            # Use SVG icon (colored white)
            icon_obj = create_white_svg_icon(self._icon_path)
            self.setIcon(icon_obj)
            self.setIconSize(QSize(24, 24))  # 24x24 icon size
        else:
            # Use emoji/text fallback
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
            # For SVG icons, we need to set icon color via stylesheet
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
            # Update icon color for SVG icons
            if self._icon_path:
                # Reload icon with white color for active state
                # Note: Qt doesn't directly support icon color changes via stylesheet for SVG
                # We'll keep the SVG as-is, or could use a colored variant
                pass
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
        
        # Set default active tool to "segment"
        self.set_active_tool("segment")
    
    def setup_ui(self):
        """Set up the UI layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 16)  # py-4 = 16px vertical padding
        layout.setSpacing(8)  # gap-2 = 8px spacing
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        
        # Get tools based on configuration
        tools = get_tools()
        
        # Create tool buttons
        for tool in tools:
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
        # fit_bbox is an action, not a tool - it doesn't change active tool state
        if tool_id == "fit_bbox":
            # Just emit the signal without changing active tool
            self.tool_changed.emit(tool_id)
        else:
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

