"""
Theme and Style Constants
Centralized color and style definitions for the application
"""

# ===================== COLORS =====================
# Main colors
BACKGROUND_MAIN = "#161a22"
ITEM_BG = "#1d222a"
ITEM_BORDER = "#2b303b"
TEXT_COLOR = "#ffffff"


# Canvas/Image colors
CANVAS_BG = BACKGROUND_MAIN

# Topbar colors (uses ITEM_BG and ITEM_BORDER)
TOPBAR_BG = ITEM_BG  # Use item background color
TOPBAR_BORDER = ITEM_BORDER  # Use item border color
TOPBAR_TEXT_MUTED = "#9ca3af"  # muted foreground

# Primary/Accent colors for active states
PRIMARY_COLOR = "#3b82f6"  # Blue color for active tool
PRIMARY_HOVER = "#2563eb"
PRIMARY_PRESSED = "#1d4ed8"


def get_main_window_style():
    """Get style sheet for main window"""
    return f"""
        background-color: {BACKGROUND_MAIN};
    """


def get_canvas_style():
    """Get style sheet for image canvas"""
    return f"""
        background-color: {CANVAS_BG};
    """


def get_topbar_style():
    """Get style sheet for topbar"""
    return f"""
        QWidget#TopBarWidget {{
            background-color: {ITEM_BG};
            border: none;
        }}
        QWidget#TopBarWidget QLabel {{
            color: {TEXT_COLOR};
            background-color: transparent;
        }}
        QWidget#TopBarWidget QLabel#muted {{
            color: {TOPBAR_TEXT_MUTED};
        }}
        /* Buttons should use main background, not item background */
        QWidget#TopBarWidget QPushButton {{
            background-color: {BACKGROUND_MAIN};
            border: 1px solid {ITEM_BORDER};
            color: {TEXT_COLOR};
            padding: 4px 12px;
            border-radius: 4px;
            min-width: 48px;
        }}
        QWidget#TopBarWidget QPushButton:hover {{
            background-color: {ITEM_BG};
        }}
        QWidget#TopBarWidget QPushButton:pressed {{
            background-color: {ITEM_BORDER};
        }}
    """


def get_button_style():
    """Get style sheet for buttons (uses main background for buttons in topbar)"""
    return f"""
        QPushButton {{
            background-color: {BACKGROUND_MAIN};
            border: 1px solid {ITEM_BORDER};
            color: {TEXT_COLOR};
            padding: 4px 12px;
            border-radius: 4px;
        }}
        QPushButton:hover {{
            background-color: {ITEM_BG};
        }}
        QPushButton:pressed {{
            background-color: {ITEM_BORDER};
        }}
    """


def get_combo_box_style():
    """Get style sheet for combo box (label selector)"""
    return f"""
        QComboBox {{
            background-color: {ITEM_BG};
            border: 1px solid {ITEM_BORDER};
            color: {TEXT_COLOR};
            padding: 4px 8px;
            border-radius: 4px;
            min-width: 150px;
        }}
        QComboBox:hover {{
            border-color: {TOPBAR_TEXT_MUTED};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {ITEM_BG};
            border: 1px solid {ITEM_BORDER};
            color: {TEXT_COLOR};
            selection-background-color: {ITEM_BORDER};
        }}
    """
