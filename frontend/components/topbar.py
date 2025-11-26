"""
TopBar component for SAM Annotator
Replicates the functionality of the React TopBar component
"""

import os
import re
from typing import List, Optional

from PySide6.QtCore import QByteArray, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from frontend.theme import (
    ITEM_BG,
    ITEM_BORDER,
    TEXT_COLOR,
    get_topbar_style,
)

# Get the directory of this file to resolve icon paths
_ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")


def create_white_svg_icon(svg_path: str) -> QIcon:
    """Load SVG file and create a white-colored version as QIcon"""
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

        # Replace stroke colors with white
        # Match stroke:#... in CSS styles (e.g., stroke:#020202;)
        svg_content = re.sub(r"stroke:#[0-9a-fA-F]{3,6}", "stroke:#ffffff", svg_content)
        # Match stroke="..." attributes
        svg_content = re.sub(r'stroke="[^"]*"', 'stroke="#ffffff"', svg_content)

        # Replace fill colors with white (except for fill="none" or fill:none)
        # Match fill:#... in CSS styles (but not fill:none)
        svg_content = re.sub(
            r"fill:#([0-9a-fA-F]{3,6})(?!\s*none)", "fill:#ffffff", svg_content
        )
        # Match fill="..." attributes (but not fill="none")
        svg_content = re.sub(r'fill="(?!none)[^"]*"', 'fill="#ffffff"', svg_content)

        # Also replace any color: attributes in CSS
        svg_content = re.sub(r"color:#[0-9a-fA-F]{3,6}", "color:#ffffff", svg_content)

        # Create QIcon from modified SVG content using QSvgRenderer
        svg_bytes = QByteArray(svg_content.encode("utf-8"))
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


class LabelColorDelegate:
    """Helper class to represent a label with color"""

    def __init__(self, label_id: str, name: str, color: str):
        self.id = label_id
        self.name = name
        self.color = color


class LabelComboDelegate(QStyledItemDelegate):
    """Custom delegate for rendering label items with color dots in combo box"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels = []

    def set_labels(self, labels: List[LabelColorDelegate]):
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
            painter.fillRect(option.rect, QColor("#2b303b"))
        else:
            painter.fillRect(option.rect, QColor("#2c3e50"))

        # Draw color dot
        dot_size = 12
        dot_x = option.rect.left() + 8
        dot_y = option.rect.center().y() - dot_size // 2

        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(label.color)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(dot_x, dot_y, dot_size, dot_size)

        # Draw text
        text_x = dot_x + dot_size + 8
        text_y = option.rect.center().y() + 5
        painter.setPen(QColor("#ffffff"))
        painter.drawText(text_x, text_y, label.name)

    def sizeHint(self, _option, _index):
        """Return size hint for the item"""
        return QSize(150, 24)


class AddLabelDialog(QDialog):
    """Dialog for adding a new label"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Label")
        self.setModal(True)
        self.setMinimumWidth(350)

        # Dialog layout
        layout = QVBoxLayout(self)

        # Name input
        name_layout = QVBoxLayout()
        name_label = QLabel("Name")
        name_label.setObjectName("muted")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Label name")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        # Color input
        color_layout = QVBoxLayout()
        color_label = QLabel("Color")
        color_label.setObjectName("muted")
        color_input_layout = QHBoxLayout()

        self.color_picker = QPushButton()
        self.color_picker.setFixedSize(64, 40)
        self.color_value = QLineEdit()
        self.color_value.setPlaceholderText("#00D9FF")
        self.current_color = "#00D9FF"
        self.update_color_button()

        self.color_picker.clicked.connect(self.open_color_dialog)
        self.color_value.textChanged.connect(self.on_color_text_changed)

        color_input_layout.addWidget(self.color_picker)
        color_input_layout.addWidget(self.color_value)
        color_layout.addWidget(color_label)
        color_layout.addLayout(color_input_layout)
        layout.addLayout(color_layout)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Style the dialog
        self.setStyleSheet("""
            QDialog {{
                background-color: #2c3e50;
                color: #ffffff;
            }}
            QLabel {{
                color: #ffffff;
                background-color: transparent;
            }}
            QLabel#muted {{
                color: #9ca3af;
            }}
            QLineEdit {{
                background-color: #1a1f2e;
                border: 1px solid #2b303b;
                color: #ffffff;
                padding: 6px;
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: #1a1f2e;
                border: 1px solid #2b303b;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #2c3e50;
            }}
            QDialogButtonBox QPushButton {{
                min-width: 80px;
                padding: 6px 16px;
            }}
        """)

    def update_color_button(self):
        """Update the color picker button appearance"""
        pixmap = QPixmap(64, 40)
        pixmap.fill(QColor(self.current_color))
        self.color_picker.setIcon(QIcon(pixmap))
        self.color_value.setText(self.current_color)

    def open_color_dialog(self):
        """Open Qt's color picker dialog"""
        color = QColorDialog.getColor(QColor(self.current_color), self, "Choose Color")
        if color.isValid():
            self.current_color = color.name()
            self.update_color_button()

    def on_color_text_changed(self, text: str):
        """Handle manual color text input"""
        if QColor(text).isValid():
            self.current_color = text
            self.update_color_button()

    def get_label_data(self):
        """Get the label data from the dialog"""
        name = self.name_input.text().strip()
        color = self.current_color
        return name, color


class LabelColorDialog(QDialog):
    """Dialog for viewing and managing label colors"""

    # Signal emitted when a label color is changed
    color_changed = Signal(str, str)  # Emits (label_id, new_color)

    def __init__(self, labels: List[LabelColorDelegate], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label Colors")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        # Store original labels list for reference
        self.labels = labels

        # Dialog layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        title_label = QLabel("Label Colors")
        title_label.setStyleSheet("font-size: 18px; font-weight: 600; color: #ffffff;")
        layout.addWidget(title_label)

        # Scroll area for labels
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {ITEM_BG};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {ITEM_BORDER};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #3b4048;
            }}
        """)

        # Container for label items
        labels_container = QWidget()
        labels_layout = QVBoxLayout(labels_container)
        labels_layout.setContentsMargins(0, 0, 0, 0)
        labels_layout.setSpacing(8)

        # Add label items
        self.label_items = []
        self.color_boxes = {}  # Store color boxes by label_id for updates
        for label in labels:
            label_item = self.create_label_item(label)
            labels_layout.addWidget(label_item)
            self.label_items.append(label_item)

        # Add stretch at the end
        labels_layout.addStretch()

        scroll_area.setWidget(labels_container)
        layout.addWidget(scroll_area)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Style the dialog
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {ITEM_BG};
                color: {TEXT_COLOR};
            }}
            QLabel {{
                color: {TEXT_COLOR};
                background-color: transparent;
            }}
            QPushButton {{
                background-color: {ITEM_BG};
                border: 1px solid {ITEM_BORDER};
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {ITEM_BORDER};
            }}
            QDialogButtonBox QPushButton {{
                min-width: 80px;
                padding: 6px 16px;
            }}
        """)

    def create_label_item(self, label: LabelColorDelegate) -> QFrame:
        """Create a label item widget with name and color box"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {ITEM_BG};
                border: 1px solid {ITEM_BORDER};
                border-radius: 4px;
                padding: 8px;
            }}
        """)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Color box - make it a clickable button
        color_box = QPushButton()
        color_box.setFixedSize(32, 32)
        color_box.setCursor(Qt.PointingHandCursor)
        color_box.setToolTip("Click to change color")
        self.update_color_box(color_box, label.color)

        # Store reference to color box and hex label for updates
        self.color_boxes[label.id] = {
            "button": color_box,
            "label": label,
            "hex_label": None,  # Will be set below
        }

        # Connect click handler
        color_box.clicked.connect(
            lambda checked, lid=label.id: self.on_color_box_clicked(lid)
        )

        # Label name
        name_label = QLabel(label.name)
        name_label.setStyleSheet("font-size: 14px; color: #ffffff;")

        # Color hex value
        color_hex_label = QLabel(label.color)
        color_hex_label.setStyleSheet("font-size: 12px; color: #9ca3af;")
        color_hex_label.setObjectName("muted")

        # Store hex label reference
        self.color_boxes[label.id]["hex_label"] = color_hex_label

        layout.addWidget(color_box)
        layout.addWidget(name_label)
        layout.addStretch()
        layout.addWidget(color_hex_label)

        return frame

    def update_color_box(self, color_box: QPushButton, color: str):
        """Update the appearance of a color box button"""
        color_box.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border: 1px solid {ITEM_BORDER};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border: 2px solid {TEXT_COLOR};
            }}
        """)

    def on_color_box_clicked(self, label_id: str):
        """Handle color box click - open color picker"""
        if label_id not in self.color_boxes:
            return

        label_info = self.color_boxes[label_id]
        label = label_info["label"]
        current_color = QColor(label.color)

        # Open color picker dialog
        new_color = QColorDialog.getColor(
            current_color, self, f"Choose color for {label.name}"
        )

        if new_color.isValid() and new_color != current_color:
            # Update label color
            new_color_hex = new_color.name()
            label.color = new_color_hex

            # Update UI
            self.update_color_box(label_info["button"], new_color_hex)
            label_info["hex_label"].setText(new_color_hex)

            # Emit signal to notify parent
            self.color_changed.emit(label_id, new_color_hex)


class TopBar(QWidget):
    """Top bar widget matching the React TopBar component"""

    # Signals for callbacks
    label_selected = Signal(str)  # Emits label_id when label is selected
    label_added = Signal(object)  # Emits Label object when label is added
    label_color_changed = Signal(
        str, str
    )  # Emits (label_id, new_color) when label color is changed
    import_requested = Signal()  # Emits when Import button is clicked
    export_requested = Signal()  # Emits when Export button is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)  # h-14 equivalent (14 * 4px = 56px)

        # Set object name so stylesheet can target this widget
        self.setObjectName("TopBarWidget")

        # Internal state
        self.image_name: Optional[str] = None
        self.labels: List[LabelColorDelegate] = []
        self.selected_label_id: Optional[str] = None

        # Ensure the widget itself has the background color using palette FIRST
        # This prevents the parent background from showing through
        from frontend.theme import ITEM_BG

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(ITEM_BG))
        self.setPalette(palette)

        # Apply styles after palette to override any parent styles
        self.setStyleSheet(get_topbar_style())

        self.setup_ui()

    def setup_ui(self):
        """Set up the UI layout"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 0, 16, 0)  # px-4 equivalent
        main_layout.setSpacing(12)

        # Left side: Title and image name
        left_layout = QHBoxLayout()
        left_layout.setSpacing(16)  # gap-4 equivalent

        self.title_label = QLabel("SAM Annotator")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600;")

        self.image_name_label = QLabel()
        self.image_name_label.setObjectName("muted")
        self.image_name_label.setStyleSheet("font-size: 14px;")
        self.image_name_label.hide()  # Hidden by default

        left_layout.addWidget(self.title_label)
        left_layout.addWidget(self.image_name_label)
        left_layout.addStretch()

        # Right side: Settings button
        right_layout = QHBoxLayout()
        right_layout.setSpacing(12)

        # Settings button
        settings_icon_path = os.path.join(_ICON_DIR, "settings-svgrepo-com.svg")
        self.settings_button = QPushButton()
        if os.path.isfile(settings_icon_path):
            icon = create_white_svg_icon(settings_icon_path)
            self.settings_button.setIcon(icon)
            self.settings_button.setIconSize(QSize(24, 24))
        self.settings_button.setFixedSize(40, 40)
        self.settings_button.setToolTip("Settings")
        self.settings_button.clicked.connect(self.show_settings_menu)
        right_layout.addWidget(self.settings_button)

        # Add layouts to main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def set_image_name(self, name: Optional[str]):
        """Set the current image name"""
        self.image_name = name
        if name:
            self.image_name_label.setText(name)
            self.image_name_label.show()
        else:
            self.image_name_label.hide()

    def set_labels(self, labels: List[dict]):
        """Set the available labels

        Args:
            labels: List of label dicts with 'id', 'name', and 'color' keys
        """
        self.labels = [
            LabelColorDelegate(
                label_item["id"], label_item["name"], label_item["color"]
            )
            for label_item in labels
        ]

    def set_selected_label(self, label_id: Optional[str]):
        """Set the currently selected label"""
        self.selected_label_id = label_id

    def open_add_label_dialog(self):
        """Open the add label dialog"""
        dialog = AddLabelDialog(self)
        if dialog.exec() == QDialog.Accepted:
            name, color = dialog.get_label_data()
            if name:
                from datetime import datetime

                label_id = str(int(datetime.now().timestamp() * 1000))
                label_data = {"id": label_id, "name": name, "color": color}
                self.labels.append(LabelColorDelegate(label_id, name, color))
                self.label_added.emit(label_data)

    def show_settings_menu(self):
        """Show the settings dropdown menu"""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {ITEM_BG};
                border: 1px solid {ITEM_BORDER};
                color: {TEXT_COLOR};
                padding: 4px;
                border-radius: 4px;
            }}
            QMenu::item {{
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {ITEM_BORDER};
            }}
        """)

        # Add "Label Color" menu item
        label_color_action = menu.addAction("Label Color")
        label_color_action.triggered.connect(self.on_label_color_clicked)

        # Position menu below the button, aligned to the right edge
        # Get button's bottom-right corner in global coordinates
        button_bottom_right = self.settings_button.mapToGlobal(
            QPoint(self.settings_button.width(), self.settings_button.height())
        )
        # Adjust x position so menu's right edge aligns with button's right edge
        # We need to estimate menu width or use sizeHint after adding items
        menu_width = menu.sizeHint().width()
        menu_x = button_bottom_right.x() - menu_width
        menu_y = button_bottom_right.y()
        menu.exec(QPoint(menu_x, menu_y))

    def on_label_color_clicked(self):
        """Handle label color settings click"""
        if not self.labels:
            # No labels to show
            return

        dialog = LabelColorDialog(self.labels, self)
        # Connect dialog's color_changed signal to TopBar's signal
        dialog.color_changed.connect(self.label_color_changed.emit)
        dialog.exec()
