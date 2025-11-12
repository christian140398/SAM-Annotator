"""
TopBar component for SAM Annotator
Replicates the functionality of the React TopBar component
"""

from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QColorDialog,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPainter, QIcon, QPixmap
from PySide6.QtWidgets import QStyledItemDelegate
from frontend.theme import (
    get_topbar_style,
)


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


class TopBar(QWidget):
    """Top bar widget matching the React TopBar component"""

    # Signals for callbacks
    label_selected = Signal(str)  # Emits label_id when label is selected
    label_added = Signal(object)  # Emits Label object when label is added
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

        # Right side: (removed label selector, import, and export buttons)
        # Keep right_layout for potential future additions but leave it empty for now
        right_layout = QHBoxLayout()
        right_layout.setSpacing(12)

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
        self.update_label_combo()

    def update_label_combo(self):
        """Update the label combo box with current labels"""
        self.label_combo.blockSignals(True)
        self.label_combo.clear()

        # Update delegate with current labels
        self.combo_delegate.set_labels(self.labels)

        for label in self.labels:
            # Item text is just the name (delegate will handle color rendering)
            self.label_combo.addItem(label.name, label.id)

        self.label_combo.blockSignals(False)

        # Restore selection
        if self.selected_label_id:
            index = next(
                (
                    i
                    for i, label in enumerate(self.labels)
                    if label.id == self.selected_label_id
                ),
                -1,
            )
            if index >= 0:
                self.label_combo.setCurrentIndex(index)

    def on_label_changed(self, index: int):
        """Handle label selection change"""
        if index >= 0 and index < len(self.labels):
            label_id = self.labels[index].id
            self.selected_label_id = label_id
            self.label_selected.emit(label_id)

    def set_selected_label(self, label_id: Optional[str]):
        """Set the currently selected label"""
        self.selected_label_id = label_id
        if label_id:
            index = next(
                (i for i, label in enumerate(self.labels) if label.id == label_id), -1
            )
            if index >= 0:
                self.label_combo.blockSignals(True)
                self.label_combo.setCurrentIndex(index)
                self.label_combo.blockSignals(False)
        else:
            self.label_combo.blockSignals(True)
            self.label_combo.setCurrentIndex(-1)
            self.label_combo.blockSignals(False)

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
                self.update_label_combo()
                self.label_combo.setCurrentIndex(len(self.labels) - 1)
                self.label_added.emit(label_data)
