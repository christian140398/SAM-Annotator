"""
Main entry point for the PySide6 frontend
"""
import sys
from PySide6.QtWidgets import QApplication
from frontend.main_window import MainWindow


def run_app():
    """Run the SAM Annotator application"""
    app = QApplication(sys.argv)
    
    # Set application style/theme here if needed
    
    window = MainWindow()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()

