# File: RawRefinery/main.py

import sys
import platform
from PySide6.QtWidgets import QApplication
from RawRefinery.application.RawRefineryApp import RawRefineryApp  # Adjust import based on your structure

def main():
    """
    The primary entry point for the RawRefinery application.
    """
    app = QApplication(sys.argv)
    
    # Use platform.system() to check the OS and set metadata
    if platform.system() == "Darwin": 
        app.setApplicationDisplayName("Raw Refinery")
        app.setOrganizationName("Ryan Mueller")
        app.setDesktopFileName("com.rawrefinery.app") 

    window = RawRefineryApp()
    window.show()
    window.loading_popup()
    
    # Start the Qt event loop
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
