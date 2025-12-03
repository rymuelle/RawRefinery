import sys
from pathlib import Path
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QFileDialog, QListWidget, QLabel,
    QSlider, QSpinBox, QHBoxLayout, QFormLayout,
    QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QMouseEvent
from PySide6.QtCore import Qt, QSize, Signal

import numpy as np
import torch

from RawRefinery.model.Cond_NAFNet import load_model
from RawRefinery.model.Restorer import make_sparse
from RawRefinery.application.viewing_utils import numpy_to_qimage_rgb, apply_gamma
from RawRefinery.application.ModelHandler import ModelHandler
from RawRefinery.application.dng_utils import to_dng


class RawRefineryApp(QMainWindow):
    """
    A PySide6 application for browsing and 'denoising' raw image files.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raw Refinery")
        self.setGeometry(200, 200, 200, 200)

        # --- Main Widget and Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- File List Panel (Left) ---
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.clicked.connect(self.open_folder_dialog)
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self.on_file_selected)
        self.left_layout.addWidget(self.btn_open_folder)
        self.left_layout.addWidget(self.file_list_widget)
        self.main_layout.addWidget(self.left_panel, 1) # 1/3 of the space

        # --- Preview and Control Panel (Right) ---
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # --- Horizontal Layout for Image Previews ---
        self.preview_row_widget = QWidget()
        self.preview_row_layout = QHBoxLayout(self.preview_row_widget)

        # Create your labels
        self.image_preview_label = QLabel("Preview")
        self.image_thumbnail_label = ClickableImageLabel("Thumbnail", proportional=True)

        # Common label styling (optional helper function for brevity)
        for label in [self.image_thumbnail_label, self.image_preview_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(400, 400)
            label.setMaximumSize(512, 512)
            label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
            self.preview_row_layout.addWidget(label)

        # Connect thumbnail click
        self.image_thumbnail_label.imageClicked.connect(self.update_preview)

        # Add the HBox layout (wrapped in a QWidget) to the main VBox
        self.right_layout.addWidget(self.preview_row_widget)

        # --- Denoise Controls ---
        self.controls_layout = QFormLayout()
        self.iso_level_slider = QSlider(Qt.Horizontal)
        self.iso_level_slider.setRange(0, 65534)
        self.iso_level_slider.setMinimumWidth(200)
        self.iso_level_spinbox = QSpinBox()
        self.iso_level_spinbox.setRange(0, 65534)

        self.grain_amount_slider = QSlider(Qt.Horizontal)
        self.grain_amount_slider.setRange(0, 100)
        self.grain_amount_slider.setMinimumWidth(200)
        self.grain_amount_spinbox = QSpinBox()
        self.grain_amount_spinbox.setRange(0, 100)

        # Connect slider and spinbox
        self.iso_level_slider.valueChanged.connect(self.iso_level_spinbox.setValue)
        self.iso_level_spinbox.valueChanged.connect(self.iso_level_slider.setValue)
        self.grain_amount_slider.valueChanged.connect(self.grain_amount_spinbox.setValue)
        self.grain_amount_spinbox.valueChanged.connect(self.grain_amount_slider.setValue)

        self.controls_layout.addRow("Image ISO: ", self.iso_level_slider)
        self.controls_layout.addRow("", self.iso_level_spinbox)
        self.controls_layout.addRow("Grain Amount:", self.grain_amount_slider)
        self.controls_layout.addRow("", self.grain_amount_spinbox)

        self.btn_preview_denoise = QPushButton("Preview Denoise")
        self.btn_preview_denoise.clicked.connect(self.preview_denoised_image)
        
        self.btn_save_full = QPushButton("Save Denoised Image")
        self.btn_save_full.clicked.connect(lambda: self.save_full_image(save_cfa=False))
        self.btn_save_full_cfa = QPushButton("Save Denoised Image (CFA)")
        self.btn_save_full_cfa.clicked.connect(lambda: self.save_full_image(save_cfa=True))

        self.right_layout.addLayout(self.controls_layout)
        self.right_layout.addWidget(self.btn_preview_denoise)
        self.right_layout.addWidget(self.btn_save_full)
        self.right_layout.addWidget(self.btn_save_full_cfa)

        self.main_layout.addWidget(self.right_panel, 2) # 2/3 of the space

        # --- State Variables ---
        self.current_folder = None
        self.current_file_path = None
        self.original_pixmap = None

        # -- Denoising Model --
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.mh = ModelHandler("RGGB_v1_trace.py", self.device, colorspace='lin_rec2020')

    def open_folder_dialog(self):
        """
        Opens a dialog to select a directory.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.current_folder = folder_path
            self.populate_file_list(folder_path)

    def populate_file_list(self, folder_path):
        """
        Lists raw image files from the selected directory.
        Supported formats: .cr2, .nef, .arw
        """
        self.file_list_widget.clear()
        raw_extensions = ['.cr2', '.nef', '.arw', '.dng', '.orf', '.raf', '.pef']
        try:
            files = os.listdir(folder_path)
            files = sorted(files)
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in raw_extensions):
                    self.file_list_widget.addItem(filename)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read directory: {e}")


    def update_preview(self, W, H):
        H_o, W_o = self.mh.rh.raw.shape
        H, W = int(H*H_o), int(W*W_o)
        w, h = 512, 512
        self.dims=(H - h//2, H + h//2, W - w//2, W + w//2)
        self.preview_denoised_image()

    def on_file_selected(self, current_item, previous_item):
        """
        Handles the selection of a file in the list.
        Displays a placeholder image as a preview.
        """
        if current_item is None:
            return
            
        filename = current_item.text()
        
        self.current_file_path = os.path.join(self.current_folder, filename)

        try:
            # Create a simple placeholder image
            self.mh.get_rh(self.current_file_path)
            self.iso_level_spinbox.setValue(self.mh.iso)

            # Reset dim location
            self.set_dims()

            self.preview_denoised_image()
            self.image_preview_label.setText("") # Clear the initial text      

            # Generate thumbnail
            img_rgb = self.mh.generate_thumbnail(min_preview_size=400)

            print("max val thumbnail", img_rgb.max())
            print(img_rgb.shape)
            image = numpy_to_qimage_rgb(img_rgb)
            self.update_image_thumbnail(image)
            self.image_thumbnail_label.setText("") # Clear the initial text
            
        except Exception as e:
            self.image_preview_label.setText(f"Could not preview file:\n{e}")
            self.original_pixmap = None
            QMessageBox.warning(self, "Preview Error", f"Could not create a preview for {filename}.\nError: {e}")

    def set_dims(self):
        W, H = self.mh.rh.raw.shape
        w, h = 800, 600
        self.dims = (H//2 - h//2, H//2 + h//2, W//2 - w//2, W//2 + w//2)

    def update_image_thumbnail(self, image):
        """
        Updates the image preview label with the given pixmap, scaled to fit.
        """
        if image:
            self.image_thumbnail_label.set_image(image)
        else:
            self.image_thumbnail_label.setText("No image to display.")

    def update_image_preview(self, pixmap):
        """
        Updates the image preview label with the given pixmap, scaled to fit.
        """
        if pixmap:
            scaled_pixmap = pixmap.scaled(self.image_preview_label.size(), 
                                          Qt.KeepAspectRatio, 
                                          Qt.SmoothTransformation)
            self.image_preview_label.setPixmap(scaled_pixmap)
        else:
            self.image_preview_label.setText("No image to display.")


    def preview_denoised_image(self):
        
        denoise_amount = self.iso_level_spinbox.value()
        grain_amount = self.grain_amount_spinbox.value()
        img_rgb, denoised = self.mh.tile([denoise_amount, grain_amount], dims = self.dims, apply_gamma=True)
        self.img_rgb = denoised
        img = numpy_to_qimage_rgb(self.img_rgb)
        denoised_pixmap = QPixmap.fromImage(img)
        self.update_image_preview(denoised_pixmap)
        print(f"Denoising with amount: {denoise_amount} { denoise_amount/1000.}")


    def save_full_image(self, save_cfa=False):
        """
        Runs the 'denoising' on the full image and saves the output.
        """
        if not self.current_file_path:
            QMessageBox.warning(self, "No File Selected", "Please select a file to denoise.")
            return

        denoise_amount = self.iso_level_spinbox.value()

        output_filename, _ = QFileDialog.getSaveFileName(
            self, "Save Denoised Image",
            os.path.splitext(self.current_file_path)[0] + f"_{denoise_amount}_denoised.DNG",
            "DNG Image (*.DNG)"
        )

        if output_filename:
            try:
                self.mh.save_dng(output_filename, [denoise_amount, 0], save_cfa=save_cfa)    
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save the image.\nError: {e}")


class ClickableImageLabel(QLabel):
    # Signal to emit when the image is clicked, passing mapped coordinates
    imageClicked = Signal(float, float)

    def __init__(self, parent=None, proportional=False):
        super().__init__(parent)
        self.original_image_size = QSize() # To store the original QImage size
        self.proportional = proportional

    def set_image(self, qimage: QImage):
        self.original_image_size = qimage.size()
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.size(), 
                                      Qt.KeepAspectRatio, 
                                      Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        self.setScaledContents(True)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Get click position relative to the QLabel
            label_x = event.pos().x()
            label_y = event.pos().y()

            # Calculate scaling factors
            label_width = self.width()
            label_height = self.height()
            image_width = self.original_image_size.width()
            image_height = self.original_image_size.height()

            if self.proportional:
                if label_width > 0 and label_height > 0 and image_width > 0 and image_height > 0:
                    # Map click coordinates back to original QImage dimensions
                    mapped_x = label_x / label_width
                    mapped_y = label_y / label_height

                    self.imageClicked.emit(mapped_x, mapped_y)
            else:
                if label_width > 0 and label_height > 0 and image_width > 0 and image_height > 0:
                    # Map click coordinates back to original QImage dimensions
                    mapped_x = int(label_x * (image_width / label_width))
                    mapped_y = int(label_y * (image_height / label_height))

                    self.imageClicked.emit(mapped_x, mapped_y)

        super().mousePressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RawRefineryApp()
    window.show()
    sys.exit(app.exec())
