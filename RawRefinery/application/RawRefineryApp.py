import sys
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
    QFileDialog, QListWidget, QLabel, QSlider, QSpinBox, 
    QHBoxLayout, QFormLayout, QComboBox, QProgressBar, QMessageBox
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, Slot

# Import utils
from RawRefinery.application.viewing_utils import numpy_to_qimage_rgb
from RawRefinery.application.ModelHandler import ModelController, MODEL_REGISTRY
from RawRefinery.application.ClickableImageLabel import ClickableImageLabel 
from RawRefinery.application.dng_utils import to_dng, convert_ccm_to_rational
from RawRefinery.application.LogarithmicSlider import LogarithmicSlider

class RawRefineryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raw Refinery")
        self.resize(1000, 700)

        #  Logic Controller
        self.controller = ModelController()
        
        # Connect Signals
        self.controller.progress_update.connect(self.update_progress)
        self.controller.preview_ready.connect(self.display_result)
        self.controller.save_finished.connect(self.reset_after_save)
        self.controller.error_occurred.connect(self.show_error)
        self.controller.model_loaded.connect(lambda n: self.status_label.setText(f"Model Loaded: {n}"))
        self.setup_ui()
        
        # Load default model
        self.controller.load_model("Tree Net Denoise")

    def loading_popup(self):
        popup = QMessageBox()
        popup.setWindowTitle("Notice")
        popup.setIcon(QMessageBox.Icon.Information)
        popup.setText(
            """
            <h3>Alpha Release Notice</h3>
            <p>This software is currently in Alpha. Please note:</p>
            <ul>
                <li>Expect memory usage up to <b>3 GB</b>.</li>
                <li>CPU processing is slow. Use <b>MPS</b> or <b>CUDA</b> for best results.</li>
            </ul>
            """
        )
        popup.exec()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        ## Left Panel 
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.clicked.connect(self.open_folder_dialog)
        
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        
        self.left_layout.addWidget(self.btn_open_folder)
        self.left_layout.addWidget(self.file_list)
        self.main_layout.addWidget(self.left_panel, 1)

        ## Right Panel 
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        # Previews
        self.preview_container = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_container)
        self.thumb_label = ClickableImageLabel("Thumbnail", proportional=True)
        self.preview_label = QLabel("Denoised Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        
        # Connect thumbnail click
        self.thumb_label.imageClicked.connect(self.on_thumbnail_click)
        
        self.preview_layout.addWidget(self.thumb_label)
        self.preview_layout.addWidget(self.preview_label)
        self.right_layout.addWidget(self.preview_container)

        ## Controls
        self.controls_layout = QFormLayout()
        
        # Model Selector
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_REGISTRY.keys())
        self.model_combo.currentTextChanged.connect(self.controller.load_model)
        self.controls_layout.addRow("Model:", self.model_combo)

        # Device Selector
        self.device_combo = QComboBox()
        self.device_combo.addItems(self.controller.devices)
        self.device_combo.currentTextChanged.connect(self.controller.set_device)
        self.controls_layout.addRow("Device:", self.device_combo)

        # ISO
        self.iso_slider = QSlider(Qt.Horizontal)
        self.iso_slider.setRange(0, 65534)
        self.iso_spin = QSpinBox()
        self.iso_spin.setRange(0, 65534)
        self.iso_slider.valueChanged.connect(self.iso_spin.setValue)
        self.iso_spin.valueChanged.connect(self.iso_slider.setValue)
        self.controls_layout.addRow("ISO:", self.iso_slider)
        self.controls_layout.addRow("", self.iso_spin)

        # Grain
        self.grain_slider = QSlider(Qt.Horizontal)
        self.grain_slider.setRange(0, 100)
        self.grain_spin = QSpinBox()
        self.grain_spin.setRange(0, 100)
        self.grain_slider.valueChanged.connect(self.grain_spin.setValue)
        self.grain_spin.valueChanged.connect(self.grain_slider.setValue)
        self.controls_layout.addRow("Grain:", self.grain_slider)
        self.controls_layout.addRow("", self.grain_spin)

        #Exposure 
        self.exposure_slider = LogarithmicSlider(0.1, 10.)
        self.exposure_slider.setNaturalValue(1.)
        self.exposure_label = QLabel(f"Exposure adjustment (for visualization only): {self.exposure_slider.get_natural_value():.1f}")
        self.exposure_slider.naturalValueChanged.connect(self.on_exposure_change)
        self.controls_layout.addRow(self.exposure_label, self.exposure_slider)

        self.right_layout.addLayout(self.controls_layout)

        # Action Buttons
        self.btn_preview = QPushButton("Update Preview")
        self.btn_preview.clicked.connect(self.trigger_preview)
        self.right_layout.addWidget(self.btn_preview)

        self.btn_save_cfa = QPushButton("Save CFA dng")
        self.btn_save_cfa.clicked.connect(self.trigger_save)
        self.right_layout.addWidget(self.btn_save_cfa)

        self.btn_save_test_patch = QPushButton("Save Test Patch")
        self.btn_save_test_patch.clicked.connect(self.trigger_save_test_patch)
        self.right_layout.addWidget(self.btn_save_test_patch)

        # Status
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self.right_layout.addWidget(self.progress_bar)
        self.right_layout.addWidget(self.status_label)

        # Add Right Panel
        self.main_layout.addWidget(self.right_panel, 2)
        
        ## State
        self.dims = None
        self.current_folder = None
        self.current_file_path = None
    
    def on_exposure_change(self, exposure):
            # Set text
            self.exposure_label.setText(f"Exposure adjustment (for visualization only): {exposure:.1f}")
            # Show thumbnail
            thumb_rgb = self.controller.generate_thumbnail()
            qimg = numpy_to_qimage_rgb(thumb_rgb, exposure=exposure)
            self.thumb_label.set_image(qimg)


    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.current_folder = folder
            self.file_list.clear()
            exts = ('.cr2', '.cr3', '.nef', '.arw', '.dng')
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
            self.file_list.addItems(files)

    def on_file_selected(self, item):
        if not item: return
        path = os.path.join(self.current_folder, item.text())
        self.current_file_path = path
        
        try:
            # Load metadata
            iso = self.controller.load_rh(path)
            self.iso_spin.setValue(iso)
            
            # Show thumbnail
            thumb_rgb = self.controller.generate_thumbnail()
            qimg = numpy_to_qimage_rgb(thumb_rgb, exposure=self.exposure_slider.get_natural_value())
            self.thumb_label.set_image(qimg)
            
            self.dims = None 
            self.preview_label.setText("Click thumbnail to preview region")
            
        except Exception as e:
            self.show_error(f"Failed to open file: {e}")

    def on_thumbnail_click(self, x_ratio, y_ratio):
        # Calculate dims based on click ratio and raw size
        rh = self.controller.rh
        H_full, W_full = rh.raw.shape
        
        # Center of click in raw coords
        c_x = int(x_ratio * W_full)
        c_y = int(y_ratio * H_full)
        
        w_preview, h_preview = 512, 512
        
        # Calculate crops
        h_start = max(0, c_y - h_preview//2)
        h_end = min(H_full, h_start + h_preview)
        w_start = max(0, c_x - w_preview//2)
        w_end = min(W_full, w_start + w_preview)
        
        self.dims = (h_start, h_end, w_start, w_end)
        self.trigger_preview()

    def disable_ui(self):
        self.btn_preview.setEnabled(False)
        self.btn_save_cfa.setEnabled(False)
        self.thumb_label.setEnabled(False)

    def enable_ui(self):
        self.btn_preview.setEnabled(True)
        self.btn_save_cfa.setEnabled(True)
        self.thumb_label.setEnabled(True)

    def trigger_preview(self):
        if not self.dims:
            self.status_label.setText("Select a region on thumbnail first.")
            return

        conditioning = [self.iso_spin.value(), self.grain_spin.value()]
        # Disable button to prevent spamming
        self.disable_ui()
        self.status_label.setText("Processing...")
        # Fire off the worker
        self.controller.run_inference(conditioning, self.dims)

    def trigger_save(self):
        if not self.current_file_path:
            self.status_label.setText("Select and image first.")
            return 
        
        output_filename, _ = QFileDialog.getSaveFileName(
            self, "Save Denoised Image",
            os.path.splitext(self.current_file_path)[0] + "_denoised.dng",
            "DNG Image (*.dng)"
        )

        conditioning = [self.iso_spin.value(), self.grain_spin.value()]
        self.disable_ui()
        self.status_label.setText("Processing...")
        self.controller.save_image(output_filename, conditioning, save_cfa=True)

    def trigger_save_test_patch(self):
        if not self.current_file_path:
            self.status_label.setText("Select and image first.")
            return 
        try:
            output_filename, _ = QFileDialog.getSaveFileName(
                self, "Save Denoised Image",
                os.path.splitext(self.current_file_path)[0] + "_test_patch.dng",
                "DNG Image (*.dng)"
            )
            cfa = self.controller.rh._input_handler(dims=self.dims)
            cfa = np.squeeze(cfa, axis=0)
            ccm1 = convert_ccm_to_rational(self.controller.rh.core_metadata.rgb_xyz_matrix[:3, :])
            to_dng(cfa, self.controller.rh, output_filename, ccm1, save_cfa=True, convert_to_cfa=False, use_orig_wb_points=True)
        except Exception as e:
            self.show_error(f"Failed to save patch: {e}")

    # Slots 
    @Slot(float)
    def update_progress(self, val):
        self.progress_bar.setValue(int(val * 100))

    @Slot(object, object)
    def display_result(self, img_rgb, denoised):
        self.enable_ui()
        self.status_label.setText("Done.")
        self.progress_bar.setValue(100)
        
        # Convert output to QImage
        qimg = numpy_to_qimage_rgb(denoised, exposure=self.exposure_slider.get_natural_value())
        pix = QPixmap.fromImage(qimg)
        
        # Scale for display
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    @Slot(float)
    def reset_after_save(self, str):
        self.enable_ui()
        self.status_label.setText(str)
        self.progress_bar.setValue(100.)

    @Slot(str)
    def show_error(self, msg):
        self.enable_ui()
        self.status_label.setText("Error.")
        QMessageBox.critical(self, "Error", msg)

    @Slot(str)
    def display_device(self, device):
        print("device used")
        self.device_label.setText(device)

import platform
if __name__ == '__main__':
    app = QApplication(sys.argv)
    if platform.system() == "Darwin":  # Darwin is the kernel for macOS
        app.setApplicationDisplayName("Raw Refinery")
        app.setOrganizationName("Ryan Mueller")
        app.setDesktopFileName("com.rawrefinery.app") 

    window = RawRefineryApp()
    window.show()
    window.loading_popup()
    sys.exit(app.exec())