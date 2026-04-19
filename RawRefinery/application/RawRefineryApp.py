import sys
import os
import numpy as np
import multiprocessing
import traceback
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
    QFileDialog, QListWidget, QListWidgetItem, QLabel, QSlider, QSpinBox, 
    QHBoxLayout, QFormLayout, QComboBox, QProgressBar, QMessageBox, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, Slot, QTimer, QThread

# Import utils
from RawRefinery.application.viewing_utils import numpy_to_qimage_rgb
from RawRefinery.application.ModelHandler import ModelController, MODEL_REGISTRY
from RawRefinery.application.ClickableImageLabel import ClickableImageLabel 
from RawRefinery.application.dng_utils import to_dng, convert_ccm_to_rational
from RawRefinery.application.LogarithmicSlider import LogarithmicSlider
from .batch_thread_worker import BatchWorker

class RawRefineryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raw Refinery")
        self.resize(1000, 700)

        self.controller = ModelController()
        self.batch_defaults = {'iso': 0, 'grain': 0}
        self.batch_destination_folder = None
        self.progress_row_map = {}
        self.progress_queue = None
        self.batch_thread = None
        self.batch_worker = None
        
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_batch_progress)
        
        self.controller.progress_update.connect(self.update_progress)
        self.controller.preview_ready.connect(self.display_result)
        self.controller.save_finished.connect(self.reset_after_save)
        self.controller.error_occurred.connect(self.show_error)
        self.controller.model_loaded.connect(lambda n: self.status_label.setText(f"Model Loaded: {n}"))
        self.setup_ui()
        
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

        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.clicked.connect(self.open_folder_dialog)
        self.select_buttons_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.select_all_files)
        self.btn_deselect_all = QPushButton("Deselect All")
        self.btn_deselect_all.clicked.connect(self.deselect_all_files)
        self.select_buttons_layout.addWidget(self.btn_select_all)
        self.select_buttons_layout.addWidget(self.btn_deselect_all)
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        self.progress_table = QTableWidget()
        self.progress_table.setColumnCount(3)
        self.progress_table.setHorizontalHeaderLabels(["File", "Status", "Progress"])
        self.progress_table.setVisible(False)
        self.left_layout.addWidget(self.btn_open_folder)
        self.left_layout.addLayout(self.select_buttons_layout)
        self.left_layout.addWidget(self.file_list)
        self.left_layout.addWidget(self.progress_table)
        self.main_layout.addWidget(self.left_panel, 1)

        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.preview_container = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_container)
        self.thumb_label = ClickableImageLabel("Thumbnail", proportional=True)
        self.preview_label = QLabel("Denoised Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.thumb_label.imageClicked.connect(self.on_thumbnail_click)
        self.preview_layout.addWidget(self.thumb_label)
        self.preview_layout.addWidget(self.preview_label)
        self.right_layout.addWidget(self.preview_container)

        self.controls_layout = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_REGISTRY.keys())
        self.model_combo.currentTextChanged.connect(self.controller.load_model)
        self.controls_layout.addRow("Model:", self.model_combo)
        self.device_combo = QComboBox()
        self.device_combo.addItems(self.controller.devices)
        self.device_combo.currentTextChanged.connect(self.controller.set_device)
        self.controls_layout.addRow("Device:", self.device_combo)
        self.iso_slider = QSlider(Qt.Horizontal)
        self.iso_slider.setRange(0, 65534)
        self.iso_spin = QSpinBox()
        self.iso_spin.setRange(0, 65534)
        self.iso_slider.valueChanged.connect(self.iso_spin.setValue)
        self.iso_spin.valueChanged.connect(self.iso_slider.setValue)
        self.controls_layout.addRow("ISO:", self.iso_slider)
        self.controls_layout.addRow("", self.iso_spin)
        self.grain_slider = QSlider(Qt.Horizontal)
        self.grain_slider.setRange(0, 100)
        self.grain_spin = QSpinBox()
        self.grain_spin.setRange(0, 100)
        self.grain_slider.valueChanged.connect(self.grain_spin.setValue)
        self.grain_spin.valueChanged.connect(self.grain_slider.setValue)
        self.controls_layout.addRow("Grain:", self.grain_slider)
        self.controls_layout.addRow("", self.grain_spin)
        self.exposure_slider = LogarithmicSlider(0.1, 10.)
        self.exposure_slider.setNaturalValue(1.)
        self.exposure_label = QLabel(f"Exposure adjustment (for visualization only): {self.exposure_slider.get_natural_value():.1f}")
        self.exposure_slider.naturalValueChanged.connect(self.on_exposure_change)
        self.controls_layout.addRow(self.exposure_label, self.exposure_slider)
        self.right_layout.addLayout(self.controls_layout)

        self.batch_group_box = QGroupBox("Batch Processing Settings")
        self.batch_controls_layout = QFormLayout(self.batch_group_box)
        self.batch_iso_override = QCheckBox("Override ISO")
        self.batch_iso_spin = QSpinBox()
        self.batch_iso_spin.setRange(0, 65534)
        self.btn_reset_batch_iso = QPushButton("Reset")
        iso_layout = QHBoxLayout()
        iso_layout.addWidget(self.batch_iso_override)
        iso_layout.addWidget(self.batch_iso_spin)
        iso_layout.addWidget(self.btn_reset_batch_iso)
        self.batch_controls_layout.addRow(iso_layout)
        self.batch_grain_override = QCheckBox("Override Grain")
        self.batch_grain_spin = QSpinBox()
        self.batch_grain_spin.setRange(0, 100)
        self.btn_reset_batch_grain = QPushButton("Reset")
        grain_layout = QHBoxLayout()
        grain_layout.addWidget(self.batch_grain_override)
        grain_layout.addWidget(self.batch_grain_spin)
        grain_layout.addWidget(self.btn_reset_batch_grain)
        self.batch_controls_layout.addRow(grain_layout)
        self.parallel_processes_spin = QSpinBox()
        self.parallel_processes_spin.setRange(1, os.cpu_count())
        self.parallel_processes_spin.setValue(os.cpu_count())
        self.batch_controls_layout.addRow("Parallel Processes:", self.parallel_processes_spin)
        self.btn_set_destination = QPushButton("Set Destination Folder")
        self.btn_set_destination.clicked.connect(self.set_destination_folder)
        self.destination_folder_label = QLabel("Destination: Not Set")
        self.batch_controls_layout.addRow(self.btn_set_destination)
        self.batch_controls_layout.addRow(self.destination_folder_label)
        self.batch_iso_spin.valueChanged.connect(self.on_batch_iso_changed)
        self.btn_reset_batch_iso.clicked.connect(self.on_reset_batch_iso)
        self.batch_grain_spin.valueChanged.connect(self.on_batch_grain_changed)
        self.btn_reset_batch_grain.clicked.connect(self.on_reset_batch_grain)
        self.right_layout.addWidget(self.batch_group_box)
        
        self.actions_group_box = QGroupBox("Actions")
        self.actions_layout = QVBoxLayout(self.actions_group_box)
        self.btn_preview = QPushButton("Update Preview")
        self.btn_preview.clicked.connect(self.trigger_preview)
        self.btn_save_cfa = QPushButton("Save CFA dng")
        self.btn_save_cfa.clicked.connect(self.trigger_save)
        actions_row1_layout = QHBoxLayout()
        actions_row1_layout.addWidget(self.btn_preview)
        actions_row1_layout.addWidget(self.btn_save_cfa)
        self.actions_layout.addLayout(actions_row1_layout)
        self.btn_run_batch = QPushButton("Run Batch")
        self.btn_run_batch.clicked.connect(self.trigger_batch_processing)
        self.btn_save_test_patch = QPushButton("Save Test Patch")
        self.btn_save_test_patch.clicked.connect(self.trigger_save_test_patch)
        actions_row2_layout = QHBoxLayout()
        actions_row2_layout.addWidget(self.btn_run_batch)
        actions_row2_layout.addWidget(self.btn_save_test_patch)
        self.actions_layout.addLayout(actions_row2_layout)
        self.right_layout.addWidget(self.actions_group_box)

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self.right_layout.addWidget(self.progress_bar)
        self.right_layout.addWidget(self.status_label)

        self.main_layout.addWidget(self.right_panel, 2)
        
        self.dims = None
        self.current_folder = None
        self.current_file_path = None
    
    def on_exposure_change(self, exposure):
        self.exposure_label.setText(f"Exposure adjustment (for visualization only): {exposure:.1f}")
        thumb_rgb = self.controller.generate_thumbnail()
        qimg = numpy_to_qimage_rgb(thumb_rgb, exposure=exposure)
        self.thumb_label.set_image(qimg)

    def select_all_files(self):
        for i in range(self.file_list.count()): self.file_list.item(i).setCheckState(Qt.Checked)
    def deselect_all_files(self):
        for i in range(self.file_list.count()): self.file_list.item(i).setCheckState(Qt.Unchecked)

    def on_batch_iso_changed(self, value):
        self.batch_iso_spin.setStyleSheet("background-color: #fceea7;" if value != self.batch_defaults['iso'] else "")
    def on_reset_batch_iso(self):
        self.batch_iso_spin.setValue(self.batch_defaults['iso'])
    def on_batch_grain_changed(self, value):
        self.batch_grain_spin.setStyleSheet("background-color: #fceea7;" if value != self.batch_defaults['grain'] else "")
    def on_reset_batch_grain(self):
        self.batch_grain_spin.setValue(self.batch_defaults['grain'])

    def set_destination_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        if folder:
            self.batch_destination_folder = folder
            self.destination_folder_label.setText(f"Destination: {folder}")

    def trigger_batch_processing(self):
        if self.batch_thread and self.batch_thread.isRunning():
            self.show_error("Batch processing is already running.")
            return

        if not self.batch_destination_folder:
            self.show_error("Please set a destination folder first.")
            return
            
        selected_files = [self.file_list.item(i).text() for i in range(self.file_list.count()) if self.file_list.item(i).checkState() == Qt.Checked]
        if not selected_files:
            self.show_error("No files selected for batch processing.")
            return

        self.file_list.setVisible(False)
        self.progress_table.setVisible(True)
        self.progress_table.setRowCount(len(selected_files))
        self.progress_row_map.clear()
        for i, filename in enumerate(selected_files):
            self.progress_row_map[filename] = i
            self.progress_table.setItem(i, 0, QTableWidgetItem(filename))
            self.progress_table.setItem(i, 1, QTableWidgetItem("Queued"))
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            self.progress_table.setCellWidget(i, 2, progress_bar)

        batch_settings = {}
        if self.batch_iso_override.isChecked(): batch_settings['iso'] = self.batch_iso_spin.value()
        if self.batch_grain_override.isChecked(): batch_settings['grain'] = self.batch_grain_spin.value()
        
        num_workers = self.parallel_processes_spin.value()
        model_name = self.model_combo.currentText()
        device_name = self.device_combo.currentText()

        self.disable_ui()
        self.status_label.setText(f"Starting batch processing with {num_workers} workers...")
        
        manager = multiprocessing.Manager()
        self.progress_queue = manager.Queue()
        self.progress_timer.start(100)

        self.batch_thread = QThread()
        self.batch_worker = BatchWorker(selected_files, self.current_folder, self.batch_destination_folder, batch_settings, model_name, device_name, num_workers, self.progress_queue)
        self.batch_worker.moveToThread(self.batch_thread)

        self.batch_thread.started.connect(self.batch_worker.run)
        self.batch_worker.file_processed.connect(self.handle_file_processed)
        self.batch_worker.batch_finished.connect(self.handle_batch_finished)
        self.batch_worker.batch_finished.connect(self.batch_thread.quit)
        self.batch_worker.batch_finished.connect(self.batch_worker.deleteLater)
        self.batch_thread.finished.connect(self.batch_thread.deleteLater)
        
        self.batch_thread.start()

    @Slot(str, str)
    def handle_file_processed(self, filename, result):
        if filename in self.progress_row_map:
            row = self.progress_row_map[filename]
            if "Error" in result:
                self.progress_table.item(row, 1).setText("Error")
                self.show_error(f"Failed to process {filename}:\n{result}")
            else:
                self.progress_table.cellWidget(row, 2).setValue(100)
                self.progress_table.item(row, 1).setText("Done")

    @Slot()
    def handle_batch_finished(self):
        self.progress_timer.stop()
        self.progress_queue = None
        self.status_label.setText("Batch processing complete.")
        self.enable_ui()
        self.progress_table.setVisible(False)
        self.file_list.setVisible(True)
        QMessageBox.information(self, "Batch Complete", f"Finished processing {len(self.progress_row_map)} files.")
        self.batch_thread = None
        self.batch_worker = None

    @Slot()
    def update_batch_progress(self):
        if not self.progress_queue: return
        while not self.progress_queue.empty():
            try:
                message = self.progress_queue.get_nowait()
                filename = message.get('filename')
                progress = message.get('progress')
                if filename and progress is not None and filename in self.progress_row_map:
                    row = self.progress_row_map[filename]
                    if self.progress_table.item(row, 1).text() != "Processing":
                        self.progress_table.item(row, 1).setText("Processing")
                    self.progress_table.cellWidget(row, 2).setValue(int(progress * 100))
            except Exception:
                break

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.current_folder = folder
            self.file_list.clear()
            exts = ('.cr2', '.cr3', '.nef', '.arw', '.dng')
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
            for f in files:
                item = QListWidgetItem(f)
                item.setCheckState(Qt.Unchecked)
                self.file_list.addItem(item)

    def on_file_selected(self, item):
        if not item: return
        path = os.path.join(self.current_folder, item.text())
        self.current_file_path = path
        try:
            iso = self.controller.load_rh(path)
            self.iso_spin.setValue(iso)
            thumb_rgb = self.controller.generate_thumbnail()
            qimg = numpy_to_qimage_rgb(thumb_rgb, exposure=self.exposure_slider.get_natural_value())
            self.thumb_label.set_image(qimg)
            self.dims = None 
            self.preview_label.setText("Click thumbnail to preview region")
        except Exception as e:
            self.show_error(f"Failed to open file: {e}")

    def on_thumbnail_click(self, x_ratio, y_ratio):
        rh = self.controller.rh
        H_full, W_full = rh.raw.shape
        c_x = int(x_ratio * W_full); c_y = int(y_ratio * H_full)
        w_preview, h_preview = 512, 512
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
        self.btn_run_batch.setEnabled(False)

    def enable_ui(self):
        self.btn_preview.setEnabled(True)
        self.btn_save_cfa.setEnabled(True)
        self.thumb_label.setEnabled(True)
        self.btn_run_batch.setEnabled(True)

    def trigger_preview(self):
        if not self.dims:
            self.status_label.setText("Select a region on thumbnail first.")
            return
        conditioning = [self.iso_spin.value(), self.grain_spin.value()]
        self.disable_ui()
        self.status_label.setText("Processing...")
        self.controller.run_inference(conditioning, self.dims)

    def trigger_save(self):
        if not self.current_file_path:
            self.status_label.setText("Select and image first.")
            return 
        output_filename, _ = QFileDialog.getSaveFileName(self, "Save Denoised Image", os.path.splitext(self.current_file_path)[0] + "_denoised.dng", "DNG Image (*.dng)")
        if not output_filename: return
        conditioning = [self.iso_spin.value(), self.grain_spin.value()]
        self.disable_ui()
        self.status_label.setText("Processing...")
        self.controller.save_image(output_filename, conditioning, save_cfa=True)

    def trigger_save_test_patch(self):
        if not self.current_file_path: return
        try:
            output_filename, _ = QFileDialog.getSaveFileName(self, "Save Denoised Image", os.path.splitext(self.current_file_path)[0] + "_test_patch.dng", "DNG Image (*.dng)")
            if not output_filename: return
            cfa = self.controller.rh._input_handler(dims=self.dims)
            cfa = np.squeeze(cfa, axis=0)
            ccm1 = convert_ccm_to_rational(self.controller.rh.core_metadata.rgb_xyz_matrix[:3, :])
            to_dng(cfa, self.controller.rh, output_filename, ccm1, save_cfa=True, convert_to_cfa=False, use_orig_wb_points=True)
        except Exception as e:
            self.show_error(f"Failed to save patch: {e}")

    @Slot(float)
    def update_progress(self, val):
        self.progress_bar.setValue(int(val * 100))

    @Slot(object, object)
    def display_result(self, img_rgb, denoised):
        self.enable_ui()
        self.status_label.setText("Done.")
        self.progress_bar.setValue(100)
        qimg = numpy_to_qimage_rgb(denoised, exposure=self.exposure_slider.get_natural_value())
        pix = QPixmap.fromImage(qimg)
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
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    if platform.system() == "Darwin":
        app.setApplicationDisplayName("Raw Refinery")
        app.setOrganizationName("Ryan Mueller")
        app.setDesktopFileName("com.rawrefinery.app") 

    window = RawRefineryApp()
    window.show()
    window.loading_popup()
    sys.exit(app.exec())
