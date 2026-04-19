import os
import concurrent.futures
from PySide6.QtCore import QObject, Signal
from .batch_worker import process_single_image_worker, init_worker

class BatchWorker(QObject):
    """
    Worker object to handle batch processing in a separate thread.
    Manages a ProcessPoolExecutor to run image processing tasks.
    """
    file_processed = Signal(str, str)  # Emits filename and result/error
    batch_finished = Signal()         # Emits when the entire batch is done

    def __init__(self, files_to_process, base_folder, dest_folder, settings, model, device, num_workers, progress_queue):
        super().__init__()
        self.files = files_to_process
        self.base_folder = base_folder
        self.dest_folder = dest_folder
        self.settings = settings
        self.model = model
        self.device = device
        self.num_workers = num_workers
        self.progress_queue = progress_queue
        self.is_cancelled = False

    def run(self):
        """
        Executes the batch processing using a ProcessPoolExecutor.
        """
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=init_worker,
            initargs=(self.model, self.device)
        ) as executor:
            futures = {}
            for filename in self.files:
                if self.is_cancelled:
                    break
                input_path = os.path.join(self.base_folder, filename)
                output_path = os.path.join(self.dest_folder, os.path.splitext(filename)[0] + "_refined.dng")
                future = executor.submit(process_single_image_worker, input_path, output_path, self.settings, self.progress_queue)
                futures[future] = filename

            for future in concurrent.futures.as_completed(futures):
                if self.is_cancelled:
                    # Cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
                
                filename = futures[future]
                try:
                    _, result = future.result()
                    self.file_processed.emit(filename, result)
                except Exception as e:
                    self.file_processed.emit(filename, f"A worker process failed: {e}")

        self.batch_finished.emit()

    def cancel(self):
        self.is_cancelled = True
