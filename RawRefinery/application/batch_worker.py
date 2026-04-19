import os
import traceback
from RawRefinery.application.ModelHandler import ModelController

worker_controller = None

def init_worker(model_name, device_name):
    """
    Initializer for each worker process. Creates a single ModelController
    instance for the lifetime of the process.
    """
    global worker_controller
    print(f"Initializing worker on process {os.getpid()} with device {device_name}")
    worker_controller = ModelController()
    worker_controller.load_model(model_name)
    worker_controller.set_device(device_name)

def process_single_image_worker(input_path, output_path, batch_settings, progress_queue):
    """
    A top-level function to be run in a separate process for batch processing.
    Uses the globally initialized ModelController.
    """
    global worker_controller
    if not worker_controller:
        return (os.path.basename(input_path), "Error: Worker process not initialized.")

    try:
        original_iso = worker_controller.load_rh(input_path)
        original_grain = 0

        final_iso = batch_settings.get('iso', original_iso)
        final_grain = batch_settings.get('grain', original_grain)
        
        conditioning = [final_iso, final_grain]
        
        # Pass the queue DIRECTLY to the save_image method.
        worker_controller.save_image(output_path, conditioning, save_cfa=True, progress_queue=progress_queue)
        
        return (os.path.basename(input_path), "Success")
    except Exception as e:
        return (os.path.basename(input_path), f"Error: {traceback.format_exc()}")
