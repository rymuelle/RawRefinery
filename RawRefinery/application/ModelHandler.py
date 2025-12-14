import torch
import numpy as np
from pathlib import Path
from platformdirs import user_data_dir
from time import perf_counter
import requests
from PySide6.QtCore import QObject, Signal, Slot, QThread
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from RawHandler.RawHandler import RawHandler
from blended_tiling import TilingModule
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from RawRefinery.application.dng_utils import convert_color_matrix, to_dng
from RawRefinery.application.postprocessing import match_colors_linear

MODEL_REGISTRY = {
    "Tree Net Denoise": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/ShadowWeightedL1.pt",
        "filename": "ShadowWeightedL1.pt"
    },
    "Tree Net Denoise Light": {
        "url": "    https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/ShadowWeightedL1_light.pt",
        "filename": "ShadowWeightedL1_light.pt"
    },
    "Tree Net Denoise Super Light": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/ShadowWeightedL1_super_light.pt",
        "filename": "ShadowWeightedL1_super_light.pt"
    },

    "Tree Net Denoise Heavy": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/ShadowWeightedL1_24_deep_500.pt",
        "filename": "ShadowWeightedL1_24_deep_500.pt"
    },

    "Deblur": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/realblur_gamma_140.pt",
        "filename": "realblur_gamma_140.pt",
        "affine": True,
    },
    "DeepSharpen": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/Deblur_deep_24.pt",
        "filename": "Deblur_deep_24.pt",
        "affine": True,
    },
}

key_string = '''-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEA8iRGMPqFIFVF0TM/AbMI
DJUqdjY1S7dGn6rYjLixnhohHLKIo2ZhFUfPaeYrDoqJblP9MxbBLm6a782/Us0A
vblTQOsdVFHOlVEiDUkG9CJrzh7arqJF+v2LLP9qPIcL5QdIHM+BCKlbNPBU/TJB
49b6a+1FfKCEeY1z9F8H6GCHGeRB43lz5/1yMoBnq//Rc7NrvinwlNcFYHHM1oj6
Hk6KPkgitya11QgTTva+XimR7cbw7h9/vJKbrS7tValApio3Ypmx7AKf6/k16S9K
BCFDN3cyWmjItQNzEWbO2nuM9d3PX2O4FcZVfsA/GU0qSuKFUrrN0KcxKGglLdu4
3Nt3JmOh+VebVWPSTeMzn2R1LDs2CsDpGG+KnHso80HBBq6RuHTugTiUZ2EwjiXN
lRS7olKFQOPwT0tm1EVkH8IxQgV4KJbCb6hAScvWfsDdsP+bu4R+QI9hfU6HCWG3
a8w1AY+5GT7zp1pzKifmnXgMXF3VnAPTGRhpIvPQfum2+tppLZueXlalobK0MDzi
n36TNhRELao1W7Tvc18fxyZn37BBgKs89JO85/cjD72yhVowW7Hy9lL7RnB+etaN
ehXoYFsJReNmD5KNgRtmXbsCUJ+D8v7BVYNGl1UgebmQnMdMWyiU/3l1Uuy8HS3L
1QJYp42f5QqONttCqVzgzrECAwEAAQ==
-----END PUBLIC KEY-----'''

class InferenceWorker(QObject):
    """
    Performs heavy lifting in a background thread.
    Does not know about UI, only knows about data.
    """
    finished = Signal(object, object) # img_rgb, denoised
    progress = Signal(float)
    error = Signal(str)

    def __init__(self, model, model_params, device, rh, conditioning, dims, img_size=128, tile_overlap=0.25, batch_size=2):
        super().__init__()
        self.model = model
        self.model_params = model_params
        self.device = device
        self.rh = rh
        self.conditioning = conditioning
        self.dims = dims
        self.img_size = img_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _tile_process(self):
        # Prepare Data
        image_RGGB = self.rh.as_rggb(dims=self.dims, colorspace='lin_rec2020')
        image_RGB = self.rh.as_rgb(dims=self.dims, demosaicing_func=demosaicing_CFA_Bayer_Malvar2004, colorspace='lin_rec2020', clip=True)
        
        tensor_image = torch.from_numpy(image_RGGB).unsqueeze(0).contiguous()
        tensor_RGB = torch.from_numpy(image_RGB).unsqueeze(0).contiguous()

        full_size = [image_RGGB.shape[1], image_RGGB.shape[2]]
        tile_size = [self.img_size, self.img_size]
        overlap = [self.tile_overlap, self.tile_overlap]

        # Tiling Setup
        tiling_module = TilingModule(tile_size=tile_size, tile_overlap=overlap, base_size=full_size)
        tiling_module_rgb = TilingModule(tile_size=[s*2 for s in tile_size], tile_overlap=overlap, base_size=[s*2 for s in full_size])
        tiling_module_rebuild = TilingModule(tile_size=[s*2 for s in tile_size], tile_overlap=overlap, base_size=[s*2 for s in full_size])

        tiles = tiling_module.split_into_tiles(tensor_image).float().to(self.device)
        tiles_rgb = tiling_module_rgb.split_into_tiles(tensor_RGB).float().to(self.device)
        
        batches = torch.split(tiles, self.batch_size)
        batches_rgb = torch.split(tiles_rgb, self.batch_size)

        # Conditioning Setup
        cond_tensor = torch.as_tensor(self.conditioning, device=self.device).float().unsqueeze(0)
        cond_tensor[:, 0] /= 6400
        cond_tensor[:, 1] = 0
        cond_tensor = cond_tensor[:, 0:1]

        processed_batches = []
        
        # Determine Dtype
        dtype_map = {'mps': torch.float16, 'cuda': torch.float16, 'cpu': torch.bfloat16}
        autocast_dtype = dtype_map.get(self.device.type, torch.float32)

        total_batches = len(batches_rgb)
        
        # Inference Loop
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                for i, (batch, batch_rgb) in enumerate(zip(batches, batches_rgb)):
                    if self._is_cancelled: return None, None
                    
                    B = batch.shape[0]
                    # Expand conditioning to match batch size
                    curr_cond = cond_tensor.expand(B, -1)
                    
                    output = self.model(batch_rgb, curr_cond)

                    # Output processing
                    if "affine" in self.model_params:
                        output, _, _ = match_colors_linear(output, batch_rgb)
                    processed_batches.append(output.cpu())
                    
                    self.progress.emit((i + 1) / total_batches)

        # Rebuild
        tiles_out = torch.cat(processed_batches, dim=0)
        stitched = tiling_module_rebuild.rebuild_with_masks(tiles_out).detach().cpu().numpy()[0]

        torch.cuda.empty_cache()

        return image_RGB.transpose(1, 2, 0), stitched.transpose(1, 2, 0)
    
    @Slot()
    def run(self):
        try:
            img, denoised_img = self._tile_process()
            
            # Post-process blending
            blend_alpha = self.conditioning[1] / 100
            final_denoised = (denoised_img * (1 - blend_alpha)) + (img * blend_alpha)
            
            self.finished.emit(img, final_denoised)
            
        except Exception as e:
            self.error.emit(str(e))

class ModelController(QObject):
    """
    Manages the LifeCycle of the Model, the RawHandler, and the Worker Thread.
    """
    progress_update = Signal(float)
    preview_ready = Signal(object, object)
    error_occurred = Signal(str)
    model_loaded = Signal(str)
    save_finished = Signal(str)

    def __init__(self):
        super().__init__()

        self.model = None
        self.rh = None
        self.iso = 100
        self.colorspace = 'lin_rec2020'

        # Manage devices
        devices = {
                   "cuda": torch.cuda.is_available(),
                   "mps": torch.backends.mps.is_available(),
                   "cpu": lambda : True
        }
        self.devices = [d for d, is_available in devices.items() if is_available]
        self.set_device(self.devices[0])
        
        # Thread management
        self.worker_thread = None
        self.worker = None
        self.filename = None
        self.save_cfa = None
        self.start_time = None
        self.model_params = {}

        self.pub = serialization.load_pem_public_key(key_string.encode('utf-8'))

    def load_rh(self, path):
        """Loads the raw file handler"""
        self.rh = RawHandler(path, colorspace=self.colorspace)
        if 'EXIF ISOSpeedRatings' in self.rh.full_metadata:
            self.iso = int(self.rh.full_metadata['EXIF ISOSpeedRatings'].values[0])
        else:
            self.iso = 100
        return self.iso

    def load_model(self, model_key):
        """Loads a model by key from the registry"""
        if model_key not in MODEL_REGISTRY:
            self.error_occurred.emit(f"Model {model_key} not found in registry.")
            return

        conf = MODEL_REGISTRY[model_key]
        self.model_params = conf
        app_name = "RawRefinery"
        data_dir = Path(user_data_dir(app_name))
        model_path = data_dir / conf["filename"]

        # Handle Download
        if not model_path.is_file():
            if conf["url"]:
                print(f"Downloading {model_key}...")
                if not self._download_file(conf["url"], model_path):
                    self.error_occurred.emit("Failed to download model.")
                    return
            else:
                 self.error_occurred.emit(f"Model file not found at {model_path}")
                 return

        try:
            print(f"Loading model: {model_path}")
            # Verify model before load
            self._verify_model(model_path, model_path.with_suffix(f'{model_path.suffix}.sig'))
            
            loaded = torch.jit.load(model_path, map_location='cpu')
            self.model = loaded.eval().to(self.device)
            self.model_loaded.emit(model_key)
        except Exception as e:
            self.error_occurred.emit(f"Failed to load model: {e}")

    def set_device(self, device):
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)
        print(f"Using Device {self.device} from {device}")

    def run_inference(self, conditioning, dims=None):
        """Starts the worker thread"""
        if not self.model or not self.rh:
            self.error_occurred.emit("Model or Image not loaded.")
            return

        if self.worker_thread is not None:
            print("Worker already running, cancelling...")
            self.worker.cancel()
            self.worker_thread.quit()
            self.worker_thread.wait()

        # Create Thread and Worker
        self.worker_thread = QThread()
        self.worker = InferenceWorker(self.model, self.model_params, self.device, self.rh, conditioning, dims)
        self.worker.moveToThread(self.worker_thread)

        # Connect Signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_result)
        self.worker.progress.connect(self.progress_update)
        self.worker.error.connect(self.error_occurred)
        
        # Clean up
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._cleanup_references)

        self.worker_thread.start()

    def _cleanup_references(self):
        """
        Called when the thread finishes. 
        Sets the Python variables to None so we don't access dead C++ objects later.
        """
        self.worker_thread = None
        self.worker = None

    def save_image(self, filename, conditioning, save_cfa=False):
        """Starts the worker thread"""
        if not self.model or not self.rh:
            self.error_occurred.emit("Model or Image not loaded.")
            return

        if self.worker_thread is not None:
            print("Worker already running, cancelling...")
            self.worker.cancel()
            self.worker_thread.quit()
            self.worker_thread.wait()

        # Store the save information
        self.filename = filename
        self.save_cfa = save_cfa
        self.start_time = perf_counter()

        # Create Thread and Worker
        self.worker_thread = QThread()
        dims = None
        self.worker = InferenceWorker(self.model, self.model_params, self.device, self.rh, conditioning, dims)
        self.worker.moveToThread(self.worker_thread)

        # Connect Signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_full_image)
        self.worker.progress.connect(self.progress_update)
        self.worker.error.connect(self.error_occurred)
        
        # Clean up
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._cleanup_references)
        
        self.worker_thread.start()


    def generate_thumbnail(self, size=400):
        if not self.rh: return None
        thumb = self.rh.generate_thumbnail(min_preview_size=size, clip=True)
        return thumb

    def _verify_model(self, dest_path, sig_path):
        try:
            data = Path(dest_path).read_bytes()
            signature = Path(sig_path).read_bytes()
            self.pub.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256(),
            )
            print(f"Model {dest_path} verified!")
            return True
        except Exception as e:
            print(e)
            # if dest_path.exists():
            #     dest_path.unlink()
            # if sig_path.exists():
            #     sig_path.unlink()
            print(f"Model {dest_path} not verified! Deleting.")
            return False


    def _download_file(self, url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Download model signature
            r = requests.get(url + '.sig', stream=True)
            r.raise_for_status()
            sig_path = dest_path.with_suffix(f'{dest_path.suffix}.sig')
            with open(sig_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("test verification")
            return self._verify_model(dest_path, sig_path)
        
        except Exception as e:
            print(e)
            return False
        
    # Slots
    @Slot(object, object)
    def handle_result(self, img, denoised):
        self.preview_ready.emit(img, denoised)

    @Slot(object, object)
    def handle_full_image(self, img, denoised):
        if not self.filename:
            self.error_occurred.emit("Controller does not have a filename.")
            return
        if not self.save_cfa:
            self.error_occurred.emit("Controller does not know if it should save image as a CFA.")
            return 
        
        transform_matrix = np.linalg.inv(
                self.rh.rgb_colorspace_transform(colorspace=self.colorspace)
                )

        CCM = self.rh.rgb_colorspace_transform(colorspace='XYZ')
        CCM = np.linalg.inv(CCM)

        transformed = denoised @ transform_matrix.T
        uint_img = np.clip(transformed * 2**16-1, 0, 2**16-1).astype(np.uint16)
        ccm1 = convert_color_matrix(CCM)
        to_dng(uint_img, self.rh, self.filename, ccm1, save_cfa=self.save_cfa, convert_to_cfa=True)
        delta_time = perf_counter() - self.start_time 
        self.save_finished.emit(f"Done in {delta_time:.1f} seconds.")