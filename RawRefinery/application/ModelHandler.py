from RawHandler.RawHandler import RawHandler
from RawRefinery.utils.viewing_utils import linear_to_srgb
import torch
from blended_tiling import TilingModule
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from RawRefinery.application.dng_utils import convert_color_matrix, to_dng

from pathlib import Path
from platformdirs import user_data_dir
import requests
from tqdm import tqdm

class ModelHandler():
    def __init__(self, model_name, device, n_batches=2, colorspace = 'lin_rec2020'):

        app_name = "RawRefinery"
        model_url = "https://github.com/rymuelle/RawRefinery/releases/download/v1.0.0-alpha/RGGB_v1_trace.pt" 

        data_dir: Path = Path(user_data_dir(app_name))
        model_path: Path = data_dir / model_name


        if model_path.is_file():
            print(f"Model weights found at: {model_path}")
        else:
            print(f"Model weights not found. Expected path: {model_path}")
            download_file(model_url, model_path)

        model = torch.jit.load(model_path)
        self.model = model.eval().to(device)

        self.rh = None
        self.iso = 100
        self.device = device
        self.n_batches = n_batches
        self.colorspace = colorspace

    def controls(self):
        controls = {
            "ISO": {"range": [0, 65534], "default": self.iso},
            "Grain": {"range": [0, 100], "default": 0}
        }
        return controls

    def get_rh(self, path):
        self.rh = RawHandler(path, colorspace=self.colorspace)
        if 'EXIF ISOSpeedRatings' in self.rh.full_metadata:
            self.iso = int(self.rh.full_metadata['EXIF ISOSpeedRatings'].values[0])
        else:
            self.iso = 100
    
    def tile(self, conditioning, dims=None, apply_gamma=False):
        img, denoised_img = tile_image_rggb(self.rh, self.device, conditioning, self.model, dims=dims)
        # img, denoised_img = tile_image_sparse(self.rh, self.device, conditioning, self.model, dims=dims)
        denoised_img = denoised_img * (1 - conditioning[1]/100) + img * conditioning[1]/100
        if apply_gamma:
            img = img ** (1/2.2)
            denoised_img = denoised_img ** (1/2.2)
        return img, denoised_img

    def save_dng(self, filename, conditioning, dims=None):
            img, denoised_img = self.tile(conditioning, dims=dims)

            transform_matrix = np.linalg.inv(
                 self.rh.rgb_colorspace_transform(colorspace=self.colorspace)
                 )

            CCM = self.rh.rgb_colorspace_transform(colorspace='XYZ')
            CCM = np.linalg.inv(CCM)

            transformed = denoised_img @ transform_matrix.T
            uint_img = np.clip(transformed * 2**16-1, 0, 2**16-1).astype(np.uint16)
            ccm1 = convert_color_matrix(CCM)

            to_dng(uint_img, self.rh, filename, ccm1)

    def generate_thumbnail(self, min_preview_size=400):
         thumbnail = self.rh.generate_thumbnail(min_preview_size=min_preview_size, clip=True)
         thumbnail = linear_to_srgb(thumbnail)
         return thumbnail

def tile_image_rggb(rh, device, conditioning, model,
                    img_size=128, tile_overlap=0.25, batch_size=1, dims=None):
    image_RGGB = rh.as_rggb(dims=dims)
    image_RGB = rh.as_rgb(dims=dims)
    tensor_image = torch.from_numpy(image_RGGB).unsqueeze(0).contiguous()

    full_size = [image_RGGB.shape[1], image_RGGB.shape[2]]
    tile_size = [img_size, img_size]
    overlap = [tile_overlap, tile_overlap]

    tiling_module = TilingModule(tile_size=tile_size, tile_overlap=overlap, base_size=full_size)
    tiling_module_rebuild = TilingModule(tile_size=[s*2 for s in tile_size],
                                         tile_overlap=overlap,
                                         base_size=[s*2 for s in full_size])

    tiles = tiling_module.split_into_tiles(tensor_image).float().to(device)
    batches = torch.split(tiles, batch_size)

    conditioning_tensor = (
        torch.as_tensor(conditioning, device=device)
        .float()
        .unsqueeze(0)
    )
    conditioning_tensor[:, 0] /= 6400
    conditioning_tensor[:, 1] = 0
    processed_batches = []


    # Set up AMP
    if device.type == 'mps':
        autocast_dtype = torch.float16
    elif device.type == 'cuda':
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.bfloat16


    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            for batch in batches:
                B = batch.shape[0]
                output = model(batch, conditioning_tensor[:B, :])
                processed_batches.append(output.cpu())  # move to CPU
                del output
                torch.cuda.empty_cache()

    tiles = torch.cat(processed_batches, dim=0)
    stitched = (
        tiling_module_rebuild.rebuild_with_masks(tiles)
        .detach()
        .cpu()
        .numpy()[0]
    )

    stitched += image_RGB
    # Blend based on grain mixer'
    alpha = conditioning[1] / 100
    stitched = (stitched * (1-alpha)) + image_RGB * alpha
    del tiles, batches, processed_batches, conditioning_tensor
    torch.cuda.empty_cache()

    return image_RGB.transpose(1, 2, 0), stitched.transpose(1, 2, 0)

def download_file(url: str, dest_path: Path):
    """Downloads a file from a URL to a destination path with a progress bar."""
    
    # 1. Ensure the destination directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Attempting to download model from: {url}")
    print(f"Saving to: {dest_path}")
    
    try:
        # Use stream=True to download large files in chunks
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Get the total file size from the header
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 KB

        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)

        print("Download complete.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        # Clean up the partial file if it exists
        if dest_path.is_file():
             dest_path.unlink()
        return False