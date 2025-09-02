from RawHandler.RawHandler import RawHandler
from RawRefinery.utils.viewing_utils import linear_to_srgb
import torch
from blended_tiling import TilingModule
import numpy as np

class ModelHandler():
    def __init__(self, model, device, n_batches=2):
        self.model = model.to(device)
        self.rh = None
        self.iso = 100
        self.device = device
        self.n_batches = n_batches
        self.colorspace = 'lin_rec2020'

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
        img, denoised_img = tile_image(self.rh, self.device, conditioning, self.model, dims=dims)
        denoised_img = denoised_img * (1 - conditioning[1]/100) + img * conditioning[1]/100
        if apply_gamma:
            img = linear_to_srgb(img)
            denoised_img = linear_to_srgb(denoised_img)
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



from RawRefinery.application.dng_utils import convert_color_matrix, to_dng

def tile_image(rh, device, conditioning, model,
               img_size = 128,
               tile_overlap=0.25,
               n_batches=2,
               dims=None,
               ):

    image_RGGB = rh.as_rggb(dims=dims)
    image_RGB = rh.as_rgb(dims=dims)
    tensor_image = torch.tensor(image_RGGB).unsqueeze(0)

    
    full_size = [image_RGGB.shape[1], image_RGGB.shape[2]]
    tile_size = [img_size, img_size]
    tile_overlap = [tile_overlap, tile_overlap] 

    tiling_module = TilingModule(
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        base_size=full_size,
    )

    full_size = [image_RGGB.shape[1]*2, image_RGGB.shape[2]*2]
    tile_size = [img_size*2, img_size*2]
    tiling_module_rebuild = TilingModule(
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        base_size=full_size,
    )

    tiles = tiling_module.split_into_tiles(tensor_image).float().to(device)
    batches = torch.split(tiles, n_batches)
    
    conditioning_tensor = [conditioning for _ in range(n_batches)]
    conditioning_tensor = torch.tensor(conditioning_tensor)
    conditioning_tensor[:, 0] =conditioning_tensor[:, 0]/6400
    conditioning_tensor = conditioning_tensor.float().to(device)  

    processed_batches = []
    with torch.no_grad():
        for batch in batches:
            B = batch.shape[0]
            processed_batches.append(model(batch, conditioning_tensor[:B,:]))

    tiles = torch.cat(processed_batches, dim=0)
    stitched = tiling_module_rebuild.rebuild_with_masks(tiles).detach().cpu().numpy()[0]
    stitched += image_RGB
    return image_RGB.transpose(1, 2, 0), stitched.transpose(1, 2, 0)
