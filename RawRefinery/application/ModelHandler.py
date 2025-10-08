from RawHandler.RawHandler import RawHandler
from RawRefinery.utils.viewing_utils import linear_to_srgb
import torch
from blended_tiling import TilingModule
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from RawRefinery.application.dng_utils import convert_color_matrix, to_dng


class ModelHandler():
    def __init__(self, model, device, n_batches=2, colorspace = 'lin_rec2020'):
        self.model = model.to(device)
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
            output = model(batch, conditioning_tensor[:B, :])
            processed_batches.append(output)

    tiles = torch.cat(processed_batches, dim=0)
    stitched = tiling_module_rebuild.rebuild_with_masks(tiles).detach().cpu().numpy()[0]
    stitched += image_RGB
    return image_RGB.transpose(1, 2, 0), stitched.transpose(1, 2, 0)


def tile_image_sparse(rh, device, conditioning, model,
               img_size = 256,
               tile_overlap=0.25,
               n_batches=2,
               dims=None,
               ):

    image_sparse = rh.as_sparse(dims=dims)
    image_RGB = rh.as_rgb(dims=dims, demosaicing_func=demosaicing_CFA_Bayer_Malvar2004)
    tensor_sparse = torch.tensor(image_sparse).unsqueeze(0)
    tensor_RGB = torch.tensor(image_RGB).unsqueeze(0)

    
    full_size = [image_sparse.shape[1], image_sparse.shape[2]]
    tile_size = [img_size, img_size]
    tile_overlap = [tile_overlap, tile_overlap] 

    tiling_module = TilingModule(
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        base_size=full_size,
    )

    tiles_sparse = tiling_module.split_into_tiles(tensor_sparse).float().to(device)
    tiles_rgb = tiling_module.split_into_tiles(tensor_RGB).float().to(device)
    sparse_batches = torch.split(tiles_sparse, n_batches)
    rgb_batches = torch.split(tiles_rgb, n_batches)
    
    conditioning_tensor = [conditioning for _ in range(n_batches)]
    conditioning_tensor = torch.tensor(conditioning_tensor)
    conditioning_tensor[:, 0] = conditioning_tensor[:, 0]/6400
    conditioning_tensor = conditioning_tensor[:, :1]
    conditioning_tensor = conditioning_tensor.float().to(device)  

    processed_batches = []
    with torch.no_grad():
        for sparse, rgb in zip(sparse_batches, rgb_batches):
            B = sparse.shape[0]
            processed_batches.append(model(sparse, conditioning_tensor[:B,], rgb))

    tiles = torch.cat(processed_batches, dim=0)
    stitched = tiling_module.rebuild_with_masks(tiles).detach().cpu().numpy()[0]
    # stitched += image_RGB
    return image_RGB.transpose(1, 2, 0), stitched.transpose(1, 2, 0)
