import unittest
import torch
import numpy as np
from RawHandler.RawHandler import BaseRawHandler, CoreRawMetadata
from RawRefinery.application.ModelHandler import ModelHandler

## This simple test was added so I could quickly verify proper functioning, but the model and rawhandler should be replaced with a mock

class TestModelHandler(unittest.TestCase):
    def test_tile(self):

        # Simulate raw file
        dim = 256
        N = dim * dim
        max_uint16 = 2**16 - 1  
        data_1d = np.arange(N, dtype=np.float32)
        scaled_data = (data_1d / N) * max_uint16
        bayer = scaled_data.astype(np.uint16).reshape(dim, dim)    
        
        core_metadata = CoreRawMetadata(
            black_level_per_channel=[0, 0, 0, 0],
            white_level=65535,
            rgb_xyz_matrix=np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]),
            raw_pattern=np.array([[1,2],[2,3]]),
            iheight=dim,
            iwidth=dim,
        )
        rh = BaseRawHandler(bayer, core_metadata, colorspace="lin_rec2020")

        # Create model handler
        device = torch.device('cpu')
        mh = ModelHandler("RGGB_v1_trace.pt", device, colorspace='lin_rec2020')
        mh.rh = rh

        # Tile
        img_rgb, denoised = mh.tile([100, 0])

        # Check shape
        assert denoised.shape == (dim, dim, 3), "Output has wrong dimensions"

        assert img_rgb.shape == (dim, dim, 3), "Output has wrong dimensions"

if __name__ == "__main__":
    unittest.main()