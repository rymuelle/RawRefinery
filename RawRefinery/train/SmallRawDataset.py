import pandas as pd
import os
from  torch.utils.data import Dataset
import imageio
from colour_demosaicing import (
    demosaicing_CFA_Bayer_Malvar2004)

from RawRefinery.utils.image_utils import cfa_to_sparse, inverse_gamma_tone_curve
import numpy as np
import torch 
import cv2

class SmallRawDataset(Dataset):
    def __init__(self, path, csv, crop_size=180, buffer=10):
        super().__init__()
        self.df = pd.read_csv(os.path.join(path, csv))
        self.path = path
        self.crop_size = crop_size
        self.buffer = buffer
        self.coordinate_iso = 6400

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get Row Matrix
        shape=(2,3)
        cols = [f"m{i}{j}" for i in range(shape[0]) for j in range(shape[1])]
        flat = np.array([row.pop(c) for c in cols], dtype=np.float32)
        warp_matrix = flat.reshape(shape)
        warp_matrix

        # Load images
        with imageio.imopen(f"{self.path}/{row.noisy_image}_bayer.jpg", "r") as image_resource:
            bayer_data = image_resource.read()

        with imageio.imopen(f"{self.path}/{row.gt_image}.jpg", "r") as image_resource:
            gt_image = image_resource.read()
        gt_image  = gt_image/255
        bayer_data = bayer_data/255
        # bayer_data = Image.open(f"{self.path}/{row.noisy_image}_bayer.jpg")
        # bayer_data = np.array(bayer_data)/255.

        # gt_image = Image.open(f"{self.path}/{row.gt_image}.jpg")
        # gt_image = np.array(gt_image)/255.

        # Align GT
        h, w, _ = gt_image.shape
        gt_image = cv2.warpAffine(gt_image, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        #Crop images
        top = np.random.randint(0 + self.buffer, h - self.crop_size - self.buffer)
        if top % 2 != 0: top = top - 1
        left = np.random.randint(0 + self.buffer, w - self.crop_size - self.buffer)
        if left % 2 != 0: left = left - 1
        bottom = top + self.crop_size
        right = left + self.crop_size
        gt_image = gt_image[top:bottom, left:right]
        bayer_data = bayer_data[top:bottom, left:right]
        h, w, _ = gt_image.shape

        # Translate to linear
        gt_image = inverse_gamma_tone_curve(gt_image)
        bayer_data = inverse_gamma_tone_curve(bayer_data)

        demosaiced_noisy = demosaicing_CFA_Bayer_Malvar2004(bayer_data)

        sparse, _ = cfa_to_sparse(bayer_data)
        rggb = bayer_data.reshape(h // 2, 2, w // 2, 2, 1).transpose(3, 1, 4, 0, 2).reshape(4, h // 2, w // 2)

        # Convert to tensors
        output = {
            "bayer": torch.tensor(bayer_data).to(float).clip(0,1), 
            "gt": torch.tensor(gt_image).to(float).permute(2, 0, 1).clip(0,1), 
            "sparse": torch.tensor(sparse).to(float).clip(0,1),
            "noisy": torch.tensor(demosaiced_noisy).to(float).permute(2, 0, 1).clip(0,1), 
            "rggb": torch.tensor(rggb).to(float).clip(0,1),
            "conditioning": torch.tensor([row.iso/self.coordinate_iso]).to(float), 
        }
        return output