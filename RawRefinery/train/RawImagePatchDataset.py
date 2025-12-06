import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import random
import cv2
from RawHandler.RawHandler import RawHandler
from RawRefinery.utils.traing_utils import normalized_cross_correlation

class RawImagePatchDataset(Dataset):
    def __init__(self, csv_path, patch_size=256, cc_threshold=0.92, colorspace='sRGB'):
        self.df = pd.read_csv(csv_path)

        # Drop rows with missing 'cc' or invalid types
        self.df = self.df[pd.to_numeric(self.df['cc'], errors='coerce').notnull()]
        self.df['cc'] = self.df['cc'].astype(float)

        # Filter based on cc threshold
        original_len = len(self.df)
        self.df = self.df[self.df['cc'] >= cc_threshold]
        filtered_len = len(self.df)

        print(f"[Dataset] Filtered out {original_len - filtered_len} rows below cc threshold of {cc_threshold}.")
        
        self.patch_size = patch_size
        self.colorspace = colorspace

    def __len__(self):
        return len(self.df)

    def blend_noise(self, proportion, min, gt, noisy):
        noisy_hsv = np.clip(noisy, 0, 1)
        noisy_hsv = rgb_to_hsv(noisy_hsv)

        gt_hsv = np.clip(gt, 0, 1)
        gt_hsv = rgb_to_hsv(gt_hsv)

        delta = noisy_hsv[:, :, 2]-gt_hsv[:, :, 2]
        scale = 10
        delta = np.tanh(delta*scale)/scale
        gt_hsv[:, :, 2] += delta**1.0 * (gt_hsv[:, :, 2]*proportion + min)

        pred_mix = hsv_to_rgb(pred_hsv)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load images using RawHandler
        gt = RawHandler(row["gt_image"], colorspace=self.colorspace)

        # Get dimensions
        H, W = gt.raw.shape  # Assume same for noisy and gt

        # Random crop coordinates
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"Image is smaller than patch size: {H}x{W} < {self.patch_size}")

        align_offset = 20
        x1 = random.randint(0 + align_offset, W - (self.patch_size + align_offset))
        y1 = random.randint(0 + align_offset, H - (self.patch_size + align_offset))
        return self.get_patches(idx, x1, y1)


    def get_patches(self, idx, x1, y1):
            row = self.df.iloc[idx]

            # Load images using RawHandler
            noisy = RawHandler(row["noisy_image"], colorspace=self.colorspace)
            gt = RawHandler(row["gt_image"], colorspace=self.colorspace)

            # Get dimensions
            H, W = gt.raw.shape  # Assume same for noisy and gt

            # Random crop coordinates
            if H < self.patch_size or W < self.patch_size:
                raise ValueError(f"Image is smaller than patch size: {H}x{W} < {self.patch_size}")

            align_offset = 20
            x2 = x1 + self.patch_size
            y2 = y1 + self.patch_size
            crop_dim = (y1, y2, x1, x2)

            # Convert to float32 numpy arrays
            expand_crop_dim = (y1-align_offset, y2+align_offset, x1-align_offset, x2+align_offset)
            noisy_patch = noisy.as_rgb(dims=expand_crop_dim).astype(np.float32)
            noisy_patch = noisy_patch[:, align_offset:-align_offset, align_offset:-align_offset]
            align_crop_dim = (y1-align_offset, y2+align_offset, x1-align_offset, x2+align_offset)
            gt_patch = gt.as_rgb(dims=align_crop_dim).astype(np.float32)

            # Align gt image
            gt_image = gt_patch.transpose(1, 2, 0)
            noisy_image = noisy_patch.transpose(1, 2, 0)

            warp_matrix = np.array([[1, 0, -row.x_warp + align_offset],
                        [0, 1, -row.y_warp + align_offset]])
            gt_image = cv2.warpAffine(gt_image, warp_matrix, 
                                (noisy_image.shape[1], noisy_image.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
            ncc = normalized_cross_correlation(gt_image, noisy_image)
            noise_level = (gt_image-noisy_image).std(axis=(0, 1))

            gt_patch = gt_image.transpose(2, 0, 1)

            # Adjust brightness
            seperate_channels = False
            if seperate_channels:
                gains = gt_patch.mean(axis=(1, 2))/noisy_patch.mean(axis=(1, 2))

                # Known issue, since the gt will clip at 1 and adjust by 2, we might have a max val of .5
                gt_patch *= 1/gains.reshape(3, 1, 1)
                gt_patch = np.clip(gt_patch, 0, 1)
            else:
                gains = gt_patch.mean()/noisy_patch.mean()
                gt_patch *= 1/gains
                gt_patch = np.clip(gt_patch, 0, 1)

            # RGGB 
            noisy_rggb = noisy.as_rggb( dims=crop_dim)
            
            # Convert to tensors and permute to CHW format
            noisy_tensor = torch.from_numpy(noisy_patch).float()
            gt_tensor = torch.from_numpy(gt_patch).float()
            noisy_rggb_tensor = torch.from_numpy(noisy_rggb).float()
            conditioning = torch.tensor([float(row['iso'])/6400, 0, 0, 0]).float()
            return {
                'noisy': noisy_tensor,
                'gt': gt_tensor,
                "noisy_rggb_tensor": noisy_rggb_tensor,
                "conditioning": conditioning, 
                'idx': idx,
                'cc': row['cc'],
                'ncc': ncc,
                'noise_level': noise_level,
                'iso': row['iso'],
                'x_warp': row['x_warp'],
                'y_warp': row['y_warp'],
                'gt_mean': row['gt_mean'],
                'noisy_mean': row['noisy_mean'],
                'gains': gains
            }
