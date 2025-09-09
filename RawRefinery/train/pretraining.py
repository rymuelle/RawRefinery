import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
import random
import numpy as np
import kagglehub
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from RawRefinery.utils.image_utils import color_jitter_0_1, simulate_sparse, bilinear_demosaic, inverse_gamma_tone_curve

def scale(x, value_range):
    min_val, max_val = value_range
    return (x - min_val) / (max_val - min_val + 1e-8)

def reverse_scale(x, value_range):
    min_val, max_val = value_range
    return x * (max_val - min_val + 1e-8) + min_val

def mix_hsv_noise(image, noise, conditioning):
    # Add noise to image
    noisy_image = image + noise

    # Compute global min/max for scaling (to normalize to [0, 1])
    combined_min = min(noisy_image.min(), image.min())
    combined_max = max(noisy_image.max(), image.max())

    # Scale both to [0, 1] for safe HSV conversion
    image_norm = scale(image, (combined_min, combined_max))
    noisy_image_norm = scale(noisy_image, (combined_min, combined_max))

    # Convert to HSV
    hsv = rgb_to_hsv(image_norm)
    hsv_noisy = rgb_to_hsv(noisy_image_norm)

    # Apply noise in HSV space
    hsv_noise = hsv_noisy - hsv
    hsv_noise[:, :, 0] *= conditioning[1]  # hue
    hsv_noise[:, :, 1] *= conditioning[2]  # saturation
    hsv_noise[:, :, 2] *= conditioning[3]  # value

    # Add noise and convert back to RGB
    blended_hsv = hsv + hsv_noise
    blended_hsv = np.clip(blended_hsv, 0.0, 1.0)  # make sure hsv stays valid
    blended_rgb = hsv_to_rgb(blended_hsv)

    # Reverse scaling
    noisy_image_final = reverse_scale(blended_rgb, (combined_min, combined_max))
    image_final = reverse_scale(image_norm, (combined_min, combined_max))

    return noisy_image_final, image_final

def transform_noise(_rgb_noise, conditioning):
    s = conditioning[2]
    v = conditioning[3]
    _hsv_noise = rgb_to_hsv( np.clip(_rgb_noise+0.5, 0.0, 1.0))
    _hsv_noise[:, :, 1] = s*_hsv_noise[:, :, 1] + (1-s)*0    # saturation
    _hsv_noise[:, :, 2] = v*_hsv_noise[:, :, 2] + (1-v)*0.5    # value
    rgb_noise = hsv_to_rgb(_hsv_noise)-0.5
    return rgb_noise

class Flickr8kDataset(Dataset):
    def __init__(self,  dataset="adityajn105/flickr8k", crop_size=180, cfa_type='bayer', max_iso=156800, min_iso=100):
        self.get_data(dataset=dataset)
        self.crop_size = crop_size
        self.annotations = []
        self.images = os.listdir(self.image_dir)
        self.max_iso=max_iso
        self.min_iso=min_iso
        self.cfa_type = cfa_type

    def get_data(self, dataset="adityajn105/flickr8k"):
        path = kagglehub.dataset_download(dataset)
        self.path = path
        self.captions_dir = os.path.join(path, 'captions.txt')
        self.image_dir = os.path.join(path, 'Images')

    def __len__(self):
        return len(self.images)
    
    def get_image(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB') 
        # Random crop image to crop size
        # If image is smaller than crop size, interpolate it up
        w, h = image.size
        if w < self.crop_size or h < self.crop_size:
            image = image.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        w, h = image.size
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        bottom = top + self.crop_size
        right = left + self.crop_size
        image = image.crop((left, top, right, bottom))
        image = np.array(image).astype(np.float16)
        image *= 1./255
        return image.astype(float)


    
    def __getitem__(self, idx):
        # Numpy array [W, H, C]
        image = self.get_image(idx)

        #Scale from 0 to 1
        image -= image.min()
        # # image *= 1/image.max()
        # image*=0
        #Inverse tone curve
        image = inverse_gamma_tone_curve(image, gamma=3)

        # Color jitter
        image = color_jitter_0_1(image)
        
        # Sparse images
        sparse_image, sparse_mask = simulate_sparse(image.transpose(2, 0, 1), cfa_type=self.cfa_type)
        #Add noise
        iso = gen_iso(low=self.min_iso, high=self.max_iso)
        # if np.random.random() < 0.1:
        #     iso = 0
        noise_levels = generate_noise_level(iso)
        conditioning = [*noise_levels]
        W, H, C = image.shape
        noise = np.random.normal(0, size = [C, W, H])*np.array(noise_levels).reshape(-1, 1, 1)
        sparse_image += noise * sparse_mask
        bilinear_image = bilinear_demosaic(sparse_image)
        sparse_image = np.clip(sparse_image, 0, 1)
        bilinear_image = np.clip(bilinear_image, 0, 1)
        image = np.clip(image, 0, 1)
        # Convert to tensor
        bilinear_image = torch.tensor(bilinear_image).float()
        sparse_image = torch.tensor(sparse_image).float()
        image  = torch.tensor(image).permute(2, 0, 1).float()
        conditioning_tensor  = torch.tensor(conditioning).float()

        output = {
                "bilinear_image": bilinear_image,
                "sparse_image": sparse_image,
                "image": image,
                "conditioning_tensor": conditioning_tensor,
                "iso": iso,
                "sparse_mask": sparse_mask,
        }
        return output
    


def generate_noise_level(iso):
    base_noise_level = (iso/12800)**.5*0.02
    r_level = base_noise_level * 0.11 / 0.07
    g_level = base_noise_level * 0.05 / 0.07
    b_level = base_noise_level * 0.03 / 0.07
    return r_level, g_level, b_level



# def gen_iso(low=25, high=65535, size=None, base=np.e):
#     log_low = np.log(low) / np.log(base)
#     log_high = np.log(high) / np.log(base)
#     return base ** np.random.uniform(log_low, log_high, size=size)

def gen_iso(low=100, high=156800):
    uniform_log_samples = np.random.uniform(low=np.log(low), high=np.log(high), size=1)
    random_logspace_numbers = np.exp(uniform_log_samples)[0]
    return random_logspace_numbers