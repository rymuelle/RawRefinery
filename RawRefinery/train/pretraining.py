import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
import random
import numpy as np
import kagglehub
from colour_demosaicing import mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_DDFAPD
from Restorer.utils import numpy_pixel_unshuffle
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from RawRefinery.utils.image_utils import color_jitter_0_1, simulate_sparse, bilinear_demosaic

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
    def __init__(self,  dataset="adityajn105/flickr8k", crop_size=180):
        self.get_data(dataset=dataset)
        self.crop_size = crop_size
        self.annotations = []
        self.images = os.listdir(self.image_dir)
        self.max_noise=0.1

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

        # Color jitter
        image = color_jitter_0_1(image)
        
        # Sparse images
        sparse_image = simulate_sparse(image.transpose(2, 0, 1))
        
        #Add noise
        iso = lognuniform(size=1, low=100/65535, high=1,)[0]*65535
        noise_levels = generate_noise_level(iso)
        conditioning = [*noise_levels]
        W, H, C = image.shape
        # print(noise_levels, np.array(noise_levels).reshape(-1, 1, 1))
        noise = np.random.normal(0, size = [C, W, H])*np.array(noise_levels).reshape(-1, 1, 1)
        sparse_image += noise
        print(sparse_image.shape)
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
        }
        return output
    


def generate_noise_level(iso):
    noise_level = (iso/12800)**.5
    r_level = noise_level * (0.0013*np.random.randn()+0.0082)
    g_level = noise_level * (0.0013*np.random.randn()+0.0082)
    b_level = noise_level * (0.0013*np.random.randn()+0.0041)
    return r_level, g_level, b_level


def lognuniform(low=0, high=1, size=None, base=np.e):
    return np.power(base, np.random.uniform(low, high, size))/base