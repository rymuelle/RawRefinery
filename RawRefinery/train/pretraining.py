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

    def compute_noise_level(self, no_noise_amount = 0.1):
        # Any negative number should not remove noise
        if random.random() < no_noise_amount:
            return 0
        return random.random() * self.max_noise
    
    def compute_conditioning(self, min=-0.1, max = 1.1, h_scale=0.0, s_scale=1, v_scale=1):
        def rand():
            slope = max-min
            return random.random() * slope + min
        noise = self.compute_noise_level()
        noise_conditioning = noise/self.max_noise
        noise_conditioning = -1 if noise_conditioning <= 0 else noise_conditioning
        return [noise_conditioning, rand()*h_scale, rand()*s_scale, rand()*v_scale], noise

    def process_mosaic(self, _mosaic_img):
        _mosaic_img = np.expand_dims(_mosaic_img,axis=-1)
        _rggb_img = numpy_pixel_unshuffle(_mosaic_img)
        _demosaic = demosaicing_CFA_Bayer_DDFAPD(_mosaic_img)
        return _rggb_img, _demosaic
    
    def compute_final_range(self, min_range=-1, max_range=5):
        rand_range = np.random.rand(2)
        rand_range.sort()
        new_range = reverse_scale(rand_range, (-1, 5))
        slope = new_range[1]-new_range[0]
        return reverse_scale(rand_range, (-1, 5)), slope
    
    def __getitem__(self, idx):
        # Numpy array [W, H, C]
        image = self.get_image(idx)

        #Add noise
        conditioning, noise_level = self.compute_conditioning()
        W, H, C = image.shape
        noise = np.random.normal(0, noise_level, [W, H])
        _, noise_demosaiced = self.process_mosaic(noise)
        conditioned_noise = transform_noise(noise_demosaiced, conditioning)
        # Randomize range of image
        # image_range = (-1.5, 5)
        # rand_range = np.random.random(2)
        # rand_range.sort()
        # rand_range = reverse_scale(rand_range, image_range)
        # scaled_image = reverse_scale(image, rand_range)
        scaled_image = image
        # Make model input
        mosaic_img = mosaicing_CFA_Bayer(scaled_image)
        noisy_rggb, noisy_demosaiced = self.process_mosaic(mosaic_img+noise)
        target_image = scaled_image + conditioned_noise
        target_image = np.clip(target_image, 0, 1)

        
        # Convert to tensor
        rggb_tensor = ToTensor()(noisy_rggb).float()
        image_tensor = ToTensor()(scaled_image).float()
        target_image_tensor  = ToTensor()(target_image).float()
        conditioning_tensor  = torch.tensor(conditioning).float()
        noisy_demosaiced_tensor = torch.tensor(noisy_demosaiced).float()
        return rggb_tensor, image_tensor, target_image_tensor, conditioning_tensor, noisy_demosaiced_tensor