import numpy as np
from PySide6.QtGui import QImage


def numpy_to_qimage_rgb(array):
    array = apply_gamma(array)
    array_uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    array_uint8 = np.ascontiguousarray(array_uint8)
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError("Expected array shape (H, W, 3) for RGB")
    
    # Convert to bytes
    bytes_per_line = 3 * width
    return QImage(array_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888)

def apply_gamma(img, gamma: float = 2.2):
    img = img.clip(0, 1)  
    return (img ** (1.0 / gamma)).clip(0, 1) 

def apply_gamma_v2(tensor):   
    img_mask = tensor > 0.0031308
    tensor[img_mask] = (
        1.055 * np.pow(tensor[img_mask], 1.0 / 2.4) - 0.055
    )
    tensor[~img_mask] *= 12.92
    return tensor
