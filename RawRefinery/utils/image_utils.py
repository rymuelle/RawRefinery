import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def simulate_sparse(image, pattern="RGGB", cfa_type="bayer"):
    """
    Simulate a sparse CFA (Color Filter Array) from an RGB image.

    Args:
        image: numpy array (3, H, W), RGB image in [0, 1] or [0, 255].
        pattern: CFA pattern string, one of {"RGGB","BGGR","GRBG","GBRG"} for Bayer,
                 or ignored if cfa_type="xtrans".
        cfa_type: "bayer" or "xtrans".

    Returns:
        cfa: numpy array (r, H, W), sparse CFA image.
    """
    _, H, W= image.shape
    cfa = np.zeros((3, H, W), dtype=image.dtype)

    if cfa_type == "bayer":
        # 2×2 Bayer masks
        masks = {
            "RGGB": np.array([["R", "G"], ["G", "B"]]),
            "BGGR": np.array([["B", "G"], ["G", "R"]]),
            "GRBG": np.array([["G", "R"], ["B", "G"]]),
            "GBRG": np.array([["G", "B"], ["R", "G"]]),
        }
        if pattern not in masks:
            raise ValueError(f"Unknown Bayer pattern: {pattern}")

        mask = masks[pattern]
        cmap = {"R": 0, "G": 1, "B": 2}

        for i in range(2):
            for j in range(2):
                ch = cmap[mask[i, j]]
                cfa[ch, i::2, j::2] = image[ch, i::2, j::2]

    elif cfa_type == "xtrans":
        # Fuji X-Trans 6×6 repeating pattern
        xtrans_pattern = np.array([
            ["G","B","R","G","R","B"],
            ["R","G","G","B","G","G"],
            ["B","G","G","R","G","G"],
            ["G","R","B","G","B","R"],
            ["B","G","G","R","G","G"],
            ["R","G","G","B","G","G"],
        ])
        cmap = {"R":0, "G":1, "B":2}

        for i in range(6):
            for j in range(6):
                ch = cmap[xtrans_pattern[i, j]]
                cfa[ch, i::6, j::6] = image[ch, i::6, j::6]

    else:
        raise ValueError(f"Unknown CFA type: {cfa_type}")

    return cfa


def color_jitter_0_1(img, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
    """
    Applies color jitter to a NumPy image array with pixel values scaled 0-1.

    Args:
        img (np.ndarray): Input image as a NumPy array (H, W, 3) with values in [0, 1].
        brightness (float): Max variation for brightness factor.
        contrast (float): Max variation for contrast factor.
        saturation (float): Max variation for saturation factor.
        hue (float): Max variation for hue factor.

    Returns:
        np.ndarray: The jittered image array with values clipped to [0, 1].
    """
    # Ensure the image is a float type for calculations
    img = img.astype(np.float32)

    # 1. Adjust brightness
    if brightness > 0:
        b_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        img = img * b_factor

    # 2. Adjust contrast
    if contrast > 0:
        c_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        img = img * c_factor + (1 - c_factor) * np.mean(img, axis=(0, 1))

    # Convert to HSV to adjust saturation and hue.
    # PIL expects uint8, so we scale from [0, 1] to [0, 255]
    img_pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    img_hsv = np.array(img_pil.convert('HSV'), dtype=np.float32)

    # 3. Adjust saturation
    if saturation > 0:
        s_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        img_hsv[:, :, 1] *= s_factor

    # 4. Adjust hue
    if hue > 0:
        h_factor = np.random.uniform(-hue, hue)
        # Hue in Pillow is 0-255, so we scale our factor
        img_hsv[:, :, 0] += h_factor * 255.0

    # Clip HSV values to [0, 255] range
    img_hsv = np.clip(img_hsv, 0, 255)

    # Convert back to RGB and scale back to [0, 1]
    img_jittered_pil = Image.fromarray(img_hsv.astype(np.uint8), 'HSV').convert('RGB')
    jittered_img = np.array(img_jittered_pil, dtype=np.float32) / 255.0

    # Final clipping to ensure output is in [0, 1]
    final_img = np.clip(jittered_img, 0, 1)

    return final_img




def bilinear_demosaic(sparse, pattern="RGGB", cfa_type="bayer"):
    """
    Simple bilinear demosaicing for Bayer and X-Trans CFAs.

    Args:
        sparse: numpy array (3, H, W) with a sparse representation of the bayer image.
        pattern: CFA pattern string, one of {"RGGB","BGGR","GRBG","GBRG"} for Bayer.
        cfa_type: "bayer" or "xtrans".

    Returns:
        rgb: numpy array (H, W, 3), bilinearly demosaiced image.
    """
    C, H, W = sparse.shape
    rgb = np.zeros((C, H, W), dtype=sparse.dtype)

    if cfa_type == "bayer":
        # Bilinear interpolation kernel
        kernel = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]], dtype=np.float32)
        kernel /= kernel.sum()

    elif cfa_type == "xtrans":
        # Bilinear interpolation kernel
        kernel = np.array([
            [1, 2, 4, 2, 1],
            [2, 4, 8, 4, 2],
            [4, 8, 16, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1],], dtype=np.float32)
        kernel /= kernel.sum()

    else:
        raise ValueError(f"Unknown CFA type: {cfa_type}")

    # Interpolate each channel
    for ch in range(3):
        rgb[ch, ...] = convolve(sparse[ch], kernel, mode="mirror")

    # Mulitply by 2 or 4 depending on how sparse the layer is.
    rgb[0,...] *= 4.0   # red
    rgb[1,...] *= 2.0   # green (already dense)
    rgb[2,...] *= 4.0   # blue
    return rgb