import numpy as np

def normalized_cross_correlation(im1, im2):
    im1 = im1 - np.mean(im1)
    im2 = im2 - np.mean(im2)
    numerator = np.sum(im1 * im2)
    denominator = np.sqrt(np.sum(im1**2) * np.sum(im2**2))
    return numerator / denominator if denominator != 0 else 0.0