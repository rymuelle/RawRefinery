import numpy as np
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *

def get_ratios(string, rh):
    return [x.as_integer_ratio() for x in rh.full_metadata[string].values]

def to_dng(uint_img, rh, filepath, ccm1):
    width = uint_img.shape[1]
    height = uint_img.shape[0]
    bpp = 16 

    exposures = get_ratios('EXIF ExposureTime', rh)
    fnumber = get_ratios('EXIF FNumber', rh)
    ExposureBiasValue = get_ratios('EXIF ExposureBiasValue', rh) 
    FocalLength = get_ratios('EXIF FocalLength', rh) 


    t = DNGTags()
    t.set(Tag.ImageWidth, width)
    t.set(Tag.ImageLength, height)
    t.set(Tag.BitsPerSample, [bpp, bpp, bpp]) # 3 channels for RGB

    t.set(Tag.SamplesPerPixel, 3) # 3 for RGB
    t.set(Tag.PlanarConfiguration, 1) # 1 for chunky (interleaved RGB)

    t.set(Tag.TileWidth, width)
    t.set(Tag.TileLength, height)
    t.set(Tag.Orientation, rh.full_metadata['Image Orientation'].values[0])
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Linear_Raw)
    t.set(Tag.BlackLevel,[0,0,0])
    t.set(Tag.WhiteLevel, [65535, 65535, 65535])

    t.set(Tag.BitsPerSample, bpp)

    t.set(Tag.ColorMatrix1, ccm1)
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
    # t.set(Tag.AsShotNeutral, [[1,1],[1,1],[1,1]])
    t.set(Tag.BaselineExposure, [[0,100]])
    t.set(Tag.Make, rh.full_metadata['Image Make'].values)
    t.set(Tag.Model, rh.full_metadata['Image Model'].values)



    t.set(Tag.FocalLength, FocalLength)
    t.set(Tag.EXIFPhotoLensModel, rh.full_metadata['EXIF LensModel'].values)
    t.set(Tag.ExposureBiasValue, ExposureBiasValue)
    t.set(Tag.ExposureTime, exposures)
    t.set(Tag.FNumber, fnumber)
    t.set(Tag.PhotographicSensitivity, rh.full_metadata['EXIF ISOSpeedRatings'].values)
    t.set(Tag.DNGVersion, DNGVersion.V1_4)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.Adobe_RGB)

    r = RAW2DNG()

    r.options(t, path="", compress=False)

    r.convert(uint_img, filename=filepath)



def convert_color_matrix(matrix):
  """
  Converts a 3x3 NumPy matrix of floats into a list of integer pairs.

  Each float value in the matrix is converted to a fractional representation
  with a denominator of 10000. The numerator is calculated by scaling the
  float value by 10000 and rounding to the nearest integer.

  Args:
    matrix: A 3x3 NumPy array with floating-point numbers.

  Returns:
    A list of 9 lists, where each inner list contains two integers
    representing the numerator and denominator.
  """
  # Ensure the input is a NumPy array
  if not isinstance(matrix, np.ndarray):
    raise TypeError("Input must be a NumPy array.")

  # Flatten the 3x3 matrix into a 1D array of 9 elements
  flattened_matrix = matrix.flatten()

  # Initialize the list for the converted matrix
  converted_list = []
  denominator = 10000

  # Iterate over each element in the flattened matrix
  for element in flattened_matrix:
    # Scale the element, round it to the nearest integer, and cast to int
    numerator = int(round(element * denominator))
    # Append the [numerator, denominator] pair to the result list
    converted_list.append([numerator, denominator])

  return converted_list