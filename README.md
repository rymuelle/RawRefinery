# RawRefinery

RawRefinery is an open-source application for raw image quality refinement. 

THe application is in it's alpha release, and offers a high quality raw image de-noising tool that works on most bayer image files, such as those from Canon, Nikon, and Sony cameras. 

Below is an example of the denoising performance!

<div align="center">
  <img src="https://github.com/rymuelle/RawRefinery/blob/main/examples/Bayer_TEST_MuseeL-bluebirds-A7C_ISO65535_sha1=eb9cb3e1d80f48b93d0aabe20458870c5c1ef2fa.jpg" alt="Noisy Image" width="400"/>
  <img src="https://github.com/rymuelle/RawRefinery/blob/main/examples/Bayer_TEST_MuseeL-bluebirds-A7C_ISO65535_sha1=eb9cb3e1d80f48b93d0aabe20458870c5c1ef2fa_65534_denoised.DNG.jpg" alt="Denoised" width="400"/>
</div>




## Usage
![RawRefinery main window](https://github.com/rymuelle/RawRefinery/blob/main/examples/RawRefinery.png)


### Installation
Download the DMG here: https://github.com/rymuelle/RawRefinery/releases/tag/v1.1.0-alpha

Windows and linux applications will be provides shortly. In addition, instructions to install from source will be provided. 

### Use
Upon first usage, the application will download the denoising model (roughly 500mb). Afterwards, you will be able to select a directory of raw files. 

Upon selecting a raw file in the left pain, you will be shown a thumbnail (left) and 100% preview (right) in the main window. By clicking on the thumbnail, you may view that section of the image at 100%. 

Below the previews, there are two sliders. The first slider conditions the model on the ISO of the image. It should default to the ISO value of the image, but you may adjust it as well. The second slider blends some of the original image into the results. This allows for some high frequency detail to be added back in. 


After adjusting a slider, you must press the "preview denoise" button for the preview to be re-rendered. 

When you are satisfied with the results, you can produce a denoised DNG with the "Save Denoised Image". The program will use several GB of memory (ram or vram, depending on the accelerator used) and take up to a minute to process the image. CUDA, CPU, and MPS backends are supported.

## Status of the program

This is a prerelease alpha, and testing is in progress. Please report any bugs encountered. Model performance is actively being improved. 

In addition, several features are actively being developed. Current priorites are:

1. Providing Windows and Linux support
2. Incorporating newly developed models that result
3. Improving user experience, including progress bar and threading so the program is operatable while denoising.

Requests for features and feedback on model performance is welcome. 