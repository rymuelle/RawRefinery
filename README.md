# RawRefinery

**RawRefinery** is an open-source application for **raw image quality refinement and denoising**.

Currently in **alpha release**, RawRefinery provides a high-quality raw image denoising tool that works directly on most Bayer image formats, including those from **Canon, Nikon, and Sony** cameras.

---

### Example: Denoising Performance

<div align="center">
  <img src="https://github.com/rymuelle/RawRefinery/blob/main/examples/Bayer_TEST_MuseeL-bluebirds-A7C_ISO65535_sha1=eb9cb3e1d80f48b93d0aabe20458870c5c1ef2fa.jpg" alt="Noisy Image" width="400"/>
  <img src="https://github.com/rymuelle/RawRefinery/blob/main/examples/Bayer_TEST_MuseeL-bluebirds-A7C_ISO65535_sha1=eb9cb3e1d80f48b93d0aabe20458870c5c1ef2fa_65534_denoised.DNG.jpg" alt="Denoised" width="400"/>
</div>

---

## Overview

![RawRefinery main window](https://github.com/rymuelle/RawRefinery/blob/main/examples/RawRefinery.png)

RawRefinery lets you visually explore and refine your raw files through a simple, intuitive interface.
It uses a deep learning–based denoising model designed to preserve fine image detail while removing noise, even at very high ISO values.

---

## Installation

Download the macOS build here:
[**RawRefinery v1.1.0-alpha**](https://github.com/rymuelle/RawRefinery/releases/tag/v1.1.0-alpha)

Windows and Linux versions are coming soon, along with setup instructions for building from source.

---

## 🚀 Usage

1. **First launch:**
   On first run, the app will download the denoising model (~500 MB).

2. **Selecting files:**
   Choose a directory containing your raw files. Selecting a file from the left panel displays:

   * A **thumbnail** (left)
   * A **100% preview** (right)

   Clicking on the thumbnail updates the preview region.

3. **Adjusting settings:**
   Two sliders are available below the preview:

   * **ISO conditioning:** Adjusts model sensitivity to match the image ISO. It defaults to the detected ISO but can be fine-tuned.
   * **Blend:** Mixes a portion of the original image back into the result to recover subtle high-frequency details.

   After changing either slider, click **“Preview Denoise”** to re-render the preview.

4. **Saving results:**
   When satisfied, click **“Save Denoised Image”** to export a denoised `.DNG`.
   Processing may take up to a minute and use several GB of memory (RAM or VRAM depending on the backend).
   Supported backends: **CUDA**, **MPS**, and **CPU**.

---

### Model Architecture and Training

The model training code is currently being documented and refactored here:

https://github.com/rymuelle/Restorer/tree/feature/mps

The feature/mps branch contains the base model architecture and the training code, however, it is a current work in progress. 

---

## Development Status

RawRefinery is currently in **pre-release alpha**, and testing is ongoing.
Bug reports and feedback on denoising performance are greatly appreciated.

### Current priorities:

1. Add Windows and Linux support
2. Integrate newer denoising models
3. Improve user experience — progress bars, threading, and real-time responsiveness

Feature requests and community contributions are welcome!

---

### Feedback

If you encounter issues or have suggestions, please open an [issue](https://github.com/rymuelle/RawRefinery/issues) on GitHub.
You can also share before/after results or performance feedback to help guide model improvements.

----

## Acknowledgments

With thanks to the creators of the RawNIND dataset.


Brummer, Benoit; De Vleeschouwer, Christophe, 2025, "Raw Natural Image Noise Dataset", https://doi.org/10.14428/DVN/DEQCIM, Open Data @ UCLouvain, V1 