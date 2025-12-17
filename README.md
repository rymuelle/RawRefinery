# RawRefinery

[![PyPI version](https://img.shields.io/pypi/v/rawrefinery.svg)](https://pypi.org/project/rawrefinery/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python version](https://img.shields.io/pypi/pyversions/rawrefinery.svg)](https://pypi.org/project/rawrefinery/)

**RawRefinery** is an open-source application for **raw image quality refinement and denoising**.

Currently in **alpha release**, RawRefinery provides a high-quality raw image denoising tool that works directly on most Bayer image formats, including those from **Canon, Nikon, and Sony** cameras.

---

### Example: Denoising Performance

Before and after image denoising + deblurring performance for an ISO 102400(!) photo taken with an A7RII. This image is not included in any training set and represents real world performance. Click for to see larger versions of either image.

<div align="center">
  <a href="https://github.com/rymuelle/RawRefinery/blob/main/examples/brushes_4k.jpg">
    <img src="https://github.com/rymuelle/RawRefinery/blob/main/examples/brushes_crop.jpg" width="400" />
  </a>
  <a href="https://github.com/rymuelle/RawRefinery/blob/main/examples/brushes_4k_denoised.jpg">
    <img src="https://github.com/rymuelle/RawRefinery/blob/main/examples/brushes_denoised_crop.jpg" width="400" />
  </a>
</div>

---

## Overview


![RawRefinery main window](https://github.com/rymuelle/RawRefinery/blob/main/examples/RawRefinery_v1.3.0.png)

RawRefinery lets you visually explore and refine your raw files through a simple, intuitive interface.
It uses a deep learning–based denoising model designed to preserve fine image detail while removing noise, even at very high ISO values.

---

## Installation 

### Install via PyPI (All OS):

RawRefinery is now on PyPI! 

https://pypi.org/project/rawrefinery/1.3.0/

You can install with pip as follows with python 3.11+:

#### Setting up the enviroment
Example instructions for setting up the enviroment in an apt based linux OS.

##### 1. Set up Python

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv build-essential python3-dev libxcb-cursor0

python3 -m venv .venv
source .venv/bin/activate
```

##### 2. Set up Torch

CUDA support depends on your GPU model.  
Before installing RawRefinery, install a compatible version of **PyTorch**.

You can find the correct PyTorch install command for your system here:
https://pytorch.org/get-started/locally/

e.g. For a 10XX GPU with CUDA 12.6, you may want:
```bash
pip install torch==2.6.0  --index-url https://download.pytorch.org/whl/test/cu126
```

For CPU usage, torch 2.8 is tested:

```bash
pip install torch~=2.8
```

##### 3. Install via PyPI

```bash
python -m pip install rawrefinery
```

##### 4. Run the application!


And run in bash:

```bash
rawrefinery
```


### Build from source from git (linux):

https://github.com/rymuelle/RawRefinery/tree/main/linux

### Build from source from git (MacOS):

Pending

### Download installer

[Download the macOS build here.](https://github.com/rymuelle/RawRefinery/releases/download/v1.3.0-alpha/RawRefinery_v1.3.0.dmg)


---

## Usage

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

With thanks to:


> Brummer, Benoit; De Vleeschouwer, Christophe. (2025).
> *Raw Natural Image Noise Dataset.*
> [https://doi.org/10.14428/DVN/DEQCIM](https://doi.org/10.14428/DVN/DEQCIM), Open Data @ UCLouvain, V1.

> Chen, Liangyu; Chu, Xiaojie; Zhang, Xiangyu; Chen, Jianhao. (2022).
> *NAFNet: Simple Baselines for Image Restoration.*
> [https://doi.org/10.48550/arXiv.2208.04677](https://doi.org/10.48550/arXiv.2208.04677), arXiv, V1.
