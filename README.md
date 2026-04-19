# RawRefinery (Leif-US Fork)

**RawRefinery** is an open-source application for **raw image quality refinement and denoising**. This version is configured to use a specific fork of the `PiDNG` library.

![RawRefinery main window](https://github.com/rymuelle/RawRefinery/blob/main/examples/RawRefinery_v1.3.0.png)

---

## Installation (from Source)

This project must be installed from the source code due to its dependency on a forked GitHub repository.

### 1. Clone the Repository

First, clone this repository to your local machine.

```bash
git clone https://github.com/Leif-US/RawRefinery.git
cd RawRefinery
```

### 2. Set up a Virtual Environment

It is highly recommended to use a Python virtual environment to manage dependencies.

**On Windows (PowerShell):**
```powershell
# If you don't have venv, you might need to install it first
# py -m pip install --user virtualenv

# Create and activate the virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1
```

*(If you get an error about execution policies, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` and try activating again.)*

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch

Before installing RawRefinery, you must install a compatible version of PyTorch. The correct version depends on your system's hardware (CUDA for NVIDIA, MPS for Apple Silicon, or CPU).

Find the correct installation command for your system on the [official PyTorch website](https://pytorch.org/get-started/locally/).

**Example (CPU-only):**
```bash
pip install torch~=2.8
```

### 4. Install RawRefinery and Dependencies

Now, install RawRefinery in "editable" mode. This will also automatically fetch and install the correct `PiDNG` version from GitHub as specified in the project's configuration.

```bash
# From the root directory of the project (RawRefinery)
pip install -e .
```

---

## Running the Application

Once the installation is complete and your virtual environment is active, you can start the application by running the following command in your terminal:

```bash
rawrefinery
```
