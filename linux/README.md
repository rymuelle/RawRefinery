
# Build from source from git:

This section will describe how to clone and run RawRefinery, and optional, how to produce a .deb in debian based OSes for installation.  The instructions are based on a debian based OS.


## Installation Guide: Cloning and Local Development Setup

This guide describes how to clone the RawRefinery repository and install it locally for development or execution.

> **Note:** I strongly recommend setting up a virtual environment (e.g. using `venv`) to manage project dependencies and avoid conflicts with your system Python installation.

Suggested bash script:

##### 1. Cloning the code

```bash
# Set up python 3.11+
sudo apt update
sudo apt install python3 python3-pip python3-venv build-essential python3-dev

# Clone the repository from GitHub
git clone https://github.com/rymuelle/RawRefinery.git
cd RawRefinery

# Create the virtual environment named '.venv'
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

##### 2. Setting up Torch

CUDA support depends on your GPU model.  
Before installing RawRefinery, install a compatible version of **PyTorch**.

You can find the correct PyTorch install command for your system here:
https://pytorch.org/get-started/locally/

For CPU usage, torch 2.8 is tested:

```bash
pip install torch~=2.8
```

##### 3. Installing

```bash
# Install the project. Use 'pip install -e .' for editable mode
pip install .
```

##### 4. Running

Now, ensuring your virtual environment is activated, you may run the program:

```bash
rawrefinery
```

## Creating a .deb to install the program

This step is unnecessary, and only applies if you wish to create an installer for the application. Doing so will enable you to run the application as any other. 


 Be sure to run this with the environment active, and in the top level directory of the repository. 

```bash
# First, we use pyinstaller to produce installable material.
python3 -m pip install pyinstaller
# Clean up previous builds 
rm -rf build dist __pycache__ 
cp linux/main.spec .
pyinstaller main.spec

#Then, we build the .deb with fpm
sudo apt update
sudo apt install ruby ruby-dev build-essential qemu-guest-agent upx-ucl
sudo gem install --no-document fpm

# Set up staging area
mkdir -p staging/opt/RawRefinery
mkdir -p staging/usr/share/applications
cp -r dist/RawRefinery/* staging/opt/RawRefinery/
cp linux/RawRefinery.desktop staging/usr/share/applications/RawRefinery.desktop

# Create the .deb
fpm -n rawrefinery -v 1.3.0 --description "A PySide6 application for Raw Data Processing." \
    -t deb -s dir \
    -C staging \
    --url "https://github.com/rymuelle/RawRefinery" \
    --vendor "Ryan Mueller" \
    --license "MIT" \
    .
```

The deb can be installed as usual:

```bash
sudo dpkg -i rawrefinery_1.3.0_amd64.deb
```