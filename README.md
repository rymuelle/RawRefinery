# RawRefinery
![RawRefinery main window](https://github.com/rymuelle/RawRefinery/blob/main/examples/RawRefinery.png)


<div align="center">
  <img src="https://github.com/rymuelle/RawRefinery/blob/main/examples/Bayer_TEST_MuseeL-bluebirds-A7C_ISO65535_sha1=eb9cb3e1d80f48b93d0aabe20458870c5c1ef2fa.jpg" alt="Noisy Image" width="600"/>
  <img src="https://github.com/rymuelle/RawRefinery/blob/main/examples/Bayer_TEST_MuseeL-bluebirds-A7C_ISO65535_sha1=eb9cb3e1d80f48b93d0aabe20458870c5c1ef2fa_65534_denoised.DNG.jpg" alt="Denoised" width="600"/>
</div>



## MacOS build instructions


### Clean up

rm -rf build dist __pycache__ 

### Producing the APP

In the 
pyinstaller main.spec


### Make DMG
create-dmg \
  --volname "RawRefinery Installer" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --app-drop-link 450 200 \
  "RawRefinery.dmg" \
  "dist/"