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