## MacOS build instructions


### Clean up

rm -rf build dist __pycache__ 

### Producing the APP

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


## Building linux executable in docker:

Build in docker:

cd <base_dir_of_repo>
docker run --rm -it --platform linux/amd64 \
  -v $(pwd):/src python:3.11-slim \
  bash -c "apt update && apt install -y binutils && cd /src && pip install pyinstaller && pyinstaller main.spec"