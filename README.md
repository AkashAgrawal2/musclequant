# MuscleQuant

MuscleQuant is a Python GUI app (napari + Qt) for muscle image segmentation and quantification.

## Run locally

```bash
python -m pip install -r requirements.txt
python musclequant_gui.py
```

## Build a double-clickable app (PyInstaller)

Build on each target OS (macOS for `.app`, Windows for `.exe`).

```bash
python -m pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --clean --windowed --name MuscleQuant \
  --collect-all napari \
  --collect-all magicgui \
  --collect-all qtpy \
  --collect-all skimage \
  --collect-all cellpose \
  --collect-all tifffile \
  --collect-all scipy \
  musclequant_gui.py
```

Artifacts:
- macOS: `dist/MuscleQuant.app`
- Windows: `dist/MuscleQuant/MuscleQuant.exe`