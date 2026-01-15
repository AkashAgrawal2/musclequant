# MuscleQuant

MuscleQuant is a Python GUI app (napari + Qt) for muscle image segmentation and quantification.

There are two options for downloading this program.

## 1. Run locally (setup from scratch)

Assumes Python is already installed. Steps below create a virtual environment and install all dependencies.

```bash
git clone https://github.com/AkashAgrawal2/musclequant.git
cd musclequant
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows (PowerShell)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies installed from `requirements.txt`:
- numpy, pandas, tifffile, scikit-image, scipy
- magicgui, napari, qtpy
- cellpose, torch

Run the app:

```bash
python musclequant_gui.py
```

## 2. Build a double-clickable app (PyInstaller)

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
