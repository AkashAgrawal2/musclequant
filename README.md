# MuscleQuant

MuscleQuant is a Python GUI app (napari + Qt) for muscle image segmentation and quantification.

There are two options for downloading this program.

## 0. Simple download (no Git required)

For users who are not familiar with programming, this is the easiest way.

1) Open the GitHub page: https://github.com/AkashAgrawal2/musclequant
2) Click the green "Code" button, then choose "Download ZIP".
3) Unzip the file.
4) Open a terminal:
   - Windows: open "Anaconda Prompt"
   - macOS: open "Terminal"
5) Go into the unzipped folder:

Windows example:
```bat
cd "C:\Users\YOUR_NAME\Downloads\musclequant-main\musclequant-main"
```

macOS example:
```bash
cd ~/Downloads/musclequant-main/musclequant-main
```

Then continue with the setup steps in "Run locally (setup from scratch)" below.

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
