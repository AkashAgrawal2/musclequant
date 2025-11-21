# MuscleQuant

MuscleQuant is a napari-based GUI for automated quantification of immunofluorescence (IHC) images, with a focus on laminin-based segmentation and multi-channel quantification (e.g., DAPI, FITC, TxRed).

It uses:

- [Cellpose](https://www.cellpose.org/) for segmentation  
- [napari](https://napari.org/) + [magicgui](https://github.com/pyapp-kit/magicgui) + Qt for the GUI  
- NumPy, pandas, scikit-image, and SciPy for measurements  

---

## Features

- **Laminin-based segmentation** with Cellpose
- **Single-image and batch processing**
- **Multiple background correction modes** (local ring, median, percentile, rolling-ball)
- Per-object metrics:
  - Area, perimeter, eccentricity, solidity, equivalent diameter
  - Mean / max / integrated intensity per channel
  - Background mean / mode
  - CTCF
  - Entropy
  - Nearest-neighbor distance (μm)
- Per-sample folders with `*_quant.csv` and a combined `combined_quant.csv`
- `metadata.json` containing processing parameters for reproducibility
- Manual addition / correction tools

---

## Installation

### Option 1 — From source

```bash
git clone https://github.com/AkashAgrawal2/musclequant.git
cd musclequant
pip install -r requirements.txt
python -m musclequant