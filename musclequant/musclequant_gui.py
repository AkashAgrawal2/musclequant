# musclequant_gui.py — napari GUI for Laminin segmentation + multi-protein quant
# IHC-first pipeline, NO p5 fallback. Local/Median/Percentile/Rolling-ball backgrounds.
# Features:
# • Single & Batch, auto RGB-plane or grayscale, cleaning, manual additions
# • Background mode toggle (Local ring | Median | Percentile | Rolling ball)
# • Raw-intensity quant with rolling/local background + entropy / NN metrics
# • Per-protein: mean, max, integrated, CTCF, bg_mean, bg_mode
# • Per-sample folders + combined CSV, metadata.json
# • Compact multi-row toolbars; re-quantify uses same BG settings

from pathlib import Path
import re
from typing import Optional, Tuple, List, Dict, Any, Set, TYPE_CHECKING

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from skimage import measure, exposure, restoration
from skimage.segmentation import find_boundaries, relabel_sequential
from skimage.measure import label as cc_label
from skimage.morphology import disk, dilation
from skimage.filters import rank

from magicgui import magicgui
import napari
from qtpy.QtWidgets import (
    QMessageBox, QWidget, QVBoxLayout, QScrollArea, QPushButton, QSlider, QLabel,
    QGridLayout, QInputDialog, QFileDialog
)

from qtpy.QtCore import Qt

import tifffile

if TYPE_CHECKING:
    from cellpose import models as _cellpose_models
    CellposeModel = _cellpose_models.CellposeModel
else:
    CellposeModel = Any

# ---------------------- CONFIG DEFAULTS ----------------------
DEFAULT_MODEL = "cyto2"        # or path to your fine-tuned model
DEFAULT_DIAMETER = 70          # px (None = auto; slower)
FLOW_THRESH = 0.45
CELLPROB_THRESH = 0.0
DEFAULT_PX_UM = 0.108

# ---------------------- LAZY DEVICE / MODEL ----------------------
_DEVICE = None
_MODEL_CACHE = None

def get_device():
    """Pick a device lazily the first time Cellpose is actually used."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    import torch  # heavy import done only once, and only if needed

    if torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    else:
        _DEVICE = torch.device("cpu")
    print("Using device:", _DEVICE)
    return _DEVICE

def load_model(pretrained: str = DEFAULT_MODEL) -> CellposeModel:
    """Lazily create and cache the Cellpose model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    from cellpose import models  # heavy import done only once, on first use

    device = get_device()
    _MODEL_CACHE = models.CellposeModel(pretrained_model=pretrained, device=device)
    return _MODEL_CACHE

# ---------------------- HELPERS ----------------------
# >>> PIXEL SIZE HELPERS
def _infer_pixel_size_um_from_tiff(path: Path) -> float | None:
    try:
        with tifffile.TiffFile(str(path)) as tif:
            try:
                ome_xml = tif.ome_metadata or ""
                import re as _re
                m = _re.search(r'PhysicalSizeX="([\d\.eE+-]+)"', ome_xml)
                if m:
                    val = float(m.group(1))
                    if 0.005 <= val <= 20.0:
                        return val
            except Exception:
                pass
            try:
                page = tif.pages[0]
                tags = page.tags
                if "XResolution" in tags and "ResolutionUnit" in tags:
                    num, den = tags["XResolution"].value
                    if den == 0:
                        raise ZeroDivisionError
                    xres = float(num) / float(den)
                    unit = int(tags["ResolutionUnit"].value)
                    if unit == 2 and xres > 0:
                        return 25400.0 / xres
                    if unit == 3 and xres > 0:
                        return 10000.0 / xres
            except Exception:
                pass
            try:
                desc = str(tif.pages[0].description or "")
                import re as _re
                m = _re.search(r'(?:PixelSize|pixelsize|PixelWidth|XPixelSize)\s*[:=]\s*([\d\.eE+-]+)\s*(?:um|µm|micron)', desc, _re.I)
                if m:
                    val = float(m.group(1))
                    if 0.005 <= val <= 20.0:
                        return val
            except Exception:
                pass
    except Exception:
        return None
    return None
# <<< PIXEL SIZE HELPERS

# ---- Stain name helpers ----
STAIN_SUFFIX_RE = re.compile(r"(?:^|_)([^_\.]*)\.(?:tif|tiff|png|jpg)$", re.IGNORECASE)

def infer_stain_from_filename(path: Path) -> str:
    m = STAIN_SUFFIX_RE.search(path.name)
    if m:
        candidate = m.group(1)
        if candidate.upper() in {"RGB","GRAY","GRAY8","GRAY16","GRAY32"}:
            return candidate.title()
        return candidate
    return "Unknown"

PLANE_TO_IDX = {"R": 0, "G": 1, "B": 2}
IDX_TO_PLANE = {0: "R", 1: "G", 2: "B"}
COLORMAP_CYCLE = ["red", "green", "blue", "magenta", "cyan", "yellow", "orange", "gray"]

def _uniquify_name(name: str, used: Set[str]) -> str:
    """Ensure name is unique by appending _2, _3, ... if needed."""
    if name not in used:
        used.add(name)
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    new_name = f"{name}_{i}"
    used.add(new_name)
    return new_name

def _score_preview_candidate(path: Path) -> Tuple[Tuple[float, int], np.ndarray, str]:
    """
    Return ((p99, nonzero_count), normalized_plane, plane_code) for a candidate protein image.
    Respects CURRENT['preview_planes'] cache to keep preview/quant planes consistent.
    """
    arr = imread(str(path))
    prev_used = CURRENT.get("preview_planes", {}).get(str(path))
    used_plane = "Gray"
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        chosen = prev_used if prev_used in ("R", "G", "B") else autodetect_plane(arr)
        ch = PLANE_TO_IDX[chosen]
        gray = arr[..., ch].astype(np.float32)
        used_plane = chosen
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        gray = arr[..., 0].astype(np.float32)
    elif arr.ndim == 2:
        gray = arr.astype(np.float32)
    else:
        return ((-1.0, 0), np.zeros((1, 1), np.float32), "Gray")
    nz = int(np.count_nonzero(gray))
    p99 = float(np.percentile(gray, 99)) if nz else 0.0
    disp = exposure.rescale_intensity(gray, in_range="image", out_range=(0, 1)) if gray.max() > 1.5 else np.clip(gray, 0, 1)
    return ((p99, nz), disp, used_plane)

def select_protein_previews(
    plist: List[Path],
    lam_path: Path,
    stain_order: List[str],
) -> List[Tuple[str, str, np.ndarray, Path]]:
    """
    Pick one best file per stain (honoring stain_order if provided) using the
    fluorescence scoring heuristic shared with batch processing.
    """
    CURRENT.setdefault("preview_planes", {})
    proteins_loaded: List[Tuple[str, str, np.ndarray, Path]] = []
    detected_files = sorted(plist)
    used_names: Set[str] = set()

    inferred_names: Dict[Path, str] = {}
    remaining_order: List[Path] = []
    for p in detected_files:
        if p == lam_path:
            continue
        inferred = (infer_stain_from_filename(p).strip() or "Protein")
        if inferred.lower() in ("background", "laminin"):
            continue
        inferred_names[p] = inferred
        remaining_order.append(p)
    remaining_set: Set[Path] = set(remaining_order)
    if not remaining_order:
        return proteins_loaded
    
    score_cache: Dict[Path, Tuple[Tuple[float, int], np.ndarray, str]] = {}

    def _get_scored(path: Path) -> Tuple[Tuple[float, int], np.ndarray, str]:
        if path not in score_cache:
            score_cache[path] = _score_preview_candidate(path)
        return score_cache[path]

    def _available_paths() -> List[Path]:
        return [p for p in remaining_order if p in remaining_set]

    def _norm_name(txt: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", txt.lower())

    def _matches_target(path: Path, target: str) -> bool:
        if not target:
            return True
        norm_target = _norm_name(target)
        if not norm_target:
            return True
        inferred_norm = _norm_name(inferred_names.get(path, ""))
        if inferred_norm and inferred_norm == norm_target:
            return True
        return norm_target in _norm_name(path.stem)

    def _choose(paths: List[Path], forced_name: Optional[str]) -> bool:
        best: Optional[Tuple[Tuple[float, int], np.ndarray, str, Path]] = None
        for path in paths:
            if path not in remaining_set:
                continue
            score, disp, used_plane = _get_scored(path)
            if best is None or score > best[0]:
                best = (score, disp, used_plane, path)
        if best is None:
            return False
        (_score_val, disp, used_plane, chosen_path) = best
        name_hint = forced_name or inferred_names.get(chosen_path, "Protein")
        clean_name = _uniquify_name(name_hint, used_names)
        CURRENT["preview_planes"][str(chosen_path)] = used_plane
        proteins_loaded.append((clean_name, used_plane, disp, chosen_path))
        remaining_set.remove(chosen_path)
        return True

    for desired in stain_order:
        if desired.lower() in ("background", "laminin"):
            continue
        matches = [p for p in _available_paths() if _matches_target(p, desired)]
        if not _choose(matches, desired):
            if not _choose(_available_paths(), desired):
                break
        if not remaining_set:
            break

    if remaining_set:
        leftovers: Dict[str, List[Path]] = {}
        for p in _available_paths():
            leftovers.setdefault(inferred_names.get(p, "Protein"), []).append(p)
        for inferred_name in sorted(leftovers.keys(), key=lambda s: s.lower()):
            if not _choose(leftovers[inferred_name], None):
                continue
            if not remaining_set:
                break

    return proteins_loaded

def autodetect_plane(arr: np.ndarray) -> str:
    # Choose channel with highest p99 (ties → most nonzero)
    assert arr.ndim == 3 and arr.shape[-1] >= 3, "Expected RGB image"
    stats = []
    for i in range(3):
        ch = arr[..., i].astype(np.float32)
        nz = np.count_nonzero(ch)
        p99 = np.percentile(ch, 99) if nz else 0.0
        stats.append((i, nz, p99))
    stats.sort(key=lambda t: (t[2], t[1]), reverse=True)
    return IDX_TO_PLANE[stats[0][0]]

def load_rgb_plane(path: Path, plane: Optional[str]) -> Tuple[np.ndarray, str]:
    """
    Accepts RGB (H,W,3) or grayscale (H,W)/(H,W,1).
    Returns float32 image in [0,1] and a tag for the used plane: 'R','G','B','Gray'.
    """
    arr = imread(str(path))
    used_plane = "Gray"

    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        gray = arr[..., 0].astype(np.float32)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        used_plane = plane if plane in ("R","G","B") else autodetect_plane(arr)
        ch = PLANE_TO_IDX[used_plane]
        gray = arr[..., ch].astype(np.float32)
    else:
        raise ValueError(f"{path.name}: unsupported image shape {arr.shape} (want (H,W) or (H,W,3))")

    # Normalize to [0,1] for processing/visualization (preserves relative intensities per image)
    if gray.max() > 1.5:
        gray = exposure.rescale_intensity(gray, in_range='image', out_range=(0,1))
    else:
        gray = np.clip(gray, 0, 1)

    return gray, used_plane

def load_raw_gray(path: Path, plane: Optional[str]) -> np.ndarray:
    """
    Load grayscale data for quantification without rescaling.
    """
    arr = imread(str(path))
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        gray = arr[..., 0].astype(np.float32)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        used_plane = plane if plane in ("R", "G", "B") else autodetect_plane(arr)
        ch = PLANE_TO_IDX[used_plane]
        gray = arr[..., ch].astype(np.float32)
    else:
        raise ValueError(f"{path.name}: unsupported image shape {arr.shape}")
    return gray

def overlay_boundaries(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    b = find_boundaries(labels, mode="outer")
    rgb = np.dstack([gray, gray, gray])
    rgb[b] = [1.0, 0.0, 0.0]
    return (rgb * 255).astype(np.uint8)

def run_cellpose(img: np.ndarray, model: CellposeModel,
                 diameter=DEFAULT_DIAMETER, flow_thr=FLOW_THRESH, cellprob_thr=CELLPROB_THRESH):
    out = model.eval(
        img,
        channels=[0, 0],   # v4 OK
        diameter=diameter,
        flow_threshold=flow_thr,
        cellprob_threshold=cellprob_thr,
        batch_size=1
    )
    if isinstance(out, tuple):
        return out[0]  # masks
    raise RuntimeError("Unexpected Cellpose eval() return.")

def clean_mask(mask: np.ndarray, min_area_px: int = 0, drop_border: bool = False):
    lab = mask.copy().astype(np.int32)
    removed_small = 0
    removed_border = 0

    if drop_border:
        border_ids = np.unique(np.concatenate([lab[0,:], lab[-1,:], lab[:,0], lab[:,-1]]))
        border_ids = border_ids[border_ids != 0]
        for b in border_ids:
            lab[lab == b] = 0
        removed_border = int(border_ids.size)

    if min_area_px and min_area_px > 0:
        counts = np.bincount(lab.ravel())
        small_ids = np.where((counts > 0) & (counts < min_area_px))[0]
        small_ids = small_ids[small_ids != 0]
        for s in small_ids:
            lab[lab == s] = 0
        removed_small = int(small_ids.size)

    lab, _, _ = relabel_sequential(lab)
    stats = dict(removed_small=removed_small, removed_border=removed_border, final_labels=int(lab.max()))
    return lab.astype(np.uint16), stats

# ---------------------- QUANT (Literature-style metrics) ----------------------
def quantify(
    mask: np.ndarray,
    intensity_raw: np.ndarray,
    px_um: float = DEFAULT_PX_UM,
    protein_name: str = "protein",
    bg_mode: str = "local",            # "local" | "median" | "percentile" | "rolling_ball"
    bg_percentile: float = 5.0,
    local_ring_px: int = 5,
    rolling_ball_radius_px: int = 50,
    add_texture: bool = True,
    add_spatial: bool = True,
) -> pd.DataFrame:
    """
    Literature-style per-cell quantification:
      - CTCF with local background by default
      - Optional rolling-ball background pre-subtraction
      - Expanded morphometrics (+texture, +nearest-neighbor)
    """
    img = intensity_raw.astype(np.float32, copy=False)

    if bg_mode == "rolling_ball":
        bg_result = restoration.rolling_ball(img, radius=rolling_ball_radius_px)
        bg_img = bg_result[0] if isinstance(bg_result, tuple) else bg_result
        img_corr = img - bg_img
        img_corr[img_corr < 0] = 0
        base_bg_mean = float(np.mean(bg_img, dtype=np.float32))
        bg_mode_used = f"rolling_ball_r{rolling_ball_radius_px}"
    else:
        img_corr = img
        base_bg_mean = float(np.median(img))
        bg_mode_used = bg_mode

    props = measure.regionprops_table(
        mask,
        intensity_image=img_corr,
        properties=(
            "label", "area", "perimeter",
            "eccentricity", "solidity", "equivalent_diameter",
            "centroid", "mean_intensity", "max_intensity",
        ),
    )
    df = pd.DataFrame(props).rename(columns={"area": "area_px"})
    df["px_um"] = float(px_um)
    df["area_um2"] = df["area_px"] * (px_um**2)

    df[f"mean_{protein_name}"] = df.pop("mean_intensity")
    df[f"max_{protein_name}"] = df.pop("max_intensity")
    df[f"integrated_{protein_name}"] = df["area_px"] * df[f"mean_{protein_name}"]

    if bg_mode in ("median", "percentile"):
        if bg_mode == "percentile":
            bg_mean = float(np.percentile(img.astype(np.float32), float(bg_percentile)))
            mode_used = f"p{int(bg_percentile)}"
        else:
            bg_mask = (mask == 0)
            if np.any(bg_mask):
                bg_mean = float(np.median(img[bg_mask]))
                mode_used = "median_outside"
            else:
                bg_mean = float(np.median(img))
                mode_used = "median_global"
        df[f"bg_mean_{protein_name}"] = bg_mean
        df[f"bg_mode_{protein_name}"] = mode_used
        df[f"ctcf_{protein_name}"] = df[f"integrated_{protein_name}"] - (df["area_px"] * bg_mean)

    elif bg_mode == "local":
        labels = np.unique(mask)
        labels = labels[labels != 0]
        local_bgs = np.zeros(len(df), dtype=np.float32)

        selem = disk(max(1, int(local_ring_px)))
        any_cell = mask > 0
        label_to_index = {int(lbl): idx for idx, lbl in enumerate(df["label"].astype(int).tolist())}

        for lbl in labels:
            idx = label_to_index[int(lbl)]
            cell_mask = (mask == lbl)
            ring = dilation(cell_mask, selem) & (~cell_mask) & (~any_cell)
            if not np.any(ring):
                ring = dilation(cell_mask, selem) & (~cell_mask)
            if np.any(ring):
                local_bgs[idx] = float(np.median(img[ring]))
            else:
                local_bgs[idx] = float(np.median(img))

        df[f"bg_mean_{protein_name}"] = local_bgs
        df[f"bg_mode_{protein_name}"] = f"local_ring_{int(local_ring_px)}px"
        df[f"ctcf_{protein_name}"] = df[f"integrated_{protein_name}"] - (df["area_px"] * local_bgs)

    elif bg_mode == "rolling_ball":
        df[f"bg_mean_{protein_name}"] = base_bg_mean
        df[f"bg_mode_{protein_name}"] = bg_mode_used
        df[f"ctcf_{protein_name}"] = df[f"integrated_{protein_name}"] - (df["area_px"] * base_bg_mean)

    else:
        raise ValueError(f"Unknown bg_mode: {bg_mode}")

    if add_texture:
        max_val = img_corr.max()
        if max_val > 0:
            img8 = np.clip((img_corr / max_val) * 255.0, 0, 255).astype(np.uint8)
            entropy_map = rank.entropy(img8, disk(4))
            ent_props = measure.regionprops_table(
                mask,
                intensity_image=entropy_map,
                properties=("label", "mean_intensity"),
            )
            ent_df = pd.DataFrame(ent_props).rename(columns={"mean_intensity": f"entropy_{protein_name}"})
            df = df.merge(ent_df, on="label", how="left")
        else:
            df[f"entropy_{protein_name}"] = 0.0

    if add_spatial:
        centroids = df[["centroid-0", "centroid-1"]].to_numpy(dtype=np.float32)
        if len(centroids) >= 2:
            # Lazy import to avoid SciPy overhead at startup
            from scipy.spatial import distance_matrix

            dist = distance_matrix(centroids, centroids)
            np.fill_diagonal(dist, np.inf)
            nn = np.min(dist, axis=1) * float(px_um)
        else:
            nn = np.full(len(df), np.nan, dtype=np.float32)
        df["nn_dist_um"] = nn

    return df

def label_stats(mask: np.ndarray) -> str:
    labels = mask[mask > 0]
    if labels.size == 0:
        return "No labels."
    n = int(mask.max())
    areas = np.bincount(mask.ravel())[1:]
    areas = areas[areas > 0]
    pct = np.percentile(areas, [5, 25, 50, 75, 95]).round(1)
    return (
        f"Labels: {n}\n"
        f"Area(px): min={areas.min()}, median={np.median(areas):.1f}, max={areas.max()}\n"
        f"Area px percentiles (5/25/50/75/95): {pct.tolist()}"
    )

def prepare_export_dir(save_root: Path, suggested: str) -> Optional[Path]:
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    name, ok = QInputDialog.getText(
        None,
        "Export folder",
        "Folder name for this sample:",
        text=suggested,
    )
    if not ok:
        return None
    name_str = str(name).strip() or str(suggested).strip() or "sample"
    outdir = save_root / name_str
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def save_metadata_json(outdir: Path, meta: dict):
    import json
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def choose_export_root(default: Path) -> Optional[Path]:
    # Use the provided path if not empty or "."; otherwise prompt for a directory.
    default_str = str(default).strip()
    if default_str not in ["", "."]:
        outdir = Path(default_str)
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    prompt = QMessageBox()
    prompt.setWindowTitle("Select results folder")
    prompt.setIcon(QMessageBox.Information)
    prompt.setText(
        "Please choose a folder to store the segmentation results.\n"
        "Click 'Choose Folder…' to open Finder/File Explorer and pick a destination.\n"
        "Select Cancel to stop the segmentation."
    )
    choose_btn = prompt.addButton("Choose Folder…", QMessageBox.AcceptRole)
    prompt.addButton(QMessageBox.Cancel)
    prompt.exec()
    if prompt.clickedButton() is not choose_btn:
        return None

    root = QFileDialog.getExistingDirectory(
        None,
        "Select a directory to save segmentation results",
    )
    if root and str(root).strip():
        outdir = Path(root)
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir
    return None

# ---------- Manual additions: state + ops ----------
CURRENT: Dict[str, Any] = {
    "base": None,
    "lam": None,              # np.ndarray
    "masks": None,            # np.ndarray (labels)
    "proteins": [],           # List[Tuple[name, plane_code, Path]]
    "preview_planes": {},     # str(path) -> plane code used in preview/quant
    "px_um": DEFAULT_PX_UM,
    "save_dir": Path.home() / "musclequant" / "results",
    "cleaned": False,
    "bg_mode": "local",
    "bg_percentile": 5.0,
    "local_ring_px": 5,
    "rolling_ball_radius_px": 50,
    "add_texture": True,
    "add_spatial": True,
}

def ensure_manual_layer(viewer: "napari.Viewer"):
    if "Manual_additions" in [L.name for L in viewer.layers]:
        return viewer.layers["Manual_additions"]
    if CURRENT["masks"] is None:
        QMessageBox.information(None, "MuscleQuant", "Run segmentation first.")
        return None
    add = np.zeros_like(CURRENT["masks"], dtype=np.uint16)
    layer = viewer.add_labels(add, name="Manual_additions", visible=True)
    layer.selected_label = int(CURRENT["masks"].max()) + 1
    layer.brush_size = 5
    QMessageBox.information(
        None,
        "Manual mode",
        "Paint new cells on 'Manual_additions'.\nTips:\n• Brush (2) to paint; label 0 erases\n• 'New label ID' for each new cell\n• 'Merge additions' when done"
    )
    return layer

def next_label_id():
    m = CURRENT["masks"]
    if m is None:
        return 1
    return int(np.max(m)) + 1

def merge_manual_additions(viewer: "napari.Viewer"):
    if CURRENT["masks"] is None or "Manual_additions" not in [L.name for L in viewer.layers]:
        QMessageBox.information(None, "MuscleQuant", "Nothing to merge (segment first, paint on 'Manual_additions').")
        return False
    masks = CURRENT["masks"].astype(np.int32).copy()
    add = np.asarray(viewer.layers["Manual_additions"].data).astype(np.int32)

    # keep original model labels if overlap
    add[masks > 0] = 0

    # connected components for manual strokes
    cc = cc_label(add > 0, connectivity=1)
    n_new = int(cc.max())
    if n_new == 0:
        QMessageBox.information(None, "MuscleQuant", "No new components to merge.")
        return False

    nid = int(masks.max()) + 1
    for i in range(1, n_new + 1):
        masks[cc == i] = nid
        nid += 1

    CURRENT["masks"] = masks.astype(np.uint16)
    viewer.layers["Manual_additions"].data = np.zeros_like(masks, dtype=np.uint16)

    labels_name = "Fibers_mask_clean" if CURRENT["cleaned"] else "Fibers_mask"
    if labels_name in [L.name for L in viewer.layers]:
        viewer.layers[labels_name].data = CURRENT["masks"]
    else:
        viewer.add_labels(CURRENT["masks"], name=labels_name, visible=True).contour = True

    ov = overlay_boundaries(CURRENT["lam"], CURRENT["masks"])
    if "Overlay" in [L.name for L in viewer.layers]:
        viewer.layers["Overlay"].data = ov
    else:
        viewer.add_image(ov, name="Overlay", blending="translucent_no_depth", visible=True, opacity=0.33)

    QMessageBox.information(None, "MuscleQuant", f"Merged {n_new} new cells. Total labels: {int(CURRENT['masks'].max())}.")
    return True

def requant_and_save():
    if CURRENT["masks"] is None or CURRENT["lam"] is None:
        QMessageBox.information(None, "MuscleQuant", "Nothing to save/quantify yet.")
        return
    base = CURRENT["base"] or "image"
    export_dir: Path = Path(CURRENT["save_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    wide_df = None
    shared_base_cols = {
        "label", "area_px", "area_um2", "perimeter", "eccentricity",
        "solidity", "equivalent_diameter", "centroid-0", "centroid-1", "px_um",
    }
    if CURRENT.get("add_spatial", True):
        shared_base_cols.add("nn_dist_um")

    for pname, plane_code, p_path in CURRENT["proteins"]:
        preview_plane = CURRENT.get("preview_planes", {}).get(str(p_path))
        plane_for_raw = preview_plane if preview_plane in PLANE_TO_IDX else (plane_code if plane_code in PLANE_TO_IDX else None)
        p_raw = load_raw_gray(Path(p_path), plane_for_raw)
        df = quantify(
            CURRENT["masks"], p_raw,
            px_um=float(CURRENT["px_um"]),
            protein_name=pname,
            bg_mode=CURRENT.get("bg_mode", "local"),
            bg_percentile=float(CURRENT.get("bg_percentile", 5.0)),
            local_ring_px=int(CURRENT.get("local_ring_px", 5)),
            rolling_ball_radius_px=int(CURRENT.get("rolling_ball_radius_px", 50)),
            add_texture=bool(CURRENT.get("add_texture", True)),
            add_spatial=bool(CURRENT.get("add_spatial", True)),
        )
        if wide_df is None:
            wide_df = df.copy()
        else:
            protein_cols = [c for c in df.columns if c not in shared_base_cols]
            merge_cols = ["label"] + protein_cols
            wide_df = pd.merge(wide_df, df[merge_cols], on="label", how="left")

    suffix = "_clean" if CURRENT["cleaned"] else ""
    edited_suffix = suffix + "_edited"

    mask_out = export_dir / f"{base}_mask{edited_suffix}.tif"
    ov_out   = export_dir / f"{base}_overlay{edited_suffix}.png"
    imwrite(mask_out, CURRENT["masks"].astype(np.uint16), compression="zlib")
    imwrite(ov_out, overlay_boundaries(CURRENT["lam"], CURRENT["masks"]))

    if wide_df is not None:
        wide_df["image_base"] = base
        csv_out = export_dir / f"{base}_quant{edited_suffix}.csv"
        wide_df.to_csv(csv_out, index=False)
        QMessageBox.information(None, "MuscleQuant",
                                f"Saved (edited) to:\n{export_dir}\n\n• {mask_out.name}\n• {ov_out.name}\n• {csv_out.name}")
    else:
        QMessageBox.information(None, "MuscleQuant",
                                f"Saved (edited) to:\n{export_dir}\n\n• {mask_out.name}\n• {ov_out.name}\n(no CSV — no proteins loaded)")

# ===================== BATCH MODE (mirrors per-sample folders) =====================
@magicgui(
    layout="vertical",
    call_button="Batch Segment + Quantify",
    model_path={"label": "Cellpose model (cyto2 or path)"},
    folder={"label": "", "visible": False},
    laminin_substring={"label": "Background stain name"},        
    stain_names={"label": "Stains (comma separated)"},
    pixel_size_um={"label": "Pixel size (µm/px)"},
    diameter_px={"label": "Diameter (px ~ fiber width)"},
    flow_thresh={"label": "Flow threshold"},
    cellprob_thresh={"label": "Cellprob threshold"},
    drop_edge_touching={"label": "Remove edge-touching cells"},
    min_area_px={"label": "Min area to keep (px)"},
    bg_mode={"choices": ["Local ring", "Median outside", "Percentile", "Rolling ball"], "label": "Background mode"},
    bg_percentile={"label": "Percentile (p) for 'Percentile' mode", "min": 0.0, "max": 100.0},
    local_ring_px={"label": "Local ring width (px)", "min": 1, "max": 50},
    rolling_ball_radius_px={"label": "Rolling-ball radius (px)", "min": 5, "max": 500},
    add_texture={"label": "Add entropy feature"},
    add_spatial={"label": "Add NN distance"},
    save_dir={"label": "Save results under…", "mode": "d"},
    preserve_preview={"label": "Keep preview layers"},
)
def batch_widget(
    viewer: "napari.Viewer",
    model_path: str = DEFAULT_MODEL,
    folder: Path = Path.home() / "musclequant" / "input_raw",
    stain_names: str = "",
    laminin_substring: str = "Background",
    pixel_size_um: float = DEFAULT_PX_UM,
    diameter_px: float = DEFAULT_DIAMETER,
    flow_thresh: float = FLOW_THRESH,
    cellprob_thresh: float = CELLPROB_THRESH,
    drop_edge_touching: bool = False,
    min_area_px: int = 0,
    bg_mode: str = "Local ring",
    bg_percentile: float = 5.0,
    local_ring_px: int = 5,
    rolling_ball_radius_px: int = 50,
    add_texture: bool = True,
    add_spatial: bool = True,
    save_dir: Path = Path("."),
    preserve_preview: bool = True,
):
    try:
        folder = Path(folder)
        files = sorted([p for p in folder.glob("*.tif*")])

        # Ask once for export root
        export_root = choose_export_root(Path(save_dir))
        if export_root is None:
            QMessageBox.information(
                None,
                "Batch canceled",
                "No export folder was selected. Segmentation has been canceled.",
            )
            return
        export_root.mkdir(parents=True, exist_ok=True)

        # Map BG selection
        bg_mode_lower = str(bg_mode).lower()
        if bg_mode_lower.startswith("rolling"):
            _bg_mode = "rolling_ball"
        elif bg_mode_lower.startswith("percentile"):
            _bg_mode = "percentile"
        elif bg_mode_lower.startswith("median"):
            _bg_mode = "median"
        else:
            _bg_mode = "local"
        # Parse stain names list (user-provided) once
        _stain_list = [s.strip() for s in stain_names.split(",") if s.strip()] if isinstance(stain_names, str) else []

        # Group by base (before _RGB_… or before "__")
        def split_base(p: Path):
            m = re.match(r"^(.*)_RGB_([^_]+)\.tif[f]?$", p.name, flags=re.IGNORECASE)
            if not m and "__" in p.stem:
                head = p.stem.split("__")[0]
                m = re.match(r"^(.*)_RGB_([^_]+)$", head, flags=re.IGNORECASE)
            return m.group(1) if m else p.stem

        groups: Dict[str, List[Path]] = {}
        for p in files:
            groups.setdefault(split_base(p), []).append(p)

        model = load_model(pretrained=model_path)

        combined_rows: List[pd.DataFrame] = []
        processed = 0

        for base, plist in groups.items():
            # find laminin
            lam_files = [p for p in plist if laminin_substring.lower() in p.name.lower()]
            if not lam_files:
                continue
            lam_path = lam_files[0]

            # per-sample folder name prompt (like Single)
            export_dir = prepare_export_dir(export_root, suggested=base)
            if export_dir is None:
                QMessageBox.information(
                    None,
                    "Batch canceled",
                    "Folder naming was canceled. Segmentation has been stopped.",
                )
                return

            # load laminin (RGB or Gray), auto-plane if RGB
            lam_gray, lam_used = load_rgb_plane(lam_path, None)

            # segment
            masks_raw = run_cellpose(
                lam_gray, model,
                diameter=None if diameter_px in (None, 0) else float(diameter_px),
                flow_thr=float(flow_thresh),
                cellprob_thr=float(cellprob_thresh)
            )

            # clean
            cleaned = False
            masks = masks_raw
            if drop_edge_touching or (min_area_px and int(min_area_px) > 0):
                masks, _ = clean_mask(
                    masks_raw,
                    min_area_px=int(min_area_px) if min_area_px else 0,
                    drop_border=bool(drop_edge_touching),
                )
                cleaned = True
            suffix = "_clean" if cleaned else ""

            proteins_loaded = select_protein_previews(plist, lam_path, _stain_list)
            if not proteins_loaded:
                continue

            print(f"[DEBUG] Sample {base}")
            print(f"  Laminin: {lam_path.name} plane={lam_used}")
            for pname, used_plane, _disp, psrc in proteins_loaded:
                print(f"  Protein: name={pname} plane={used_plane} src={Path(psrc).name}")

            # show layers (optional visualization during batch)
            if not preserve_preview:
                viewer.layers.clear()
                viewer.add_image(
                    lam_gray,
                    name=f"{base} — Background [{lam_used}]",
                    colormap="gray",
                    blending="additive",
                    visible=True,
                )
                for pname, used_plane, p_gray, _ in proteins_loaded:
                    viewer.add_image(
                        p_gray,
                        name=f"{base} — {pname} [{used_plane}]",
                        colormap="magenta",
                        blending="additive",
                        visible=True,
                    )
            labels_name = "Fibers_mask_clean" if cleaned else "Fibers_mask"
            viewer.add_labels(masks.astype(np.uint16), name=f"{base} — {labels_name}", visible=True).contour = True
            viewer.add_image(overlay_boundaries(lam_gray, masks), name=f"{base} — Overlay", blending="translucent_no_depth", visible=True, opacity=0.33)

            # wide quant with BG traceability
            wide_df = None
            shared_base_cols = {
                "label", "area_px", "area_um2", "perimeter", "eccentricity",
                "solidity", "equivalent_diameter", "centroid-0", "centroid-1", "px_um",
            }
            if add_spatial:
                shared_base_cols.add("nn_dist_um")

            for pname, used_plane, _disp, orig_path in proteins_loaded:
                preview_plane = CURRENT.get("preview_planes", {}).get(str(orig_path))
                plane_for_raw = preview_plane if preview_plane in PLANE_TO_IDX else (used_plane if used_plane in PLANE_TO_IDX else None)
                print(f"[DEBUG] Quant: {base} → {pname} ({used_plane}) from {Path(orig_path).name}")
                p_raw = load_raw_gray(Path(orig_path), plane_for_raw)
                df = quantify(
                    masks, p_raw,
                    px_um=float(pixel_size_um),
                    protein_name=pname,
                    bg_mode=_bg_mode,
                    bg_percentile=float(bg_percentile),
                    local_ring_px=int(local_ring_px),
                    rolling_ball_radius_px=int(rolling_ball_radius_px),
                    add_texture=bool(add_texture),
                    add_spatial=bool(add_spatial),
                )
                if wide_df is None:
                    wide_df = df.copy()
                else:
                    protein_cols = [c for c in df.columns if c not in shared_base_cols]
                    right = df[["label"] + protein_cols].copy()
                    collide = [c for c in protein_cols if c in wide_df.columns]
                    if collide:
                        for c in collide:
                            newc = c
                            k = 2
                            while newc in wide_df.columns:
                                newc = f"{c}_{k}"
                                k += 1
                            right.rename(columns={c: newc}, inplace=True)
                    wide_df = pd.merge(wide_df, right, on="label", how="left")

            # save to per-sample dir
            base_out = base.replace("/", "_")
            export_dir.mkdir(parents=True, exist_ok=True)

            mask_out = export_dir / f"{base_out}_mask{suffix}.tif"
            ov_out   = export_dir / f"{base_out}_overlay{suffix}.png"
            imwrite(mask_out, masks.astype(np.uint16), compression="zlib")
            imwrite(ov_out, overlay_boundaries(lam_gray, masks))

            for pname, used_plane, p_gray, _ in proteins_loaded:
                imwrite(export_dir / f"{base_out}_{pname}_plane-{used_plane}.png",
                        (np.clip(p_gray, 0, 1) * 255).astype(np.uint8))

            # annotate & save CSV + metadata
            wide_df["image_base"] = base
            wide_df["laminin_file"] = lam_path.name
            for pname, used_plane, *_ in proteins_loaded:
                meta_col = f"plane_{pname}"
                if meta_col in wide_df.columns:
                    j = 2
                    while f"{meta_col}_{j}" in wide_df.columns:
                        j += 1
                    meta_col = f"{meta_col}_{j}"
                wide_df[meta_col] = used_plane
            csv_out = export_dir / f"{base_out}_quant{suffix}.csv"
            wide_df.to_csv(csv_out, index=False)

            meta = {
                "base": base,
                "export_dir": str(export_dir),
                "pixel_size_um": float(pixel_size_um),
                "diameter_px": None if diameter_px in (None, 0) else float(diameter_px),
                "flow_threshold": float(flow_thresh),
                "cellprob_threshold": float(cellprob_thresh),
                "drop_edge_touching": bool(drop_edge_touching),
                "min_area_px": int(min_area_px) if min_area_px else 0,
                "cleaned": cleaned,
                "labels": int(masks.max()),
                "laminin_file": lam_path.name,
                "laminin_plane": lam_used,
                "background_mode": _bg_mode,
                "background_percentile": float(bg_percentile) if _bg_mode == "percentile" else None,
                "local_ring_px": int(local_ring_px) if _bg_mode == "local" else None,
                "rolling_ball_radius_px": int(rolling_ball_radius_px) if _bg_mode == "rolling_ball" else None,
                "add_texture": bool(add_texture),
                "add_spatial": bool(add_spatial),
                "proteins": [{"name": name, "plane": plane_code, "file": str(path)} for (name, plane_code, _img, path) in proteins_loaded],
            }
            save_metadata_json(export_dir, meta)

            # Store state for manual → keep same folder + bg settings
            CURRENT["base"] = base
            CURRENT["lam"] = lam_gray
            CURRENT["masks"] = masks.astype(np.uint16)
            CURRENT["proteins"] = [(pname, plane_code, Path(p_path)) for (pname, plane_code, _img, p_path) in proteins_loaded]
            CURRENT["px_um"] = float(pixel_size_um)
            CURRENT["save_dir"] = export_dir
            CURRENT["cleaned"] = cleaned
            CURRENT["bg_mode"] = _bg_mode
            CURRENT["bg_percentile"] = float(bg_percentile)
            CURRENT["local_ring_px"] = int(local_ring_px)
            CURRENT["rolling_ball_radius_px"] = int(rolling_ball_radius_px)
            CURRENT["add_texture"] = bool(add_texture)
            CURRENT["add_spatial"] = bool(add_spatial)

            combined_rows.append(wide_df.assign(sample_folder=export_dir.name))
            processed += 1

        # combined CSV at export root
        if combined_rows:
            big = pd.concat(combined_rows, ignore_index=True)
            big.to_csv(export_root / "combined_quant.csv", index=False)
            QMessageBox.information(
                None,
                "MuscleQuant (Batch)",
                f"Processed {processed} sample(s).\nPer-sample folders under:\n{export_root}\n\nCombined table: combined_quant.csv",
            )
        else:
            QMessageBox.information(None, "MuscleQuant (Batch)", "No valid Background+protein groups found.")

    except Exception as e:
        QMessageBox.critical(None, "MuscleQuant — Error", str(e))
        raise

# ---------------------- EXTRA UI (scroll, compact toolbars, manual tools) ----------------------
def make_quick_toolbar(viewer):
    """Compact 2-row grid toolbar."""
    bar = QWidget()
    grid = QGridLayout(bar)
    grid.setContentsMargins(6,6,6,6)
    # Row 0
    grid.addWidget(QLabel("Overlay α"), 0, 0)
    slider = QSlider(Qt.Horizontal); slider.setMinimum(0); slider.setMaximum(100); slider.setValue(33)
    grid.addWidget(slider, 0, 1, 1, 2)

    btn_auto = QPushButton("Auto-contrast (1–99%)")
    grid.addWidget(btn_auto, 0, 3)

    # Row 1
    btn_toggle = QPushButton("Toggle proteins"); grid.addWidget(btn_toggle, 1, 0)
    btn_stats  = QPushButton("Label stats");     grid.addWidget(btn_stats, 1, 1)
    btn_export = QPushButton("Export view PNG"); grid.addWidget(btn_export, 1, 2)

    def on_slide(val):
        layer = next((L for L in viewer.layers if L.name.endswith("Overlay") or L.name.startswith("Overlay")), None)
        if layer is not None:
            layer.opacity = val/100.0
    slider.valueChanged.connect(on_slide)

    def do_autocontrast():
        for L in viewer.layers:
            if hasattr(L, "contrast_limits") and hasattr(L, "data") and L.visible:
                data = np.asarray(L.data, dtype=np.float32)
                if data.ndim == 2:
                    lo, hi = np.percentile(data, (1, 99))
                    if hi > lo:
                        L.contrast_limits = (float(lo), float(hi))
    btn_auto.clicked.connect(do_autocontrast)

    def do_toggle():
        for L in viewer.layers:
            if any(L.name.find(prefix) >= 0 for prefix in ("Protein",)) or \
               ("[" in L.name and "]" in L.name and "Background" not in L.name and "Overlay" not in L.name and "mask" not in L.name):
                L.visible = not L.visible
    btn_toggle.clicked.connect(do_toggle)

    def do_stats():
        lab = None
        for L in viewer.layers:
            if L.__class__.__name__.lower().startswith("labels") and "Manual_additions" not in L.name:
                lab = L.data
        if lab is None:
            QMessageBox.information(None, "MuscleQuant", "No Labels layer found.")
            return
        QMessageBox.information(None, "MuscleQuant — Label stats", label_stats(np.asarray(lab)))
    btn_stats.clicked.connect(do_stats)

    def do_export():
        img = viewer.screenshot(canvas_only=True)
        out = Path.home() / "musclequant" / "results" / "canvas_export.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = img if getattr(img, "dtype", None) == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
        imwrite(out, arr)
        QMessageBox.information(None, "MuscleQuant", f"Saved: {out}")
    btn_export.clicked.connect(do_export)

    return bar

def make_manual_toolbar(viewer):
    """Manual additions controls in 2-row grid."""
    bar = QWidget()
    grid = QGridLayout(bar)
    grid.setContentsMargins(6,6,6,6)
    grid.addWidget(QLabel("Manual additions"), 0, 0)

    btn_start = QPushButton("Start (add layer)"); grid.addWidget(btn_start, 0, 1)
    btn_newid = QPushButton("New label ID");      grid.addWidget(btn_newid, 0, 2)
    btn_merge = QPushButton("Merge additions");   grid.addWidget(btn_merge, 1, 1)
    btn_save  = QPushButton("Re-quantify & Save");grid.addWidget(btn_save, 1, 2)

    btn_start.clicked.connect(lambda: ensure_manual_layer(viewer))
    def _newid():
        layer = ensure_manual_layer(viewer)
        if layer is not None:
            layer.selected_label = next_label_id()
            QMessageBox.information(None, "Manual mode", f"Selected label set to {layer.selected_label}.")
    btn_newid.clicked.connect(_newid)
    def _merge():
        ok = merge_manual_additions(viewer)
        if ok and "Manual_additions" in [L.name for L in viewer.layers]:
            viewer.layers["Manual_additions"].selected_label = next_label_id()
    btn_merge.clicked.connect(_merge)
    btn_save.clicked.connect(lambda: requant_and_save())

    return bar

def make_scrollable_panel(viewer, *widgets):
    container = QWidget()
    vbox = QVBoxLayout(container)
    for w in widgets:
        vbox.addWidget(w.native if hasattr(w, "native") else w)
    vbox.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(container)
    return scroll

# ---------------------- APP ENTRY (napari ≥ 0.5) ----------------------





@magicgui(layout="horizontal", call_button="Preview", folder={"label": "Images (Folder)", "mode": "d"})
def folder_preview(viewer: "napari.Viewer", folder: Path = Path(".")) -> None:
    try:
        batch_widget.folder.value = folder
    except Exception:
        pass
    files = sorted([p for p in Path(folder).glob("*.tif*")])
    if not files:
        QMessageBox.information(None, "No files", "No TIFF files found in the selected folder.")
        return
    viewer.layers.clear()
    CURRENT["preview_planes"] = {}
    detected = []
    pix_size_found = None
    bg_indices: List[int] = []
    fg_indices: List[int] = []
    for idx, p in enumerate(files):
        stem_lower = p.stem.lower()
        if "lam" in stem_lower or "background" in stem_lower:
            bg_indices.append(idx)
        else:
            fg_indices.append(idx)

    bg_count = len(bg_indices)
    fg_count = len(fg_indices)
    if bg_count > 0 and fg_count > 0:
        bg_share = 0.4
        fg_share = 1.0 - bg_share
    elif bg_count > 0:
        bg_share = 1.0
        fg_share = 0.0
    else:
        bg_share = 0.0
        fg_share = 1.0

    bg_opacity = (bg_share / bg_count) if bg_count else 0.0
    fg_opacity = (fg_share / fg_count) if fg_count else 0.0

    fg_color_idx = 0
    for idx, p in enumerate(files):
        plane, used = load_rgb_plane(p, None)
        CURRENT["preview_planes"][str(p)] = used
        stem_lower = p.stem.lower()
        is_background = idx in bg_indices
        if is_background:
            cmap = "gray"
            opacity = bg_opacity or 0.5
        else:
            cmap = COLORMAP_CYCLE[fg_color_idx % len(COLORMAP_CYCLE)]
            opacity = fg_opacity or 1.0
            fg_color_idx += 1
        layer = viewer.add_image(plane, name=p.stem, blending="additive", colormap=cmap, visible=True)
        layer.opacity = float(opacity)
        detected.append(infer_stain_from_filename(p))
        if pix_size_found is None:
            pix = _infer_pixel_size_um_from_tiff(p)
            if isinstance(pix, float) and pix > 0:
                pix_size_found = pix
    seen=set(); ordered=[]
    for n in detected:
        if n not in seen:
            seen.add(n); ordered.append(n)
    try:
        batch_widget.stain_names.value = ", ".join(ordered)
    except Exception:
        pass
    try:
        if ordered and hasattr(batch_widget, "laminin_substring"):
            batch_widget.laminin_substring.value = ordered[0]
    except Exception:
        pass
    try:
        if pix_size_found is not None and hasattr(batch_widget, "pixel_size_um"):
            batch_widget.pixel_size_um.value = float(pix_size_found)
    except Exception:
        pass

def main():
    from qtpy.QtWidgets import QApplication, QSplashScreen
    from qtpy.QtGui import QPixmap

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    pixmap = QPixmap(400, 200)
    pixmap.fill(Qt.white)

    splash = QSplashScreen(pixmap)
    splash.showMessage(
        "Loading MuscleQuant...",
        Qt.AlignCenter | Qt.AlignBottom
    )
    splash.show()
    app.processEvents()

    viewer = napari.Viewer()

    quickbar = make_quick_toolbar(viewer)
    manualbar = make_manual_toolbar(viewer)
    scroll_panel = make_scrollable_panel(viewer, folder_preview, batch_widget, manualbar, quickbar)

    viewer.window.add_dock_widget(scroll_panel, area="right", name="MuscleQuant")

    splash.close()

    print("""
MuscleQuant tips:
  • The right dock scrolls; controls are arranged across multiple rows to save space.
  • Background mode: 'Local ring' (default) with Median, Percentile, or Rolling-ball options.
  • Batch: prompts for export root once, and folder name per sample; writes per-sample folders + combined CSV.
  • Manual additions: Start (add layer) → paint → Merge additions → Re-quantify & Save (into the same sample folder).
""")

    napari.run()


if __name__ == "__main__":
    main()