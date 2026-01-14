# musclequant_gui.py — napari GUI for background segmentation + multi-protein quant
# IHC-first pipeline, NO p5 fallback. Local/Median/Percentile/Rolling-ball backgrounds.
# Features:
# • Single & Batch, auto RGB-plane or grayscale, cleaning, manual additions
# • Background mode: center-based intracellular background subtraction
# • Raw-intensity quant with rolling/local background + entropy / NN metrics
# • Per-protein: mean, max, integrated, CTCF, bg_mean, bg_mode
# • Per-sample folders + combined CSV, metadata.json
# • Compact multi-row toolbars; re-quantify uses same BG settings

import json
import copy
import weakref
from pathlib import Path
import time
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Set, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from skimage import color, measure, exposure
from skimage.measure import label as cc_label
from skimage.morphology import disk, dilation, erosion, remove_small_objects, binary_closing
from skimage.transform import resize
from skimage.util import img_as_float
from skimage.filters import threshold_otsu, gaussian
from scipy.ndimage import binary_fill_holes

from magicgui import magicgui
import napari
from napari.utils.colormaps import Colormap
from napari.utils import progress
from napari.utils.notifications import show_info as _show_info, show_warning as _show_warning
import musclequant.segmentation as segmentation
from qtpy.QtWidgets import (
    QApplication,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QPushButton,
    QSlider,
    QLabel,
    QGridLayout,
    QLayout,
    QFormLayout,
    QInputDialog,
    QFileDialog,
    QCheckBox,
    QDialog,
    QSplashScreen,
    QSizePolicy,
    QTabWidget,
    QDockWidget,
    QAbstractItemView,
    QListView,
    QTreeView,
    QStackedWidget,
)

from qtpy.QtCore import Qt, QTimer, QObject, QEvent
from qtpy.QtGui import QPixmap

import tifffile

from musclequant.config import (
    DEFAULT_MODEL,
    DEFAULT_DIAMETER,
    FLOW_THRESH,
    CELLPROB_THRESH,
    DEFAULT_PX_UM,
    SYNAPSE_RING_PX,
    SYNAPSE_PERCENTILE,
    CHUNK_COUNT_DEFAULT,
)
from musclequant.segmentation import (
    load_model,
    run_cellpose,
    clean_mask,
    overlay_boundaries,
)
from musclequant.quantification import (
    quantify,
    quantify_geometry,
    _filter_quant_columns,
    _split_metadata_columns,
    label_stats,
    prepare_export_dir,
    save_metadata_json,
    choose_export_root,
    normalize_bg_mode,
    _sanitize_metrics,
    compute_center_masks,
)

if TYPE_CHECKING:
    from cellpose import models as _cellpose_models
    CellposeModel = _cellpose_models.CellposeModel
else:
    CellposeModel = Any

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
STAIN_SUFFIX_RE = re.compile(r"(?:^|[_\s-])([^_\s\.-]+)\.(?:tif|tiff|png|jpg)$", re.IGNORECASE)

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
HEMATOXYLIN_CMAP = Colormap([[1.0, 1.0, 1.0, 1.0], [0.42, 0.24, 0.6, 1.0]], name="Hematoxylin")
EOSIN_CMAP = Colormap([[1.0, 1.0, 1.0, 1.0], [0.95, 0.6, 0.72, 1.0]], name="Eosin")
HED_DMAX = 3.0
DAB_CMAP = Colormap([[1.0, 1.0, 1.0, 1.0], [0.65, 0.45, 0.2, 1.0]], name="DAB")


def _stain_colormap(name: str, default: str, plane: Optional[str] = None) -> str:
    """Pick a consistent colormap for common stains."""
    n = (name or "").lower()
    if "hematox" in n or "haematox" in n:
        return HEMATOXYLIN_CMAP
    if "eosin" in n:
        return EOSIN_CMAP
    if "dab" in n:
        return DAB_CMAP
    if "dapi" in n:
        return "blue"
    if "fitc" in n:
        return "green"
    if "txred" in n or "texas" in n:
        return "red"
    if plane in ("R", "G", "B"):
        return {"R": "red", "G": "green", "B": "blue"}[plane]
    return default


def _stain_colormap_from_rgb(color: Optional[np.ndarray], fallback, *, transparent_background: bool = False):
    if color is None:
        return fallback
    try:
        rgb = np.clip(np.asarray(color, dtype=float)[:3], 0, 1)
        if transparent_background:
            return Colormap([[0.0, 0.0, 0.0, 0.0], [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]], name="stain")
        return Colormap([[1.0, 1.0, 1.0, 1.0], [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]], name="stain")
    except Exception:
        return fallback


@dataclass
class ProteinChannel:
    """Container describing a quantifiable channel (file-backed or virtual)."""
    name: str
    plane: str
    path: Optional[Path] = None
    data: Optional[np.ndarray] = None
    preview_id: Optional[str] = None
    origin: Optional[str] = None

    def key(self) -> str:
        if self.preview_id:
            return self.preview_id
        if self.path is not None:
            return str(self.path)
        if self.origin:
            return f"virtual::{self.origin}::{self.name}"
        return f"virtual::{self.name}"


def _parse_stain_list(stain_names: str) -> List[str]:
    if not isinstance(stain_names, str):
        return []
    return [s.strip() for s in stain_names.split(",") if s.strip()]


def _virtual_key(base: str, channel_name: str) -> str:
    return f"virtual::{base}::{channel_name}"


_HE_HINTS = ("h&e", "hand e", "h and e", "hematox", "haematox", "eosin", "hemaeosin", "hed", "brightfield", "bright-field")


def _text_has_he_hint(text: str) -> bool:
    if not text:
        return False
    txt = text.lower()
    variants = [txt, txt.replace("&", "and"), txt.replace(" ", "")]
    for hint in _HE_HINTS:
        if any(hint in variant for variant in variants):
            return True
    return False


def _should_use_he_mode(files: List[Path], stain_tokens: List[str]) -> bool:
    if len(files) != 1:
        return False
    candidates = [files[0].stem, files[0].name] + stain_tokens
    return any(_text_has_he_hint(c) for c in candidates if c)


def _set_quant_method(method: str, viewer: Optional["napari.Viewer"] = None):
    """Update selected quantification method and toggle method-specific controls."""
    CURRENT["quant_method"] = method
    show_synaptic = method == SYNAPTIC_METHOD
    show_standard = method == STANDARD_METHOD
    for w in list(_synaptic_controls):
        try:
            w.setVisible(show_synaptic)
        except Exception:
            pass
    for w in list(_standard_controls):
        try:
            w.setVisible(show_standard)
        except Exception:
            pass
    try:
        if _metrics_panel_widget is not None:
            _metrics_panel_widget.setVisible(method == STANDARD_METHOD)
    except Exception:
        pass

def infer_mode_from_files(files: List[Path], stain_tokens: List[str]) -> str:
    if not files:
        return "IHC"
    candidates = [p.stem for p in files] + [p.name for p in files] + stain_tokens
    if any(_text_has_he_hint(c) for c in candidates if c):
        return "H&E"
    if len(files) == 1:
        return "H&E"
    return "IHC"


def _build_protein_channels(
    base: str,
    preview_records: List[Tuple[str, str, np.ndarray, Optional[Path]]],
    preview_source: Dict[str, str],
    virtual_payload: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[List[ProteinChannel], Dict[str, str]]:
    channels: List[ProteinChannel] = []
    preview_map: Dict[str, str] = {}
    for name, plane_code, _disp, raw_path in preview_records:
        if isinstance(raw_path, Path):
            src_path = raw_path
        elif raw_path is None:
            src_path = None
        else:
            src_path = Path(raw_path)
        data = (virtual_payload or {}).get(name)
        key = str(src_path) if src_path is not None else _virtual_key(base, name)
        preview_plane = preview_source.get(str(raw_path)) if src_path is not None else None
        if not preview_plane:
            preview_plane = plane_code if plane_code in PLANE_TO_IDX else "Gray"
        preview_map[key] = preview_plane
        channels.append(
            ProteinChannel(
                name=name,
                plane=plane_code,
                path=src_path,
                data=data,
                preview_id=key,
                origin=base if src_path is None else None,
            )
        )
    return channels, preview_map

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
        if inferred.lower() in ("background",):
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
        if desired.lower() in ("background",):
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


def _load_channel_raw(channel: ProteinChannel, preview_planes: Dict[str, str]) -> np.ndarray:
    key = channel.key()
    preview_plane = preview_planes.get(key)
    plane_for_raw = preview_plane if preview_plane in PLANE_TO_IDX else (channel.plane if channel.plane in PLANE_TO_IDX else None)
    if channel.data is not None:
        return channel.data.astype(np.float32, copy=False)
    if channel.path is None:
        raise ValueError(f"Channel '{channel.name}' has no backing data or file.")
    return load_raw_gray(channel.path, plane_for_raw)


def _channel_preview_image(channel: ProteinChannel) -> Optional[np.ndarray]:
    if channel.data is not None:
        arr = np.asarray(channel.data, dtype=np.float32)
    elif channel.path is not None:
        gray, _ = load_rgb_plane(channel.path, channel.plane if channel.plane in PLANE_TO_IDX else None)
        arr = np.asarray(gray, dtype=np.float32)
    else:
        return None
    if arr.size == 0:
        return None
    if arr.max() > 1.0 + 1e-3 or arr.min() < -1e-3:
        arr = exposure.rescale_intensity(arr, in_range="image", out_range=(0, 1))
    else:
        arr = np.clip(arr, 0, 1)
    return arr


def _combine_laminin_planes(lam_files: List[Path], return_raw: bool = False) -> Union[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, np.ndarray, List[str]]]:
    if not lam_files:
        raise ValueError("No background files supplied.")
    lam_grays: List[np.ndarray] = []
    lam_raws: List[np.ndarray] = []
    lam_used_list: List[str] = []
    target_shape = None
    resized = False
    for lp in lam_files:
        lg, lu = load_rgb_plane(lp, None)
        raw = load_raw_gray(lp, lu if lu in PLANE_TO_IDX else None)
        if target_shape is None:
            target_shape = lg.shape
        elif lg.shape != target_shape:
            lg = resize(lg, target_shape, order=1, preserve_range=True, anti_aliasing=True)
            raw = resize(raw, target_shape, order=1, preserve_range=True, anti_aliasing=True)
            resized = True
        lam_grays.append(lg.astype(np.float32, copy=False))
        lam_raws.append(raw.astype(np.float32, copy=False))
        lam_used_list.append(lu)
    lam_gray = lam_grays[0] if len(lam_grays) == 1 else np.maximum.reduce(lam_grays)
    lam_raw = lam_raws[0] if len(lam_raws) == 1 else np.maximum.reduce(lam_raws)
    if resized:
        show_warning("Background channels had different sizes; resized to match the first background image.")
    if return_raw:
        return lam_gray, lam_raw, lam_used_list
    return lam_gray, lam_used_list


def _compute_membrane_and_interior_masks(masks: np.ndarray, dilation_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-cell membrane and interior label maps using an erosion-based shell.
    Membrane = cell minus eroded interior (thickness ~ dilation_px).
    """
    dilation_px = max(1, int(dilation_px or 0))
    selem = disk(dilation_px)
    membrane = np.zeros_like(masks, dtype=np.uint16)
    interior = np.zeros_like(masks, dtype=np.uint16)
    labels = np.unique(masks.astype(np.int64, copy=False))
    labels = labels[labels > 0]
    for lbl in labels:
        cell_mask = masks == lbl
        if not np.any(cell_mask):
            continue
        eroded = erosion(cell_mask, selem)
        if not np.any(eroded):
            eroded = cell_mask
        interior[eroded] = lbl
        mem_region = cell_mask & (~eroded)
        if np.any(mem_region):
            membrane[mem_region] = lbl
    return membrane, interior


def highlight_membranes(viewer: Optional["napari.Viewer"], dilation_px: int):
    """
    Preview membrane shells on the current segmentation using the chosen dilation.
    """
    _activate_viewer_state(viewer)
    if viewer is None:
        show_info("No viewer available.")
        return
    masks = CURRENT.get("masks")
    if masks is None:
        labels_name = "Fibers_mask_clean" if CURRENT.get("cleaned") else "Fibers_mask"
        label_layer = None
        for layer in viewer.layers:
            if layer.name == labels_name:
                label_layer = layer
                break
        if label_layer is None:
            for layer in viewer.layers:
                if layer.__class__.__name__.lower().startswith("labels"):
                    label_layer = layer
                    break
        if label_layer is not None:
            try:
                masks = np.asarray(label_layer.data, dtype=np.uint16)
                CURRENT["masks"] = masks
            except Exception:
                masks = None
    if masks is None:
        mq_info(None, "Highlight membranes", "Run segmentation first (no masks loaded).")
        return
    mem_layer, _ = _compute_membrane_and_interior_masks(np.asarray(masks), int(dilation_px))
    _add_or_replace_layer(viewer, "Membranes", mem_layer.astype(np.uint16, copy=False), opacity=0.6)
    CURRENT["membrane_px"] = int(dilation_px)
    CURRENT["membrane_layer"] = mem_layer
    show_info(f"Membrane preview updated ({int(dilation_px)} px shell).")


def _membrane_area_maps(membrane_labels: np.ndarray, px_um: float) -> Tuple[Dict[int, int], Dict[int, float]]:
    labels = np.unique(membrane_labels.astype(np.int64, copy=False))
    labels = labels[labels > 0]
    area_px_map: Dict[int, int] = {}
    area_um2_map: Dict[int, float] = {}
    scale = float(px_um) ** 2
    for lbl in labels:
        area_px = int(np.count_nonzero(membrane_labels == lbl))
        area_px_map[int(lbl)] = area_px
        area_um2_map[int(lbl)] = area_px * scale
    return area_px_map, area_um2_map


def _add_membrane_intensity_columns(
    df: pd.DataFrame,
    intensity_raw: np.ndarray,
    masks: np.ndarray,
    membrane_labels: np.ndarray,
    interior_labels: np.ndarray,
    protein_name: str,
) -> pd.DataFrame:
    """Append membrane mean/CTCF/MFI columns using intracellular background."""
    img = np.asarray(intensity_raw, dtype=np.float32)
    mem_mean_map: Dict[int, float] = {}
    mem_ctcf_map: Dict[int, float] = {}
    mem_bg_map: Dict[int, float] = {}
    mem_mfi_map: Dict[int, float] = {}

    labels = np.unique(masks.astype(np.int64, copy=False))
    labels = labels[labels > 0]
    for lbl in labels:
        mem_region = membrane_labels == lbl
        if not np.any(mem_region):
            continue
        interior = interior_labels == lbl
        cell_mask = masks == lbl
        if np.any(interior):
            bg_val = float(np.mean(img[interior]))
        elif np.any(cell_mask):
            bg_val = float(np.median(img[cell_mask]))
        else:
            bg_val = 0.0
        mem_mean = float(np.mean(img[mem_region])) if np.any(mem_region) else 0.0
        area = int(np.count_nonzero(mem_region))
        ctcf_val = (mem_mean * area) - (bg_val * area)
        mfi_val = ctcf_val / float(area) if area > 0 else 0.0

        mem_mean_map[int(lbl)] = mem_mean
        mem_ctcf_map[int(lbl)] = ctcf_val
        mem_bg_map[int(lbl)] = bg_val
        mem_mfi_map[int(lbl)] = mfi_val

    df[f"{protein_name} Membrane Background Mean"] = df["label"].map(mem_bg_map).fillna(0.0)
    df[f"{protein_name} Membrane Mean"] = df["label"].map(mem_mean_map).fillna(0.0)
    df[f"{protein_name} Membrane CTCF"] = df["label"].map(mem_ctcf_map).fillna(0.0)
    df[f"{protein_name} Membrane MFI"] = df["label"].map(mem_mfi_map).fillna(0.0)
    return df


def _debug_log_image_info(tag: str, arr: np.ndarray):
    try:
        print(f"[DEBUG] {tag}: shape={arr.shape}, dtype={arr.dtype}, ndim={arr.ndim}, "
              f"min={float(np.nanmin(arr))}, max={float(np.nanmax(arr))}")
    except Exception:
        pass


def _filter_channels_by_viewer(viewer: Optional["napari.Viewer"], channels: List[ProteinChannel]) -> List[ProteinChannel]:
    """
    Drop channels whose image layers were deleted in the viewer.
    Keeps virtual channels. Matches by layer name or file stem.
    """
    if viewer is None:
        return channels
    # For imported runs, avoid over-filtering: keep everything if we cannot confidently map layers.
    imported_mode = CURRENT.get("mode_source") == "import"
    layer_names = {L.name for L in viewer.layers if L.__class__.__name__.lower().startswith("image")}
    layer_names_lower = {n.lower() for n in layer_names}
    stems: Set[str] = set()
    for L in viewer.layers:
        if not L.__class__.__name__.lower().startswith("image"):
            continue
        candidate = getattr(L, "source", None)
        if hasattr(candidate, "path") and candidate.path:
            cand_val = candidate.path
        else:
            cand_val = candidate
        try:
            stems.add(Path(cand_val).stem)
        except Exception:
            try:
                stems.add(Path(str(cand_val)).stem)
            except Exception:
                continue
    filtered: List[ProteinChannel] = []
    # If we cannot derive any mapping from the viewer (no image stems), keep everything to avoid dropping channels.
    no_mapping = (imported_mode and not stems) or (not stems and not layer_names)
    for ch in channels:
        if ch.path is None:
            filtered.append(ch)
            continue
        if no_mapping:
            filtered.append(ch)
            continue
        stem = Path(ch.path).stem
        if stem in layer_names or stem in stems or ch.name in layer_names:
            filtered.append(ch)
        else:
            # Fallback: keep if channel name appears in any image layer label (for imported runs with custom names).
            if any(ch.name.lower() in lname for lname in layer_names_lower):
                filtered.append(ch)
            else:
                print(f"[S1/quant] Skipping channel removed from viewer: {stem}")
    return filtered


def _ensure_masks_from_viewer(viewer: Optional["napari.Viewer"]) -> Optional[np.ndarray]:
    if viewer is None:
        return CURRENT.get("masks")
    if CURRENT.get("masks") is not None:
        return CURRENT.get("masks")
    label_layer = None
    for layer in viewer.layers:
        if layer.__class__.__name__.lower().startswith("labels") and "fibers_mask" in layer.name.lower():
            label_layer = layer
            break
    if label_layer is None:
        active = getattr(viewer.layers, "selection", None)
        if active and len(active) == 1:
            only = list(active)[0]
            if only.__class__.__name__.lower().startswith("labels"):
                label_layer = only
    if label_layer is None:
        for layer in viewer.layers:
            if layer.__class__.__name__.lower().startswith("labels"):
                label_layer = layer
                break
    if label_layer is None:
        return None
    try:
        masks = np.asarray(label_layer.data, dtype=np.uint16)
    except Exception:
        return None
    CURRENT["masks"] = masks
    CURRENT["cleaned"] = "clean" in label_layer.name.lower()
    return masks


def _fallback_proteins_from_viewer(viewer: Optional["napari.Viewer"]) -> List[ProteinChannel]:
    if viewer is None:
        return []
    proteins: List[ProteinChannel] = []
    for layer in viewer.layers:
        if not layer.__class__.__name__.lower().startswith("image"):
            continue
        lname = layer.name.lower()
        if "overlay" in lname or "white background" in lname:
            continue
        path = _extract_layer_path(layer)
        data = None
        if path is None:
            try:
                data = np.asarray(layer.data, dtype=np.float32)
            except Exception:
                continue
        proteins.append(
            ProteinChannel(
                name=layer.name,
                plane="Gray",
                path=path,
                data=data,
                preview_id=str(path) if path is not None else None,
                origin="viewer",
            )
        )
    return proteins

def _extract_layer_path(layer) -> Optional[Path]:
    """Best-effort to recover the filesystem path for a napari image layer."""
    candidate = getattr(layer, "source", None)
    path_attr = None
    if hasattr(candidate, "path") and getattr(candidate, "path"):
        path_attr = getattr(candidate, "path")
    elif hasattr(layer, "metadata"):
        meta = layer.metadata or {}
        path_attr = meta.get("path") or meta.get("source") or meta.get("file")
    if path_attr:
        try:
            return Path(path_attr)
        except Exception:
            return None
    return None


def _guess_pixel_size_from_layer(layer) -> Optional[float]:
    """Try to infer pixel size from layer metadata/scale."""
    meta = getattr(layer, "metadata", {}) or {}
    for key in ("PhysicalSizeX", "physicalSizeX", "pixel_size_um", "px_um", "pixelSize", "PixelSize"):
        val = meta.get(key)
        try:
            if val is not None:
                val_f = float(val)
                if val_f > 0:
                    return val_f
        except Exception:
            continue
    scale = getattr(layer, "scale", None)
    try:
        if scale is not None and len(scale) >= 1:
            val_f = float(scale[0])
            if val_f > 0 and val_f != 1.0:
                return val_f
    except Exception:
        pass
    return None


def _parse_px_um(candidate: Any) -> Optional[float]:
    """Convert candidate pixel size to float microns/pixel when possible."""
    try:
        val = float(candidate)
        if val > 0:
            return val
    except Exception:
        return None
    return None


def _px_um_from_sample(sample: Dict[str, Any], default: float = DEFAULT_PX_UM) -> float:
    """Resolve pixel size (µm/px) from a sample dict or its metadata."""
    meta = sample.get("metadata") or {}
    for cand in (sample.get("px_um"), meta.get("pixel_size_um"), meta.get("px_um")):
        val = _parse_px_um(cand)
        if val is not None:
            return val
    return float(default)


def _propagate_px_um(px_um_val: float):
    """Update CURRENT and cached samples with a new pixel size."""
    try:
        px = float(px_um_val)
    except Exception:
        return
    if px <= 0:
        return
    CURRENT["px_um"] = px
    for sample in SEGMENTED_CACHE:
        sample["px_um"] = px
        meta = sample.get("metadata") or {}
        meta["pixel_size_um"] = px
        meta["px_um"] = px
        sample["metadata"] = meta


def _sample_from_viewer_layers(
    viewer: "napari.Viewer",
    lam_keys: List[str],
    stain_tokens: List[str],
    default_px_um: float,
) -> Optional[Dict[str, Any]]:
    """Build a virtual sample from images already loaded in the viewer."""
    image_layers = [L for L in viewer.layers if L.__class__.__name__.lower().startswith("image")]
    if not image_layers:
        return None
    entries: List[Dict[str, Any]] = []
    used_names: Set[str] = set()
    px_guess: Optional[float] = None

    lam_key_lower = [k.lower() for k in lam_keys]
    for idx, layer in enumerate(image_layers):
        arr = np.asarray(layer.data)
        if arr.ndim < 2:
            continue
        arr = np.squeeze(arr)
        plane_tag = "Gray"
        try:
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                plane_tag = autodetect_plane(arr)
                arr = arr[..., PLANE_TO_IDX[plane_tag]]
        except Exception:
            plane_tag = "Gray"
        raw = np.asarray(arr, dtype=np.float32)
        disp = raw
        if disp.max() > 1.5:
            disp = exposure.rescale_intensity(disp, in_range="image", out_range=(0, 1))
        else:
            disp = np.clip(disp, 0, 1)
        name = (layer.name or f"Layer{idx+1}").strip() or f"Layer{idx+1}"
        inferred = infer_stain_from_filename(Path(name))
        if inferred and inferred.lower() != "unknown":
            name = inferred
        name = _uniquify_name(name, used_names)
        used_names.add(name)
        lname = name.lower()
        is_bg = False
        if lam_key_lower:
            is_bg = any(k in lname for k in lam_key_lower)
        if not is_bg:
            is_bg = ("lam" in lname) or ("background" in lname)
        source_path = _extract_layer_path(layer)
        px_guess = px_guess or _guess_pixel_size_from_layer(layer)
        entries.append(
            {
                "name": name,
                "raw": raw,
                "disp": disp,
                "plane": plane_tag,
                "is_bg": is_bg,
                "path": source_path,
            }
        )

    if not entries:
        return None
    lam_entry = next((e for e in entries if e["is_bg"]), None) or entries[0]
    proteins = [e for e in entries if e is not lam_entry]

    virtual_payload: Dict[str, np.ndarray] = {}
    preview_source: Dict[str, str] = {}
    proteins_loaded: List[Tuple[str, str, np.ndarray, Optional[Path]]] = []
    used_names.clear()
    for p in proteins:
        pname = _uniquify_name(p["name"], used_names)
        used_names.add(pname)
        proteins_loaded.append((pname, p["plane"], p["disp"], p["path"]))
        virtual_payload[pname] = p["raw"]
        if p["path"]:
            preview_source[str(p["path"])] = p["plane"]

    base = lam_entry["name"] or "viewer_import"
    lam_files = [lam_entry["path"]] if lam_entry["path"] else []
    return {
        "_source": "viewer",
        "base": base,
        "lam_gray": lam_entry["disp"],
        "lam_raw": lam_entry["raw"],
        "lam_used_list": [lam_entry["plane"]],
        "lam_path": lam_entry["path"],
        "lam_files": lam_files,
        "proteins_loaded": proteins_loaded,
        "virtual_payload": virtual_payload,
        "preview_source": preview_source,
        "pixel_size_um": float(px_guess) if px_guess else float(default_px_um),
        "sample_files": [],
        "he_mode": False,
    }

def _he_quant_mode() -> bool:
    return bool(CURRENT.get("he_quant_mode", True))

def _he_dmax() -> float:
    try:
        return float(CURRENT.get("he_dmax", HED_DMAX))
    except Exception:
        return HED_DMAX

def _estimate_stain_color(rgb_float: np.ndarray, channel: np.ndarray) -> Optional[np.ndarray]:
    if rgb_float is None or channel is None:
        return None
    try:
        vals = channel.astype(np.float32, copy=False)
        if vals.size == 0:
            return None
        thr = np.nanpercentile(vals, 99.5)
        mask = vals >= thr
        if not np.any(mask):
            thr = np.nanpercentile(vals, 95)
            mask = vals >= thr
        if not np.any(mask):
            return None
        # Prefer darker tissue pixels to avoid white background bias.
        rgb = rgb_float[mask]
        if rgb.size == 0:
            return None
        mean = np.nanmean(rgb, axis=0)
        mean = np.clip(mean, 0, 1)
        if np.nanmean(mean) > 0.95:
            return None
        return mean.astype(np.float32, copy=False)
    except Exception:
        return None


def _separate_he_channels(
    path: Path,
    *,
    include_dab: bool = True,
    quant_mode: bool = True,
    dmax: float = HED_DMAX,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Robust H&E stain separation.

    - Accepts RGB, RGBA, or single-channel grayscale TIFFs.
    - Ensures a proper RGB float image in [0, 1].
    - Uses rgb2hed (which handles OD internally).
    - Returns raw OD density (hed_raw) plus display-friendly channels.
    """
    try:
        arr = imread(str(path))
    except Exception:
        return None

    _debug_log_image_info(f"H&E source {path.name}", arr)

    # Normalize to RGB
    if arr.ndim == 2:
        rgb = np.dstack([arr, arr, arr])
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        rgb = np.dstack([arr[..., 0], arr[..., 0], arr[..., 0]])
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3]
    else:
        return None

    rgb_float = img_as_float(rgb)
    if quant_mode:
        rgb_in = np.clip(rgb_float, 0, 1)
    else:
        # White-balance using bright pixels to stabilize stain separation.
        try:
            bg = np.percentile(rgb_float.reshape(-1, 3), 99.5, axis=0)
            bg = np.clip(bg, 1e-3, None)
            rgb_in = np.clip(rgb_float / bg, 0, 1)
        except Exception:
            rgb_in = np.clip(rgb_float, 0, 1)

    try:
        stains = color.separate_stains(rgb_in, color.hed_from_rgb)
    except Exception:
        try:
            stains = color.rgb2hed(rgb_in)
        except Exception:
            return None

    hed_raw = stains.astype(np.float32, copy=False)
    dmax_val = float(dmax) if dmax is not None else None
    if dmax_val is not None:
        hed_display = np.clip(hed_raw, 0, dmax_val)
    else:
        hed_display = hed_raw

    channels: Dict[str, np.ndarray] = {}
    h_raw = hed_raw[..., 0]
    e_raw = hed_raw[..., 1]
    h_disp = hed_display[..., 0]
    e_disp = hed_display[..., 1]
    h_color = _estimate_stain_color(rgb_in, h_disp)
    e_color = _estimate_stain_color(rgb_in, e_disp)
    channels["Hematoxylin"] = h_disp
    channels["Eosin"] = e_disp

    if include_dab:
        dab_raw = hed_raw[..., 2]
        dab_disp = hed_display[..., 2]
        if np.nanmax(dab_raw) - np.nanmin(dab_raw) > 1e-3:
            if np.nanmean(dab_raw) > 0.03 or np.nanpercentile(dab_raw, 99) > 0.1:
                channels["DAB"] = dab_disp
    stain_colors: Dict[str, np.ndarray] = {}
    if h_color is not None:
        stain_colors["Hematoxylin"] = h_color
    if e_color is not None:
        stain_colors["Eosin"] = e_color
    if include_dab and "DAB" in channels:
        dab_color = _estimate_stain_color(rgb_float, channels["DAB"])
        if dab_color is not None:
            stain_colors["DAB"] = dab_color
    if stain_colors:
        channels["_stain_colors"] = stain_colors
    try:
        recon = color.hed2rgb(hed_raw)
        channels["Composite"] = np.clip(recon, 0, 1).astype(np.float32, copy=False)
    except Exception:
        pass
    raw_channels = {
        "Hematoxylin": h_raw,
        "Eosin": e_raw,
    }
    if include_dab and "DAB" in channels:
        raw_channels["DAB"] = dab_raw
    channels["_raw"] = raw_channels
    channels["_hed_raw"] = hed_raw
    channels["_hed_display"] = hed_display
    channels["_rgb_in"] = rgb_in.astype(np.float32, copy=False)
    channels["_dmax"] = dmax_val
    return channels

def _add_he_layers(viewer: "napari.Viewer", base_name: str, he_payload: Dict[str, np.ndarray]) -> None:
    if viewer is None:
        return
    rgb_in = he_payload.get("_rgb_in")
    if isinstance(rgb_in, np.ndarray):
        viewer.add_image(
            rgb_in,
            name=f"{base_name} — Original (H&E input)",
            rgb=True,
            blending="opaque",
            visible=False,
        )
    hed_raw = he_payload.get("_hed_raw")
    if isinstance(hed_raw, np.ndarray):
        CURRENT["he_hed_raw"] = hed_raw
        alpha_h = float(CURRENT.get("he_alpha_h", 1.0))
        alpha_e = float(CURRENT.get("he_alpha_e", 1.0))
        alpha_d = float(CURRENT.get("he_alpha_d", 1.0))
        recon = _he_reconstruct(hed_raw, alpha_h, alpha_e, alpha_d)
        if recon is not None:
            recon_layer = viewer.add_image(
                recon,
                name=f"{base_name} — Recon",
                rgb=True,
                blending="opaque",
                visible=True,
            )
            CURRENT["he_recon_layer_name"] = recon_layer.name
    dmax = he_payload.get("_dmax", HED_DMAX)
    dmax_val = float(dmax) if dmax is not None else None
    for chan_name in ("Hematoxylin", "Eosin", "DAB"):
        arr = he_payload.get(chan_name)
        if arr is None:
            continue
        layer = viewer.add_image(
            np.asarray(arr, dtype=np.float32),
            name=f"{base_name} — {chan_name}",
            colormap="gray",
            blending="translucent",
            visible=False,
        )
        if dmax_val is not None:
            layer.contrast_limits = (0.0, dmax_val)
        layer.opacity = 1.0

def _update_he_recon_layer(viewer: Optional["napari.Viewer"]) -> None:
    if viewer is None:
        return
    hed_raw = CURRENT.get("he_hed_raw")
    if hed_raw is None:
        return
    try:
        alpha_h = float(CURRENT.get("he_alpha_h", 1.0))
        alpha_e = float(CURRENT.get("he_alpha_e", 1.0))
        alpha_d = float(CURRENT.get("he_alpha_d", 1.0))
    except Exception:
        return
    recon = _he_reconstruct(hed_raw, alpha_h, alpha_e, alpha_d)
    if recon is None:
        return
    layer_name = CURRENT.get("he_recon_layer_name")
    if not layer_name:
        return
    layer = next((L for L in viewer.layers if L.name == layer_name), None)
    if layer is not None:
        layer.data = recon

def _he_reconstruct(hed_raw: np.ndarray, alpha_h: float, alpha_e: float, alpha_d: float) -> Optional[np.ndarray]:
    try:
        hed_scaled = np.array(hed_raw, dtype=np.float32, copy=True)
        if hed_scaled.shape[-1] > 0:
            hed_scaled[..., 0] *= alpha_h
        if hed_scaled.shape[-1] > 1:
            hed_scaled[..., 1] *= alpha_e
        if hed_scaled.shape[-1] > 2:
            hed_scaled[..., 2] = 0.0
        recon = color.hed2rgb(hed_scaled)
        return np.clip(recon, 0, 1).astype(np.float32, copy=False)
    except Exception:
        return None


# ---------- Manual additions: state + ops ----------
CURRENT_TEMPLATE: Dict[str, Any] = {
    "base": None,
    "lam": None,              # np.ndarray
    "lam_raw": None,          # np.ndarray (un-rescaled, matched to mask)
    "lam_paths": [],          # list of Path
    "lam_used_list": [],
    "masks": None,            # np.ndarray (labels)
    "proteins": [],           # List[ProteinChannel]
    "preview_planes": {},     # preview_id -> plane code used in preview/quant
    "px_um": DEFAULT_PX_UM,
    "save_dir": None,
    "cleaned": False,
    "bg_mode": "center",
    "bg_percentile": 5.0,
    "local_ring_px": 5,
    "rolling_ball_radius_px": 50,
    "add_texture": True,
    "add_spatial": True,
    "drop_edge_touching": True,
    "drop_edge_buffer_px": 3,
    "chunk_first": False,
    "chunk_count": CHUNK_COUNT_DEFAULT,
    "metrics": {"area": True, "ctcf": False, "eccentricity": False, "solidity": False},
    "mode": "IHC",
    "mode_source": "auto",
    "export_all_images": False,
    "quant_use_raw": True,    # use raw intensities (ImageJ-style) for quant/S1 (no per-image rescale)
    "he_quant_mode": True,    # H&E: disable per-image white balance for quantitative separation
    "he_dmax": HED_DMAX,      # H&E: fixed OD clip for display/quant stability
    "he_alpha_h": 1.0,
    "he_alpha_e": 1.0,
    "he_alpha_d": 1.0,
    "last_preview_folder": None,
    "synapse_stain": "",
    "s1_min_mult": 5.0,
    "s1_min_hits": 5,
    "s1_min_raw": 0.0,
    "s1_membrane_px": 4,
    "membrane_px": 4,
    "modified": False,
    "s1_manual_keep_ids": [],
    "quant_method": None,
    "synaptic_results": None,
}

CURRENT: Dict[str, Any] = copy.deepcopy(CURRENT_TEMPLATE)
CURRENT_BY_VIEWER: Dict[int, Dict[str, Any]] = {}
_ACTIVE_VIEWER: Optional["napari.Viewer"] = None

# Cache recent segmentations so quantification can be triggered separately.
SEGMENTED_CACHE: List[Dict[str, Any]] = []
SEGMENT_EXPORT_ROOT: Optional[Path] = None
SYNAPTIC_METHOD = "Synaptic analysis"
STANDARD_METHOD = "Standard quantification"
_synaptic_controls: List[Any] = []
_standard_controls: List[Any] = []
_metrics_panel_widget = None
_single_only_widgets: List[QWidget] = []
_method_combo_widget: Optional[QWidget] = None
_single_only_panels: List[QWidget] = []
_quickbar_controls: Dict[str, QWidget] = {}
_quickbar_grid = None
_quickbar_row_cell = 1


def show_info(message: str, *, duration: float = 6.0, **kwargs):
    """Info notification that auto-dismisses after a short duration (best-effort)."""
    try:
        return _show_info(message, duration=duration, **kwargs)
    except TypeError:
        # Older napari versions may not support duration; fallback silently.
        return _show_info(message, **kwargs)


def show_warning(message: str, *, duration: float = 8.0, **kwargs):
    """Warning notification that auto-dismisses after a short duration (best-effort)."""
    try:
        return _show_warning(message, duration=duration, **kwargs)
    except TypeError:
        return _show_warning(message, **kwargs)


_ACTIVE_MSGBOXES: List[QMessageBox] = []


def _activate_viewer_state(viewer: Optional["napari.Viewer"]) -> None:
    global CURRENT, _ACTIVE_VIEWER
    if viewer is None:
        return
    _ACTIVE_VIEWER = viewer
    key = id(viewer)
    state = CURRENT_BY_VIEWER.get(key)
    if state is None:
        state = copy.deepcopy(CURRENT_TEMPLATE)
        CURRENT_BY_VIEWER[key] = state
    CURRENT = state


def _sync_magicgui_viewer(viewer: Optional["napari.Viewer"]) -> None:
    if viewer is None:
        return
    try:
        if hasattr(folder_preview, "viewer"):
            folder_preview.viewer.value = viewer
    except Exception:
        pass
    try:
        if hasattr(batch_widget, "viewer"):
            batch_widget.viewer.value = viewer
    except Exception:
        pass


def _sync_batch_stain_visibility():
    try:
        if _BATCH_USE_ALL_STRAINS is None:
            return
        use_all = _BATCH_USE_ALL_STRAINS.isChecked()
        if hasattr(batch_widget, "stain_names"):
            batch_widget.stain_names.visible = not use_all
    except Exception:
        pass


def _set_ui_mode(mode: str, viewer: Optional["napari.Viewer"]) -> None:
    if _UI_MODE_STACK is None:
        return
    is_batch = str(mode).strip().lower() == "batch"
    for combo in _UI_MODE_COMBOS:
        try:
            combo.blockSignals(True)
            combo.setCurrentText("Batch" if is_batch else "Single")
        except Exception:
            pass
        finally:
            try:
                combo.blockSignals(False)
            except Exception:
                pass
    try:
        _UI_MODE_STACK.setCurrentIndex(1 if is_batch else 0)
    except Exception:
        pass

    # Move shared widgets between panels.
    if _SINGLE_BODY is not None and _BATCH_BODY is not None:
        try:
            target = _BATCH_BODY if is_batch else _SINGLE_BODY
            if batch_widget.native.parentWidget() is not target:
                batch_widget.native.setParent(None)
                target.layout().addWidget(batch_widget.native)
        except Exception:
            pass
        if _metrics_panel_widget is not None:
            try:
                target = _BATCH_BODY if is_batch else _SINGLE_BODY
                if _metrics_panel_widget.parentWidget() is not target:
                    _metrics_panel_widget.setParent(None)
                    target.layout().addWidget(_metrics_panel_widget)
            except Exception:
                pass

    # Hide single-only widgets in batch mode.
    for w in _single_only_widgets:
        try:
            w.setVisible(not is_batch)
        except Exception:
            pass
    for w in _single_only_panels:
        try:
            w.setVisible(not is_batch)
        except Exception:
            pass
    if _method_combo_widget is not None:
        try:
            _method_combo_widget.setEnabled(not is_batch)
            if is_batch:
                _method_combo_widget.setCurrentText(STANDARD_METHOD)
        except Exception:
            pass
    _set_batch_call_button_visible(not is_batch)
    if is_batch:
        _update_batch_action_ui(viewer)
    if is_batch:
        try:
            if hasattr(batch_widget, "save_dir"):
                batch_widget.save_dir.visible = False
        except Exception:
            pass
        _set_batch_call_button_visible(False)
        try:
            if hasattr(batch_widget, "stain_names"):
                _sync_batch_stain_visibility()
        except Exception:
            pass

    if is_batch and viewer is not None:
        _activate_viewer_state(viewer)
        _sync_magicgui_viewer(viewer)
        _update_batch_action_ui(viewer)
    if not is_batch:
        try:
            if hasattr(batch_widget, "stain_names"):
                batch_widget.stain_names.visible = True
        except Exception:
            pass
        for _key, btn in _quickbar_controls.items():
            try:
                btn.setVisible(True)
            except Exception:
                pass


def _show_timed_message(parent, title: str, message: str, icon, *, duration: float) -> QMessageBox:
    box = QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setText(message)
    box.setStandardButtons(QMessageBox.Ok)
    box.setModal(False)
    box.show()
    _ACTIVE_MSGBOXES.append(box)

    def _cleanup(*_args):
        if box in _ACTIVE_MSGBOXES:
            _ACTIVE_MSGBOXES.remove(box)
    try:
        box.finished.connect(_cleanup)
    except Exception:
        pass
    QTimer.singleShot(int(max(duration, 0.1) * 1000), box.accept)
    return box


def mq_info(parent, title: str, message: str, *, duration: float = 6.0) -> QMessageBox:
    return _show_timed_message(parent, title, message, QMessageBox.Information, duration=duration)


def mq_warning(parent, title: str, message: str, *, duration: float = 8.0) -> QMessageBox:
    return _show_timed_message(parent, title, message, QMessageBox.Warning, duration=duration)


def mq_critical(parent, title: str, message: str, *, duration: float = 12.0) -> QMessageBox:
    return _show_timed_message(parent, title, message, QMessageBox.Critical, duration=duration)


def _clean_path_candidate(candidate: Any) -> Optional[Path]:
    if candidate is None:
        return None
    try:
        p = Path(candidate)
    except Exception:
        return None
    if str(p).strip() in ("", "."):
        return None
    if p.is_file():
        p = p.parent
    return p


def _default_export_root(base_default: Optional[Path] = None) -> Path:
    """Pick an export root, preferring the last preview/import folder."""
    fallback = Path.home() / "musclequant" / "results"
    for cand in (
        CURRENT.get("last_preview_folder"),
        CURRENT.get("save_dir"),
        base_default,
        fallback,
    ):
        path = _clean_path_candidate(cand)
        if path is None:
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return path
    return fallback


def _safe_basename(name: Any) -> str:
    """Filesystem-safe stem for naming outputs."""
    try:
        stem = Path(str(name)).stem
    except Exception:
        stem = str(name or "")
    stem = stem.strip() or "sample"
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    return stem or "sample"

METRIC_OPTIONS = [
    ("area", "Area / size metrics"),
    ("ctcf", "Intensity & CTCF metrics"),
    ("eccentricity", "Eccentricity"),
    ("solidity", "Solidity"),
]

MODE_PRESETS = {
    "IHC": {
        "metrics": {"area": True, "ctcf": False, "eccentricity": False, "solidity": False},
    },
    "H&E": {
        "metrics": {"area": True, "ctcf": False, "eccentricity": False, "solidity": False},
    },
}

_metric_checks: Dict[str, QCheckBox] = {}
_metric_block = False

def metric_selection_widget() -> QWidget:
    panel = QWidget()
    global _metrics_panel_widget
    _metrics_panel_widget = panel
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.addWidget(QLabel("Quantification metrics"))
    defaults = _sanitize_metrics(CURRENT.get("metrics"))
    for key, label in METRIC_OPTIONS:
        cb = QCheckBox(label)
        cb.setChecked(defaults.get(key, True))
        cb.stateChanged.connect(_on_metric_checkbox_changed)
        _metric_checks[key] = cb
        layout.addWidget(cb)
    layout.addStretch(1)
    return panel

def _on_metric_checkbox_changed(*_):
    if _metric_block:
        return
    CURRENT["metrics"] = selected_metrics()

def selected_metrics() -> Dict[str, bool]:
    if not _metric_checks:
        metrics = _sanitize_metrics(CURRENT.get("metrics"))
        CURRENT["metrics"] = metrics
        return metrics
    metrics = {key: bool(cb.isChecked()) for key, cb in _metric_checks.items()}
    metrics = _sanitize_metrics(metrics)
    CURRENT["metrics"] = metrics
    return metrics

def apply_mode_presets(mode: str):
    presets = MODE_PRESETS.get(mode, MODE_PRESETS["IHC"])
    metrics = _sanitize_metrics(presets.get("metrics"))
    CURRENT["metrics"] = metrics
    global _metric_block
    _metric_block = True
    try:
        for key, cb in _metric_checks.items():
            target = metrics.get(key, True)
            if cb.isChecked() != target:
                cb.setChecked(target)
    finally:
        _metric_block = False

MODE_CHOICES = ["IHC", "H&E"]
_mode_combo = None
_mode_combo_block = False
_mode_viewer = None

def set_mode(mode: str, source: str = "auto", apply_presets_flag: bool = True):
    global _mode_combo_block
    norm = "H&E" if str(mode).strip().lower().startswith("h") else "IHC"
    CURRENT["mode"] = norm
    CURRENT["mode_source"] = source
    if apply_presets_flag:
        apply_mode_presets(norm)
    if _mode_combo is not None:
        if _mode_combo.currentText() != norm:
            _mode_combo_block = True
            try:
                _mode_combo.setCurrentText(norm)
            finally:
                _mode_combo_block = False
    _update_he_slider_visibility()

def make_mode_selector(viewer: "napari.Viewer") -> QWidget:
    from qtpy.QtWidgets import QHBoxLayout, QComboBox

    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.addWidget(QLabel("Image mode"))
    combo = QComboBox()
    combo.addItems(MODE_CHOICES)
    combo.setCurrentText(CURRENT.get("mode", "IHC"))

    def _on_change(text: str):
        global _mode_combo_block
        if _mode_combo_block:
            return
        v = _resolve_viewer(None, viewer)
        if v is None:
            return
        set_mode(text, source="manual", apply_presets_flag=True)
        last_folder = CURRENT.get("last_preview_folder")
        if last_folder:
            folder_path = Path(last_folder)
            try:
                folder_preview(v, folder=folder_path)
            except Exception as exc:
                show_warning(f"Failed to regenerate preview: {exc}")

    combo.currentTextChanged.connect(_on_change)
    layout.addWidget(combo)
    layout.addStretch(1)

    global _mode_combo, _mode_viewer
    _mode_combo = combo
    _mode_viewer = viewer
    return container


def make_mode_row(viewer: "napari.Viewer") -> QWidget:
    """Deprecated: retained for compatibility (no-op placeholder)."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    return row

def ensure_manual_layer(viewer: "napari.Viewer"):
    if "Manual_additions" in [L.name for L in viewer.layers]:
        return viewer.layers["Manual_additions"]
    if CURRENT["masks"] is None:
        _ensure_masks_from_viewer(viewer)
    if CURRENT["masks"] is None:
        mq_info(None, "MuscleQuant", "Run segmentation first.")
        return None
    add = np.zeros_like(CURRENT["masks"], dtype=np.uint16)
    layer = viewer.add_labels(add, name="Manual_additions", visible=True)
    layer.selected_label = int(CURRENT["masks"].max()) + 1
    layer.brush_size = 5
    mq_info(
        None,
        "Manual mode",
        "Paint new cells on 'Manual_additions'.\nTips:\n• Brush (2) to paint; label 0 erases\n• 'New label ID' for each new cell\n• 'Merge additions' when done"
    )
    return layer


def ensure_s1_manual_layer(viewer: "napari.Viewer"):
    """Create or return an editable copy of the current masks for S1-only tweaking."""
    base = CURRENT.get("masks")
    if base is None:
        base = _ensure_masks_from_viewer(viewer)
    if base is None:
        mq_info(None, "S1 manual mask", "Run segmentation first (no masks to copy).")
        return None
    if "S1_manual_mask" in [L.name for L in viewer.layers]:
        layer = viewer.layers["S1_manual_mask"]
    else:
        layer = viewer.add_labels(np.asarray(base, dtype=np.uint16), name="S1_manual_mask", visible=True)
        layer.contour = True
    layer.selected_label = int(np.max(layer.data)) + 1
    layer.brush_size = 5
    show_info(
        "S1 manual mask ready:\n"
        "• Paint/erase on 'S1_manual_mask' to tweak cells for S1 quant only.\n"
        "• Re-run S1 analysis to use these edits (base segmentation stays unchanged)."
    )
    return layer


def _seed_membrane_from_masks(masks: np.ndarray, ring_px: int) -> np.ndarray:
    ring_px = max(1, int(ring_px))
    selem = disk(ring_px)
    out = np.zeros_like(masks, dtype=np.uint16)
    labels = np.unique(masks)
    labels = labels[labels > 0]
    for lbl in labels:
        cell = masks == lbl
        if not np.any(cell):
            continue
        interior = erosion(cell, selem)
        if not np.any(interior):
            interior = cell
        ring = cell & (~interior)
        out[ring] = lbl
    return out


def ensure_s1_region_manual_layer(viewer: "napari.Viewer", region: str):
    """
    Create/return a manual region layer for S1 (membrane/synapse/extrasynaptic).
    Seeds from preview layers when available; otherwise empty (or ring for membrane).
    """
    region_key = region.lower().strip()
    name_map = {
        "membrane": "S1_manual_membrane",
        "synapse": "S1_manual_synapse",
        "extrasyn": "S1_manual_extrasyn",
        "extrasynaptic": "S1_manual_extrasyn",
    }
    target_name = name_map.get(region_key)
    if target_name is None:
        show_warning("Choose membrane, synapse, or extrasynaptic for S1 manual regions.")
        return None
    masks = CURRENT.get("masks")
    if masks is None:
        masks = _ensure_masks_from_viewer(viewer)
    if masks is None:
        mq_info(None, "S1 manual region", "Run segmentation/import first (no masks to copy).")
        return None
    seed = None
    preview_candidates = []
    if region_key == "membrane":
        preview_candidates = ["S1 membrane", "S1 membrane (preview)"]
    elif region_key == "synapse":
        preview_candidates = ["S1 synapse", "S1 synapse (preview)"]
    else:
        preview_candidates = ["S1 extrasyn", "S1 extrasyn (preview)"]

    for cand in preview_candidates:
        if cand in [L.name for L in viewer.layers]:
            try:
                arr = np.asarray(viewer.layers[cand].data)
                if arr.shape == masks.shape:
                    seed = arr.astype(np.uint16, copy=False)
                    break
            except Exception:
                continue

    if seed is None and region_key == "membrane":
        seed = _seed_membrane_from_masks(masks.astype(np.uint16, copy=False), CURRENT.get("s1_membrane_px", 4))
    if seed is None:
        seed = np.zeros_like(masks, dtype=np.uint16)

    if target_name in [L.name for L in viewer.layers]:
        layer = viewer.layers[target_name]
        layer.data = seed
    else:
        layer = viewer.add_labels(seed, name=target_name, visible=True)
        layer.contour = True
    layer.selected_label = int(np.max(layer.data)) + 1
    layer.brush_size = 5
    show_info(
        f"S1 manual {region_key} mask ready:\n"
        f"• Paint label IDs for each cell's {region_key} region on '{target_name}'.\n"
        "• Re-run S1 analysis to use these edits."
    )
    return layer


def sync_s1_region_masks(viewer: "napari.Viewer"):
    """
    Sync manual and preview S1 region masks.

    Priority: manual layer -> preview; if manual missing, preview -> manual.
    """
    if CURRENT.get("masks") is None and _ensure_masks_from_viewer(viewer) is None:
        mq_info(None, "S1 region mask", "Run/import a segmentation first.")
        return
    shape = np.asarray(CURRENT["masks"]).shape
    regions = [
        ("membrane", "S1_manual_membrane", "S1 membrane (preview)"),
        ("synapse", "S1_manual_synapse", "S1 synapse (preview)"),
        ("extrasynaptic", "S1_manual_extrasyn", "S1 extrasyn (preview)"),
    ]
    updated = []
    for _r, manual_name, preview_name in regions:
        manual_arr = _get_manual_region_array(viewer, manual_name, shape)
        preview_arr = _get_manual_region_array(viewer, preview_name, shape)
        src = None
        if manual_arr is not None:
            src = manual_arr
        elif preview_arr is not None:
            src = preview_arr
        if src is None:
            continue
        _add_or_replace_layer(viewer, preview_name, src.astype(np.uint16, copy=False), opacity=0.7)
        _add_or_replace_layer(viewer, manual_name, src.astype(np.uint16, copy=False), opacity=0.7)
        updated.append(manual_name)
    if updated:
        show_info(f"Updated S1 region mask(s): {', '.join(updated)}")

def ensure_selection_layer(viewer: "napari.Viewer"):
    if "Manual_selection" in [L.name for L in viewer.layers]:
        layer = viewer.layers["Manual_selection"]
        return layer
    layer = viewer.add_shapes(
        name="Manual_selection",
        shape_type="polygon",
        edge_color="yellow",
        face_color=[1.0, 1.0, 0.0, 0.15],
        blending="translucent_no_depth",
        visible=True,
    )
    layer.mode = "add_polygon"
    show_info(
        "Selection mode:\n"
        "• Use the Shapes layer tools (polygon, rectangle, etc.) to outline regions.\n"
        "• Use the built-in napari selection handles to move/resize.\n"
        "• Click 'Apply selection → manual' when ready."
    )
    return layer


def ensure_roi_layer(viewer: "napari.Viewer"):
    """Ensure a dedicated ROI layer exists for per-stain ROI statistics."""
    if "ROI_regions" in [L.name for L in viewer.layers]:
        return viewer.layers["ROI_regions"]
    layer = viewer.add_shapes(
        name="ROI_regions",
        shape_type="polygon",
        edge_color="magenta",
        face_color=[1.0, 0.0, 1.0, 0.15],
        blending="translucent_no_depth",
        visible=True,
    )
    layer.mode = "add_polygon"
    show_info("ROI mode: draw polygon(s) on 'ROI_regions', then click 'Compute ROI stats'.")
    return layer


def ensure_cell_query_layer(viewer: "napari.Viewer"):
    """Shapes layer used to select regions for cell ID lookup."""
    if "Cell_query" in [L.name for L in viewer.layers]:
        layer = viewer.layers["Cell_query"]
        _attach_cell_query_stats(viewer, layer)
        return layer
    layer = viewer.add_shapes(
        name="Cell_query",
        shape_type="polygon",
        edge_color="cyan",
        face_color=[0.0, 1.0, 1.0, 0.15],
        blending="translucent_no_depth",
        visible=True,
    )
    layer.mode = "add_polygon"
    _attach_cell_query_stats(viewer, layer)
    show_info("Cell ID query: draw polygon(s) on 'Cell_query', then click 'List cells in selection'.")
    return layer


def _attach_cell_query_stats(viewer: "napari.Viewer", layer) -> None:
    if layer is None:
        return
    if getattr(layer, "_mq_cell_query_stats", False):
        return
    try:
        layer.events.data.connect(lambda event=None: _schedule_cell_query_stats(viewer, layer))
    except Exception:
        return
    layer._mq_cell_query_stats = True


def _update_cell_query_stats(viewer: "napari.Viewer", layer) -> None:
    if viewer is None or layer is None:
        return
    _activate_viewer_state(viewer)
    masks = CURRENT.get("masks")
    if masks is None:
        _clear_cell_query_stats(viewer)
        return
    if not len(layer.data):
        _clear_cell_query_stats(viewer)
        return
    try:
        labels = layer.to_labels(masks.shape)
    except Exception:
        _clear_cell_query_stats(viewer)
        return
    sel_mask = labels > 0
    if not np.any(sel_mask):
        _clear_cell_query_stats(viewer)
        return
    sel_ids = np.unique(masks[sel_mask])
    sel_ids = sel_ids[sel_ids > 0]
    if sel_ids.size == 0:
        _clear_cell_query_stats(viewer)
        return
    areas = np.bincount(masks.ravel())
    avg_area = float(np.mean(areas[sel_ids]))
    text = f"avg area: {avg_area:.1f} px"
    if _update_selection_stats_overlay(viewer, text):
        _clear_selection_stats_points(viewer)
        return
    _update_selection_stats_points(viewer, sel_mask, text)


def _schedule_cell_query_stats(viewer: "napari.Viewer", layer) -> None:
    if layer is None:
        return
    if getattr(layer, "_mq_stats_pending", False):
        return
    layer._mq_stats_pending = True

    def _run():
        layer._mq_stats_pending = False
        _update_cell_query_stats(viewer, layer)

    QTimer.singleShot(0, _run)


def _clear_cell_query_stats(viewer: "napari.Viewer") -> None:
    if viewer is None:
        return
    _update_selection_stats_overlay(viewer, "")
    _clear_selection_stats_points(viewer)


class _OverlayFollower(QObject):
    def __init__(self, anchor_widget: QWidget, overlay: QWidget, margin: int = 12):
        super().__init__(anchor_widget)
        self._anchor = anchor_widget
        self._overlay = overlay
        self._margin = margin

    def eventFilter(self, obj, event):
        if obj is self._anchor and event.type() in (QEvent.Resize, QEvent.Show):
            self._reposition()
        return super().eventFilter(obj, event)

    def _reposition(self):
        if self._anchor is None or self._overlay is None:
            return
        rect = self._anchor.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        size = self._overlay.sizeHint()
        x = max(self._margin, rect.width() - size.width() - self._margin)
        y = self._margin
        self._overlay.move(x, y)


def _ensure_selection_stats_overlay(viewer: "napari.Viewer") -> Optional[QLabel]:
    if viewer is None:
        return None
    cached = _SELECTION_OVERLAYS.get(id(viewer))
    if cached is not None:
        ref, overlay, _follower = cached
        if ref() is viewer:
            return overlay
        _SELECTION_OVERLAYS.pop(id(viewer), None)
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if qt_viewer is None:
        return None
    canvas = getattr(qt_viewer, "canvas", None)
    anchor = getattr(canvas, "native", None)
    if anchor is None:
        return None

    label = QLabel(anchor)
    label.setObjectName("SelectionStatsOverlay")
    label.setText("")
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    label.setStyleSheet(
        "QLabel#SelectionStatsOverlay {"
        "background-color: rgba(0, 0, 0, 140);"
        "color: white;"
        "padding: 6px 8px;"
        "border-radius: 6px;"
        "font-size: 11px;"
        "}"
    )
    label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    label.adjustSize()
    label.hide()

    follower = _OverlayFollower(anchor, label)
    anchor.installEventFilter(follower)
    follower._reposition()
    _SELECTION_OVERLAYS[id(viewer)] = (weakref.ref(viewer), label, follower)
    return label


def _update_selection_stats_overlay(viewer: "napari.Viewer", text: str) -> bool:
    overlay = _ensure_selection_stats_overlay(viewer)
    if overlay is None:
        return False
    if text:
        overlay.setText(text)
        overlay.adjustSize()
        overlay.show()
    else:
        overlay.setText("")
        overlay.hide()
    cached = _SELECTION_OVERLAYS.get(id(viewer))
    if cached is not None:
        _ref, _overlay, follower = cached
        try:
            follower._reposition()
        except Exception:
            pass
    return True


_SELECTION_OVERLAYS: Dict[int, Tuple["weakref.ReferenceType", QLabel, _OverlayFollower]] = {}

_COMPARE_VIEWER: Optional["napari.Viewer"] = None
_COMPARE_TABS: Optional[QTabWidget] = None
_COMPARE_MAIN_QT = None
_COMPARE_COMP_QT = None
_COMPARE_LAYERS_DOCK: Optional[QDockWidget] = None
_CONTROLS_TABS: Optional[QTabWidget] = None
_CONTROLS_MAIN_PAGE: Optional[QWidget] = None
_CONTROLS_COMPARE_PAGE: Optional[QWidget] = None
_MAIN_DOCK = None
_MAIN_WINDOW = None
_DOCK_SNAPSHOT = None
_UI_MODE_STACK = None
_SINGLE_PANEL = None
_BATCH_PANEL = None
_SINGLE_BODY = None
_BATCH_BODY = None
_BATCH_FOLDERS: List[Path] = []
_BATCH_FOLDER_LABEL: Optional[QLabel] = None
_BATCH_ACTION_COMBO = None
_UI_MODE_COMBOS: List[Any] = []
_BATCH_RUN_BUTTON: Optional[QPushButton] = None
_BATCH_NAV_WIDGET: Optional[QWidget] = None
_BATCH_PROGRESS_LABEL: Optional[QLabel] = None
_BATCH_SYNAPTIC_SAMPLES: List[Dict[str, Any]] = []
_BATCH_SYNAPTIC_INDEX: int = -1
_BATCH_SYNAPTIC_SAVED: Set[int] = set()
_BATCH_USE_ALL_STRAINS: Optional[QCheckBox] = None
_BATCH_VERIFY_CHECK: Optional[QCheckBox] = None
_BATCH_VERIFY_DONE: bool = False
_BATCH_VERIFY_PENDING: bool = False
_BATCH_CANCELLED: bool = False
_BATCH_VERIFY_WIDGET: Optional[QWidget] = None
_BATCH_RUN_INDEX: int = 0
_BATCH_VERIFY_TIMEOUT_MS: int = 5000
_BATCH_VERIFY_PAUSED: bool = False
_BATCH_VERIFY_SELECTION: Optional[Set[Path]] = None
_BATCH_SYNAPTIC_ACTIVE: bool = False
_BATCH_SAMPLE_LABEL: Optional[QLabel] = None
_GAMMA_PATCHED: bool = False
_SINGLE_FOLDER_LABEL: Optional[QLabel] = None
_HE_SLIDER_WIDGETS: List[QWidget] = []
_FOLDER_LABEL_MAX = 16
_PANEL_MARGIN = 6
_PANEL_SPACING = 6
_PANEL_GRID_SPACING = 6


def _apply_panel_layout(layout: Optional[QLayout]) -> None:
    if layout is None:
        return
    try:
        layout.setContentsMargins(_PANEL_MARGIN, _PANEL_MARGIN, _PANEL_MARGIN, _PANEL_MARGIN)
    except Exception:
        pass

def _update_he_slider_visibility() -> None:
    visible = bool(str(CURRENT.get("mode", "IHC")).upper() == "H&E")
    for widget in list(_HE_SLIDER_WIDGETS):
        try:
            widget.setVisible(visible)
        except Exception:
            pass


def _set_batch_call_button_visible(visible: bool) -> None:
    try:
        if hasattr(batch_widget, "call_button"):
            try:
                batch_widget.call_button.setVisible(visible)
            except Exception:
                pass
            try:
                batch_widget.call_button.setEnabled(visible)
            except Exception:
                pass
            try:
                batch_widget.call_button.visible = visible
            except Exception:
                pass
    except Exception:
        pass


def _patch_gamma_slider_limits(min_val: float = 0.0, max_val: float = 5.0) -> None:
    global _GAMMA_PATCHED
    if _GAMMA_PATCHED:
        return
    try:
        from napari._qt.layer_controls.widgets.qt_gamma_slider import QtGammaSliderControl
    except Exception:
        return
    orig_init = getattr(QtGammaSliderControl, "__init__", None)
    if not orig_init:
        return

    def _patched_init(self, parent, layer):  # type: ignore[override]
        orig_init(self, parent, layer)
        try:
            self.gamma_slider.setMinimum(float(min_val))
            self.gamma_slider.setMaximum(float(max_val))
        except Exception:
            pass

    QtGammaSliderControl.__init__ = _patched_init  # type: ignore[assignment]
    _GAMMA_PATCHED = True


def _snapshot_dock_sizes(main_win):
    if main_win is None:
        return None
    docks = list(main_win.findChildren(QDockWidget))
    if not docks:
        return None
    sizes = []
    for dock in docks:
        try:
            sizes.append((dock, dock.width()))
        except Exception:
            sizes.append((dock, None))
    return sizes


def _restore_dock_sizes(main_win, snapshot):
    if main_win is None or not snapshot:
        return
    docks = []
    widths = []
    for dock, width in snapshot:
        if dock is None or width is None:
            continue
        docks.append(dock)
        widths.append(int(width))
    if docks and widths:
        try:
            main_win.resizeDocks(docks, widths, Qt.Horizontal)
        except Exception:
            pass


def _find_layers_dock(main_win: QDockWidget):
    if main_win is None:
        return None
    titles = {"Layers", "Layer list", "Layer List", "layer list", "Layer list (napari)"}
    for dock in main_win.findChildren(QDockWidget):
        try:
            if dock.windowTitle() in titles:
                return dock
        except Exception:
            continue
    return None


def _ensure_compare_layers_dock(main_viewer: "napari.Viewer", comp_viewer: "napari.Viewer") -> Optional[QDockWidget]:
    global _COMPARE_LAYERS_DOCK
    if main_viewer is None or comp_viewer is None:
        return None
    main_win = getattr(main_viewer.window, "_qt_window", None)
    if main_win is None:
        return None
    if _COMPARE_LAYERS_DOCK is not None:
        try:
            _COMPARE_LAYERS_DOCK.show()
        except Exception:
            pass
        return _COMPARE_LAYERS_DOCK

    comp_qt = getattr(comp_viewer.window, "_qt_viewer", None)
    if comp_qt is None:
        return None
    layer_list = getattr(comp_qt, "_layer_list", None)
    if layer_list is None:
        layer_list = getattr(comp_qt, "layers", None)
    if layer_list is None:
        return None
    layer_widget = getattr(layer_list, "native", None) or layer_list

    dock = QDockWidget("Compare Layers", main_win)
    dock.setObjectName("CompareLayersDock")
    dock.setWidget(layer_widget)
    main_win.addDockWidget(Qt.LeftDockWidgetArea, dock)
    main_layers = _find_layers_dock(main_win)
    if main_layers is not None:
        try:
            main_win.tabifyDockWidget(main_layers, dock)
        except Exception:
            pass
    _COMPARE_LAYERS_DOCK = dock
    return dock


def _ensure_compare_viewer(main_viewer: "napari.Viewer") -> Optional["napari.Viewer"]:
    global _COMPARE_VIEWER
    if main_viewer is None:
        return None
    if _COMPARE_VIEWER is None:
        try:
            _COMPARE_VIEWER = napari.Viewer(show=False)
        except TypeError:
            _COMPARE_VIEWER = napari.Viewer()
        try:
            _COMPARE_VIEWER.window._qt_window.hide()
        except Exception:
            pass
    return _COMPARE_VIEWER


def _ensure_compare_tabs(main_viewer: "napari.Viewer", comp_viewer: "napari.Viewer") -> Optional[QTabWidget]:
    global _COMPARE_TABS, _COMPARE_MAIN_QT, _COMPARE_COMP_QT
    if _COMPARE_TABS is not None:
        return _COMPARE_TABS
    main_qt = getattr(main_viewer.window, "_qt_viewer", None)
    comp_qt = getattr(comp_viewer.window, "_qt_viewer", None)
    main_win = getattr(main_viewer.window, "_qt_window", None)
    if main_qt is None or comp_qt is None or main_win is None:
        return None
    tabs = QTabWidget()
    tabs.setTabPosition(QTabWidget.South)
    tabs.setMovable(False)
    tabs.setTabsClosable(False)
    if hasattr(tabs, "setTabBarAutoHide"):
        tabs.setTabBarAutoHide(True)
    tabs.addTab(main_qt, "Main View")
    main_win.setCentralWidget(tabs)
    _COMPARE_TABS = tabs
    _COMPARE_MAIN_QT = main_qt
    _COMPARE_COMP_QT = comp_qt
    def _on_tab_change(idx: int):
        widget = tabs.widget(idx)
        if widget is comp_qt:
            _activate_viewer_state(comp_viewer)
            _sync_magicgui_viewer(comp_viewer)
        else:
            _activate_viewer_state(main_viewer)
            _sync_magicgui_viewer(main_viewer)
    tabs.currentChanged.connect(_on_tab_change)
    _activate_viewer_state(main_viewer)
    _sync_magicgui_viewer(main_viewer)
    return tabs


def toggle_compare_view(main_viewer: "napari.Viewer") -> None:
    global _COMPARE_TABS, _DOCK_SNAPSHOT
    if main_viewer is None:
        return
    comp_viewer = _ensure_compare_viewer(main_viewer)
    if comp_viewer is None:
        return
    tabs = _ensure_compare_tabs(main_viewer, comp_viewer)
    if tabs is None:
        return
    comp_qt = _COMPARE_COMP_QT
    if comp_qt is None:
        return
    if _MAIN_WINDOW is not None:
        try:
            _DOCK_SNAPSHOT = _snapshot_dock_sizes(_MAIN_WINDOW)
        except Exception:
            _DOCK_SNAPSHOT = None
    idx = tabs.indexOf(comp_qt)
    if idx == -1:
        tabs.addTab(comp_qt, "Compare View")
        tabs.setCurrentWidget(comp_qt)
        _ensure_compare_layers_dock(main_viewer, comp_viewer)
        if _CONTROLS_TABS is not None and _CONTROLS_COMPARE_PAGE is not None:
            if _CONTROLS_TABS.indexOf(_CONTROLS_COMPARE_PAGE) == -1:
                _CONTROLS_TABS.addTab(_CONTROLS_COMPARE_PAGE, "Compare Controls")
            _CONTROLS_TABS.setCurrentWidget(_CONTROLS_COMPARE_PAGE)
            _activate_viewer_state(comp_viewer)
            _sync_magicgui_viewer(comp_viewer)
        if _MAIN_WINDOW is not None:
            QTimer.singleShot(0, lambda: _restore_dock_sizes(_MAIN_WINDOW, _DOCK_SNAPSHOT))
        show_info("Compare view opened. Drag/drop images into the Compare View tab.")
    else:
        tabs.removeTab(idx)
        comp_qt.setParent(None)
        if _COMPARE_LAYERS_DOCK is not None:
            try:
                _COMPARE_LAYERS_DOCK.hide()
            except Exception:
                pass
        if _CONTROLS_TABS is not None and _CONTROLS_COMPARE_PAGE is not None:
            cidx = _CONTROLS_TABS.indexOf(_CONTROLS_COMPARE_PAGE)
            if cidx != -1:
                _CONTROLS_TABS.removeTab(cidx)
            if _CONTROLS_MAIN_PAGE is not None:
                _CONTROLS_TABS.setCurrentWidget(_CONTROLS_MAIN_PAGE)
            _activate_viewer_state(main_viewer)
            _sync_magicgui_viewer(main_viewer)
        if _MAIN_WINDOW is not None:
            QTimer.singleShot(0, lambda: _restore_dock_sizes(_MAIN_WINDOW, _DOCK_SNAPSHOT))


def _update_selection_stats_points(viewer: "napari.Viewer", sel_mask: np.ndarray, text: str) -> None:
    coords = np.argwhere(sel_mask)
    if coords.size == 0:
        _clear_selection_stats_points(viewer)
        return
    centroid = coords.mean(axis=0)
    pts = np.asarray([[float(centroid[0]), float(centroid[1])]], dtype=np.float32)
    text_cfg = {"string": [text], "color": "cyan", "size": 10}
    name = "Selection stats"
    if name in [L.name for L in viewer.layers]:
        pts_layer = viewer.layers[name]
        pts_layer.data = pts
        pts_layer.text = text_cfg
    else:
        viewer.add_points(
            pts,
            name=name,
            size=6,
            face_color="transparent",
            text=text_cfg,
        )


def _clear_selection_stats_points(viewer: "napari.Viewer") -> None:
    name = "Selection stats"
    if name in [L.name for L in viewer.layers]:
        pts_layer = viewer.layers[name]
        pts_layer.data = np.zeros((0, 2), dtype=np.float32)
        pts_layer.text = {"string": [], "color": "cyan", "size": 10}


def apply_selection_to_manual(viewer: "napari.Viewer"):
    sel_layer = ensure_selection_layer(viewer)
    if sel_layer is None:
        return
    if not len(sel_layer.data):
        mq_info(None, "Selection", "No shapes drawn yet. Add polygons/rectangles first.")
        return
    manual_layer = ensure_manual_layer(viewer)
    if manual_layer is None:
        return
    try:
        labels = sel_layer.to_labels(manual_layer.data.shape)
    except Exception as exc:
        mq_warning(None, "Selection", f"Could not rasterize shapes: {exc}")
        return
    mask = labels > 0
    if not np.any(mask):
        mq_info(None, "Selection", "Shapes produced an empty selection.")
        return
    if manual_layer.selected_label == 0:
        manual_layer.selected_label = next_label_id()
    label_val = int(manual_layer.selected_label)
    manual_layer.data = np.where(mask, label_val, manual_layer.data)
    sel_layer.data = []
    show_info(f"Selection applied into Manual_additions (label {label_val}).")


def _merge_shapes_into_masks(viewer: "napari.Viewer" = None) -> bool:
    if viewer is None or "Manual_selection" not in [L.name for L in viewer.layers]:
        return False
    if CURRENT["masks"] is None:
        _ensure_masks_from_viewer(viewer)
    if CURRENT["masks"] is None:
        mq_info(None, "Manual mode", "No masks available to edit.")
        return False
    sel_layer = viewer.layers["Manual_selection"]
    if not len(sel_layer.data):
        return False
    try:
        labels = sel_layer.to_labels(CURRENT["masks"].shape)
    except Exception as exc:
        mq_warning(None, "Selection", f"Could not rasterize shapes: {exc}")
        return False
    new_mask = CURRENT["masks"].astype(np.int32).copy()
    next_id = int(new_mask.max()) + 1
    added = 0
    for val in np.unique(labels):
        if val <= 0:
            continue
        new_mask[labels == val] = next_id
        next_id += 1
        added += 1
    if added == 0:
        return False
    CURRENT["masks"] = new_mask.astype(np.uint16)
    CURRENT["modified"] = True
    CURRENT["s1_layers_dirty"] = True
    sel_layer.data = []
    labels_name = "Fibers_mask_clean" if CURRENT["cleaned"] else "Fibers_mask"
    if labels_name in [L.name for L in viewer.layers]:
        viewer.layers[labels_name].data = CURRENT["masks"]
    show_info(f"Added {added} new label(s) from selections.")
    return True


def apply_label_changes(viewer: "napari.Viewer"):
    """
    Merge new additions (Manual_selection) and sync CURRENT masks to the latest Labels layer.
    Refresh overlay and cell IDs; no quantification or saving.
    """
    if CURRENT.get("masks") is None:
        _ensure_masks_from_viewer(viewer)
    if CURRENT.get("masks") is None:
        mq_info(None, "Manual mode", "No masks available. Run segmentation first.")
        return
    _merge_shapes_into_masks(viewer)

    labels_name = "Fibers_mask_clean" if CURRENT.get("cleaned") else "Fibers_mask"
    label_layer = None
    for layer in viewer.layers:
        if layer.name == labels_name:
            label_layer = layer
            break
    if label_layer is None:
        # Fallback: any labels layer containing Fibers_mask, else active labels layer
        for layer in viewer.layers:
            if layer.__class__.__name__.lower().startswith("labels") and "fibers_mask" in layer.name.lower():
                label_layer = layer
                break
    if label_layer is None:
        active = getattr(viewer.layers, "selection", None)
        if active and len(active) == 1:
            only = list(active)[0]
            if only.__class__.__name__.lower().startswith("labels"):
                label_layer = only
    new_mask = CURRENT["masks"]
    if label_layer is not None:
        try:
            new_mask = np.asarray(label_layer.data, dtype=np.uint16)
        except Exception:
            new_mask = CURRENT["masks"]
    CURRENT["masks"] = new_mask.astype(np.uint16, copy=False)
    CURRENT["s1_layers_dirty"] = True
    CURRENT["modified"] = True
    CURRENT["s1_layers_dirty"] = True
    if label_layer is not None:
        label_layer.data = CURRENT["masks"]

    if CURRENT.get("lam") is not None:
        ov = overlay_boundaries(CURRENT["lam"], CURRENT["masks"])
        overlay_layer = None
        base = CURRENT.get("base")
        candidates = ["Overlay"]
        if base:
            candidates.insert(0, f"{base} — Overlay")
        for nm in candidates:
            if nm in [L.name for L in viewer.layers]:
                overlay_layer = viewer.layers[nm]
                break
        if overlay_layer is None:
            for L in viewer.layers:
                if L.__class__.__name__.lower().startswith("image") and "overlay" in L.name.lower():
                    overlay_layer = L
                    break
        if overlay_layer is not None:
            overlay_layer.data = ov
            try:
                overlay_layer.visible = True
                overlay_layer.opacity = 0.33
                overlay_layer.blending = "translucent_no_depth"
            except Exception:
                pass
        else:
            overlay_name = candidates[0] if candidates else "Overlay"
            viewer.add_image(ov, name=overlay_name, blending="translucent_no_depth", visible=True, opacity=0.33)

    # Masks changed: drop prior S1 preview/final layers so S1 uses updated boundaries.
    _clear_s1_layers(viewer, include_final=True)
    CURRENT["s1_layers_dirty"] = False
    _add_cell_ids_layer(viewer, CURRENT["masks"])
    show_info("Label changes applied (manual edits + additions merged).")


def _sync_current_masks_from_labels(viewer: Optional["napari.Viewer"]):
    """
    Ensure CURRENT['masks'] reflects the latest labels layer (for downstream S1/quant).
    """
    if viewer is None:
        return
    if CURRENT.get("masks") is None:
        return
    current_mask = np.asarray(CURRENT["masks"])
    labels_name = "Fibers_mask_clean" if CURRENT.get("cleaned") else "Fibers_mask"
    label_layer = None
    for layer in viewer.layers:
        if layer.name == labels_name:
            label_layer = layer
            break
    if label_layer is None:
        for layer in viewer.layers:
            if layer.__class__.__name__.lower().startswith("labels") and "fibers_mask" in layer.name.lower():
                label_layer = layer
                break
    if label_layer is None:
        active = getattr(viewer.layers, "selection", None)
        if active and len(active) == 1:
            only = list(active)[0]
            if only.__class__.__name__.lower().startswith("labels"):
                label_layer = only
    if label_layer is None:
        return
    try:
        new_mask = np.asarray(label_layer.data, dtype=np.uint16)
    except Exception:
        return
    if new_mask.shape != np.asarray(CURRENT["masks"]).shape:
        show_warning(f"Labels layer shape {new_mask.shape} does not match CURRENT masks {np.asarray(CURRENT['masks']).shape}.")
        return
    new_mask = new_mask.astype(np.uint16, copy=False)
    if new_mask.shape != current_mask.shape:
        show_warning(f"Labels layer shape {new_mask.shape} does not match CURRENT masks {np.asarray(CURRENT['masks']).shape}.")
        return
    if np.array_equal(new_mask, current_mask):
        return
    CURRENT["masks"] = new_mask
    CURRENT["s1_layers_dirty"] = True


def _update_length_annotations(layer):
    """Annotate each path/line shape with its length in px and microns."""
    if layer is None or not hasattr(layer, "data"):
        return
    px_um = float(CURRENT.get("px_um", DEFAULT_PX_UM))
    texts = []
    for coords in layer.data:
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] < 2:
            texts.append("length: n/a")
            continue
        diffs = np.diff(coords, axis=0)
        length_px = float(np.sum(np.linalg.norm(diffs, axis=1)))
        length_um = length_px * px_um
        texts.append(f"{length_px:.1f}px / {length_um:.2f}um")
    if texts:
        layer.text = {"string": texts, "color": "orange", "size": 10}


def ensure_length_layer(viewer: "napari.Viewer"):
    """Create or return a shapes layer for drawing lines/paths to measure length."""
    created = False
    if "Length_measure" in [L.name for L in viewer.layers]:
        layer = viewer.layers["Length_measure"]
    else:
        layer = viewer.add_shapes(
            name="Length_measure",
            shape_type="path",
            edge_color="orange",
            face_color=[1.0, 0.5, 0.0, 0.1],
            blending="translucent_no_depth",
            visible=True,
        )
        layer.mode = "add_path"
        layer.tooltip = "Draw lines/paths to measure length"
        try:
            layer.events.data.connect(lambda event=None: _update_length_annotations(layer))
        except Exception:
            pass
        created = True
    if created:
        show_info("Length tool: draw a line/path on 'Length_measure', then click 'Update lengths'.")
    return layer


def update_length_measurements(viewer: "napari.Viewer"):
    """Recalculate line/path lengths for the Length_measure layer."""
    layer = ensure_length_layer(viewer)
    if layer is None:
        return
    _update_length_annotations(layer)
    n = len(layer.data)
    if n == 0:
        show_info("No lines/paths drawn yet.")
        return
    last_txt = ""
    if isinstance(layer.text, dict):
        strings = layer.text.get("string", [])
        if strings:
            last_txt = strings[-1]
    show_info(f"Updated lengths for {n} line(s). {last_txt}")


def load_mask_from_file(viewer: "napari.Viewer"):
    """
    Load an external mask file and set it as the current segmentation.
    """
    _activate_viewer_state(viewer)
    _reset_s1_state(viewer)
    path, _ = QFileDialog.getOpenFileName(None, "Select mask file", str(Path.home()), "TIFF Files (*.tif *.tiff);;All Files (*)")
    if not path:
        return
    try:
        mask = imread(path)
    except Exception as exc:
        mq_warning(None, "Load mask", f"Could not read mask: {exc}")
        return
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        mq_warning(None, "Load mask", f"Mask must be 2D. Got shape {mask.shape}.")
        return
    if mask.dtype != np.uint16:
        mask = mask.astype(np.uint16, copy=False)
    CURRENT["masks"] = mask
    CURRENT["cleaned"] = True
    CURRENT["modified"] = False
    labels_name = "Fibers_mask_clean"
    if labels_name in [L.name for L in viewer.layers]:
        viewer.layers[labels_name].data = mask
    else:
        viewer.add_labels(mask, name=labels_name, visible=True).contour = True
    if CURRENT.get("lam") is not None:
        ov = overlay_boundaries(CURRENT["lam"], mask)
        if "Overlay" in [L.name for L in viewer.layers]:
            viewer.layers["Overlay"].data = ov
        else:
            viewer.add_image(ov, name="Overlay", blending="translucent_no_depth", visible=True, opacity=0.33)
    _reset_click_keep_state(viewer)
    _clear_s1_layers(viewer, include_final=True)
    CURRENT["s1_layers_dirty"] = False
    _add_cell_ids_layer(viewer, mask)
    show_info(f"Loaded mask: {Path(path).name}")

def _import_saved_run_from_folder(viewer: "napari.Viewer", folder: Path) -> bool:
    _activate_viewer_state(viewer)
    _reset_s1_state(viewer)
    _clear_batch_verify_overlay(viewer)
    folder = Path(folder)
    CURRENT["last_preview_folder"] = str(folder)

    meta_path = folder / "metadata.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as exc:
            show_warning(f"metadata.json unreadable: {exc}")

    # Find mask
    mask_path: Optional[Path] = None
    meta_mask = meta.get("mask_file")
    if meta_mask:
        candidate = folder / meta_mask
        if candidate.exists():
            mask_path = candidate
    if mask_path is None:
        candidates = sorted([
            p for p in folder.glob("*.tif*")
            if re.search(r"mask", p.stem, re.IGNORECASE)
        ])
        if candidates:
            mask_path = candidates[0]
    if mask_path is None:
        mq_warning(None, "Import saved run", "No mask file found in the selected folder.")
        return False

    try:
        mask = imread(mask_path)
    except Exception as exc:
        mq_warning(None, "Import saved run", f"Could not read mask: {exc}")
        return False
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        mq_warning(None, "Import saved run", f"Mask must be 2D. Got shape {mask.shape}.")
        return False
    if mask.dtype != np.uint16:
        mask = mask.astype(np.uint16, copy=False)

    # Collect raw images in folder (excluding mask/overlay)
    all_tifs = list(folder.glob("*.tif*"))
    raw_paths: List[Path] = []
    s1_syn = None
    s1_mem = None
    s1_extra = None
    for p in all_tifs:
        stem_lower = p.stem.lower()
        if p == mask_path or "overlay" in stem_lower:
            continue
        if "s1_syn" in stem_lower or "synapse" in stem_lower:
            s1_syn = s1_syn or p
            continue
        if "s1_mem" in stem_lower or "membrane" in stem_lower:
            s1_mem = s1_mem or p
            continue
        if "s1_extra" in stem_lower or "extrasyn" in stem_lower:
            s1_extra = s1_extra or p
            continue
        if "mask" in stem_lower:
            continue
        raw_paths.append(p)

    lam_files = [p for p in raw_paths if re.search(r"(background)", p.stem, re.IGNORECASE)]
    if not lam_files and raw_paths:
        lam_files = [raw_paths[0]]

    # Let user pick background via dropdown (optional)
    if raw_paths:
        options = [p.name for p in raw_paths]
        try:
            default_idx = min(len(options) - 1, raw_paths.index(lam_files[0])) if lam_files else 0
        except Exception:
            default_idx = 0
        choice, ok = QInputDialog.getItem(
            None,
            "Background stain",
            "Select background image:",
            options,
            current=default_idx,
            editable=False,
        )
        if ok and choice:
            try:
                chosen_idx = options.index(choice)
                lam_files = [raw_paths[chosen_idx]]
            except Exception:
                pass

    protein_paths = [p for p in raw_paths if p not in lam_files]

    lam_gray = None
    lam_raw = None
    lam_used_list: List[str] = []
    lam_display_name = None
    if lam_files:
        lam_grays: List[np.ndarray] = []
        lam_raws: List[np.ndarray] = []
        for lp in lam_files:
            lg, lu = load_rgb_plane(lp, None)
            raw = load_raw_gray(lp, lu if lu in PLANE_TO_IDX else None)
            if lg.shape != mask.shape:
                lg = resize(lg, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
            if raw.shape != mask.shape:
                raw = resize(raw, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
            lam_grays.append(np.asarray(lg, dtype=np.float32))
            lam_raws.append(np.asarray(raw, dtype=np.float32))
            lam_used_list.append(lu)
        lam_gray = lam_grays[0] if len(lam_grays) == 1 else np.maximum.reduce(lam_grays)
        lam_raw = lam_raws[0] if len(lam_raws) == 1 else np.maximum.reduce(lam_raws)
        lam_display_name = infer_stain_from_filename(lam_files[0])

    px_um = float(meta.get("pixel_size_um") or meta.get("px_um") or DEFAULT_PX_UM)
    if not meta.get("pixel_size_um") and lam_files:
        for lp in lam_files:
            guess = _infer_pixel_size_um_from_tiff(lp)
            if isinstance(guess, float) and guess > 0:
                px_um = float(guess)
                break

    preview_planes: Dict[str, str] = {}
    protein_channels: List[ProteinChannel] = []
    protein_layers: List[Tuple[str, np.ndarray, str]] = []
    s1_layers: List[Tuple[str, np.ndarray]] = []
    for idx, pp in enumerate(protein_paths):
        try:
            disp, used_plane = load_rgb_plane(pp, None)
            raw_gray = load_raw_gray(pp, used_plane if used_plane in PLANE_TO_IDX else None)
        except Exception as exc:
            show_warning(f"Skipped {pp.name}: {exc}")
            continue
        if raw_gray.shape != mask.shape:
            raw_gray = resize(raw_gray, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
        if disp.shape != mask.shape:
            disp = resize(disp, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
        name = infer_stain_from_filename(pp)
        key = str(pp)
        preview_planes[key] = used_plane if used_plane in PLANE_TO_IDX else "Gray"
        ch = ProteinChannel(
            name=name,
            plane=used_plane,
            path=pp,
            data=np.asarray(raw_gray, dtype=np.float32),
            preview_id=key,
            origin="import",
        )
        protein_channels.append(ch)
        protein_layers.append((name, disp, used_plane))
        if s1_syn and pp == s1_syn:
            s1_layers.append(("S1 synapse", disp))

    try:
        viewer.layers.clear()
    except Exception:
        pass

    base = re.sub(r"(_mask(_clean)?|_labels?)$", "", mask_path.stem, flags=re.IGNORECASE)
    base = base or mask_path.stem
    if lam_gray is not None:
        lam_cmap = _stain_colormap(lam_display_name or "Background", "gray", None)
        viewer.add_image(
            lam_gray,
            name=f"{lam_display_name or 'Background'}",
            colormap=lam_cmap,
            blending="additive",
            visible=True,
        )
    # Add S1 region layers if supplied
    if s1_syn and s1_syn.exists():
        try:
            syn_arr, _ = load_rgb_plane(s1_syn, None)
            if syn_arr.shape != mask.shape:
                syn_arr = resize(syn_arr, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
            viewer.add_labels((syn_arr > 0).astype(np.uint16), name="S1 synapse (preview)", opacity=0.7)
        except Exception:
            pass
    if s1_mem and s1_mem.exists():
        try:
            mem_arr, _ = load_rgb_plane(s1_mem, None)
            if mem_arr.shape != mask.shape:
                mem_arr = resize(mem_arr, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
            viewer.add_labels((mem_arr > 0).astype(np.uint16), name="S1 membrane (preview)", opacity=0.5)
        except Exception:
            pass
    if s1_extra and s1_extra.exists():
        try:
            extra_arr, _ = load_rgb_plane(s1_extra, None)
            if extra_arr.shape != mask.shape:
                extra_arr = resize(extra_arr, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
            viewer.add_labels((extra_arr > 0).astype(np.uint16), name="S1 extrasyn (preview)", opacity=0.7)
        except Exception:
            pass
    color_idx = 0
    for name, disp, used_plane in protein_layers:
        default_cmap = COLORMAP_CYCLE[color_idx % len(COLORMAP_CYCLE)]
        cmap = _stain_colormap(name, default_cmap, used_plane)
        color_idx += 1
        layer = viewer.add_image(
            disp,
            name=name,
            colormap=cmap,
            blending="additive",
            visible=False,
        )
        layer.opacity = 0.6

    labels_name = "Fibers_mask_imported"
    viewer.add_labels(mask, name=labels_name, visible=True).contour = True
    if lam_gray is not None:
        ov = overlay_boundaries(lam_gray, mask)
        viewer.add_image(ov, name=f"{base} — Overlay", blending="translucent_no_depth", visible=True, opacity=0.33)
    _add_cell_ids_layer(viewer, mask)

    mode_from_meta = meta.get("mode")
    if mode_from_meta:
        set_mode(str(mode_from_meta), source="auto", apply_presets_flag=False)
    else:
        try:
            stain_tokens = _parse_stain_list(batch_widget.stain_names.value)
        except Exception:
            stain_tokens = []
        auto_mode = infer_mode_from_files(raw_paths, stain_tokens)
        set_mode(auto_mode, source="auto", apply_presets_flag=False)

    CURRENT["base"] = base
    CURRENT["lam"] = lam_gray
    CURRENT["lam_raw"] = lam_raw
    CURRENT["lam_paths"] = lam_files
    CURRENT["lam_used_list"] = lam_used_list
    CURRENT["masks"] = mask
    CURRENT["proteins"] = protein_channels
    CURRENT["preview_planes"] = preview_planes
    CURRENT["px_um"] = float(px_um)
    CURRENT["save_dir"] = mask_path.parent
    CURRENT["cleaned"] = True
    CURRENT["bg_mode"] = normalize_bg_mode(meta.get("background_mode", CURRENT.get("bg_mode", "center")))
    CURRENT["bg_percentile"] = float(meta.get("background_percentile", CURRENT.get("bg_percentile", 5.0)))
    CURRENT["local_ring_px"] = int(meta.get("local_ring_px", CURRENT.get("local_ring_px", 5)))
    CURRENT["rolling_ball_radius_px"] = int(meta.get("rolling_ball_radius_px", CURRENT.get("rolling_ball_radius_px", 50)))
    CURRENT["add_texture"] = bool(meta.get("add_texture", CURRENT.get("add_texture", True)))
    CURRENT["add_spatial"] = bool(meta.get("add_spatial", CURRENT.get("add_spatial", True)))
    CURRENT["stain_filter"] = meta.get("stain_filter", CURRENT.get("stain_filter", []))
    CURRENT["metrics"] = _sanitize_metrics(meta.get("metrics", CURRENT.get("metrics")))
    CURRENT["drop_edge_touching"] = bool(meta.get("drop_edge_touching", CURRENT.get("drop_edge_touching", True)))
    CURRENT["drop_edge_buffer_px"] = int(meta.get("drop_edge_buffer_px", CURRENT.get("drop_edge_buffer_px", 0)))
    CURRENT["export_all_images"] = bool(meta.get("export_all_images", CURRENT.get("export_all_images", False)))
    CURRENT["mode_source"] = "import"

    global SEGMENTED_CACHE, SEGMENT_EXPORT_ROOT
    SEGMENTED_CACHE.clear()
    SEGMENT_EXPORT_ROOT = meta.get("export_root")
    sample = {
        "base": base,
        "base_out": base,
        "export_dir": meta.get("export_dir"),
        "export_root": SEGMENT_EXPORT_ROOT,
        "lam_path": lam_files[0] if lam_files else None,
        "lam_paths": lam_files,
        "lam": lam_gray,
        "lam_raw": lam_raw,
        "lam_used": lam_used_list[0] if lam_used_list else "Gray",
        "masks": mask.astype(np.uint16),
        "cleaned": True,
        "proteins": protein_channels,
        "px_um": float(px_um),
        "preview_planes": preview_planes,
        "suffix": "_import",
        "metadata": meta,
        "suggested_folder": base,
        "drop_edge_touching": bool(meta.get("drop_edge_touching", True)),
        "drop_edge_buffer_px": int(meta.get("drop_edge_buffer_px", 0)),
        "metrics": CURRENT.get("metrics"),
        "mode": CURRENT.get("mode", "IHC"),
        "export_all_images": bool(meta.get("export_all_images", False)),
        "modified": bool(meta.get("modified", False)),
    }
    SEGMENTED_CACHE.append(sample)

    info_parts = [
        f"Mask: {mask_path.name}",
        f"Background: {len(lam_files)} file(s)",
        f"Proteins: {len(protein_channels)} channel(s)",
        f"px_um: {px_um:.4f}",
    ]
    if meta:
        info_parts.append("metadata.json loaded")
    show_info("Imported saved run:\n" + "\n".join(info_parts))
    return True


def import_saved_run(viewer: "napari.Viewer"):
    """
    Load a previously exported segmentation + raw images so it can be re-used for quant/S1.
    Select the folder containing metadata/mask/raw files.
    """
    folder = QFileDialog.getExistingDirectory(None, "Select folder with mask/metadata/raw images", str(Path.home()))
    if not folder:
        return
    _import_saved_run_from_folder(viewer, Path(folder))


def _load_saved_run_for_batch(folder: Path, bg_keys: List[str], stain_names: str, pixel_size_um: float) -> Optional[Dict[str, Any]]:
    folder = Path(folder)
    meta_path = folder / "metadata.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as exc:
            show_warning(f"{folder.name}: metadata.json unreadable ({exc})")

    # Find mask
    mask_path: Optional[Path] = None
    meta_mask = meta.get("mask_file")
    if meta_mask:
        candidate = folder / meta_mask
        if candidate.exists():
            mask_path = candidate
    if mask_path is None:
        candidates = sorted([
            p for p in folder.glob("*.tif*")
            if re.search(r"mask", p.stem, re.IGNORECASE)
        ])
        if candidates:
            mask_path = candidates[0]
    if mask_path is None:
        show_warning(f"{folder.name}: no mask file found.")
        return None

    try:
        mask = imread(mask_path)
    except Exception as exc:
        show_warning(f"{folder.name}: could not read mask ({exc})")
        return None
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        show_warning(f"{folder.name}: mask must be 2D (got {mask.shape}).")
        return None
    if mask.dtype != np.uint16:
        mask = mask.astype(np.uint16, copy=False)

    # Collect raw images in folder (excluding mask/overlay)
    all_tifs = list(folder.glob("*.tif*"))
    raw_paths: List[Path] = []
    s1_syn = None
    s1_mem = None
    s1_extra = None
    for p in all_tifs:
        stem_lower = p.stem.lower()
        if p == mask_path or "overlay" in stem_lower:
            continue
        if "s1_syn" in stem_lower or "synapse" in stem_lower:
            s1_syn = s1_syn or p
            continue
        if "s1_mem" in stem_lower or "membrane" in stem_lower:
            s1_mem = s1_mem or p
            continue
        if "s1_extra" in stem_lower or "extrasyn" in stem_lower:
            s1_extra = s1_extra or p
            continue
        if "mask" in stem_lower:
            continue
        raw_paths.append(p)

    lam_files: List[Path] = []
    for key in bg_keys:
        matches = [p for p in raw_paths if key in p.stem.lower()]
        if matches:
            lam_files = matches
            break
    if not lam_files and raw_paths:
        lam_files = [raw_paths[0]]
    protein_paths = [p for p in raw_paths if p not in lam_files]

    lam_gray = None
    lam_raw = None
    lam_used_list: List[str] = []
    if lam_files:
        lam_grays: List[np.ndarray] = []
        lam_raws: List[np.ndarray] = []
        for lp in lam_files:
            lg, lu = load_rgb_plane(lp, None)
            raw = load_raw_gray(lp, lu if lu in PLANE_TO_IDX else None)
            if lg.shape != mask.shape:
                lg = resize(lg, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
            if raw.shape != mask.shape:
                raw = resize(raw, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
            lam_grays.append(np.asarray(lg, dtype=np.float32))
            lam_raws.append(np.asarray(raw, dtype=np.float32))
            lam_used_list.append(lu)
        lam_gray = lam_grays[0] if len(lam_grays) == 1 else np.maximum.reduce(lam_grays)
        lam_raw = lam_raws[0] if len(lam_raws) == 1 else np.maximum.reduce(lam_raws)

    px_um = float(meta.get("pixel_size_um") or meta.get("px_um") or pixel_size_um or DEFAULT_PX_UM)
    if not meta.get("pixel_size_um") and lam_files:
        for lp in lam_files:
            guess = _infer_pixel_size_um_from_tiff(lp)
            if isinstance(guess, float) and guess > 0:
                px_um = float(guess)
                break

    if stain_names:
        _stain_list = _parse_stain_list(stain_names)
    else:
        _stain_list = [infer_stain_from_filename(p) for p in protein_paths]
    proteins_loaded = select_protein_previews(protein_paths, lam_files[0] if lam_files else None, _stain_list) if protein_paths else []
    if not proteins_loaded and lam_gray is not None:
        fallback_name = lam_files[0].name if lam_files else "Background"
        proteins_loaded = [(fallback_name, lam_used_list[0] if lam_used_list else "Gray", lam_gray, lam_files[0] if lam_files else None)]

    preview_planes: Dict[str, str] = {}
    protein_channels: List[ProteinChannel] = []
    for name, plane_code, _disp, raw_path in proteins_loaded:
        key = str(raw_path) if raw_path is not None else f"{folder}/{name}"
        preview_planes[key] = plane_code if plane_code in PLANE_TO_IDX else "Gray"
    protein_channels, preview_map = _build_protein_channels(
        folder.name,
        proteins_loaded,
        preview_planes,
        None,
    )

    cleaned = bool(re.search(r"clean", mask_path.stem, re.IGNORECASE)) or bool(meta.get("cleaned", False))
    suffix = "_clean" if cleaned else ""
    sample = {
        "base": folder.name,
        "base_out": folder.name,
        "export_dir": None,
        "export_root": folder,
        "source_folder": str(folder),
        "files": list(raw_paths),
        "lam_files": list(lam_files),
        "lam_path": lam_files[0] if lam_files else None,
        "lam_paths": lam_files,
        "mask_path": mask_path,
        "lam": lam_gray,
        "lam_raw": lam_raw,
        "lam_used": ",".join(lam_used_list) if lam_used_list else "Gray",
        "masks": mask.astype(np.uint16),
        "cleaned": cleaned,
        "proteins": protein_channels,
        "px_um": float(px_um),
        "preview_planes": preview_map,
        "suffix": suffix,
        "metadata": meta,
        "suggested_folder": folder.name,
        "drop_edge_touching": bool(meta.get("drop_edge_touching", True)),
        "drop_edge_buffer_px": int(meta.get("drop_edge_buffer_px", 0)),
        "metrics": CURRENT.get("metrics"),
        "mode": meta.get("mode", CURRENT.get("mode", "IHC")),
        "export_all_images": bool(meta.get("export_all_images", False)),
        "modified": bool(meta.get("modified", False)),
    }
    return sample

def next_label_id():
    m = CURRENT["masks"]
    if m is None:
        return 1
    return int(np.max(m)) + 1

def merge_manual_additions(viewer: "napari.Viewer"):
    if CURRENT["masks"] is None:
        _ensure_masks_from_viewer(viewer)
    if CURRENT["masks"] is None or "Manual_additions" not in [L.name for L in viewer.layers]:
        mq_info(None, "MuscleQuant", "Nothing to merge (segment first, paint on 'Manual_additions').")
        return False
    masks = CURRENT["masks"].astype(np.int32).copy()
    add = np.asarray(viewer.layers["Manual_additions"].data).astype(np.int32)

    # keep original model labels if overlap
    add[masks > 0] = 0

    # connected components for manual strokes
    cc = cc_label(add > 0, connectivity=1)
    n_new = int(cc.max())
    if n_new == 0:
        mq_info(None, "MuscleQuant", "No new components to merge.")
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

    if CURRENT.get("lam") is not None:
        ov = overlay_boundaries(CURRENT["lam"], CURRENT["masks"])
        overlay_layer = None
        base = CURRENT.get("base")
        candidates = ["Overlay"]
        if base:
            candidates.insert(0, f"{base} — Overlay")
        for nm in candidates:
            if nm in [L.name for L in viewer.layers]:
                overlay_layer = viewer.layers[nm]
                break
        if overlay_layer is None:
            for L in viewer.layers:
                if L.__class__.__name__.lower().startswith("image") and "overlay" in L.name.lower():
                    overlay_layer = L
                    break
        if overlay_layer is not None:
            overlay_layer.data = ov
            try:
                overlay_layer.visible = True
                overlay_layer.opacity = 0.33
                overlay_layer.blending = "translucent_no_depth"
            except Exception:
                pass
        else:
            overlay_name = candidates[0] if candidates else "Overlay"
            viewer.add_image(ov, name=overlay_name, blending="translucent_no_depth", visible=True, opacity=0.33)

    mq_info(None, "MuscleQuant", f"Merged {n_new} new cells. Total labels: {int(CURRENT['masks'].max())}.")
    return True


def compute_roi_stats(viewer: "napari.Viewer"):
    """
    Compute area and mean fluorescence per stain for user-drawn ROI polygons.
    Uses all image layers currently in the viewer (including background).
    """
    roi_layer = ensure_roi_layer(viewer)
    if roi_layer is None:
        return

    image_layers = [L for L in viewer.layers if L.__class__.__name__.lower().startswith("image")]
    if not image_layers:
        mq_info(None, "ROI", "No image layers found. Load images before computing ROI stats.")
        return
    base_shape = np.asarray(image_layers[0].data).shape

    try:
        roi_labels = roi_layer.to_labels(base_shape)
    except Exception as exc:
        mq_warning(None, "ROI", f"Could not rasterize ROI shapes: {exc}")
        return

    n_roi = int(np.max(roi_labels))
    if n_roi <= 0:
        mq_info(None, "ROI", "ROI polygons produced an empty mask.")
        return

    px_um = float(CURRENT.get("px_um", DEFAULT_PX_UM))

    points: List[Tuple[float, float]] = []
    texts: List[str] = []

    def _short_label(raw: str) -> str:
        stem = Path(raw).stem
        stain_guess = infer_stain_from_filename(Path(raw))
        if stain_guess and stain_guess.lower() != "unknown":
            return stain_guess
        if "—" in raw:
            tail = raw.split("—")[-1].strip()
            if tail:
                stem = tail
        if "[" in stem:
            stem = stem.split("[", 1)[0].strip()
        tokens = re.split(r"[ _]+", stem)
        if tokens:
            candidate = tokens[-1] or tokens[0]
        else:
            candidate = stem
        candidate = candidate.strip()
        return candidate if candidate else stem

    for rid in range(1, n_roi + 1):
        roi_mask = (roi_labels == rid)
        area_px = int(np.count_nonzero(roi_mask))
        area_um2 = area_px * (px_um ** 2)
        coords = np.argwhere(roi_mask)
        centroid = coords.mean(axis=0) if coords.size else np.array([0.0, 0.0])

        summaries = []
        for layer in image_layers:
            arr = np.asarray(layer.data, dtype=np.float32)
            if arr.shape != roi_mask.shape:
                try:
                    arr = resize(arr, roi_mask.shape, order=1, preserve_range=True, anti_aliasing=True)
                except Exception:
                    continue
            mean_val = float(np.mean(arr[roi_mask])) if area_px else 0.0
            lname = _short_label(str(layer.name))
            summaries.append(f"{lname}: {mean_val:.4g}")

        points.append((float(centroid[0]), float(centroid[1])))
        text_lines = [f"ROI{rid} area_px={area_px} area_um2={area_um2:.2f}"]
        if summaries:
            text_lines.extend(summaries)
        texts.append("\n".join(text_lines))

    pts_data = np.asarray(points, dtype=np.float32)
    text_cfg = {"string": texts, "color": "magenta", "size": 10}
    if "ROI stats" in [L.name for L in viewer.layers]:
        pts = viewer.layers["ROI stats"]
        pts.data = pts_data
        pts.text = text_cfg
    else:
        viewer.add_points(
            pts_data,
            name="ROI stats",
            size=8,
            face_color="transparent",
            text=text_cfg,
        )
    show_info("ROI statistics computed and shown in the viewer.")


def list_cells_in_selection(viewer: "napari.Viewer"):
    """
    Report cell label IDs that fall inside user-drawn polygons (Cell_query layer).
    Requires segmentation to be loaded (CURRENT['masks']).
    """
    if CURRENT.get("masks") is None:
        mq_info(None, "Cells in selection", "Run segmentation first (no labels available).")
        return
    query_layer = ensure_cell_query_layer(viewer)
    if query_layer is None:
        return
    if not len(query_layer.data):
        mq_info(None, "Cells in selection", "Draw one or more polygons on 'Cell_query' first.")
        return
    try:
        sel_labels = query_layer.to_labels(CURRENT["masks"].shape)
    except Exception as exc:
        mq_warning(None, "Cells in selection", f"Could not rasterize shapes: {exc}")
        return
    sel_mask = sel_labels > 0
    if not np.any(sel_mask):
        mq_info(None, "Cells in selection", "Selection polygons are empty.")
        return
    intersect = CURRENT["masks"][sel_mask]
    ids = np.unique(intersect)
    ids = ids[ids > 0]
    if ids.size == 0:
        mq_info(None, "Cells in selection", "No labeled cells intersect the selection.")
        return
    ids_list = sorted(int(i) for i in ids.tolist())
    msg = f"Cells in selection ({len(ids_list)}):\n" + ", ".join(map(str, ids_list))
    show_info(msg)
    return ids_list


def highlight_stained_regions(viewer: "napari.Viewer"):
    """
    For each labeled cell, threshold the background stain to highlight stained area per cell.
    Produces/updates a 'Stained regions' labels layer.
    """
    masks = CURRENT.get("masks")
    lam = CURRENT.get("lam")
    if masks is None or lam is None:
        mq_info(None, "Stained regions", "Run segmentation first (no masks/background).")
        return
    masks = np.asarray(masks)
    lam_arr = np.asarray(lam, dtype=np.float32)
    # Smooth to reduce noise before per-cell thresholding
    lam_arr = gaussian(lam_arr, sigma=1.0, preserve_range=True)
    if lam_arr.shape != masks.shape:
        try:
            lam_arr = resize(lam_arr, masks.shape, order=1, preserve_range=True, anti_aliasing=True)
        except Exception:
            mq_warning(None, "Stained regions", "Could not resize background to mask shape.")
            return
    highlight = np.zeros_like(masks, dtype=np.uint16)
    labels = np.unique(masks)
    labels = labels[labels > 0]
    for lbl in labels:
        cell_mask = (masks == lbl)
        vals = lam_arr[cell_mask]
        if vals.size == 0:
            continue
        try:
            thr = float(threshold_otsu(vals))
        except Exception:
            thr = float(np.median(vals))
        stained = cell_mask & (lam_arr >= thr)
        # drop tiny specks (1% of cell area or minimum 5 px)
        min_keep = max(5, int(round(vals.size * 0.01)))
        stained = remove_small_objects(stained, min_size=min_keep)
        # bridge gaps and fill holes inside the cell to avoid fragmented rings
        closing_radius = max(1, int(round(np.sqrt(vals.size) / 25)))
        stained = binary_closing(stained, footprint=disk(closing_radius))
        stained = binary_fill_holes(stained & cell_mask) & cell_mask
        if not np.any(stained):
            stained = cell_mask & (lam_arr >= thr)  # fallback to raw threshold if cleaning removed everything
        highlight[stained] = lbl
    if "Stained regions" in [L.name for L in viewer.layers]:
        layer = viewer.layers["Stained regions"]
        layer.data = highlight
        layer.contour = True
    else:
        viewer.add_labels(highlight, name="Stained regions", visible=True).contour = True
    show_info("Stained regions highlighted (per-cell threshold on background stain).")

def _get_synapse_channel_and_data(
    stain_name: str,
    proteins: List[ProteinChannel],
    preview_map,
    masks: np.ndarray,
    quant_use_raw: bool = False,
):
    target_lower = stain_name.lower()
    syn_channel = None
    for ch in proteins:
        if target_lower in ch.name.lower():
            syn_channel = ch
            break
    if syn_channel is None:
        return None, None
    syn_raw = _load_channel_raw(syn_channel, preview_map)
    if syn_raw.shape != masks.shape:
        try:
            syn_raw = resize(syn_raw, masks.shape, order=1, preserve_range=True, anti_aliasing=True)
        except Exception:
            return syn_channel, None
    syn_raw = np.asarray(syn_raw, dtype=np.float32)
    if quant_use_raw:
        syn_raw = np.maximum(syn_raw, 0.0)
    else:
        syn_max = float(np.nanmax(syn_raw)) if syn_raw.size else 0.0
        if syn_max > 1.5:
            syn_raw = exposure.rescale_intensity(syn_raw, in_range="image", out_range=(0, 1))
        else:
            syn_raw = np.clip(syn_raw, 0, 1)
    return syn_channel, syn_raw


def _compute_s1_keep_ids(masks: np.ndarray, syn_raw: np.ndarray, mult: float, min_hits: int, min_raw: float):
    labels = np.unique(masks.astype(np.int64, copy=False))
    labels = labels[labels > 0]
    keep_ids: List[int] = []
    if mult <= 0 and min_hits <= 0 and min_raw <= 0:
        return [int(lbl) for lbl in labels]
    for lbl in labels:
        cell_mask = (masks == lbl)
        vals_cell = syn_raw[cell_mask]
        if vals_cell.size == 0:
            continue
        thr_cell = max(float(mult) * float(np.median(vals_cell)), float(min_raw))
        cell_max = float(np.max(vals_cell))
        if cell_max < float(min_raw):
            continue
        hits = np.count_nonzero((syn_raw >= thr_cell) & cell_mask)
        if hits >= int(min_hits):
            keep_ids.append(int(lbl))
    return keep_ids


def _get_active_s1_mask(viewer: "napari.Viewer") -> Tuple[np.ndarray | None, str]:
    """Return the mask to use for S1 (manual override if present)."""
    if CURRENT.get("masks") is None:
        _ensure_masks_from_viewer(viewer)
    _sync_current_masks_from_labels(viewer)
    if CURRENT.get("s1_layers_dirty"):
        _clear_s1_layers(viewer, include_final=True)
        CURRENT["s1_layers_dirty"] = False
    base = CURRENT.get("masks")
    if base is None:
        return None, "missing"
    masks = np.asarray(base, dtype=np.uint16)
    preview_mask = None
    preview_source = None
    if viewer is not None:
        for name in ("S1 kept cells (preview)", "S1 kept cells"):
            if name in [L.name for L in viewer.layers]:
                preview_layer = viewer.layers[name]
                try:
                    preview_mask = np.asarray(preview_layer.data)
                    preview_source = name
                except Exception:
                    preview_mask = None
                break
    if viewer is None:
        return masks, "current segmentation"
    if "S1_manual_mask" in [L.name for L in viewer.layers]:
        layer = viewer.layers["S1_manual_mask"]
        try:
            manual = np.asarray(layer.data)
        except Exception as exc:
            show_warning(f"S1 manual mask unreadable: {exc}")
            return masks, "current segmentation"
        if manual.shape != masks.shape:
            show_warning(
                f"S1 manual mask shape {manual.shape} does not match masks {masks.shape}; ignoring manual layer."
            )
        else:
            masks = manual.astype(np.uint16, copy=False)
            return masks, "S1_manual_mask"
    if preview_mask is not None:
        if preview_mask.shape != masks.shape:
            show_warning(
                f"S1 preview mask shape {preview_mask.shape} does not match masks {masks.shape}; using segmentation instead."
            )
        else:
            masks = preview_mask.astype(np.uint16, copy=False)
            return masks, preview_source or "S1 preview"
    return masks, "current segmentation"


def _get_manual_region_array(viewer: Optional["napari.Viewer"], layer_name: str, shape: tuple[int, ...]) -> np.ndarray | None:
    if viewer is None:
        return None
    if layer_name not in [L.name for L in viewer.layers]:
        return None
    layer = viewer.layers[layer_name]
    try:
        arr = np.asarray(layer.data)
    except Exception:
        return None
    if arr.shape != shape:
        show_warning(f"{layer_name} shape {arr.shape} does not match masks {shape}; ignoring.")
        return None
    return arr


def _is_seed_like_region(arr: np.ndarray) -> bool:
    """Return True when a region mask only contains 0–1 pixel per label (seed dots)."""
    positive = arr[arr > 0]
    if positive.size == 0:
        return True
    _, counts = np.unique(positive, return_counts=True)
    return np.all(counts <= 1)


def _choose_region_layer(
    viewer: Optional["napari.Viewer"],
    names: list[str],
    shape: tuple[int, ...],
    skip_seed_like: bool = False,
) -> np.ndarray | None:
    for nm in names:
        arr = _get_manual_region_array(viewer, nm, shape)
        if arr is not None:
            if skip_seed_like and _is_seed_like_region(arr):
                continue
            return arr
    return None


def _seed_synapse_labels(viewer: Optional["napari.Viewer"], labels: List[int]) -> None:
    """Place a seed pixel (centroid) for each label into the manual synapse layer."""
    if viewer is None or CURRENT.get("masks") is None or not labels:
        return
    try:
        labels_int = [int(l) for l in labels if int(l) > 0]
    except Exception:
        return
    if not labels_int:
        return
    masks = np.asarray(CURRENT["masks"])
    syn_layer = ensure_s1_region_manual_layer(viewer, "synapse")
    if syn_layer is None:
        return
    data = np.array(syn_layer.data, dtype=np.uint16, copy=True)
    props = measure.regionprops(masks)
    centers = {int(p.label): tuple(int(round(v)) for v in p.centroid) for p in props}
    for lbl in labels_int:
        rc = centers.get(int(lbl))
        if rc and len(rc) >= 2:
            r, c = rc[0], rc[1]
            if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
                data[r, c] = int(lbl)
    syn_layer.data = data
    sync_s1_region_masks(viewer)


def _refresh_kept_preview_from_manual(viewer: "napari.Viewer"):
    """Update the preview kept-cells layer using manual keep IDs (no stain prompt)."""
    masks = CURRENT.get("masks")
    if masks is None or viewer is None:
        return
    keep_ids = list(set(map(int, CURRENT.get("s1_manual_keep_ids", []) or [])))
    if not keep_ids:
        return
    kept_mask = np.isin(masks, np.array(keep_ids, dtype=masks.dtype))
    kept_labels = masks * kept_mask
    _add_or_replace_layer(viewer, "S1 kept cells (preview)", kept_labels.astype(masks.dtype, copy=False), opacity=0.6)

def _stash_preview_regions(viewer: Optional["napari.Viewer"], shape: Optional[Tuple[int, ...]] = None):
    """Capture current preview region masks into CURRENT for reuse after layers are cleared."""
    if viewer is None:
        return
    if shape is None:
        shape = np.asarray(CURRENT.get("masks")).shape if CURRENT.get("masks") is not None else None
    if shape is None:
        return
    preview_keys = [
        ("S1 synapse (preview)", "s1_preview_syn"),
        ("S1 membrane (preview)", "s1_preview_mem"),
        ("S1 extrasyn (preview)", "s1_preview_extra"),
    ]
    for layer_name, cache_key in preview_keys:
        arr = _get_manual_region_array(viewer, layer_name, shape)
        if arr is not None:
            CURRENT[cache_key] = np.array(arr, dtype=np.uint16, copy=True)

def _get_cached_preview_region(cache_key: str, shape: Tuple[int, ...]):
    arr = CURRENT.get(cache_key)
    if arr is None:
        return None
    if np.asarray(arr).shape != tuple(shape):
        return None
    return np.asarray(arr)

def _clear_s1_layers(viewer: Optional["napari.Viewer"], include_final: bool = True):
    """
    Remove S1 preview/final layers tied to prior masks so new boundaries are respected.
    Keeps manual edit layers (S1_manual_*).
    """
    if viewer is None:
        return
    names = [
        "S1 kept cells (preview)",
        "S1 synapse (preview)",
        "S1 membrane (preview)",
        "S1 extrasyn (preview)",
        "Kept Cell IDs (preview)",
        "S1_keep_clicks",
    ]
    if include_final:
        names.extend([
            "S1 kept cells",
            "S1 synapse",
            "S1 membrane",
            "S1 extrasyn",
            "Kept Cell IDs",
        ])
    for nm in names:
        try:
            if nm in [L.name for L in viewer.layers]:
                viewer.layers.remove(nm)
        except Exception:
            pass
    CURRENT["s1_preview_syn"] = None
    CURRENT["s1_preview_mem"] = None
    CURRENT["s1_preview_extra"] = None

def _reset_click_keep_state(viewer: Optional["napari.Viewer"]):
    """Clear click-to-keep points and manual keep IDs."""
    CURRENT["s1_manual_keep_ids"] = []
    if viewer is None:
        return
    layer_name = "S1_keep_clicks"
    try:
        if layer_name in [L.name for L in viewer.layers]:
            viewer.layers.remove(layer_name)
    except Exception:
        pass


def _reset_manual_keep_selection(viewer: Optional["napari.Viewer"]):
    """Clear manual keep state and related preview layers."""
    _reset_click_keep_state(viewer)
    if viewer is None:
        return
    for name in ("S1 kept cells (preview)", "Kept Cell IDs (preview)"):
        try:
            if name in [L.name for L in viewer.layers]:
                viewer.layers.remove(name)
        except Exception:
            pass


def _reset_s1_state(viewer: Optional["napari.Viewer"]):
    """Reset all S1 preview/final layers, caches, and manual keep state."""
    CURRENT["s1_manual_keep_ids"] = []
    CURRENT["s1_layers_dirty"] = False
    CURRENT["s1_preview_syn"] = None
    CURRENT["s1_preview_mem"] = None
    CURRENT["s1_preview_extra"] = None
    names = [
        "S1 kept cells (preview)",
        "S1 synapse (preview)",
        "S1 membrane (preview)",
        "S1 extrasyn (preview)",
        "Kept Cell IDs (preview)",
        "S1_keep_clicks",
        "S1 kept cells",
        "S1 synapse",
        "S1 membrane",
        "S1 extrasyn",
        "Kept Cell IDs",
        "S1_manual_mask",
        "S1_manual_membrane",
        "S1_manual_synapse",
        "S1_manual_extrasyn",
    ]
    if viewer is None:
        return
    for nm in names:
        try:
            if nm in [L.name for L in viewer.layers]:
                viewer.layers.remove(nm)
        except Exception:
            pass

def ensure_s1_click_layer(viewer: "napari.Viewer"):
    """Create or return a points layer for click-to-select S1 cells."""
    layer_name = "S1_keep_clicks"
    if layer_name in [L.name for L in viewer.layers]:
        layer = viewer.layers[layer_name]
    else:
        layer = viewer.add_points(
            name=layer_name,
            size=6,
            face_color="yellow",
            opacity=0.8,
        )
    layer.mode = "add"
    layer.visible = True
    show_info("Click on cells to mark them. Use 'Keep clicked cells' to apply.")
    return layer

def keep_cells_from_clicks(viewer: "napari.Viewer"):
    """Convert clicked points into keep-ids for S1."""
    if CURRENT.get("masks") is None and _ensure_masks_from_viewer(viewer) is None:
        mq_info(None, "S1 manual keep", "Run/import a segmentation first.")
        return
    layer_name = "S1_keep_clicks"
    if layer_name not in [L.name for L in viewer.layers]:
        mq_info(None, "S1 manual keep", "Use 'Click-to-keep mode' first to add points.")
        return
    pts_layer = viewer.layers[layer_name]
    pts = np.asarray(getattr(pts_layer, "data", []))
    if pts.size == 0:
        mq_info(None, "S1 manual keep", "No points placed. Click on cells first.")
        return
    masks = np.asarray(CURRENT["masks"])
    keep_list = set(map(int, CURRENT.get("s1_manual_keep_ids", []) or []))
    for pt in pts:
        if len(pt) < 2:
            continue
        r = int(round(pt[0]))
        c = int(round(pt[1]))
        if r < 0 or c < 0 or r >= masks.shape[0] or c >= masks.shape[1]:
            continue
        lbl = int(masks[r, c])
        if lbl > 0:
            keep_list.add(lbl)
    CURRENT["s1_manual_keep_ids"] = sorted(keep_list)

    pts_layer.data = np.empty((0, 2), dtype=float)
    try:
        viewer.layers.remove(pts_layer)
    except Exception:
        pass
    # Drop cached previews so membranes/synapse regions refresh on next preview/analysis.
    CURRENT["s1_preview_syn"] = None
    CURRENT["s1_preview_mem"] = None
    CURRENT["s1_preview_extra"] = None
    for lname in (
        "S1 synapse (preview)",
        "S1 membrane (preview)",
        "S1 extrasyn (preview)",
    ):
        try:
            if lname in [L.name for L in viewer.layers]:
                viewer.layers.remove(lname)
        except Exception:
            pass
    show_info(f"Marked {len(keep_list)} total cell(s) to keep for S1.")


def run_s1_analysis(viewer: "napari.Viewer", export_results: bool = False):
    """
    Synaptic analysis:
      1) Keep cells with synapse stain (>=5 pixels >0.75).
      2) Highlight synapse region per cell (smoothed threshold).
      3) Membrane ring = inner erosion of 3px; extrasyn region = membrane minus synapse.
      4) Compute per-cell areas and fluorescence stats (synapse/extrasyn) per channel.
      5) Optionally export CSV when requested (else cache results).
    """
    masks, mask_source = _get_active_s1_mask(viewer)
    if masks is None:
        masks = _ensure_masks_from_viewer(viewer)
        if masks is not None:
            mask_source = "viewer labels"
    proteins: List[ProteinChannel] = _filter_channels_by_viewer(viewer, CURRENT.get("proteins") or [])
    if not proteins:
        proteins = _fallback_proteins_from_viewer(viewer)
        if proteins:
            CURRENT["proteins"] = list(proteins)
    if masks is None or not proteins:
        mq_info(None, "Synaptic analysis", "Run segmentation and load proteins first.")
        return
    # Pick synapse stain
    stain_name = str(CURRENT.get("synapse_stain", "") or "").strip()
    if not stain_name:
        stain_name, ok = QInputDialog.getText(
            None,
            "Synapse stain",
            "Enter synapse stain name (e.g., bungarotoxin):",
            text="",
        )
        if not ok or not str(stain_name).strip():
            return
        stain_name = str(stain_name).strip()
        CURRENT["synapse_stain"] = stain_name
    target_lower = stain_name.lower()
    preview_map = CURRENT.get("preview_planes", {}) or {}
    quant_use_raw = bool(CURRENT.get("quant_use_raw", True))
    syn_channel, syn_raw = _get_synapse_channel_and_data(stain_name, proteins, preview_map, masks, quant_use_raw=True)
    if syn_channel is None or syn_raw is None:
        mq_info(None, "Synaptic analysis", f"No stain matching '{stain_name}' found or load failed.")
        return
    syn_raw = np.maximum(syn_raw, 0.0)
    syn_norm_for_gate = np.clip(
        exposure.rescale_intensity(syn_raw, in_range="image", out_range=(0, 1)) if float(np.nanmax(syn_raw)) > 1.5 else syn_raw,
        0,
        1,
    )
    print(f"[S1] Synapse channel '{syn_channel.name}' range after norm: min={syn_raw.min():.3f} max={syn_raw.max():.3f}")
    syn_smooth = gaussian(syn_norm_for_gate, sigma=1.0, preserve_range=True)

    labels = np.unique(masks)
    labels = labels[labels > 0]
    mult = float(CURRENT.get("s1_min_mult", 5.0))
    min_hits = int(CURRENT.get("s1_min_hits", 5))
    min_raw = float(CURRENT.get("s1_min_raw", 0.0))
    manual_keep = list(map(int, CURRENT.get("s1_manual_keep_ids", []) or []))
    if manual_keep:
        keep_ids = manual_keep
    else:
        keep_ids = _compute_s1_keep_ids(masks, syn_norm_for_gate, mult, min_hits, min_raw)
        if mult <= 0 and min_hits <= 0 and min_raw <= 0:
            # Safety fallback: keep all labels when gating is disabled.
            keep_ids = [int(lbl) for lbl in labels]
    total_cells = int(labels.size)
    if not keep_ids:
        mq_info(None, "Synaptic analysis", f"No cells met the synapse stain threshold (total {total_cells}).")
        return
    print(f"[S1] Using mask source: {mask_source}")

    kept_mask = np.isin(masks, np.array(keep_ids, dtype=masks.dtype))
    kept_labels = masks * kept_mask
    _add_or_replace_layer(viewer, "S1 kept cells", kept_labels.astype(masks.dtype, copy=False), opacity=0.6)
    _add_cell_ids_layer(viewer, kept_labels.astype(np.uint16, copy=False), layer_name="Kept Cell IDs")
    print(f"[S1] Kept {len(keep_ids)} / {total_cells} cells with ≥5 px >0.75 in '{syn_channel.name}'")

    # Capture preview masks (with manual edits) before removing preview layers so gating can reuse them
    _stash_preview_regions(viewer, masks.shape)
    preview_mem_layer = _get_manual_region_array(viewer, "S1 membrane (preview)", masks.shape)
    if preview_mem_layer is None:
        preview_mem_layer = _get_cached_preview_region("s1_preview_mem", masks.shape)
    preview_syn_layer = _get_manual_region_array(viewer, "S1 synapse (preview)", masks.shape)
    if preview_syn_layer is None:
        preview_syn_layer = _get_cached_preview_region("s1_preview_syn", masks.shape)
    preview_extra_layer = _get_manual_region_array(viewer, "S1 extrasyn (preview)", masks.shape)
    if preview_extra_layer is None:
        preview_extra_layer = _get_cached_preview_region("s1_preview_extra", masks.shape)

    # Drop preview-only layers now that we are running the full analysis
    preview_layers_to_remove = [
        "S1 kept cells (preview)",
        "S1 synapse (preview)",
        "S1 membrane (preview)",
        "S1 extrasyn (preview)",
        "S1_keep_clicks",
        "Kept Cell IDs (preview)",
    ]
    for lname in preview_layers_to_remove:
        try:
            if lname in [L.name for L in viewer.layers]:
                viewer.layers.remove(lname)
        except Exception:
            pass

    syn_layer = np.zeros_like(masks, dtype=np.uint16)
    extra_layer = np.zeros_like(masks, dtype=np.uint16)
    membrane_layer = np.zeros_like(masks, dtype=np.uint16)

    rows: List[Dict[str, Any]] = []
    interior_selem = disk(max(1, int(CURRENT.get("s1_membrane_px", 4))))
    px_um = float(CURRENT.get("px_um", DEFAULT_PX_UM))
    px_area_um2 = px_um * px_um
    all_gates_off = mult <= 0 and min_hits <= 0 and min_raw <= 0
    manual_mem_arr = _choose_region_layer(viewer, ["S1_manual_membrane", "S1 membrane (preview)"], masks.shape)
    manual_syn_arr = _choose_region_layer(
        viewer,
        ["S1_manual_synapse", "S1 synapse (preview)"],
        masks.shape,
        skip_seed_like=True,
    )
    manual_extra_arr = _choose_region_layer(viewer, ["S1_manual_extrasyn", "S1 extrasyn (preview)"], masks.shape)

    # Build channel list for quant, including background/lam as a stain
    stains_for_quant: List[ProteinChannel] = list(proteins)
    lam_arr_base = CURRENT.get("lam_raw")
    if lam_arr_base is None:
        lam_paths = CURRENT.get("lam_paths") or []
        if lam_paths:
            try:
                plane_hint = None
                if CURRENT.get("lam_used_list"):
                    hint = CURRENT["lam_used_list"][0]
                    plane_hint = hint if hint in PLANE_TO_IDX else None
                lam_arr_base = load_raw_gray(Path(lam_paths[0]), plane_hint)
                CURRENT["lam_raw"] = lam_arr_base
            except Exception:
                lam_arr_base = CURRENT.get("lam")
    if lam_arr_base is not None:
        lam_path0 = Path(CURRENT.get("lam_paths", [None])[0]) if CURRENT.get("lam_paths") else None
        bg_name = infer_stain_from_filename(lam_path0) if lam_path0 else "Background"
        stains_for_quant.append(
            ProteinChannel(
                name=bg_name,
                plane="Gray",
                path=None,
                data=np.asarray(lam_arr_base, dtype=np.float32),
                preview_id="__background__",
                origin="background",
            )
        )

    for lbl in keep_ids:
        cell_mask = (masks == lbl)
        interior = erosion(cell_mask, interior_selem)
        if not np.any(interior):
            interior = cell_mask
        cell_area_px = int(np.count_nonzero(cell_mask))

        if preview_mem_layer is not None:
            membrane = preview_mem_layer == lbl
        elif manual_mem_arr is not None:
            membrane = (manual_mem_arr == lbl)
        else:
            membrane = cell_mask & (~interior)
        membrane_layer[membrane] = lbl

        if all_gates_off:
            syn_region = np.zeros_like(cell_mask, dtype=bool)
            extra_region = np.zeros_like(cell_mask, dtype=bool)
        else:
            if preview_syn_layer is not None:
                syn_region = preview_syn_layer == lbl
            elif manual_syn_arr is not None:
                syn_region = manual_syn_arr == lbl
            else:
                vals = syn_smooth[cell_mask]
                if vals.size == 0:
                    continue
                try:
                    thr = float(threshold_otsu(vals))
                except Exception:
                    thr = float(np.median(vals))
                syn_region = cell_mask & (syn_smooth >= thr)
                syn_region = remove_small_objects(syn_region, min_size=max(5, int(round(vals.size * 0.01))))
                closing_radius = max(1, int(round(np.sqrt(vals.size) / 25)))
                syn_region = binary_closing(syn_region, footprint=disk(closing_radius))
                syn_region = binary_fill_holes(syn_region & cell_mask) & cell_mask
                if not np.any(syn_region):
                    syn_region = cell_mask & (syn_smooth >= thr)
            # If synapse is oversized and dim, drop it and let extrasyn cover the membrane
            if manual_syn_arr is None and manual_extra_arr is None and cell_area_px > 0:
                syn_area_px_tmp = int(np.count_nonzero(syn_region))
                syn_median_tmp = float(np.median(syn_norm_for_gate[syn_region])) if syn_area_px_tmp else 0.0
                syn_frac_tmp = syn_area_px_tmp / float(cell_area_px)
                if syn_frac_tmp > 0.35 and syn_median_tmp < 0.5:
                    syn_region = np.zeros_like(cell_mask, dtype=bool)
            syn_layer[syn_region] = lbl
            if preview_extra_layer is not None:
                extra_region = preview_extra_layer == lbl
            elif manual_extra_arr is not None:
                extra_region = manual_extra_arr == lbl
            else:
                extra_region = membrane & (~syn_region)
                if np.any(extra_region):
                    labeled_extra = cc_label(extra_region)
                    if labeled_extra.max() > 1:
                        counts = np.bincount(labeled_extra.ravel())[1:]
                        if counts.size:
                            keep_lbl = 1 + int(np.argmax(counts))
                            extra_region = labeled_extra == keep_lbl
            extra_layer[extra_region] = lbl

        cell_area = int(np.count_nonzero(cell_mask))
        syn_area = int(np.count_nonzero(syn_region))
        extra_area = int(np.count_nonzero(extra_region))
        membrane_area = int(np.count_nonzero(membrane))
        cell_area_px = cell_area
        syn_area_px = syn_area
        extra_area_px = extra_area
        membrane_area_px = membrane_area
        cell_area_um2 = cell_area * px_area_um2
        syn_area_um2 = syn_area * px_area_um2
        membrane_area_um2 = membrane_area * px_area_um2
        extra_area_um2 = extra_area * px_area_um2

        # Synapse stain stats
        bg_intra = float(np.mean(syn_raw[interior])) if np.any(interior) else float(np.median(syn_raw))
        syn_mean = float(np.mean(syn_raw[syn_region])) if syn_area else 0.0
        syn_integrated = syn_mean * syn_area
        syn_ctcf = syn_integrated - syn_area * bg_intra

        row: Dict[str, Any] = {
            "Cell ID": lbl,
            "Cell Area (px)": cell_area_px,
            "Cell Area (um^2)": cell_area_um2,
            "Synaptic Area (px)": syn_area_px,
            "Synaptic Area (um^2)": syn_area_um2,
            "Membrane Area (px)": membrane_area_px,
            "Membrane Area (um^2)": membrane_area_um2,
            "Extrasynaptic Area (px)": extra_area_px,
            "Extrasynaptic Area (um^2)": extra_area_um2,
            f"{syn_channel.name} Background Mean": bg_intra,
            f"{syn_channel.name} Mean Synaptic Fluorescence": syn_mean,
            f"{syn_channel.name} Synaptic CTCF": syn_ctcf,
            f"{syn_channel.name} Synaptic Integrated": syn_integrated,
        }
        if all_gates_off:
            mem_mean = float(np.mean(syn_raw[membrane])) if membrane_area else 0.0
            mem_int = mem_mean * membrane_area
            ctcf_mem_intra = mem_int - membrane_area * bg_intra
            row.update({
                f"{syn_channel.name} Mean Membrane Fluorescence": mem_mean,
                f"{syn_channel.name} Membrane CTCF": ctcf_mem_intra,
                f"{syn_channel.name} Membrane Integrated": mem_int,
            })

        for ch in stains_for_quant:
            p_raw = _load_channel_raw(ch, preview_map)
            if p_raw.shape != masks.shape:
                try:
                    p_raw = resize(p_raw, masks.shape, order=1, preserve_range=True, anti_aliasing=True)
                except Exception:
                    continue
            p_raw = np.asarray(p_raw, dtype=np.float32)
            if quant_use_raw:
                p_raw = np.maximum(p_raw, 0.0)
            else:
                p_max = float(np.nanmax(p_raw)) if p_raw.size else 0.0
                if p_max > 1.5:
                    p_raw = exposure.rescale_intensity(p_raw, in_range="image", out_range=(0, 1))
                else:
                    p_raw = np.clip(p_raw, 0, 1)
            bg_intra_c = float(np.mean(p_raw[interior])) if np.any(interior) else float(np.median(p_raw))
            if all_gates_off:
                mem_mean_c = float(np.mean(p_raw[membrane])) if membrane_area else 0.0
                mem_int_c = mem_mean_c * membrane_area
                ctcf_mem_intra_c = mem_int_c - membrane_area * bg_intra_c
                row.update({
                    f"{ch.name} Background Mean": bg_intra_c,
                    f"{ch.name} Mean Membrane Fluorescence": mem_mean_c,
                    f"{ch.name} Membrane CTCF": ctcf_mem_intra_c,
                    f"{ch.name} Membrane Integrated": mem_int_c,
                })
            else:
                syn_mean_c = float(np.mean(p_raw[syn_region])) if syn_area else 0.0
                syn_int_c = syn_mean_c * syn_area
                ctcf_syn_intra = syn_int_c - syn_area * bg_intra_c

                extra_mean_c = float(np.mean(p_raw[extra_region])) if extra_area else 0.0
                extra_int_c = extra_mean_c * extra_area
                ctcf_extra_intra = extra_int_c - extra_area * bg_intra_c

                row.update({
                    f"{ch.name} Background Mean": bg_intra_c,
                    f"{ch.name} Mean Synaptic Fluorescence": syn_mean_c,
                    f"{ch.name} Synaptic CTCF": ctcf_syn_intra,
                    f"{ch.name} Synaptic Integrated": syn_int_c,
                    f"{ch.name} Mean Extrasynaptic Fluorescence": extra_mean_c,
                    f"{ch.name} Extrasynaptic CTCF": ctcf_extra_intra,
                    f"{ch.name} Extrasynaptic Integrated": extra_int_c,
                })
        rows.append(row)

    _add_or_replace_layer(viewer, "S1 membrane", membrane_layer.astype(masks.dtype, copy=False), opacity=0.5)
    if not all_gates_off:
        _add_or_replace_layer(viewer, "S1 synapse", syn_layer.astype(masks.dtype, copy=False), opacity=0.7)
        _add_or_replace_layer(viewer, "S1 extrasyn", extra_layer.astype(masks.dtype, copy=False), opacity=0.7)

    # Preserve latest regions so subsequent previews start from user-edited masks.
    CURRENT["s1_preview_mem"] = membrane_layer.copy()
    CURRENT["s1_preview_syn"] = syn_layer.copy()
    CURRENT["s1_preview_extra"] = extra_layer.copy()

    if not rows:
        mq_info(None, "Synaptic analysis", "No rows generated.")
        return

    df = pd.DataFrame(rows)
    CURRENT["synaptic_results"] = {
        "df": df,
        "base": CURRENT.get("base") or "image",
        "keep_ids": keep_ids,
        "mask_source": mask_source,
    }
    if export_results:
        export_synaptic_results(viewer, use_cached=True)
    else:
        show_info(f"Synaptic analysis done.\nKept cells: {len(keep_ids)}\nMask source: {mask_source}\nUse 'Export Results' to save.")
    return df


def export_synaptic_results(viewer: "napari.Viewer", use_cached: bool = False):
    """
    Export the latest Synaptic analysis results to CSV.
    If no cached results exist, runs the analysis without exporting first.
    """
    res = CURRENT.get("synaptic_results")
    if use_cached and res is None:
        mq_info(None, "Synaptic analysis", "No synaptic results available. Run Synaptic analysis first.")
        return
    if not use_cached or res is None:
        run_s1_analysis(viewer, export_results=False)
        res = CURRENT.get("synaptic_results")
        if res is None:
            mq_info(None, "Synaptic analysis", "No synaptic results available to export.")
            return

    df = res.get("df")
    if df is None or df.empty:
        mq_info(None, "Synaptic analysis", "No synaptic rows to export.")
        return
    base = res.get("base") or CURRENT.get("base") or "image"
    default_root = _default_export_root(SEGMENT_EXPORT_ROOT or (Path.home() / "musclequant" / "results"))
    export_root = choose_export_root(default_root)
    if export_root is None:
        mq_info(None, "Synaptic analysis", "Export canceled (no folder selected).")
        return
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    base_safe = _safe_basename(base)
    csv_out = _unique_path(export_root / f"{base_safe}_synaptic_quant.csv")
    df.to_csv(csv_out, index=False)
    show_info(f"Synaptic analysis exported.\nCSV: {csv_out}")


def preview_s1_filter(viewer: "napari.Viewer"):
    """
    Apply Synaptic keep-logic only and show kept cells layer for quick inspection.
    """
    masks, mask_source = _get_active_s1_mask(viewer)
    if masks is None:
        masks = _ensure_masks_from_viewer(viewer)
        if masks is not None:
            mask_source = "viewer labels"
    proteins: List[ProteinChannel] = _filter_channels_by_viewer(viewer, CURRENT.get("proteins") or [])
    if not proteins:
        proteins = _fallback_proteins_from_viewer(viewer)
        if proteins:
            CURRENT["proteins"] = list(proteins)
    if masks is None or not proteins:
        mq_info(None, "Synaptic preview", "Run segmentation and load proteins first.")
        return
    stain_name = str(CURRENT.get("synapse_stain", "") or "").strip()
    if not stain_name:
        stain_name, ok = QInputDialog.getText(
            None,
            "Synapse stain",
            "Enter synapse stain name (e.g., bungarotoxin):",
            text="",
        )
        if not ok or not str(stain_name).strip():
            return
        stain_name = str(stain_name).strip()
        CURRENT["synapse_stain"] = stain_name
    preview_map = CURRENT.get("preview_planes", {}) or {}
    quant_use_raw = bool(CURRENT.get("quant_use_raw", True))
    syn_channel, syn_raw = _get_synapse_channel_and_data(stain_name, proteins, preview_map, masks, quant_use_raw=True)
    if syn_channel is None or syn_raw is None:
        mq_info(None, "Synaptic preview", f"No stain matching '{stain_name}' found or load failed.")
        return
    syn_raw = np.maximum(syn_raw, 0.0)
    syn_norm_for_gate = np.clip(
        exposure.rescale_intensity(syn_raw, in_range="image", out_range=(0, 1)) if float(np.nanmax(syn_raw)) > 1.5 else syn_raw,
        0,
        1,
    )
    mult = float(CURRENT.get("s1_min_mult", 5.0))
    min_hits = int(CURRENT.get("s1_min_hits", 5))
    min_raw = float(CURRENT.get("s1_min_raw", 0.0))
    manual_keep = list(map(int, CURRENT.get("s1_manual_keep_ids", []) or []))
    if manual_keep:
        keep_ids = manual_keep
    else:
        keep_ids = _compute_s1_keep_ids(masks, syn_norm_for_gate, mult, min_hits, min_raw)
    labels = np.unique(masks); labels = labels[labels > 0]
    all_gates_off = mult <= 0 and min_hits <= 0 and min_raw <= 0
    if all_gates_off and not manual_keep:
        keep_ids = [int(lbl) for lbl in labels]
    total_cells = int(labels.size)
    if not keep_ids:
        mq_info(None, "Synaptic preview", f"No cells met the synapse stain threshold (total {total_cells}).")
        return

    preview_mem_layer = _get_manual_region_array(viewer, "S1 membrane (preview)", masks.shape)
    if preview_mem_layer is None:
        preview_mem_layer = _get_cached_preview_region("s1_preview_mem", masks.shape)
    preview_syn_layer = _get_manual_region_array(viewer, "S1 synapse (preview)", masks.shape)
    if preview_syn_layer is None:
        preview_syn_layer = _get_cached_preview_region("s1_preview_syn", masks.shape)
    preview_extra_layer = _get_manual_region_array(viewer, "S1 extrasyn (preview)", masks.shape)
    if preview_extra_layer is None:
        preview_extra_layer = _get_cached_preview_region("s1_preview_extra", masks.shape)

    # Capture existing preview masks (with any manual edits) before we clear them.
    _stash_preview_regions(viewer, masks.shape)
    # Drop prior preview layers so morphology updates (e.g., membrane dilation)
    for lname in (
        "S1 kept cells (preview)",
        "S1 synapse (preview)",
        "S1 membrane (preview)",
        "S1 extrasyn (preview)",
        "S1_keep_clicks",
        "Kept Cell IDs (preview)",
    ):
        try:
            if lname in [L.name for L in viewer.layers]:
                viewer.layers.remove(lname)
        except Exception:
            pass

    kept_mask = np.isin(masks, np.array(keep_ids, dtype=masks.dtype))
    kept_labels = masks * kept_mask
    _add_or_replace_layer(viewer, "S1 kept cells (preview)", kept_labels.astype(masks.dtype, copy=False), opacity=0.6)
    _add_cell_ids_layer(viewer, kept_labels.astype(np.uint16, copy=False), layer_name="Kept Cell IDs (preview)")

    # Highlight regions for preview
    syn_layer = np.zeros_like(masks, dtype=masks.dtype)
    extra_layer = np.zeros_like(masks, dtype=masks.dtype)
    membrane_layer = np.zeros_like(masks, dtype=masks.dtype)
    interior_selem = disk(max(1, int(CURRENT.get("s1_membrane_px", 4))))
    syn_smooth = syn_norm_for_gate if all_gates_off else gaussian(syn_norm_for_gate, sigma=1.0, preserve_range=True)
    manual_mem_arr = _choose_region_layer(viewer, ["S1_manual_membrane", "S1 membrane (preview)"], masks.shape)
    manual_syn_arr = _choose_region_layer(viewer, ["S1_manual_synapse", "S1 synapse (preview)"], masks.shape, skip_seed_like=True)
    manual_extra_arr = _choose_region_layer(viewer, ["S1_manual_extrasyn", "S1 extrasyn (preview)"], masks.shape)

    # Build overlays for all cells so membrane/extrasyn show even for non-kept cells.
    for lbl in labels:
        cell_mask = (masks == lbl)
        if not np.any(cell_mask):
            continue
        interior = erosion(cell_mask, interior_selem)
        if not np.any(interior):
            interior = cell_mask
        if preview_mem_layer is not None:
            membrane = preview_mem_layer == lbl
        elif manual_mem_arr is not None:
            membrane = manual_mem_arr == lbl
        else:
            membrane = cell_mask & (~interior)
        membrane_layer[membrane] = lbl
        if preview_syn_layer is not None:
            syn_region = preview_syn_layer == lbl
        elif manual_syn_arr is not None:
            syn_region = manual_syn_arr == lbl
        elif all_gates_off:
            syn_region = np.zeros_like(cell_mask, dtype=bool)
        else:
            vals = syn_smooth[cell_mask]
            if vals.size == 0:
                syn_region = np.zeros_like(cell_mask, dtype=bool)
            else:
                try:
                    thr = float(threshold_otsu(vals))
                except Exception:
                    thr = float(np.median(vals))
                syn_region = cell_mask & (syn_smooth >= thr)
                syn_region = remove_small_objects(syn_region, min_size=max(5, int(round(vals.size * 0.01))))
                closing_radius = max(1, int(round(np.sqrt(vals.size) / 25)))
                syn_region = binary_closing(syn_region, footprint=disk(closing_radius))
                syn_region = binary_fill_holes(syn_region & cell_mask) & cell_mask
                if not np.any(syn_region):
                    syn_region = cell_mask & (syn_smooth >= thr)
        # If synapse is oversized and dim, drop it and let extrasyn cover the membrane
        if preview_syn_layer is None and manual_syn_arr is None and manual_extra_arr is None and not all_gates_off:
            cell_area_px = int(np.count_nonzero(cell_mask))
            syn_area_px_tmp = int(np.count_nonzero(syn_region))
            syn_median_tmp = float(np.median(syn_norm_for_gate[syn_region])) if syn_area_px_tmp else 0.0
            syn_frac_tmp = (syn_area_px_tmp / float(cell_area_px)) if cell_area_px > 0 else 0.0
            if syn_frac_tmp > 0.35 and syn_median_tmp < 0.5:
                syn_region = np.zeros_like(cell_mask, dtype=bool)
        syn_layer[syn_region] = lbl
        if preview_extra_layer is not None:
            extra_region = preview_extra_layer == lbl
        elif manual_extra_arr is not None:
            extra_region = manual_extra_arr == lbl
        else:
            extra_region = membrane & (~syn_region)
            if np.any(extra_region):
                labeled_extra = cc_label(extra_region)
                if labeled_extra.max() > 1:
                    counts = np.bincount(labeled_extra.ravel())[1:]
                    if counts.size:
                        keep_lbl = 1 + int(np.argmax(counts))
                        extra_region = labeled_extra == keep_lbl
        extra_layer[extra_region] = lbl

    _add_or_replace_layer(viewer, "S1 membrane (preview)", membrane_layer, opacity=0.5)
    _add_or_replace_layer(viewer, "S1 synapse (preview)", syn_layer, opacity=0.7)
    _add_or_replace_layer(viewer, "S1 extrasyn (preview)", extra_layer, opacity=0.7)
    CURRENT["s1_preview_mem"] = membrane_layer.copy()
    CURRENT["s1_preview_syn"] = syn_layer.copy()
    CURRENT["s1_preview_extra"] = extra_layer.copy()

    show_info(
        f"Synaptic preview: kept {len(keep_ids)} / {total_cells} cells "
        f"(mask: {mask_source}) using mult={mult} hits={min_hits} min_raw={min_raw:.2f}"
    )

def quant_selection(viewer: "napari.Viewer"):
    """
    Quantify only the cells intersecting the Cell_query polygons and save a CSV.
    """
    if CURRENT.get("masks") is None or CURRENT.get("lam") is None:
        mq_info(None, "Quantify selection", "Run segmentation first.")
        return
    ids = list_cells_in_selection(viewer)
    if not ids:
        return
    labels_subset = np.isin(CURRENT["masks"], np.array(ids, dtype=CURRENT["masks"].dtype))
    subset_mask = CURRENT["masks"] * labels_subset
    if not np.any(subset_mask):
        mq_info(None, "Quantify selection", "No matching labels found in current masks.")
        return

    px_um = _parse_px_um(CURRENT.get("px_um")) or float(DEFAULT_PX_UM)

    wide_df = None
    shared_base_cols = {
        "label", "area_px", "area_um2", "perimeter", "eccentricity",
        "solidity", "equivalent_diameter", "centroid-0", "centroid-1", "px_um",
    }
    if CURRENT.get("add_spatial", True):
        shared_base_cols.add("nn_dist_um")
    shared_base_cols.update({"membrane_area_px", "membrane_area_um2"})

    stain_filter = [s.strip().lower() for s in CURRENT.get("stain_filter", []) if s.strip()]
    metrics = _sanitize_metrics(selected_metrics())
    CURRENT["metrics"] = metrics
    preview_map = CURRENT.get("preview_planes", {})

    mode = CURRENT.get("mode", "IHC")
    intensity_needed = metrics.get("ctcf", True)
    mem_px = int(CURRENT.get("membrane_px", CURRENT.get("s1_membrane_px", 4)))
    membrane_labels, interior_labels = _compute_membrane_and_interior_masks(subset_mask, mem_px)
    mem_area_px_map, mem_area_um2_map = _membrane_area_maps(membrane_labels, float(px_um))
    center_masks = compute_center_masks(subset_mask) if intensity_needed else None

    # Always include background/lam as a stain in standard quant.
    proteins_for_quant: List[ProteinChannel] = list(_filter_channels_by_viewer(viewer, CURRENT.get("proteins") or []))
    lam_arr_base = CURRENT.get("lam_raw") if CURRENT.get("lam_raw") is not None else CURRENT.get("lam")
    if lam_arr_base is not None:
        lam_path0 = Path(CURRENT.get("lam_paths", [None])[0]) if CURRENT.get("lam_paths") else None
        bg_name = infer_stain_from_filename(lam_path0) if lam_path0 else "Background"
        bg_plane = (CURRENT.get("lam_used_list", ["Gray"])[0]) if CURRENT.get("lam_used_list") else "Gray"
        proteins_for_quant = [
            ProteinChannel(
                name=bg_name,
                plane=bg_plane if bg_plane in PLANE_TO_IDX else "Gray",
                path=None,
                data=np.asarray(lam_arr_base, dtype=np.float32),
                preview_id="__background__",
                origin="background",
            ),
            *proteins_for_quant,
        ]

    if mode.upper() == "H&E" and not intensity_needed:
        df = quantify_geometry(
            subset_mask,
            px_um=float(px_um),
            add_spatial=bool(CURRENT.get("add_spatial", True)),
        )
        df["membrane_area_px"] = df["label"].map(mem_area_px_map).fillna(0).astype(np.int32)
        df["membrane_area_um2"] = df["label"].map(mem_area_um2_map).fillna(0.0).astype(np.float32)
        df = _filter_quant_columns(df, metrics, "geometry")
        wide_df = df.copy()
    else:
        for channel in proteins_for_quant:
            pname = channel.name
            if stain_filter and pname.strip().lower() not in stain_filter:
                continue
            p_raw = _load_channel_raw(channel, preview_map)
            quant_out = quantify(
                subset_mask, p_raw,
                px_um=float(px_um),
                protein_name=pname,
                bg_mode=CURRENT.get("bg_mode", "center"),
                bg_percentile=float(CURRENT.get("bg_percentile", 5.0)),
                local_ring_px=int(CURRENT.get("local_ring_px", 5)),
                rolling_ball_radius_px=int(CURRENT.get("rolling_ball_radius_px", 50)),
                synapse_ring_px=SYNAPSE_RING_PX,
                synapse_percentile=SYNAPSE_PERCENTILE,
                add_texture=bool(CURRENT.get("add_texture", True)),
                add_spatial=bool(CURRENT.get("add_spatial", True)),
                return_masks=False,
                center_masks=center_masks,
            )
            df = quant_out if not isinstance(quant_out, tuple) else quant_out[0]
            df["membrane_area_px"] = df["label"].map(mem_area_px_map).fillna(0).astype(np.int32)
            df["membrane_area_um2"] = df["label"].map(mem_area_um2_map).fillna(0.0).astype(np.float32)
            if metrics.get("ctcf", True):
                df = _add_membrane_intensity_columns(df, p_raw, subset_mask, membrane_labels, interior_labels, pname)
            df = _filter_quant_columns(df, metrics, pname)
            if wide_df is None:
                wide_df = df.copy()
            else:
                protein_cols = [c for c in df.columns if c not in shared_base_cols]
                merge_cols = ["label"] + protein_cols
                wide_df = pd.merge(wide_df, df[merge_cols], on="label", how="left")

    if wide_df is None:
        mq_info(None, "Quantify selection", "No proteins quantified.")
        return

    px_area_um2 = float(px_um) ** 2
    if "area_px" in wide_df.columns:
        wide_df["area_um2"] = wide_df["area_px"].astype(float) * px_area_um2
    if "membrane_area_px" in wide_df.columns:
        wide_df["membrane_area_um2"] = wide_df["membrane_area_px"].astype(float) * px_area_um2
    wide_df["px_um"] = float(px_um)

    base_name = _safe_basename(CURRENT.get("base") or CURRENT.get("base_out") or "image")
    default_root = _default_export_root(SEGMENT_EXPORT_ROOT or (Path.home() / "musclequant" / "results"))
    export_root = choose_export_root(default_root)
    if export_root is None:
        mq_info(None, "Quantify selection", "Canceled (no folder selected).")
        return
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    base_safe = _safe_basename(base_name)
    csv_out = _unique_path(export_root / f"{base_safe}_selection_quant.csv")
    wide_df.to_csv(csv_out, index=False)
    show_info(f"Selection quantification saved to:\n{csv_out}")

def _add_or_replace_layer(viewer: "napari.Viewer", name: str, data, **kwargs):
    if name in [L.name for L in viewer.layers]:
        layer = viewer.layers[name]
        layer.data = data
        for k, v in kwargs.items():
            if hasattr(layer, k):
                setattr(layer, k, v)
    else:
        viewer.add_labels(data, name=name, **kwargs)


def _add_cell_ids_layer(viewer: "napari.Viewer", mask: np.ndarray, layer_name: str = "Cell IDs"):
    labels = mask[mask > 0]
    if labels.size == 0:
        return
    props = measure.regionprops(mask)
    coords = np.array([p.centroid for p in props], dtype=np.float32)
    texts = [str(int(p.label)) for p in props]
    # Napari's text encoding can get out of sync when labels are sparse or added manually.
    # Always pad/trim the text array to the number of points so TextManager stays aligned.
    n_pts = coords.shape[0]
    text_arr = np.asarray(texts, dtype=str)
    if text_arr.shape[0] != n_pts:
        if text_arr.shape[0] < n_pts:
            pad = np.full(n_pts - text_arr.shape[0], "", dtype=str)
            text_arr = np.concatenate([text_arr, pad], axis=0)
        else:
            text_arr = text_arr[:n_pts]
    # Napari TextManager expects a sequence of Python strings, not a NumPy array.
    text_cfg = {"string": text_arr.astype(str).tolist()}
    if layer_name in [L.name for L in viewer.layers]:
        pts = viewer.layers[layer_name]
        pts.data = coords
        pts.text = text_cfg
        try:
            # Ensure internal caches match the new data/text lengths.
            pts.text.apply(pts.features)
        except Exception:
            pass
    else:
        viewer.add_points(
            coords,
            name=layer_name,
            size=6,
            face_color="transparent",
            text={**text_cfg, "color": "yellow", "size": 10},
            visible=False,
        )

# ===================== BATCH MODE (mirrors per-sample folders) =====================
@magicgui(
    layout="vertical",
    auto_call=False,  # avoid auto-trigger on value change
    call_button="Run Segmentation",
    model_path={"label": "Cellpose model (cyto2 or path)", "visible": False},
    folder={"label": "", "visible": False},
    laminin_substring={"label": "Background stain name(s) (comma separated)"},
    stain_names={"label": "Stains (comma separated)"},
    pixel_size_um={"label": "Pixel size (µm/px)", "visible": False},
    diameter_px={"label": "Diameter (px ~ fiber width)"},
    flow_thresh={"label": "Flow threshold"},
    cellprob_thresh={"label": "Cellprob threshold", "min": -10.0, "max": 10.0},
    drop_edge_touching={"label": "Remove edge-touching cells", "visible": False},
    drop_edge_buffer_px={"label": "Edge buffer (px)", "min": 0, "max": 100, "visible": False},
    min_area_px={"label": "Min area to keep (px)"},
    bg_mode={"choices": ["Center (intracellular)"], "label": "Background mode", "visible": False},
    bg_percentile={"label": "Percentile (p)", "min": 0.0, "max": 100.0, "visible": False},
    local_ring_px={"label": "Local ring width (px)", "min": 1, "max": 50, "visible": False},
    rolling_ball_radius_px={"label": "Rolling-ball radius (px)", "min": 5, "max": 500, "visible": False},
    chunk_first={"label": "Chunk segmentation first", "tooltip": "Run tiled/chunked segmentation upfront", "visible": False},
    chunk_count={"label": "Chunks (per dimension)", "min": 1, "max": 16, "visible": False},
    add_texture={"label": "Add entropy feature", "visible": False},
    add_spatial={"label": "Add NN distance", "visible": False},
    save_dir={"label": "Export results under…", "mode": "d"},
    export_all_images={"label": "Export mask/overlay/planes", "visible": False},
    reset_cache={"visible": False},
    show_in_viewer={"visible": False},
)
def batch_widget(
    viewer: "napari.Viewer",
    model_path: str = DEFAULT_MODEL,
    folder: Path = Path.home() / "musclequant" / "input_raw",
    stain_names: str = "",
    laminin_substring: str = "",
    pixel_size_um: float = DEFAULT_PX_UM,
    diameter_px: float = DEFAULT_DIAMETER,
    flow_thresh: float = FLOW_THRESH,
    cellprob_thresh: float = CELLPROB_THRESH,
    drop_edge_touching: bool = True,
    drop_edge_buffer_px: int = 3,
    min_area_px: int = 0,
    bg_mode: str = "Center (intracellular)",
    bg_percentile: float = 5.0,
    local_ring_px: int = 5,
    rolling_ball_radius_px: int = 50,
    chunk_first: bool = False,
    chunk_count: int = CHUNK_COUNT_DEFAULT,
    add_texture: bool = True,
    add_spatial: bool = True,
    save_dir: Path = Path("."),
    export_all_images: bool = False,
    reset_cache: bool = True,
    show_in_viewer: bool = True,
):
    _activate_viewer_state(viewer)
    if segmentation._SEGMENT_RUNNING:
        show_info("Segmentation already running; please wait for it to finish.")
        return
    segmentation._SEGMENT_RUNNING = True
    segmentation._SEGMENT_RUN_COUNTER += 1
    run_id = segmentation._SEGMENT_RUN_COUNTER
    start_ts = time.strftime("%H:%M:%S")
    print(f"[SEGMENT] Run {run_id} start @ {start_ts} | folder={folder} | stains='{stain_names}' | background='{laminin_substring}'")
    try:
        global SEGMENT_EXPORT_ROOT
        if reset_cache:
            SEGMENTED_CACHE.clear()
            SEGMENT_EXPORT_ROOT = None

        folder = Path(folder)
        CURRENT["last_preview_folder"] = str(folder)
        files = sorted([p for p in folder.glob("*.tif*")])

        save_root_candidate = Path(save_dir).expanduser() if isinstance(save_dir, (str, Path)) else Path(".")
        save_root_str = str(save_root_candidate).strip()
        default_export_root = save_root_candidate if save_root_str not in ("", ".") else None
        SEGMENT_EXPORT_ROOT = default_export_root

        # Map BG selection
        _bg_mode = normalize_bg_mode(bg_mode)
        # Parse stain names list (user-provided) once
        _stain_list = _parse_stain_list(stain_names)
        CURRENT["stain_filter"] = _stain_list
        lam_keys = [s.strip().lower() for s in str(laminin_substring).split(",") if s.strip()]
        viewer_sample = None
        if not files:
            viewer_sample = _sample_from_viewer_layers(
                viewer,
                lam_keys,
                _stain_list,
                default_px_um=float(pixel_size_um),
            )
            if viewer_sample:
                CURRENT["mode_source"] = "import"

        # Group by base (before _RGB_… or before "__")
        def split_base(p: Path):
            # Handle composite files like *_RGB.tif and channel files like *_RGB_FITC.tif
            m = re.match(r"^(.*)_RGB(?:_[^_]+)?\.tif[f]?$", p.name, flags=re.IGNORECASE)
            if not m and "__" in p.stem:
                head = p.stem.split("__")[0]
                m = re.match(r"^(.*)_RGB(?:_[^_]+)?$", head, flags=re.IGNORECASE)
            return m.group(1) if m else p.stem

        groups: Dict[str, Any] = {}
        if viewer_sample:
            groups[viewer_sample["base"]] = viewer_sample
        for p in files:
            groups.setdefault(split_base(p), []).append(p)

        model = load_model(pretrained=model_path)

        processed = 0

        for base, plist in groups.items():
            viewer_driven = isinstance(plist, dict) and plist.get("_source") == "viewer"
            sample_files: List[Path] = []
            if viewer_driven:
                sample_files = plist.get("sample_files", [])
                lam_files = plist.get("lam_files", [])
            else:
                sample_files = sorted(plist)
                if lam_keys:
                    lam_files = [p for p in sample_files if any(k in p.name.lower() for k in lam_keys)]
                else:
                    lam_files = [p for p in sample_files if "lam" in p.name.lower() or "background" in p.name.lower()]

            requested_mode = str(CURRENT.get("mode", "IHC")).upper()
            use_he_mode = requested_mode == "H&E"

            he_payload: Dict[str, np.ndarray] = {}
            he_mode = False
            lam_used_list: List[str] = []
            lam_path: Optional[Path] = None
            lam_gray: Optional[np.ndarray] = None
            lam_raw: Optional[np.ndarray] = None
            virtual_payload: Optional[Dict[str, np.ndarray]] = None
            preview_override: Dict[str, str] = {}
            proteins_loaded: List[Tuple[str, str, np.ndarray, Optional[Path]]] = []
            pixel_size_um_local = float(pixel_size_um)

            if viewer_driven:
                base = plist.get("base", base)
                lam_gray = plist.get("lam_gray")
                lam_raw = plist.get("lam_raw", lam_gray)
                lam_used_list = plist.get("lam_used_list") or ["Gray"]
                lam_path = plist.get("lam_path")
                virtual_payload = plist.get("virtual_payload", {})
                preview_override = plist.get("preview_source", {})
                proteins_loaded = plist.get("proteins_loaded", [])
                pixel_size_um_local = float(plist.get("pixel_size_um", pixel_size_um))
                he_mode = bool(plist.get("he_mode", False))
                use_he_mode = he_mode
            elif lam_files and not use_he_mode:
                lam_gray, lam_raw, lam_used_list = _combine_laminin_planes(lam_files, return_raw=True)
                lam_path = lam_files[0]
            elif use_he_mode:
                he_candidate = _separate_he_channels(
                    sample_files[0],
                    include_dab=False,
                    quant_mode=_he_quant_mode(),
                    dmax=_he_dmax(),
                )
                if he_candidate and "Hematoxylin" in he_candidate:
                    he_mode = True
                    he_payload = he_candidate
                    he_raw = he_candidate.get("_raw", {})
                    lam_gray = he_candidate["Hematoxylin"]
                    lam_raw = he_raw.get("Hematoxylin", lam_gray)
                    lam_used_list = ["Hematoxylin (H&E)"]
                    lam_path = sample_files[0]
                    lam_files = [sample_files[0]]
                    if requested_mode == "H&E":
                        show_info(f"{base}: H&E mode enabled — separating Hematoxylin/Eosin.")
                    else:
                        show_info(f"{base}: detected H&E image — separating Hematoxylin/Eosin.")
                else:
                    show_warning(f"{base}: H&E mode selected but stain separation failed; skipping sample.")
                    continue
            else:
                show_warning(f"{base}: no background files found; skipping sample.")
                continue

            if lam_gray is None:
                continue
            lam_used = ",".join(lam_used_list) if lam_used_list else "Gray"

            base_out = base.replace("/", "_")

            # segment
            masks_raw = run_cellpose(
                lam_gray, model,
                diameter=None if diameter_px in (None, 0) else float(diameter_px),
                flow_thr=float(flow_thresh),
                cellprob_thr=float(cellprob_thresh),
                chunk_first=bool(chunk_first),
                chunk_count=int(chunk_count),
            )

            # clean
            cleaned = False
            masks = masks_raw
            if drop_edge_touching or (min_area_px and int(min_area_px) > 0):
                masks, _ = clean_mask(
                    masks_raw,
                    min_area_px=int(min_area_px) if min_area_px else 0,
                    drop_border=bool(drop_edge_touching),
                    edge_buffer_px=int(drop_edge_buffer_px) if drop_edge_buffer_px else 0,
                )
                cleaned = True
            suffix = "_clean" if cleaned else ""

            if he_mode:
                proteins_loaded = []
                he_raw = he_payload.get("_raw", {})
                for chan_name in ("Hematoxylin", "Eosin", "DAB"):
                    arr = he_payload.get(chan_name)
                    if arr is None:
                        continue
                    plane_tag = chan_name[0].upper()
                    proteins_loaded.append((chan_name, plane_tag, arr, None))
                virtual_payload = he_raw
                if not proteins_loaded:
                    continue
            elif viewer_driven:
                if not proteins_loaded:
                    fallback_name = lam_path.name if isinstance(lam_path, Path) else (str(lam_path) if lam_path else "Background")
                    proteins_loaded = [(fallback_name, lam_used_list[0] if lam_used_list else "Gray", lam_gray, lam_path)]
                    virtual_payload = virtual_payload or {}
                    virtual_payload[fallback_name] = lam_raw
            else:
                proteins_loaded = select_protein_previews(sample_files, lam_path, _stain_list)
                if not proteins_loaded:
                    continue

            lam_name = lam_path.name if isinstance(lam_path, Path) else ("<virtual>" if lam_path is None else str(lam_path))
            print(f"[DEBUG] Sample {base}")
            print(f"  Background: {lam_name} plane={lam_used}")
            for pname, used_plane, _disp, psrc in proteins_loaded:
                src_name = Path(psrc).name if psrc else "virtual"
                print(f"  Protein: name={pname} plane={used_plane} src={src_name}")

            preview_source = {} if he_mode else dict(CURRENT.get("preview_planes", {}))
            if not he_mode and preview_override:
                preview_source.update(preview_override)
            virtual_payload = virtual_payload if not he_mode else (he_payload.get("_raw", {}) if he_payload else {})
            protein_channels, preview_map = _build_protein_channels(
                base,
                proteins_loaded,
                preview_source,
                virtual_payload,
            )
            CURRENT["preview_planes"] = preview_map
            metrics_sel = _sanitize_metrics(CURRENT.get("metrics"))

            labels_name = "Fibers_mask_clean" if cleaned else "Fibers_mask"
            if show_in_viewer and viewer is not None:
                viewer.add_labels(masks.astype(np.uint16), name=f"{base} — {labels_name}", visible=True).contour = True
                _add_cell_ids_layer(viewer, masks.astype(np.uint16), layer_name="Cell IDs")
                viewer.add_image(overlay_boundaries(lam_gray, masks), name=f"{base} — Overlay", blending="translucent_no_depth", visible=True, opacity=0.33)

            # annotate & save CSV + metadata
            meta = {
                "base": base,
                "base_out": base_out,
                "export_dir": None,
                "export_root": str(default_export_root) if default_export_root else None,
                "pixel_size_um": float(pixel_size_um_local),
                "diameter_px": None if diameter_px in (None, 0) else float(diameter_px),
                "flow_threshold": float(flow_thresh),
                "cellprob_threshold": float(cellprob_thresh),
                "drop_edge_touching": bool(drop_edge_touching),
                "drop_edge_buffer_px": int(drop_edge_buffer_px),
                "min_area_px": int(min_area_px) if min_area_px else 0,
                "cleaned": cleaned,
                "labels": int(masks.max()),
                "background_stain": lam_path.name if isinstance(lam_path, Path) else (str(lam_path) if lam_path else None),
                "background_plane": lam_used,
                "background_files": [p.name if isinstance(p, Path) else str(p) for p in lam_files],
                "background_planes": lam_used_list,
                "background_mode": _bg_mode,
                "background_percentile": float(bg_percentile) if _bg_mode == "percentile" else None,
                "local_ring_px": int(local_ring_px) if _bg_mode == "local" else None,
                "rolling_ball_radius_px": int(rolling_ball_radius_px) if _bg_mode == "rolling_ball" else None,
                "stain_filter": _stain_list,
            "add_texture": bool(add_texture),
            "add_spatial": bool(add_spatial),
                "chunk_first": bool(chunk_first),
                "chunk_count": int(chunk_count),
                "proteins": [
                    {
                        "name": ch.name,
                        "plane": ch.plane,
                        "file": str(ch.path) if ch.path else None,
                        "virtual": ch.path is None,
                    }
                    for ch in protein_channels
                ],
                "he_mode": bool(he_mode),
                "quantified": False,
                "metrics": metrics_sel,
                "mode": CURRENT.get("mode", "IHC"),
                "export_all_images": bool(export_all_images),
            }
            proteins_for_cache = list(protein_channels)
            preview_map_cache = dict(preview_map)
            SEGMENTED_CACHE.append({
                "base": base,
                "base_out": base_out,
                "export_dir": None,
                "export_root": default_export_root,
                "source_folder": str(folder),
                "lam_path": lam_path,
                "lam_paths": [Path(p) if not isinstance(p, Path) else p for p in lam_files],
                "lam": lam_gray,
                "lam_raw": lam_raw,
                "lam_used": lam_used,
                "masks": masks.astype(np.uint16),
                "cleaned": cleaned,
                "proteins": proteins_for_cache,
                "px_um": float(pixel_size_um_local),
                "preview_planes": preview_map_cache,
                "suffix": suffix,
                "metadata": meta,
                "suggested_folder": base,
                "drop_edge_touching": bool(drop_edge_touching),
                "drop_edge_buffer_px": int(drop_edge_buffer_px),
                "metrics": metrics_sel,
                "mode": CURRENT.get("mode", "IHC"),
                "export_all_images": bool(export_all_images),
            })

            # Store state for manual → keep same folder + bg settings
            if show_in_viewer:
                CURRENT["base"] = base
                CURRENT["lam"] = lam_gray
                CURRENT["lam_raw"] = lam_raw
                CURRENT["lam_paths"] = [p for p in lam_files]
                CURRENT["lam_used_list"] = lam_used_list
                CURRENT["masks"] = masks.astype(np.uint16)
                CURRENT["proteins"] = list(protein_channels)
                CURRENT["preview_planes"] = preview_map
                CURRENT["px_um"] = float(pixel_size_um_local)
                CURRENT["save_dir"] = None
                CURRENT["cleaned"] = cleaned
                CURRENT["bg_mode"] = _bg_mode
                CURRENT["bg_percentile"] = float(bg_percentile)
                CURRENT["local_ring_px"] = int(local_ring_px)
                CURRENT["rolling_ball_radius_px"] = int(rolling_ball_radius_px)
                CURRENT["add_texture"] = bool(add_texture)
                CURRENT["add_spatial"] = bool(add_spatial)
                CURRENT["stain_filter"] = _stain_list
                CURRENT["chunk_first"] = bool(chunk_first)
                CURRENT["chunk_count"] = int(chunk_count)
                CURRENT["metrics"] = metrics_sel
                CURRENT["drop_edge_touching"] = bool(drop_edge_touching)
                CURRENT["drop_edge_buffer_px"] = int(drop_edge_buffer_px)
                CURRENT["export_all_images"] = bool(export_all_images)

            processed += 1

        if processed:
            mq_info(
                None,
                "MuscleQuant (Segmentation)",
                f"Segmented {processed} sample(s).\nClick 'Run Quantification' to export results.",
            )
        else:
            mq_info(None, "MuscleQuant (Segmentation)", "No valid Background+protein groups found.")

    except Exception as e:
        mq_critical(None, "MuscleQuant — Error", str(e))
        raise
    finally:
        end_ts = time.strftime("%H:%M:%S")
        print(f"[SEGMENT] Run {run_id} end   @ {end_ts}")
        segmentation._SEGMENT_RUNNING = False


def _parse_bg_or_keys(raw: str) -> List[str]:
    if raw is None:
        return []
    parts = re.split(r"\s*(?:\bor\b|\|)\s*", str(raw), flags=re.IGNORECASE)
    keys: List[str] = []
    for part in parts:
        for token in str(part).split(","):
            token = token.strip().lower()
            if token:
                keys.append(token)
    return keys


def _guess_background_files(files: List[Path]) -> List[Path]:
    if not files:
        return []
    priority = ["cy5", "txred", "texas", "fitc", "dapi"]
    remaining: List[Path] = list(files)
    for key in priority:
        matches = [p for p in remaining if key in p.stem.lower()]
        if matches:
            return matches
    return [remaining[0]]


def _select_multiple_folders(parent=None, title: str = "Select folders") -> List[Path]:
    start_dir = str(CURRENT.get("last_preview_folder") or Path.home())
    dialog = QFileDialog(parent, title, start_dir)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.Directory)
    for view in dialog.findChildren(QListView) + dialog.findChildren(QTreeView):
        try:
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        except Exception:
            pass
    if dialog.exec():
        return [Path(p) for p in dialog.selectedFiles() if p]
    return []


def batch_segment_folders(viewer: Optional["napari.Viewer"]) -> None:
    v = _resolve_viewer(None, viewer)
    _activate_viewer_state(v)
    if segmentation._SEGMENT_RUNNING:
        show_info("Segmentation already running; please wait for it to finish.")
        return
    roots = _select_multiple_folders()
    folders: List[Path] = []
    for root in roots:
        if root.is_dir():
            folders.append(root)
            for sub in root.rglob("*"):
                if sub.is_dir():
                    folders.append(sub)
    # Keep only folders that contain TIFFs, de-duplicate while preserving order.
    seen: Set[Path] = set()
    valid_folders: List[Path] = []
    for folder in folders:
        if folder in seen:
            continue
        seen.add(folder)
        if any(folder.glob("*.tif*")):
            valid_folders.append(folder)
    folders = valid_folders
    if not folders:
        return

    default_bg = ""
    try:
        default_bg = str(batch_widget.laminin_substring.value)
    except Exception:
        default_bg = ""
    bg_text, ok = QInputDialog.getText(
        None,
        "Background stain name(s)",
        "Enter background stain name(s). Use OR or | for alternatives:",
        text=default_bg or "Background",
    )
    if not ok:
        return
    bg_keys = _parse_bg_or_keys(bg_text) or ["background"]

    try:
        model_path = batch_widget.model_path.value
        diameter_px = float(batch_widget.diameter_px.value)
        flow_thresh = float(batch_widget.flow_thresh.value)
        cellprob_thresh = float(batch_widget.cellprob_thresh.value)
        drop_edge_touching = bool(batch_widget.drop_edge_touching.value)
        drop_edge_buffer_px = int(batch_widget.drop_edge_buffer_px.value)
        min_area_px = int(batch_widget.min_area_px.value) if batch_widget.min_area_px.value else 0
        chunk_first = bool(batch_widget.chunk_first.value)
        chunk_count = int(batch_widget.chunk_count.value)
    except Exception as exc:
        mq_warning(None, "Batch segmentation", f"Could not read segmentation settings: {exc}")
        return

    segmentation._SEGMENT_RUNNING = True
    segmentation._SEGMENT_RUN_COUNTER += 1
    run_id = segmentation._SEGMENT_RUN_COUNTER
    start_ts = time.strftime("%H:%M:%S")
    print(f"[SEGMENT] Multi-folder run {run_id} start @ {start_ts}")
    processed = 0
    skipped = 0
    try:
        model = load_model(pretrained=model_path)

        def _base_from_path(p: Path) -> str:
            stem = p.stem
            if "__" in stem:
                stem = stem.split("__")[0]
            m = re.match(r"^(.*)_RGB(?:_[^_]+)?$", stem, flags=re.IGNORECASE)
            return m.group(1) if m else stem

        with progress(total=len(folders)) as pbar:
            for idx_folder, folder in enumerate(folders, start=1):
                try:
                    pbar.set_description(f"Segmenting {folder.name} ({idx_folder}/{len(folders)})")
                except Exception:
                    pass
                files = sorted([p for p in folder.glob("*.tif*")])
                if not files:
                    show_warning(f"No TIFF files found in {folder}.")
                    skipped += 1
                    pbar.update(1)
                    continue
                lam_files: List[Path] = []
                for key in bg_keys:
                    matches = [p for p in files if key in p.name.lower()]
                    if matches:
                        lam_files = matches
                        break
                if not lam_files:
                    show_warning(f"No background stain match in {folder} for keys: {', '.join(bg_keys)}")
                    skipped += 1
                    pbar.update(1)
                    continue

                try:
                    lam_gray, _lam_raw, _lam_used_list = _combine_laminin_planes(lam_files, return_raw=True)
                except Exception as exc:
                    show_warning(f"{folder}: failed to load background stain ({exc})")
                    skipped += 1
                    pbar.update(1)
                    continue

                masks_raw = run_cellpose(
                    lam_gray,
                    model,
                    diameter=None if diameter_px in (None, 0) else float(diameter_px),
                    flow_thr=float(flow_thresh),
                    cellprob_thr=float(cellprob_thresh),
                    chunk_first=bool(chunk_first),
                    chunk_count=int(chunk_count),
                )

                cleaned = False
                masks = masks_raw
                if drop_edge_touching or (min_area_px and int(min_area_px) > 0):
                    masks, _ = clean_mask(
                        masks_raw,
                        min_area_px=int(min_area_px) if min_area_px else 0,
                        drop_border=bool(drop_edge_touching),
                        edge_buffer_px=int(drop_edge_buffer_px) if drop_edge_buffer_px else 0,
                    )
                    cleaned = True

                base = _base_from_path(lam_files[0]) if lam_files else folder.name
                labels_name = "Fibers_mask_clean" if cleaned else "Fibers_mask"
                out_path = folder / f"{base}_{labels_name}.tif"
                if out_path.exists():
                    idx = 2
                    while True:
                        candidate = folder / f"{base}_{labels_name}_{idx}.tif"
                        if not candidate.exists():
                            out_path = candidate
                            break
                        idx += 1
                imwrite(out_path, masks.astype(np.uint16))
                processed += 1
                pbar.update(1)

        mq_info(
            None,
            "Batch segmentation",
            f"Processed {processed} folder(s), skipped {skipped}.\nMasks saved into each folder.",
        )
    finally:
        end_ts = time.strftime("%H:%M:%S")
        print(f"[SEGMENT] Multi-folder run {run_id} end   @ {end_ts}")
        segmentation._SEGMENT_RUNNING = False


def _collect_batch_folders(roots: List[Path]) -> List[Path]:
    folders: List[Path] = []
    for root in roots:
        if root.is_dir():
            folders.append(root)
            for sub in root.rglob("*"):
                if sub.is_dir():
                    folders.append(sub)
    seen: Set[Path] = set()
    valid: List[Path] = []
    for folder in folders:
        if folder in seen:
            continue
        seen.add(folder)
        if any(folder.glob("*.tif*")):
            valid.append(folder)
    return valid


def _update_batch_folder_label():
    if _BATCH_FOLDER_LABEL is None:
        return
    count = len(_BATCH_FOLDERS)
    if count == 0:
        _BATCH_FOLDER_LABEL.setText("No folders")
    elif count == 1:
        _BATCH_FOLDER_LABEL.setText("1 folder")
    else:
        _BATCH_FOLDER_LABEL.setText(f"{count} folders")


def _truncate_folder_label(path: Path) -> str:
    name = path.name
    if len(name) <= _FOLDER_LABEL_MAX:
        return name
    return name[: max(0, _FOLDER_LABEL_MAX - 1)] + "…"


def _update_single_folder_label(path: Path):
    if _SINGLE_FOLDER_LABEL is None:
        return
    _SINGLE_FOLDER_LABEL.setText(_truncate_folder_label(path))


def _select_single_folder(viewer: Optional["napari.Viewer"]) -> None:
    v = _resolve_viewer(None, viewer)
    if v is None:
        show_info("No viewer available.")
        return
    _clear_batch_verify_overlay(v)
    start_dir = str(CURRENT.get("last_preview_folder") or Path.home())
    folder = QFileDialog.getExistingDirectory(None, "Select folder with images", start_dir)
    if not folder:
        return
    folder_path = Path(folder)
    # If the folder looks like a saved run (mask/metadata), prefer saved-run import.
    has_meta = (folder_path / "metadata.json").exists()
    has_mask = any(re.search(r"mask", p.stem, re.IGNORECASE) for p in folder_path.glob("*.tif*"))
    if has_meta or has_mask:
        if _import_saved_run_from_folder(v, folder_path):
            _update_single_folder_label(folder_path)
            return
    _update_single_folder_label(folder_path)
    folder_preview(v, folder=folder_path)


def _choose_batch_verify_samples(folders: List[Path]) -> Optional[Set[Path]]:
    if not folders:
        return set()
    dialog = QDialog(None)
    dialog.setWindowTitle("Select samples to verify")
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    layout.addWidget(QLabel("Choose which samples to verify during this batch run:"))

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    list_widget = QWidget()
    list_layout = QVBoxLayout(list_widget)
    list_layout.setContentsMargins(4, 4, 4, 4)
    list_layout.setSpacing(4)

    checks: List[Tuple[Path, QCheckBox]] = []
    for folder in folders:
        cb = QCheckBox(folder.name)
        cb.setChecked(True)
        list_layout.addWidget(cb)
        checks.append((folder, cb))
    list_layout.addStretch(1)
    scroll.setWidget(list_widget)
    layout.addWidget(scroll, stretch=1)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    btn_cancel = QPushButton("Cancel")
    btn_ok = QPushButton("OK")
    btn_row.addWidget(btn_cancel)
    btn_row.addWidget(btn_ok)
    layout.addLayout(btn_row)

    btn_ok.clicked.connect(dialog.accept)
    btn_cancel.clicked.connect(dialog.reject)

    result = dialog.exec()
    if result != QDialog.Accepted:
        return None
    selected: Set[Path] = set()
    for folder, cb in checks:
        if cb.isChecked():
            selected.add(folder)
    return selected


def _ensure_batch_verify_overlay(viewer: "napari.Viewer") -> Optional[QLabel]:
    if viewer is None:
        return None
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if qt_viewer is None:
        return None
    canvas = getattr(qt_viewer, "canvas", None)
    anchor = getattr(canvas, "native", None)
    if anchor is None:
        return None
    label = getattr(viewer, "_mq_batch_verify_overlay", None)
    if label is not None:
        return label
    label = QLabel(anchor)
    label.setObjectName("BatchVerifyOverlay")
    label.setText("")
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    label.setStyleSheet(
        "QLabel#BatchVerifyOverlay {"
        "background-color: rgba(0, 0, 0, 140);"
        "color: white;"
        "padding: 8px 10px;"
        "border-radius: 6px;"
        "font-size: 11px;"
        "}"
    )
    label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    label.adjustSize()
    label.hide()
    try:
        viewer._mq_batch_verify_overlay = label
    except Exception:
        pass
    return label


def _ensure_batch_verify_widget(viewer: "napari.Viewer") -> Optional[QWidget]:
    if viewer is None:
        return None
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if qt_viewer is None:
        return None
    canvas = getattr(qt_viewer, "canvas", None)
    anchor = getattr(canvas, "native", None)
    if anchor is None:
        return None
    widget = getattr(viewer, "_mq_batch_verify_widget", None)
    if widget is not None:
        return widget
    widget = QWidget(anchor)
    widget.setObjectName("BatchVerifyWidget")
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(6, 4, 6, 4)
    layout.setSpacing(6)
    btn_pause = QPushButton("Pause")
    btn_continue = QPushButton("Continue batch")
    btn_cancel = QPushButton("Cancel batch")
    btn_pause.clicked.connect(_batch_verify_pause)
    btn_continue.clicked.connect(_batch_verify_continue)
    btn_cancel.clicked.connect(_batch_verify_cancel)
    layout.addWidget(btn_pause)
    layout.addWidget(btn_continue)
    layout.addWidget(btn_cancel)
    widget.setStyleSheet(
        "QWidget#BatchVerifyWidget {"
        "background-color: rgba(0, 0, 0, 140);"
        "color: white;"
        "border-radius: 6px;"
        "}"
    )
    widget.hide()
    try:
        viewer._mq_batch_verify_widget = widget
        viewer._mq_batch_verify_pause_btn = btn_pause
    except Exception:
        pass
    return widget


def _update_batch_verify_overlay(viewer: "napari.Viewer", text: str) -> None:
    label = _ensure_batch_verify_overlay(viewer)
    if label is None:
        return
    if text:
        label.setText(text)
        label.adjustSize()
        label.show()
        rect = label.parentWidget().rect()
        margin = 12
        x = max(margin, rect.width() - label.width() - margin)
        y = margin
        label.move(x, y)
        widget = _ensure_batch_verify_widget(viewer)
        if widget is not None:
            widget.adjustSize()
            widget.show()
            wx = max(margin, rect.width() - widget.width() - margin)
            wy = y + label.height() + 6
            widget.move(wx, wy)
    else:
        label.setText("")
        label.hide()
        widget = _ensure_batch_verify_widget(viewer)
        if widget is not None:
            widget.hide()


def _clear_batch_verify_overlay(viewer: Optional["napari.Viewer"]) -> None:
    if viewer is None:
        return
    try:
        _update_batch_verify_overlay(viewer, "")
    except Exception:
        pass


def _build_batch_verify_text(masks: np.ndarray, membrane_labels: np.ndarray, channels: List[ProteinChannel], preview_map: Dict[str, str]) -> str:
    labels = np.unique(masks.astype(np.int64, copy=False))
    labels = labels[labels > 0]
    areas = np.bincount(masks.ravel())[1:]
    avg_area = float(np.mean(areas)) if areas.size else 0.0
    mem_mask = membrane_labels > 0
    lines = [f"avg area: {avg_area:.1f} px"]
    for ch in channels:
        p_raw = _load_channel_raw(ch, preview_map)
        if p_raw is None or p_raw.shape != masks.shape:
            continue
        mem_mean = float(np.mean(p_raw[mem_mask])) if np.any(mem_mask) else 0.0
        lines.append(f"{ch.name} mem mean: {mem_mean:.4g}")
    return "\n".join(lines)


def _run_batch_verification(
    viewer: Optional["napari.Viewer"],
    folder: Path,
    bg_keys: List[str],
    stain_names: str,
    pixel_size_um: float,
) -> bool:
    global _BATCH_VERIFY_PAUSED
    _BATCH_VERIFY_PAUSED = False
    v = _resolve_viewer(None, viewer)
    if v is None:
        show_info("No viewer available.")
        return False
    sample = _load_saved_run_for_batch(folder, bg_keys, stain_names, pixel_size_um)
    if not sample:
        return False
    if not sample.get("files"):
        show_warning("Verification skipped: no raw images available to load into viewer.")
        return False
    mask_path = sample.get("mask_path")
    if mask_path is None or not Path(mask_path).exists():
        show_warning("Verification requires existing masks for quantification.")
        return False
    ok = _load_sample_into_viewer(v, sample, Path(mask_path), stain_names)
    if not ok:
        return False
    mem_px = int(CURRENT.get("membrane_px", CURRENT.get("s1_membrane_px", 4)))
    highlight_membranes(v, mem_px)

    def _finish_verify():
        _set_batch_verify_pause_button(v, False)
        masks = np.asarray(CURRENT.get("masks"), dtype=np.uint16)
        membrane_labels, _ = _compute_membrane_and_interior_masks(masks, mem_px)
        preview_map = CURRENT.get("preview_planes", {}) or {}
        channels = list(_filter_channels_by_viewer(v, CURRENT.get("proteins") or []))
        text = _build_batch_verify_text(masks, membrane_labels, channels, preview_map)
        _update_batch_verify_overlay(v, text)
        QTimer.singleShot(_BATCH_VERIFY_TIMEOUT_MS, _batch_verify_autocontinue)

    QTimer.singleShot(0, _finish_verify)
    return True


def _batch_verify_continue():
    global _BATCH_VERIFY_PENDING, _BATCH_VERIFY_DONE, _BATCH_VERIFY_PAUSED
    _BATCH_VERIFY_PENDING = False
    _BATCH_VERIFY_DONE = True
    _BATCH_VERIFY_PAUSED = False
    v = _ACTIVE_VIEWER
    _update_batch_verify_overlay(v, "")
    show_info("Batch verification continued.")
    run_batch_pipeline(v, start_index=_BATCH_RUN_INDEX)


def _batch_verify_autocontinue():
    if not _BATCH_VERIFY_PENDING or _BATCH_CANCELLED or _BATCH_VERIFY_PAUSED:
        return
    _batch_verify_continue()


def _batch_verify_cancel():
    global _BATCH_VERIFY_PENDING, _BATCH_CANCELLED, _BATCH_RUN_INDEX, _BATCH_VERIFY_PAUSED
    _BATCH_VERIFY_PENDING = False
    _BATCH_CANCELLED = True
    _BATCH_RUN_INDEX = 0
    _BATCH_VERIFY_PAUSED = False
    _update_batch_verify_overlay(_ACTIVE_VIEWER, "")
    show_info("Batch verification canceled.")
    show_info("Batch run canceled.")


def _set_batch_verify_pause_button(viewer: Optional["napari.Viewer"], paused: bool) -> None:
    if viewer is None:
        return
    btn = getattr(viewer, "_mq_batch_verify_pause_btn", None)
    if btn is None:
        return
    try:
        btn.setText("Resume" if paused else "Pause")
    except Exception:
        pass


def _batch_verify_pause():
    global _BATCH_VERIFY_PAUSED
    _BATCH_VERIFY_PAUSED = not _BATCH_VERIFY_PAUSED
    _set_batch_verify_pause_button(_ACTIVE_VIEWER, _BATCH_VERIFY_PAUSED)
    show_info("Batch verification paused." if _BATCH_VERIFY_PAUSED else "Batch verification resumed.")


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    idx = 2
    while True:
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _split_sample_base(p: Path) -> str:
    stem = p.stem
    if "__" in stem:
        stem = stem.split("__")[0]
    m = re.match(r"^(.*)_RGB(?:_[^_]+)?$", stem, flags=re.IGNORECASE)
    return m.group(1) if m else stem


def _find_mask_for_base(folder: Path, base: str) -> Optional[Path]:
    candidates = sorted([p for p in folder.glob("*.tif*") if re.search(r"mask", p.stem, re.IGNORECASE)])
    if not candidates:
        return None
    for p in candidates:
        if base.lower() in p.stem.lower():
            return p
    return candidates[0] if candidates else None


def _build_batch_samples(bg_keys: List[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for folder in _BATCH_FOLDERS:
        files = []
        for p in sorted([pp for pp in folder.glob("*.tif*")]):
            stem_lower = p.stem.lower()
            if "mask" in stem_lower or "overlay" in stem_lower:
                continue
            if "s1_syn" in stem_lower or "s1_mem" in stem_lower or "s1_extra" in stem_lower:
                continue
            files.append(p)
        if not files:
            continue
        groups: Dict[str, List[Path]] = {}
        for p in files:
            groups.setdefault(_split_sample_base(p), []).append(p)
        for base, sample_files in groups.items():
            lam_files: List[Path] = []
            for key in bg_keys:
                matches = [p for p in sample_files if key in p.name.lower()]
                if matches:
                    lam_files = matches
                    break
            if not lam_files:
                lam_files = [p for p in sample_files if "background" in p.name.lower()]
            if not lam_files:
                lam_files = _guess_background_files(sample_files)
            mask_path = _find_mask_for_base(folder, base)
            samples.append({
                "folder": folder,
                "base": base,
                "files": sample_files,
                "lam_files": lam_files,
                "mask_path": mask_path,
            })
    return samples


def _export_synaptic_results_to_folder(viewer: "napari.Viewer", folder: Path) -> Optional[Path]:
    res = CURRENT.get("synaptic_results")
    if res is None:
        run_s1_analysis(viewer, export_results=False)
        res = CURRENT.get("synaptic_results")
    if res is None:
        mq_info(None, "Synaptic analysis", "No synaptic results available to export.")
        return None
    df = res.get("df")
    if df is None or df.empty:
        mq_info(None, "Synaptic analysis", "No synaptic rows to export.")
        return None
    base = res.get("base") or CURRENT.get("base") or "image"
    base_safe = _safe_basename(base)
    out_path = _unique_path(Path(folder) / f"{base_safe}_synaptic_quant.csv")
    df.to_csv(out_path, index=False)
    show_info(f"Synaptic analysis exported.\nCSV: {out_path}")
    return out_path


def _load_sample_into_viewer(viewer: "napari.Viewer", sample: Dict[str, Any], mask_path: Optional[Path], stain_names: str) -> bool:
    if viewer is None:
        return False
    _clear_batch_verify_overlay(viewer)
    _reset_s1_state(viewer)
    try:
        viewer.layers.clear()
    except Exception:
        pass

    sample_files = list(sample.get("files") or [])
    lam_files = list(sample.get("lam_files") or [])
    if not lam_files:
        lam_files = list(sample.get("lam_paths") or [])
    if not lam_files:
        lam_path_fallback = sample.get("lam_path")
        if lam_path_fallback:
            lam_files = [Path(lam_path_fallback)]
    base = sample.get("base") or "sample"
    if not sample_files:
        return False
    _stain_list = _parse_stain_list(stain_names)

    requested_mode = str(CURRENT.get("mode", "IHC")).upper()
    use_he_mode = requested_mode == "H&E"

    he_payload: Dict[str, np.ndarray] = {}
    he_mode = False
    lam_used_list: List[str] = []
    lam_path: Optional[Path] = None
    lam_gray: Optional[np.ndarray] = None
    lam_raw: Optional[np.ndarray] = None
    preview_override: Dict[str, str] = {}
    proteins_loaded: List[Tuple[str, str, np.ndarray, Optional[Path]]] = []
    virtual_payload: Optional[Dict[str, np.ndarray]] = None

    if lam_files and not use_he_mode:
        lam_gray, lam_raw, lam_used_list = _combine_laminin_planes(lam_files, return_raw=True)
        lam_path = lam_files[0]
    elif use_he_mode:
        he_candidate = _separate_he_channels(
            sample_files[0],
            include_dab=False,
            quant_mode=_he_quant_mode(),
            dmax=_he_dmax(),
        )
        if he_candidate and "Hematoxylin" in he_candidate:
            he_mode = True
            he_payload = he_candidate
            he_raw = he_candidate.get("_raw", {})
            lam_gray = he_candidate["Hematoxylin"]
            lam_raw = he_raw.get("Hematoxylin", lam_gray)
            lam_used_list = ["Hematoxylin (H&E)"]
            lam_path = sample_files[0]
            lam_files = [sample_files[0]]
        else:
            show_warning(f"{base}: H&E mode selected but stain separation failed; skipping.")
            return False
    else:
        show_warning(f"{base}: no background stain found; skipping.")
        return False

    if lam_gray is None:
        return False

    if he_mode:
        proteins_loaded = []
        he_raw = he_payload.get("_raw", {})
        for chan_name in ("Hematoxylin", "Eosin", "DAB"):
            arr = he_payload.get(chan_name)
            if arr is None:
                continue
            plane_tag = chan_name[0].upper()
            proteins_loaded.append((chan_name, plane_tag, arr, None))
        virtual_payload = he_raw
        if not proteins_loaded:
            return False
    else:
        proteins_loaded = select_protein_previews(sample_files, lam_path, _stain_list)
        if not proteins_loaded:
            fallback_name = lam_path.name if isinstance(lam_path, Path) else "Background"
            proteins_loaded = [(fallback_name, lam_used_list[0] if lam_used_list else "Gray", lam_gray, lam_path)]

    preview_source = {} if he_mode else dict(CURRENT.get("preview_planes", {}))
    if not he_mode and preview_override:
        preview_source.update(preview_override)
    virtual_payload = virtual_payload if not he_mode else (he_payload.get("_raw", {}) if he_payload else {})
    protein_channels, preview_map = _build_protein_channels(
        base,
        proteins_loaded,
        preview_source,
        virtual_payload,
    )

    mask = None
    if mask_path and mask_path.exists():
        try:
            mask = imread(mask_path)
        except Exception:
            mask = None
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.dtype != np.uint16:
            mask = mask.astype(np.uint16, copy=False)
        if lam_gray is not None and lam_gray.shape != mask.shape:
            try:
                lam_gray = resize(lam_gray, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
                if lam_raw is not None:
                    lam_raw = resize(lam_raw, mask.shape, order=1, preserve_range=True, anti_aliasing=True)
                proteins_loaded = [
                    (name, used_plane, resize(np.asarray(disp), mask.shape, order=1, preserve_range=True, anti_aliasing=True), src)
                    for name, used_plane, disp, src in proteins_loaded
                ]
            except Exception:
                pass

    if he_mode:
        _add_he_layers(viewer, base, he_payload)
    else:
        lam_cmap = _stain_colormap(infer_stain_from_filename(lam_path) if lam_path else "Background", "gray", None)
        viewer.add_image(lam_gray, name=Path(lam_path).stem if lam_path else "Background", colormap=lam_cmap, blending="additive", visible=True)

    color_idx = 0
    stain_colors = he_payload.get("_stain_colors", {}) if he_mode else {}
    for name, used_plane, disp, _src in proteins_loaded:
        if he_mode:
            continue
        default_cmap = COLORMAP_CYCLE[color_idx % len(COLORMAP_CYCLE)]
        base_cmap = _stain_colormap(name, default_cmap, used_plane)
        cmap = _stain_colormap_from_rgb(stain_colors.get(name), base_cmap, transparent_background=he_mode)
        color_idx += 1
        layer = viewer.add_image(
            np.asarray(disp, dtype=np.float32),
            name=name,
            colormap=cmap,
            blending="opaque" if name.lower().startswith(("hematox", "haematox", "eosin")) else "additive",
            visible=False,
        )
        layer.opacity = 0.6
        if name.lower().startswith(("hematox", "haematox", "eosin")):
            layer.opacity = 1.0
            layer.contrast_limits = (0.0, 1.0)
    labels_name = "Fibers_mask_clean" if re.search(r"clean", mask_path.stem, re.IGNORECASE) else "Fibers_mask"
    labels_layer = viewer.add_labels(mask, name=labels_name, visible=True)
    labels_layer.contour = True
    labels_layer.opacity = 0.4
    overlay_layer = viewer.add_image(
        overlay_boundaries(lam_gray, mask),
        name=f"{base} — Overlay",
        blending="translucent_no_depth",
        visible=True,
        opacity=0.33,
    )
    _add_cell_ids_layer(viewer, mask)
    try:
        viewer.layers.move(viewer.layers.index(overlay_layer), len(viewer.layers) - 1)
    except Exception:
        pass
    try:
        viewer.layers.move(viewer.layers.index(labels_layer), len(viewer.layers) - 1)
    except Exception:
        pass

    CURRENT["base"] = base
    CURRENT["lam"] = lam_gray
    CURRENT["lam_raw"] = lam_raw
    CURRENT["lam_paths"] = [p for p in lam_files]
    CURRENT["lam_used_list"] = lam_used_list
    CURRENT["masks"] = mask.astype(np.uint16) if mask is not None else None
    CURRENT["proteins"] = list(protein_channels)
    CURRENT["preview_planes"] = preview_map
    save_dir = sample.get("folder") or sample.get("export_root") or sample.get("source_folder")
    if save_dir is not None:
        CURRENT["save_dir"] = Path(save_dir)
    try:
        px_um = _infer_pixel_size_um_from_tiff(lam_path) if lam_path else None
    except Exception:
        px_um = None
    CURRENT["px_um"] = float(px_um) if px_um else float(getattr(batch_widget.pixel_size_um, "value", DEFAULT_PX_UM))
    CURRENT["mode_source"] = "import"
    return True


def _select_batch_folders() -> None:
    global _BATCH_FOLDERS
    roots = _select_multiple_folders()
    if not roots:
        return
    try:
        CURRENT["last_preview_folder"] = str(roots[0])
    except Exception:
        pass
    _BATCH_FOLDERS = _collect_batch_folders(roots)
    _update_batch_folder_label()
    global _BATCH_VERIFY_DONE
    _BATCH_VERIFY_DONE = False
    global _BATCH_VERIFY_PENDING, _BATCH_CANCELLED
    _BATCH_VERIFY_PENDING = False
    _BATCH_CANCELLED = False
    # Do not initialize synaptic batches on folder selection; only on Run batch.


def _save_masks_from_cache(folder: Path) -> int:
    saved = 0
    for sample in SEGMENTED_CACHE:
        base_out = sample.get("base_out") or sample.get("base") or folder.name
        cleaned = bool(sample.get("cleaned", False))
        labels_name = "Fibers_mask_clean" if cleaned else "Fibers_mask"
        masks = np.asarray(sample.get("masks"), dtype=np.uint16)
        out_path = Path(folder) / f"{base_out}_{labels_name}.tif"
        if out_path.exists():
            idx = 2
            while True:
                candidate = Path(folder) / f"{base_out}_{labels_name}_{idx}.tif"
                if not candidate.exists():
                    out_path = candidate
                    break
                idx += 1
        imwrite(out_path, masks.astype(np.uint16))
        saved += 1
    return saved


def _segment_sample_for_synaptic(sample: Dict[str, Any]) -> Optional[Path]:
    folder = Path(sample.get("folder"))
    lam_files = list(sample.get("lam_files") or [])
    if not lam_files:
        return None
    try:
        model_path = batch_widget.model_path.value
        diameter_px = float(batch_widget.diameter_px.value)
        flow_thresh = float(batch_widget.flow_thresh.value)
        cellprob_thresh = float(batch_widget.cellprob_thresh.value)
        drop_edge_touching = bool(batch_widget.drop_edge_touching.value)
        drop_edge_buffer_px = int(batch_widget.drop_edge_buffer_px.value)
        min_area_px = int(batch_widget.min_area_px.value) if batch_widget.min_area_px.value else 0
        chunk_first = bool(batch_widget.chunk_first.value)
        chunk_count = int(batch_widget.chunk_count.value)
    except Exception:
        return None
    try:
        lam_gray, _lam_raw, _lam_used_list = _combine_laminin_planes(lam_files, return_raw=True)
    except Exception:
        return None
    model = load_model(pretrained=model_path)
    masks_raw = run_cellpose(
        lam_gray,
        model,
        diameter=None if diameter_px in (None, 0) else float(diameter_px),
        flow_thr=float(flow_thresh),
        cellprob_thr=float(cellprob_thresh),
        chunk_first=bool(chunk_first),
        chunk_count=int(chunk_count),
    )
    cleaned = False
    masks = masks_raw
    if drop_edge_touching or (min_area_px and int(min_area_px) > 0):
        masks, _ = clean_mask(
            masks_raw,
            min_area_px=int(min_area_px) if min_area_px else 0,
            drop_border=bool(drop_edge_touching),
            edge_buffer_px=int(drop_edge_buffer_px) if drop_edge_buffer_px else 0,
        )
        cleaned = True
    base = sample.get("base") or folder.name
    labels_name = "Fibers_mask_clean" if cleaned else "Fibers_mask"
    out_path = _unique_path(folder / f"{base}_{labels_name}.tif")
    imwrite(out_path, masks.astype(np.uint16))
    return out_path


def _batch_synaptic_load(index: int, viewer: Optional["napari.Viewer"]) -> None:
    global _BATCH_SYNAPTIC_INDEX
    v = _resolve_viewer(None, viewer)
    if v is None:
        show_info("No viewer available.")
        return
    if not _BATCH_SYNAPTIC_SAMPLES:
        show_info("No batch samples found.")
        return
    if index < 0 or index >= len(_BATCH_SYNAPTIC_SAMPLES):
        return
    sample = _BATCH_SYNAPTIC_SAMPLES[index]
    mask_path = sample.get("mask_path")
    if mask_path is None or not Path(mask_path).exists():
        new_mask = _segment_sample_for_synaptic(sample)
        if new_mask is not None:
            mask_path = new_mask
            sample["mask_path"] = new_mask
    if mask_path is None:
        show_warning(f"{sample.get('base')}: no mask found.")
        return
    ok = _load_sample_into_viewer(v, sample, Path(mask_path), batch_widget.stain_names.value)
    if not ok:
        show_warning(f"{sample.get('base')}: failed to load sample.")
        return
    _BATCH_SYNAPTIC_INDEX = index
    _update_batch_synaptic_progress()
    _update_batch_action_ui(_resolve_viewer(None, viewer))


def _update_batch_synaptic_progress():
    total = len(_BATCH_SYNAPTIC_SAMPLES)
    idx = _BATCH_SYNAPTIC_INDEX + 1 if _BATCH_SYNAPTIC_INDEX >= 0 else 0
    if _BATCH_PROGRESS_LABEL is not None:
        _BATCH_PROGRESS_LABEL.setText(f"{idx}/{total}" if total else "0/0")
    if _BATCH_SAMPLE_LABEL is not None:
        name = "—"
        if 0 <= _BATCH_SYNAPTIC_INDEX < len(_BATCH_SYNAPTIC_SAMPLES):
            sample = _BATCH_SYNAPTIC_SAMPLES[_BATCH_SYNAPTIC_INDEX]
            folder = sample.get("folder")
            try:
                name = Path(folder).name if folder else "—"
            except Exception:
                name = str(folder) if folder else "—"
        _BATCH_SAMPLE_LABEL.setText(f"Sample: {name}")


def _batch_synaptic_confirm_unsaved() -> bool:
    if _BATCH_SYNAPTIC_INDEX in _BATCH_SYNAPTIC_SAVED:
        return True
    if _BATCH_SYNAPTIC_INDEX < 0:
        return True
    res = QMessageBox.question(
        None,
        "Unsaved synaptic results",
        "Current sample has not been saved. Continue without saving?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    return res == QMessageBox.Yes


def _batch_synaptic_prev():
    if not _batch_synaptic_confirm_unsaved():
        return
    if not _BATCH_SYNAPTIC_SAMPLES:
        if not _BATCH_FOLDERS:
            show_info("Select one or more folders first.")
            return
        _init_batch_synaptic_samples()
    if not _BATCH_SYNAPTIC_SAMPLES:
        show_warning("No synaptic samples found.")
        return
    target = max(0, _BATCH_SYNAPTIC_INDEX - 1)
    _batch_synaptic_load(target, _ACTIVE_VIEWER)


def _batch_synaptic_next():
    if not _batch_synaptic_confirm_unsaved():
        return
    if not _BATCH_SYNAPTIC_SAMPLES:
        if not _BATCH_FOLDERS:
            show_info("Select one or more folders first.")
            return
        _init_batch_synaptic_samples()
    if not _BATCH_SYNAPTIC_SAMPLES:
        show_warning("No synaptic samples found.")
        return
    if _BATCH_SYNAPTIC_INDEX < 0:
        target = 0
    else:
        target = min(len(_BATCH_SYNAPTIC_SAMPLES) - 1, _BATCH_SYNAPTIC_INDEX + 1)
    _batch_synaptic_load(target, _ACTIVE_VIEWER)


def _batch_synaptic_save():
    if _BATCH_SYNAPTIC_INDEX < 0 or _BATCH_SYNAPTIC_INDEX >= len(_BATCH_SYNAPTIC_SAMPLES):
        return
    sample = _BATCH_SYNAPTIC_SAMPLES[_BATCH_SYNAPTIC_INDEX]
    folder = Path(sample.get("folder"))
    v = _resolve_viewer(None, _ACTIVE_VIEWER)
    if v is None:
        return
    mask_path = sample.get("mask_path")
    masks = CURRENT.get("masks")
    if masks is None:
        masks = _ensure_masks_from_viewer(v)
    if mask_path and masks is not None:
        try:
            imwrite(Path(mask_path), np.asarray(masks, dtype=np.uint16))
            CURRENT["modified"] = False
            show_info(f"Mask updated: {Path(mask_path).name}")
        except Exception as exc:
            show_warning(f"Failed to update mask ({exc})")
    out = _export_synaptic_results_to_folder(v, folder)
    if out is not None:
        _BATCH_SYNAPTIC_SAVED.add(_BATCH_SYNAPTIC_INDEX)


def _run_batch_synaptic_interactive(
    viewer: Optional["napari.Viewer"],
    folder: Path,
    bg_keys: List[str],
    stain_names: str,
    pixel_size_um: float,
) -> None:
    global _BATCH_SYNAPTIC_SAMPLES, _BATCH_SYNAPTIC_INDEX, _BATCH_SYNAPTIC_SAVED
    v = _resolve_viewer(None, viewer)
    if v is None:
        show_info("No viewer available.")
        return
    _BATCH_SYNAPTIC_SAMPLES = _build_batch_samples(bg_keys)
    _BATCH_SYNAPTIC_SAVED = set()
    _BATCH_SYNAPTIC_INDEX = -1
    targets = [s for s in _BATCH_SYNAPTIC_SAMPLES if Path(s.get("folder")) == Path(folder)]
    if not targets:
        show_warning(f"No synaptic samples found in {folder}.")
        return
    _BATCH_SYNAPTIC_SAMPLES = targets
    _BATCH_SYNAPTIC_INDEX = -1
    _BATCH_SYNAPTIC_SAVED = set()
    _update_batch_synaptic_progress()
    _batch_synaptic_next()


def _update_batch_action_ui(viewer: Optional["napari.Viewer"]) -> None:
    global _BATCH_SYNAPTIC_ACTIVE
    if _BATCH_ACTION_COMBO is None:
        return
    action = str(_BATCH_ACTION_COMBO.currentText())
    is_synaptic = _method_combo_widget is not None and str(_method_combo_widget.currentText()) == SYNAPTIC_METHOD
    lock_active = bool(_BATCH_SYNAPTIC_ACTIVE and _BATCH_SYNAPTIC_INDEX >= 0)
    if _BATCH_RUN_BUTTON is not None:
        _BATCH_RUN_BUTTON.setVisible(True)
    if _BATCH_NAV_WIDGET is not None:
        _BATCH_NAV_WIDGET.setVisible(is_synaptic)
    for w in _single_only_panels:
        try:
            w.setVisible(is_synaptic)
        except Exception:
            pass
    if is_synaptic and _BATCH_NAV_WIDGET is not None and _BATCH_BODY is not None:
        try:
            layout = _BATCH_BODY.layout()
            if layout is not None:
                nav_parent = _BATCH_NAV_WIDGET.parentWidget()
                if nav_parent is not _BATCH_BODY:
                    _BATCH_NAV_WIDGET.setParent(None)
                    layout.addWidget(_BATCH_NAV_WIDGET)
                try:
                    batch_native = getattr(batch_widget, "native", None)
                    if batch_native is not None:
                        idx = layout.indexOf(batch_native)
                        if idx != -1:
                            layout.removeWidget(_BATCH_NAV_WIDGET)
                            layout.insertWidget(idx + 1, _BATCH_NAV_WIDGET)
                except Exception:
                    pass
        except Exception:
            pass
    if not is_synaptic:
        _BATCH_SYNAPTIC_ACTIVE = False
    if is_synaptic:
        _set_quant_method(SYNAPTIC_METHOD, viewer)
        if _method_combo_widget is not None:
            try:
                _method_combo_widget.setEnabled(not lock_active)
            except Exception:
                pass
        for key, btn in _quickbar_controls.items():
            show = key in {"roi", "roi_stats", "region", "cell_list"}
            try:
                btn.setVisible(show)
            except Exception:
                pass
        if _quickbar_grid is not None and "cell_list" in _quickbar_controls:
            try:
                btn = _quickbar_controls["cell_list"]
                _quickbar_grid.addWidget(btn, _quickbar_row_cell, 1, 1, 2)
            except Exception:
                pass
    else:
        _set_quant_method(STANDARD_METHOD, viewer)
        if _method_combo_widget is not None:
            try:
                _method_combo_widget.setEnabled(True)
            except Exception:
                pass
        for key, btn in _quickbar_controls.items():
            try:
                btn.setVisible(True)
            except Exception:
                pass
        if _quickbar_grid is not None and "cell_list" in _quickbar_controls:
            try:
                btn = _quickbar_controls["cell_list"]
                _quickbar_grid.addWidget(btn, _quickbar_row_cell, 1)
            except Exception:
                pass
    try:
        if _method_combo_widget is not None and is_synaptic:
            _method_combo_widget.setEnabled(not lock_active)
    except Exception:
        pass
    if _BATCH_ACTION_COMBO is not None:
        try:
            _BATCH_ACTION_COMBO.setEnabled(not lock_active)
        except Exception:
            pass


def _init_batch_synaptic_samples():
    global _BATCH_SYNAPTIC_SAMPLES, _BATCH_SYNAPTIC_INDEX, _BATCH_SYNAPTIC_SAVED
    bg_keys = _parse_bg_or_keys(getattr(batch_widget.laminin_substring, "value", "") or "")
    if not bg_keys:
        bg_keys = ["background"]
    _BATCH_SYNAPTIC_SAMPLES = _build_batch_samples(bg_keys)
    _BATCH_SYNAPTIC_INDEX = -1
    _BATCH_SYNAPTIC_SAVED = set()
    _update_batch_synaptic_progress()


def run_batch_pipeline(viewer: Optional["napari.Viewer"], *, start_index: int = 0) -> None:
    v = _resolve_viewer(None, viewer)
    _activate_viewer_state(v)
    if not _BATCH_FOLDERS:
        show_info("Select one or more folders first.")
        return
    if _BATCH_ACTION_COMBO is None:
        show_warning("Batch action selector unavailable.")
        return
    action = str(_BATCH_ACTION_COMBO.currentText())
    do_seg = "Segmentation" in action
    do_quant = "Quantification" in action
    bg_raw = getattr(batch_widget.laminin_substring, "value", "") or ""
    bg_keys = _parse_bg_or_keys(bg_raw)
    is_synaptic = _method_combo_widget is not None and str(_method_combo_widget.currentText()) == SYNAPTIC_METHOD
    if do_seg and not bg_keys:
        mq_info(None, "Batch segmentation", "Background stain name(s) required for segmentation.")
        return
    global _BATCH_VERIFY_DONE, _BATCH_VERIFY_PENDING, _BATCH_CANCELLED, _BATCH_RUN_INDEX, _BATCH_VERIFY_SELECTION, _BATCH_SYNAPTIC_ACTIVE
    if start_index <= 0:
        _BATCH_CANCELLED = False
        _BATCH_RUN_INDEX = 0
    _BATCH_SYNAPTIC_ACTIVE = bool(is_synaptic)
    _update_batch_action_ui(v)
    if is_synaptic:
        _BATCH_SYNAPTIC_SAMPLES = _build_batch_samples(bg_keys)
        _BATCH_SYNAPTIC_SAVED = set()
        _BATCH_SYNAPTIC_INDEX = -1
        if not _BATCH_SYNAPTIC_SAMPLES:
            show_warning("No synaptic samples found in selected folders.")
            return
        _update_batch_synaptic_progress()
        _batch_synaptic_next()
        return
    stain_names = ""
    if _BATCH_USE_ALL_STRAINS is not None and not _BATCH_USE_ALL_STRAINS.isChecked():
        stain_names = getattr(batch_widget.stain_names, "value", "")
    pixel_size_um = float(getattr(batch_widget.pixel_size_um, "value", DEFAULT_PX_UM))
    verify_samples = bool(_BATCH_VERIFY_CHECK is not None and _BATCH_VERIFY_CHECK.isChecked())
    if verify_samples and start_index <= 0:
        selection = _choose_batch_verify_samples(_BATCH_FOLDERS)
        if selection is None:
            show_info("Batch run canceled.")
            return
        _BATCH_VERIFY_SELECTION = selection

    processed = 0
    with progress(total=len(_BATCH_FOLDERS)) as pbar:
        if start_index:
            try:
                pbar.update(start_index)
            except Exception:
                pass
        for idx in range(start_index, len(_BATCH_FOLDERS)):
            folder = _BATCH_FOLDERS[idx]
            if _BATCH_CANCELLED:
                show_info("Batch run canceled.")
                break
            try:
                pbar.set_description(f"{folder.name} ({idx + 1}/{len(_BATCH_FOLDERS)})")
            except Exception:
                pass
            if do_seg:
                batch_widget(
                    v,
                    folder=folder,
                    save_dir=folder,
                    reset_cache=True,
                    show_in_viewer=False,
                )
                _save_masks_from_cache(folder)
            else:
                SEGMENTED_CACHE.clear()
                sample = _load_saved_run_for_batch(folder, bg_keys, stain_names, pixel_size_um)
                if sample:
                    SEGMENTED_CACHE.append(sample)
                else:
                    pbar.update(1)
                    continue

            if do_quant and SEGMENTED_CACHE:
                global SEGMENT_EXPORT_ROOT
                SEGMENT_EXPORT_ROOT = Path(folder)
                if _method_combo_widget is not None and str(_method_combo_widget.currentText()) == SYNAPTIC_METHOD:
                    _run_batch_synaptic_interactive(v, folder, bg_keys, stain_names, pixel_size_um)
                else:
                    quantify_cached_segments(
                        None,
                        batch_widget.bg_mode.value,
                        batch_widget.bg_percentile.value,
                        batch_widget.local_ring_px.value,
                        batch_widget.rolling_ball_radius_px.value,
                        batch_widget.add_texture.value,
                        batch_widget.add_spatial.value,
                    )
            processed += 1
            pbar.update(1)
            if verify_samples:
                if _BATCH_VERIFY_SELECTION is not None and folder not in _BATCH_VERIFY_SELECTION:
                    continue
                show_info(f"Verifying sample {idx + 1}/{len(_BATCH_FOLDERS)}…")
                ok = _run_batch_verification(v, folder, bg_keys, stain_names, pixel_size_um)
                if ok:
                    _BATCH_VERIFY_PENDING = True
                    _BATCH_RUN_INDEX = idx + 1
                    show_info("Sample loaded for verification. Use Continue/Cancel to proceed.")
                    return
                show_warning(f"Verification skipped for {folder.name}.")
    show_info(f"Batch run complete ({processed} folder(s)).")

# ---------- Quantification (separate button; uses cached segmentation) ----------
def quantify_cached_segments(
    viewer: "napari.Viewer",
    bg_mode: str,
    bg_percentile: float,
    local_ring_px: int,
    rolling_ball_radius_px: int,
    add_texture: bool,
    add_spatial: bool,
):
    global SEGMENT_EXPORT_ROOT
    if not SEGMENTED_CACHE:
        mq_info(None, "MuscleQuant", "Run segmentation first (no cached masks to quantify).")
        return

    _bg_mode = normalize_bg_mode(bg_mode)
    combined_rows: List[pd.DataFrame] = []
    quantified = 0

    metrics = _sanitize_metrics(selected_metrics())
    CURRENT["metrics"] = metrics
    mem_px = int(CURRENT.get("membrane_px", CURRENT.get("s1_membrane_px", 4)))

    export_root = SEGMENT_EXPORT_ROOT
    if export_root is None:
        root_hint = None
        if SEGMENTED_CACHE:
            hint_val = SEGMENTED_CACHE[0].get("export_root")
            if hint_val:
                try:
                    root_hint = Path(hint_val).expanduser()
                except Exception:
                    root_hint = None
        if root_hint is None:
            root_hint = Path.home() / "musclequant" / "results"
        root_hint = _default_export_root(root_hint)
        export_root = choose_export_root(root_hint)
        if export_root is None:
            mq_info(None, "MuscleQuant (Quantification)", "Quantification canceled (no export folder selected).")
            return
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    SEGMENT_EXPORT_ROOT = export_root

    training_root = Path("/Users/akashagr/musclequant/app/musclequant/musclequant/Training Data")
    training_root.mkdir(parents=True, exist_ok=True)

    combined_rows: List[pd.DataFrame] = []

    for sample in SEGMENTED_CACHE:
        base = sample.get("base", "image")
        base_out = sample.get("base_out", base.replace("/", "_"))
        lam_paths = sample.get("lam_paths") or []
        lam_path = Path(sample.get("lam_path", "")) if sample.get("lam_path") else (Path(lam_paths[0]) if lam_paths else None)
        masks = np.asarray(sample["masks"])
        protein_channels: List[ProteinChannel] = sample.get("proteins", []) or []
        # Always include the background/lam stain as a quant channel.
        lam_arr_base = sample.get("lam_raw") if sample.get("lam_raw") is not None else sample.get("lam")
        if lam_arr_base is not None:
            lam_name = infer_stain_from_filename(lam_path) if lam_path else "Background"
            lam_plane = (sample.get("lam_used") or "Gray").split(",")[0]
            protein_channels = [
                ProteinChannel(
                    name=lam_name,
                    plane=lam_plane if lam_plane in PLANE_TO_IDX else "Gray",
                    path=None,
                    data=np.asarray(lam_arr_base, dtype=np.float32),
                    preview_id="__background__",
                    origin="background",
                ),
                *protein_channels,
            ]
        px_um = _px_um_from_sample(sample)
        preview_map = sample.get("preview_planes", {}) or {}
        cleaned = bool(sample.get("cleaned", False))
        suffix = sample.get("suffix", "_clean" if cleaned else "")
        stain_filter = [s.strip().lower() for s in sample.get("metadata", {}).get("stain_filter", []) if s.strip()]
        modified_flag = bool(sample.get("metadata", {}).get("modified", sample.get("modified", False)))

        if not protein_channels:
            continue

        membrane_labels, interior_labels = _compute_membrane_and_interior_masks(masks, mem_px)
        mem_area_px_map, mem_area_um2_map = _membrane_area_maps(membrane_labels, px_um)
        center_masks = compute_center_masks(masks) if metrics.get("ctcf", True) else None

        wide_df = None
        shared_base_cols = {
            "label", "area_px", "area_um2", "perimeter", "eccentricity",
            "solidity", "equivalent_diameter", "centroid-0", "centroid-1", "px_um",
        }
        if add_spatial:
            shared_base_cols.add("nn_dist_um")
        shared_base_cols.update({"membrane_area_px", "membrane_area_um2"})

        sample_mode = sample.get("mode", CURRENT.get("mode", "IHC"))
        intensity_needed = metrics.get("ctcf", True)

        if sample_mode.upper() == "H&E" and not intensity_needed:
            df = quantify_geometry(
                masks,
                px_um=float(px_um),
                add_spatial=bool(add_spatial),
            )
            df["membrane_area_px"] = df["label"].map(mem_area_px_map).fillna(0).astype(np.int32)
            df["membrane_area_um2"] = df["label"].map(mem_area_um2_map).fillna(0.0).astype(np.float32)
            df = _filter_quant_columns(df, metrics, "geometry")
            wide_df = df.copy()
        else:
            for channel in _filter_channels_by_viewer(viewer, protein_channels):
                pname = channel.name
                if stain_filter and pname.strip().lower() not in stain_filter:
                    continue
                src_path = channel.path
                src_name = Path(src_path).name if src_path else "virtual"
                print(f"[DEBUG] Quant: {base} → {pname} ({channel.plane}) from {src_name}")
                p_raw = _load_channel_raw(channel, preview_map)
                quant_out = quantify(
                    masks, p_raw,
                    px_um=float(px_um),
                    protein_name=pname,
                    bg_mode=_bg_mode,
                    bg_percentile=float(bg_percentile),
                    local_ring_px=int(local_ring_px),
                    rolling_ball_radius_px=int(rolling_ball_radius_px),
                    synapse_ring_px=SYNAPSE_RING_PX,
                    synapse_percentile=SYNAPSE_PERCENTILE,
                    add_texture=bool(add_texture),
                    add_spatial=bool(add_spatial),
                    return_masks=True if viewer is not None else False,
                    center_masks=center_masks,
                )
                if isinstance(quant_out, tuple):
                    df, overlays = quant_out
                    if viewer is not None:
                        bg_mask = overlays.get("bg_mask")
                        syn_mask = overlays.get("syn_mask")
                        if bg_mask is not None and np.any(bg_mask):
                            _add_or_replace_layer(viewer, f"BG region — {pname}", bg_mask, opacity=1.0)
                        if syn_mask is not None and np.any(syn_mask):
                            _add_or_replace_layer(viewer, f"Synaptic region — {pname}", syn_mask, opacity=1.0)
                else:
                    df = quant_out
                df["membrane_area_px"] = df["label"].map(mem_area_px_map).fillna(0).astype(np.int32)
                df["membrane_area_um2"] = df["label"].map(mem_area_um2_map).fillna(0.0).astype(np.float32)
                if metrics.get("ctcf", True):
                    df = _add_membrane_intensity_columns(df, p_raw, masks, membrane_labels, interior_labels, pname)
                df = _filter_quant_columns(df, metrics, pname)
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

        if wide_df is None:
            continue

        wide_df["image_base"] = base
        if lam_path:
            wide_df["background_stain"] = lam_path.name
        elif lam_paths:
            wide_df["background_stain"] = ",".join([Path(p).name for p in lam_paths])
        for channel in protein_channels:
            meta_col = f"plane_{channel.name}"
            if meta_col in wide_df.columns:
                j = 2
                while f"{meta_col}_{j}" in wide_df.columns:
                    j += 1
                meta_col = f"{meta_col}_{j}"
            wide_df[meta_col] = channel.plane

        px_area_um2 = float(px_um) ** 2
        if "area_px" in wide_df.columns:
            wide_df["area_um2"] = wide_df["area_px"].astype(float) * px_area_um2
        if "membrane_area_px" in wide_df.columns:
            wide_df["membrane_area_um2"] = wide_df["membrane_area_px"].astype(float) * px_area_um2
        wide_df["px_um"] = float(px_um)

        # Simplify output: drop metadata/plane/geometry extras, keep one combined CSV with friendly names.
        if "area_um2" in wide_df.columns:
            wide_df.rename(columns={"area_um2": "Cell Area (um^2)"}, inplace=True)
        drop_fixed = {
            "image_base",
            "background_stain",
            "sample",
            "nn_dist_um",
            "px_um",
            "equivalent_diameter",
            "perimeter",
        }
        drop_prefixes = ("plane_", "entropy_", "centroid-", "bg_mode_")
        cols_to_drop = set()
        for c in list(wide_df.columns):
            if c in drop_fixed:
                cols_to_drop.add(c)
            else:
                for prefix in drop_prefixes:
                    if c.startswith(prefix):
                        cols_to_drop.add(c)
                        break
        if cols_to_drop:
            wide_df = wide_df.drop(columns=[c for c in cols_to_drop if c in wide_df.columns])

        combined_rows.append(wide_df)
        quantified += 1

    if combined_rows:
        big = pd.concat(combined_rows, ignore_index=True)
        first_base = _safe_basename(SEGMENTED_CACHE[0].get("base_out") or SEGMENTED_CACHE[0].get("base") or "image") if SEGMENTED_CACHE else "image"
        if quantified == 1:
            fname = f"{first_base}_standard_quant.csv"
        else:
            fname = f"{first_base}_plus{quantified-1}_standard_quant_combined.csv"
        out_path = Path(export_root) / fname
        big.to_csv(out_path, index=False)
        mq_info(
            None,
            "MuscleQuant (Quantification)",
            f"Quantified {quantified} sample(s).\nCombined CSV saved:\n{out_path}",
        )
    else:
        mq_info(None, "MuscleQuant (Quantification)", "No proteins were quantified (check segmentation cache).")


def _run_selected_quant(method: str, widget):
    CURRENT["quant_method"] = method
    if method == SYNAPTIC_METHOD:
        run_s1_analysis(widget.viewer.value, export_results=False)
    else:
        quantify_cached_segments(
            widget.viewer.value,
            widget.bg_mode.value,
            widget.bg_percentile.value,
            widget.local_ring_px.value,
            widget.rolling_ball_radius_px.value,
            widget.add_texture.value,
            widget.add_spatial.value,
        )


def _export_selected_quant(method: str, widget):
    CURRENT["quant_method"] = method
    if method == SYNAPTIC_METHOD:
        export_synaptic_results(widget.viewer.value, use_cached=True)
    else:
        quantify_cached_segments(
            widget.viewer.value,
            widget.bg_mode.value,
            widget.bg_percentile.value,
            widget.local_ring_px.value,
            widget.rolling_ball_radius_px.value,
            widget.add_texture.value,
            widget.add_spatial.value,
        )


def _resolve_viewer(widget=None, fallback=None):
    """Safely resolve an active viewer from a widget or fallback/global."""
    if widget is not None:
        try:
            v = getattr(widget, "viewer", None)
            if v is not None:
                val = getattr(v, "value", None)
                if val is not None:
                    _activate_viewer_state(val)
                    return val
        except Exception:
            pass
    if _ACTIVE_VIEWER is not None:
        _activate_viewer_state(_ACTIVE_VIEWER)
        return _ACTIVE_VIEWER
    if fallback is not None:
        _activate_viewer_state(fallback)
        return fallback
    if _mode_viewer is not None:
        _activate_viewer_state(_mode_viewer)
        return _mode_viewer
    return None


def _set_scale_from_line(widget):
    """
    Let the user draw a calibration line and set pixel size (µm/px) from its real-world length.
    Click once to create the line layer, draw a line, then click again to confirm scale.
    """
    viewer = _resolve_viewer(widget, widget.viewer.value if hasattr(widget, "viewer") else None)
    if viewer is None:
        show_info("No viewer available.")
        return
    layer_name = "Scale calibration"
    cal_layer = None
    for L in viewer.layers:
        if getattr(L, "name", "") == layer_name:
            cal_layer = L
            break
    if cal_layer is None:
        try:
            cal_layer = viewer.add_shapes(
                name=layer_name,
                shape_type="line",
                edge_color="yellow",
                face_color="transparent",
                edge_width=3,
            )
            cal_layer.mode = "add_line"
            show_info("Draw a single calibration line, then click 'Set scale from line' again to apply.")
        except Exception as exc:
            show_warning(f"Could not create calibration layer: {exc}")
        return
    if len(getattr(cal_layer, "data", [])) == 0:
        try:
            cal_layer.mode = "add_line"
        except Exception:
            pass
        show_info("Draw a single calibration line, then click 'Set scale from line' again to apply.")
        return

    coords = np.asarray(cal_layer.data[-1])
    if coords.shape[0] < 2:
        show_warning("Calibration line is incomplete. Draw a single straight line.")
        return
    px_len = float(np.linalg.norm(coords[-1] - coords[0]))
    px_len = float(round(px_len, 0))
    if px_len <= 0:
        show_warning("Calibration line length is zero.")
        return
    current_px_um = float(CURRENT.get("px_um", DEFAULT_PX_UM))
    default_real_um = max(1e-6, px_len * current_px_um)
    real_len_um, ok = QInputDialog.getDouble(
        None,
        "Set scale from line",
        f"Line length: {px_len:.2f} px\nEnter real-world length (µm):",
        value=float(default_real_um),
        min=1e-6,
        decimals=4,
    )
    if not ok or real_len_um <= 0:
        return
    px_um_val = float(real_len_um) / px_len
    _propagate_px_um(px_um_val)
    try:
        widget.pixel_size_um.value = float(px_um_val)
    except Exception:
        pass
    try:
        cal_layer.mode = "select"
    except Exception:
        pass
    show_info(f"Pixel size set to {px_um_val:.4f} µm/px from calibration line.")


def _attach_synaptic_controls(widget, viewer: "napari.Viewer", method_combo):
    panel = QWidget()
    grid = QGridLayout(panel)
    grid.setContentsMargins(6, 6, 6, 6)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(6)

    row = 0
    mem_label = QLabel(f"Membrane dilation (px): {int(CURRENT.get('s1_membrane_px', 4))}")
    mem_slider = QSlider(Qt.Horizontal); mem_slider.setMinimum(1); mem_slider.setMaximum(50)
    mem_slider.setValue(int(CURRENT.get("s1_membrane_px", 4)))
    grid.addWidget(mem_label, row, 0)
    grid.addWidget(mem_slider, row, 1)
    row += 1
    btn_click_keep = QPushButton("Cell Selection"); grid.addWidget(btn_click_keep, row, 0)
    btn_apply_click_keep = QPushButton("Keep Selected Cells"); grid.addWidget(btn_apply_click_keep, row, 1)
    btn_reset_click_keep = QPushButton("Reset Selection"); grid.addWidget(btn_reset_click_keep, row, 2)
    row += 1
    btn_preview_s1 = QPushButton("Preview Synaptic Filter"); grid.addWidget(btn_preview_s1, row, 0, 1, 3)

    def on_mem_change(val: int):
        CURRENT["s1_membrane_px"] = int(val)
        mem_label.setText(f"Membrane dilation (px): {int(val)}")
        # Drop cached/preview membranes so the next preview respects the new dilation.
        CURRENT["s1_preview_mem"] = None
        CURRENT["s1_preview_extra"] = None
        v = _resolve_viewer(widget, viewer)
        if v is not None:
            for lname in ("S1 membrane (preview)", "S1 extrasyn (preview)"):
                try:
                    if lname in [L.name for L in v.layers]:
                        v.layers.remove(lname)
                except Exception:
                    pass
    mem_slider.valueChanged.connect(on_mem_change)

    def do_click_keep():
        v = _resolve_viewer(widget, viewer)
        if v is None:
            show_info("No viewer available.")
            return
        ensure_s1_click_layer(v)
    btn_click_keep.clicked.connect(do_click_keep)

    def do_apply_click_keep():
        v = _resolve_viewer(widget, viewer)
        if v is None:
            show_info("No viewer available.")
            return
        keep_cells_from_clicks(v)
        _refresh_kept_preview_from_manual(v)
    btn_apply_click_keep.clicked.connect(do_apply_click_keep)

    def do_reset_click_keep():
        v = _resolve_viewer(widget, viewer)
        if v is None:
            show_info("No viewer available.")
            return
        _reset_manual_keep_selection(v)
        show_info("Cleared manual cell selection.")
    btn_reset_click_keep.clicked.connect(do_reset_click_keep)

    def do_preview_s1():
        v = _resolve_viewer(widget, viewer)
        if v is None:
            show_info("No viewer available.")
            return
        preview_s1_filter(v)
    btn_preview_s1.clicked.connect(do_preview_s1)

    layout = widget.native.layout()
    try:
        layout.addRow(panel)
    except Exception:
        layout.addWidget(panel)

    _synaptic_controls.extend([
        panel,
        mem_label,
        mem_slider,
        btn_click_keep,
        btn_apply_click_keep,
        btn_reset_click_keep,
        btn_preview_s1,
    ])
    _set_quant_method(method_combo.currentText(), viewer)

# Add a dedicated button onto the batch widget for quantification.
def _attach_quant_button(widget):
    from qtpy.QtWidgets import QComboBox

    _synaptic_controls.clear()
    _standard_controls.clear()

    method_combo = QComboBox()
    method_combo.addItems([STANDARD_METHOD, SYNAPTIC_METHOD])
    method_combo.setCurrentText(CURRENT.get("quant_method") or STANDARD_METHOD)

    btn_quant = QPushButton("Run Quantification")
    btn_quant.setToolTip("Run the selected quantification method.")
    btn_export = QPushButton("Export Results")
    btn_export.setToolTip("Export results for the selected quantification method.")
    layout = widget.native.layout()
    row_widget = QWidget()
    row_layout = QHBoxLayout(row_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(6)
    row_layout.addWidget(method_combo)
    row_layout.addWidget(btn_quant)
    try:
        layout.addRow(row_widget)
        layout.addRow(btn_export)
    except Exception:
        layout.addWidget(row_widget)
        layout.addWidget(btn_export)

    mem_row = QWidget()
    mem_layout = QHBoxLayout(mem_row)
    mem_layout.setContentsMargins(0, 0, 0, 0)
    mem_layout.setSpacing(6)
    mem_label = QLabel(f"Membrane dilation (px): {int(CURRENT.get('membrane_px', CURRENT.get('s1_membrane_px', 4)))}")
    mem_slider = QSlider(Qt.Horizontal)
    mem_slider.setMinimum(1)
    mem_slider.setMaximum(50)
    mem_slider.setValue(int(CURRENT.get("membrane_px", CURRENT.get("s1_membrane_px", 4))))
    btn_highlight_mem = QPushButton("Highlight membranes")
    mem_layout.addWidget(mem_label)
    mem_layout.addWidget(mem_slider, stretch=1)
    mem_layout.addWidget(btn_highlight_mem)
    try:
        layout.addRow(mem_row)
    except Exception:
        layout.addWidget(mem_row)

    def _on_mem_slider(val: int):
        CURRENT["membrane_px"] = int(val)
        mem_label.setText(f"Membrane dilation (px): {int(val)}")
    mem_slider.valueChanged.connect(_on_mem_slider)

    def _on_highlight_mem():
        v = _resolve_viewer(widget, widget.viewer.value if hasattr(widget, "viewer") else None)
        highlight_membranes(v, mem_slider.value())
    btn_highlight_mem.clicked.connect(_on_highlight_mem)

    _standard_controls.extend([mem_row, mem_label, mem_slider, btn_highlight_mem])
    global _single_only_widgets, _method_combo_widget
    _single_only_widgets.extend([btn_quant, btn_export, mem_row, btn_highlight_mem])
    _method_combo_widget = method_combo

    def _on_method_change(text: str):
        _set_quant_method(text, widget.viewer.value)
        _update_batch_action_ui(widget.viewer.value)
    method_combo.currentTextChanged.connect(_on_method_change)
    _on_method_change(method_combo.currentText())

    btn_quant.clicked.connect(
        lambda: _run_selected_quant(
            method_combo.currentText(),
            widget,
        )
    )

    btn_export.clicked.connect(
        lambda: _export_selected_quant(
            method_combo.currentText(),
            widget,
        )
    )
    _attach_synaptic_controls(widget, widget.viewer.value, method_combo)

_attach_quant_button(batch_widget)

# Collapsible advanced settings toggle to declutter UI
def _attach_advanced_toggle(widget):
    adv_fields = [
        # Segmentation-affecting controls only
        "pixel_size_um", "bg_mode", "drop_edge_touching",
        "diameter_px", "flow_thresh", "cellprob_thresh",
        "min_area_px", "model_path",
        "chunk_first", "chunk_count", "drop_edge_buffer_px",
        "export_all_images",
    ]
    for name in adv_fields:
        try:
            getattr(widget, name).visible = False
        except Exception:
            pass

    btn_toggle = QPushButton("Show advanced settings")
    btn_toggle.setToolTip("Toggle visibility of advanced controls to reduce clutter.")
    btn_set_scale = QPushButton("Set scale from line")
    btn_set_scale.setToolTip("Draw a line on the image, then click again to set pixel size (µm/px).")
    btn_set_scale.setVisible(False)
    btn_set_scale.clicked.connect(lambda: _set_scale_from_line(widget))

    def _toggle():
        currently_visible = any(getattr(widget, n).visible for n in adv_fields if hasattr(widget, n))
        target = not currently_visible
        for name in adv_fields:
            try:
                getattr(widget, name).visible = target
            except Exception:
                pass
        btn_set_scale.setVisible(target)
        btn_toggle.setText("Hide advanced settings" if target else "Show advanced settings")

    btn_toggle.clicked.connect(_toggle)

    layout = widget.native.layout()
    # Place toggle at the very top (above Run Segmentation button)
    if hasattr(layout, "insertRow"):
        layout.insertRow(0, "", btn_toggle)
        inserted = False
        try:
            if isinstance(layout, QFormLayout) and hasattr(widget, "pixel_size_um"):
                px_native = getattr(widget.pixel_size_um, "native", None)
                if px_native is not None:
                    for r in range(layout.rowCount()):
                        item = layout.itemAt(r, QFormLayout.ItemRole.FieldRole)
                        w = item.widget() if item else None
                        if w is px_native:
                            layout.insertRow(r + 1, "", btn_set_scale)
                            inserted = True
                            break
        except Exception:
            inserted = False
        if not inserted:
            layout.insertRow(1, "", btn_set_scale)
    elif hasattr(layout, "insertWidget"):
        layout.insertWidget(0, btn_toggle)
        try:
            px_native = getattr(getattr(widget, "pixel_size_um", None), "native", None)
            if px_native is not None:
                idx = layout.indexOf(px_native)
                if idx != -1:
                    layout.insertWidget(idx + 1, btn_set_scale)
                else:
                    layout.insertWidget(1, btn_set_scale)
            else:
                layout.insertWidget(1, btn_set_scale)
        except Exception:
            layout.insertWidget(1, btn_set_scale)
    else:
        layout.addWidget(btn_toggle)
        layout.addWidget(btn_set_scale)

_attach_advanced_toggle(batch_widget)

# ---------------------- EXTRA UI (scroll, compact toolbars, manual tools) ----------------------
def make_quick_toolbar(viewer):
    """Compact toolbar focused on ROI and selection tools."""
    bar = QWidget()
    grid = QGridLayout(bar)
    _apply_panel_layout(grid)
    grid.setColumnMinimumWidth(0, 240)
    row = 0
    btn_roi = QPushButton("Add ROI"); grid.addWidget(btn_roi, row, 0)
    btn_roi_stats = QPushButton("Compute ROI stats"); grid.addWidget(btn_roi_stats, row, 1, 1, 2)
    row += 1
    btn_cell_layer = QPushButton("Region Selection"); grid.addWidget(btn_cell_layer, row, 0)
    btn_cell_list  = QPushButton("List cells in selection"); grid.addWidget(btn_cell_list, row, 1)
    btn_cell_quant = QPushButton("Quantify selection"); grid.addWidget(btn_cell_quant, row, 2)
    row += 1
    btn_compare = QPushButton("Toggle Compare View"); grid.addWidget(btn_compare, row, 0, 1, 3)
    row += 1
    h_label = QLabel("H alpha")
    grid.addWidget(h_label, row, 0)
    he_h_slider = QSlider(Qt.Horizontal)
    he_h_slider.setMinimum(0)
    he_h_slider.setMaximum(200)
    he_h_slider.setValue(int(float(CURRENT.get("he_alpha_h", 1.0)) * 100))
    grid.addWidget(he_h_slider, row, 1, 1, 2)
    row += 1
    e_label = QLabel("E alpha")
    grid.addWidget(e_label, row, 0)
    he_e_slider = QSlider(Qt.Horizontal)
    he_e_slider.setMinimum(0)
    he_e_slider.setMaximum(200)
    he_e_slider.setValue(int(float(CURRENT.get("he_alpha_e", 1.0)) * 100))
    grid.addWidget(he_e_slider, row, 1, 1, 2)
    _quickbar_controls.update({
        "roi": btn_roi,
        "roi_stats": btn_roi_stats,
        "region": btn_cell_layer,
        "cell_list": btn_cell_list,
        "cell_quant": btn_cell_quant,
        "compare": btn_compare,
    })
    global _quickbar_grid, _quickbar_row_cell
    _quickbar_grid = grid
    _quickbar_row_cell = 1

    def do_roi():
        v = _resolve_viewer(None, viewer)
        if v is not None:
            ensure_roi_layer(v)
    btn_roi.clicked.connect(do_roi)

    def do_roi_stats():
        v = _resolve_viewer(None, viewer)
        if v is not None:
            compute_roi_stats(v)
    btn_roi_stats.clicked.connect(do_roi_stats)

    def do_cell_layer():
        v = _resolve_viewer(None, viewer)
        if v is not None:
            ensure_cell_query_layer(v)
    btn_cell_layer.clicked.connect(do_cell_layer)

    def do_cell_list():
        v = _resolve_viewer(None, viewer)
        if v is not None:
            list_cells_in_selection(v)
    btn_cell_list.clicked.connect(do_cell_list)

    def do_cell_quant():
        v = _resolve_viewer(None, viewer)
        if v is not None:
            quant_selection(v)
    btn_cell_quant.clicked.connect(do_cell_quant)

    def do_compare():
        v = _resolve_viewer(None, viewer)
        if v is not None:
            toggle_compare_view(v)
    btn_compare.clicked.connect(do_compare)

    def _on_he_h(val: int):
        CURRENT["he_alpha_h"] = float(val) / 100.0
        v = _resolve_viewer(None, viewer)
        _update_he_recon_layer(v)
    he_h_slider.valueChanged.connect(_on_he_h)

    def _on_he_e(val: int):
        CURRENT["he_alpha_e"] = float(val) / 100.0
        v = _resolve_viewer(None, viewer)
        _update_he_recon_layer(v)
    he_e_slider.valueChanged.connect(_on_he_e)

    global _HE_SLIDER_WIDGETS
    _HE_SLIDER_WIDGETS = [h_label, he_h_slider, e_label, he_e_slider]
    _update_he_slider_visibility()

    return bar

def make_manual_toolbar(viewer):
    """Manual additions controls in 2-row grid."""
    bar = QWidget()
    grid = QGridLayout(bar)
    _apply_panel_layout(grid)
    grid.addWidget(QLabel("Manual additions"), 0, 0, 1, 3)

    btn_add = QPushButton("New Label")
    grid.addWidget(btn_add, 1, 0)
    btn_delete = QPushButton("Delete Selected Label")
    grid.addWidget(btn_delete, 1, 1)
    btn_save  = QPushButton("Apply Changes")
    grid.addWidget(btn_save, 1, 2)

    def _start_shapes():
        v = _resolve_viewer(None, viewer)
        if v is None:
            show_info("No viewer available.")
            return
        layer = ensure_selection_layer(v)
        if layer is not None:
            layer.mode = "add_polygon"
            show_info("Draw a polygon/rectangle for the new label. Repeat as needed, then click 'Apply label changes'.")
    btn_add.clicked.connect(_start_shapes)
    def _delete_selected():
        v = _resolve_viewer(None, viewer)
        if v is None:
            show_info("No viewer available.")
            return
        label_layer = None
        labels_name = "Fibers_mask_clean" if CURRENT.get("cleaned") else "Fibers_mask"
        # Prefer explicit names, else any labels layer containing "Fibers_mask", else the active labels layer
        for layer in v.layers:
            if layer.name == labels_name:
                label_layer = layer
                break
        if label_layer is None:
            for layer in v.layers:
                if layer.__class__.__name__.lower().startswith("labels") and "fibers_mask" in layer.name.lower():
                    label_layer = layer
                    break
        if label_layer is None:
            active = getattr(v.layers, "selection", None)
            if active and len(active) == 1:
                only = list(active)[0]
                if only.__class__.__name__.lower().startswith("labels"):
                    label_layer = only
        if label_layer is None:
            show_info("No Fibers_mask layer to edit (select a labels layer).")
            return
        label_id = int(getattr(label_layer, "selected_label", 0))
        if label_id <= 0:
            show_info("Use the Pick tool on the Fibers mask to choose a label to delete.")
            return
        mask = CURRENT.get("masks")
        if mask is None:
            show_info("No masks in memory.")
            return
        if not np.any(mask == label_id):
            show_info(f"Label {label_id} not found in current mask.")
            return
        mask = mask.copy()
        mask[mask == label_id] = 0
        CURRENT["masks"] = mask
        label_layer.data = mask
        show_info(f"Deleted label {label_id}.")

    btn_delete.clicked.connect(_delete_selected)

    def _apply_labels():
        v = _resolve_viewer(None, viewer)
        if v is not None:
            apply_label_changes(v)
    btn_save.clicked.connect(_apply_labels)

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





@magicgui(layout="horizontal", call_button="Preview", folder={"label": "Images", "mode": "d"})
def folder_preview(viewer: "napari.Viewer", folder: Path = Path(".")) -> None:
    _activate_viewer_state(viewer)
    _reset_s1_state(viewer)
    try:
        batch_widget.folder.value = folder
    except Exception:
        pass
    folder = Path(folder)

    files: List[Path] = []
    # If user dropped a file path, allow multi-file selection via QFileDialog
    if folder.is_file():
        parent = folder.parent
        selection, _ = QFileDialog.getOpenFileNames(
            None,
            "Select TIFF images",
            str(parent),
            "TIFF Files (*.tif *.tiff);;All Files (*)",
        )
        if selection:
            files = [Path(p) for p in selection]
    else:
        # Also allow user to pick multiple files from the folder prompt if empty
        files = sorted([p for p in folder.glob("*.tif*")])
        if not files:
            selection, _ = QFileDialog.getOpenFileNames(
                None,
                "Select TIFF images",
                str(folder),
                "TIFF Files (*.tif *.tiff);;All Files (*)",
            )
            if selection:
                files = [Path(p) for p in selection]

    if not files:
        mq_info(None, "No files", "No TIFF files found in the selected folder.")
        return

    try:
        CURRENT["last_preview_folder"] = str(files[0].parent)
    except Exception:
        CURRENT["last_preview_folder"] = str(folder)

    try:
        stain_tokens = _parse_stain_list(batch_widget.stain_names.value)
    except Exception:
        stain_tokens = []

    manual_override = CURRENT.get("mode_source") == "manual"
    if not manual_override:
        auto_mode = CURRENT.get("mode", "IHC")
        set_mode(auto_mode, source="auto", apply_presets_flag=True)
    else:
        auto_mode = CURRENT.get("mode", "IHC")

    viewer.layers.clear()
    CURRENT["preview_planes"] = {}
    detected = []
    pix_size_found = None
    bg_indices: List[int] = []
    fg_indices: List[int] = []

    if CURRENT.get("mode", "IHC").upper() == "H&E":
        he_channels = _separate_he_channels(
            files[0],
            include_dab=False,
            quant_mode=_he_quant_mode(),
            dmax=_he_dmax(),
        )
        if he_channels and "Hematoxylin" in he_channels:
            base_name = files[0].stem
            _add_he_layers(viewer, base_name, he_channels)
            for chan_name in ("Hematoxylin", "Eosin", "DAB"):
                arr = he_channels.get(chan_name)
                if arr is None:
                    continue
                CURRENT["preview_planes"][_virtual_key(base_name, chan_name)] = "Gray"
                detected.append(chan_name)
            pix_size_found = _infer_pixel_size_um_from_tiff(files[0])
            show_info("H&E mode: generated Hematoxylin/Eosin previews.")
            try:
                batch_widget.stain_names.value = ", ".join(detected)
            except Exception:
                pass
            try:
                if pix_size_found is not None and hasattr(batch_widget, "pixel_size_um"):
                    batch_widget.pixel_size_um.value = float(pix_size_found)
            except Exception:
                pass
            return
        else:
            show_warning("H&E mode selected but stain separation failed; falling back to standard preview.")
            set_mode("IHC", source="auto", apply_presets_flag=True)

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
        stain_name = infer_stain_from_filename(p)
        is_background = idx in bg_indices
        if is_background:
            cmap = _stain_colormap(stain_name, "gray", used)
            opacity = 1.0
        else:
            default_cmap = COLORMAP_CYCLE[fg_color_idx % len(COLORMAP_CYCLE)]
            cmap = _stain_colormap(stain_name, default_cmap, used)
            opacity = 1.0
            fg_color_idx += 1
        layer = viewer.add_image(plane, name=stain_name, blending="additive", colormap=cmap, visible=True)
        layer.opacity = float(opacity)
        detected.append(stain_name)
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
    if pix_size_found is None:
        try:
            current_px = getattr(batch_widget, "pixel_size_um", None)
            px_val = float(current_px.value) if current_px is not None else DEFAULT_PX_UM
        except Exception:
            px_val = DEFAULT_PX_UM
        show_warning(
            f"No pixel size metadata found in selected files; using current setting ({px_val:.4f} µm). "
            "Update 'Pixel size (µm)' if you know the correct value."
        )


def make_io_panel(viewer: "napari.Viewer") -> QWidget:
    """Single-mode top row: folder select + mode + image mode."""
    panel = QWidget()
    layout = QVBoxLayout(panel)
    _apply_panel_layout(layout)

    row_top = QHBoxLayout()
    row_top.setSpacing(_PANEL_SPACING)
    from qtpy.QtWidgets import QComboBox
    row_top.addWidget(QLabel("Mode"))
    mode_combo = QComboBox()
    mode_combo.addItems(["Single", "Batch"])
    mode_combo.setCurrentText("Single")
    mode_combo.currentTextChanged.connect(lambda text: _set_ui_mode(text, viewer))
    _UI_MODE_COMBOS.append(mode_combo)
    row_top.addWidget(mode_combo)
    row_top.addSpacing(_PANEL_SPACING)
    row_top.addWidget(make_mode_selector(viewer))
    row_top.addStretch(1)
    layout.addLayout(row_top)

    row_bottom = QHBoxLayout()
    row_bottom.setSpacing(_PANEL_SPACING)
    btn_select = QPushButton("Select folder…")
    btn_select.clicked.connect(lambda: _select_single_folder(viewer))
    row_bottom.addWidget(btn_select)
    global _SINGLE_FOLDER_LABEL
    _SINGLE_FOLDER_LABEL = QLabel("No folder selected")
    row_bottom.addWidget(_SINGLE_FOLDER_LABEL)
    row_bottom.addStretch(1)
    layout.addLayout(row_bottom)
    return panel


def make_batch_panel(viewer: "napari.Viewer") -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout(panel)
    _apply_panel_layout(layout)

    top_row = QHBoxLayout()
    top_row.setSpacing(_PANEL_SPACING)
    top_row.addWidget(QLabel("Mode"))
    from qtpy.QtWidgets import QComboBox
    mode_combo = QComboBox()
    mode_combo.addItems(["Single", "Batch"])
    mode_combo.setCurrentText("Batch")
    mode_combo.currentTextChanged.connect(lambda text: _set_ui_mode(text, viewer))
    _UI_MODE_COMBOS.append(mode_combo)
    top_row.addWidget(mode_combo)
    top_row.addSpacing(_PANEL_SPACING)
    top_row.addWidget(make_mode_selector(viewer))
    top_row.addStretch(1)
    layout.addLayout(top_row)

    bottom_row = QHBoxLayout()
    bottom_row.setSpacing(_PANEL_SPACING)
    btn_select = QPushButton("Select folders…")
    btn_select.clicked.connect(_select_batch_folders)
    bottom_row.addWidget(btn_select)
    global _BATCH_FOLDER_LABEL
    _BATCH_FOLDER_LABEL = QLabel("No folders selected")
    bottom_row.addWidget(_BATCH_FOLDER_LABEL)
    bottom_row.addStretch(1)
    layout.addLayout(bottom_row)

    use_all_row = QHBoxLayout()
    use_all_row.setSpacing(_PANEL_SPACING)
    use_all_row.addWidget(QLabel("Use all stains"))
    use_all = QCheckBox()
    use_all.setChecked(True)
    use_all_row.addWidget(use_all)
    use_all_row.addStretch(1)
    global _BATCH_USE_ALL_STRAINS
    _BATCH_USE_ALL_STRAINS = use_all
    use_all.stateChanged.connect(lambda _v: _sync_batch_stain_visibility())

    verify_row = QHBoxLayout()
    verify_row.setSpacing(_PANEL_SPACING)
    verify_row.addWidget(QLabel("Verify samples"))
    verify_check = QCheckBox()
    verify_check.setChecked(True)
    verify_row.addWidget(verify_check)
    verify_row.addStretch(1)
    layout.addLayout(verify_row)
    global _BATCH_VERIFY_CHECK
    _BATCH_VERIFY_CHECK = verify_check

    action_row = QHBoxLayout()
    action_row.setSpacing(_PANEL_SPACING)
    action_row.addWidget(QLabel("Batch action"))
    combo = QComboBox()
    combo.addItems([
        "Segmentation only",
        "Quantification only",
        "Segmentation + Quantification",
    ])
    global _BATCH_ACTION_COMBO
    _BATCH_ACTION_COMBO = combo
    action_row.addWidget(combo, stretch=1)
    btn_run = QPushButton("Run batch")
    btn_run.clicked.connect(lambda: run_batch_pipeline(viewer))
    action_row.addWidget(btn_run)
    global _BATCH_RUN_BUTTON
    _BATCH_RUN_BUTTON = btn_run
    layout.addLayout(action_row)

    body = QWidget()
    body_layout = QVBoxLayout(body)
    body_layout.setContentsMargins(0, 0, 0, 0)
    body_layout.setSpacing(_PANEL_SPACING)
    body_layout.addLayout(use_all_row)
    layout.addWidget(body, stretch=1)

    nav = QWidget()
    nav_layout = QHBoxLayout(nav)
    nav_layout.setContentsMargins(0, 0, 0, 0)
    nav_layout.setSpacing(_PANEL_SPACING)
    btn_prev = QPushButton("Previous")
    btn_save = QPushButton("Save")
    btn_next = QPushButton("Next")
    btn_prev.clicked.connect(_batch_synaptic_prev)
    btn_save.clicked.connect(_batch_synaptic_save)
    btn_next.clicked.connect(_batch_synaptic_next)
    nav_layout.addWidget(btn_prev)
    nav_layout.addWidget(btn_save)
    nav_layout.addWidget(btn_next)
    nav_layout.addStretch(1)
    progress_label = QLabel("0/0")
    nav_layout.addWidget(progress_label)
    nav.setVisible(False)
    layout.addWidget(nav)
    sample_row = QWidget()
    sample_row_layout = QHBoxLayout(sample_row)
    sample_row_layout.setContentsMargins(0, 0, 0, 0)
    sample_row_layout.setSpacing(_PANEL_SPACING)
    sample_label = QLabel("Sample: —")
    sample_row_layout.addWidget(sample_label)
    sample_row_layout.addStretch(1)
    layout.addWidget(sample_row)
    global _BATCH_NAV_WIDGET, _BATCH_PROGRESS_LABEL, _BATCH_SAMPLE_LABEL
    _BATCH_NAV_WIDGET = nav
    _BATCH_PROGRESS_LABEL = progress_label
    _BATCH_SAMPLE_LABEL = sample_label

    global _BATCH_BODY
    _BATCH_BODY = body
    combo.currentTextChanged.connect(lambda _t: _update_batch_action_ui(viewer))
    _update_batch_action_ui(viewer)
    _set_batch_call_button_visible(False)
    return panel

def launch_viewer():
    """Create the napari viewer, attach all widgets, and start the event loop."""
    app = QApplication.instance() or QApplication([])

    _patch_gamma_slider_limits(0.01, 5.0)

    pixmap = QPixmap(400, 200)
    pixmap.fill(Qt.white)
    splash = QSplashScreen(pixmap)
    splash.showMessage("Loading MuscleQuant...", Qt.AlignCenter | Qt.AlignBottom)
    splash.show()
    app.processEvents()

    viewer = napari.Viewer()
    _activate_viewer_state(viewer)
    _sync_magicgui_viewer(viewer)

    quickbar = make_quick_toolbar(viewer)
    manualbar = make_manual_toolbar(viewer)
    metrics_panel = metric_selection_widget()
    # Ensure method-dependent visibility reflects current selection once all widgets exist.
    _set_quant_method(CURRENT.get("quant_method") or STANDARD_METHOD, viewer)

    single_panel = QWidget()
    single_layout = QVBoxLayout(single_panel)
    single_layout.setContentsMargins(0, 0, 0, 0)
    single_layout.setSpacing(6)
    single_layout.addWidget(make_io_panel(viewer))
    single_body = QWidget()
    single_body_layout = QVBoxLayout(single_body)
    single_body_layout.setContentsMargins(0, 0, 0, 0)
    single_body_layout.setSpacing(6)
    single_layout.addWidget(single_body, stretch=1)
    single_scroll = make_scrollable_panel(viewer, single_panel)

    batch_panel = make_batch_panel(viewer)
    batch_scroll = make_scrollable_panel(viewer, batch_panel)

    global _UI_MODE_STACK, _SINGLE_PANEL, _BATCH_PANEL, _SINGLE_BODY
    _UI_MODE_STACK = QStackedWidget()
    _UI_MODE_STACK.addWidget(single_scroll)
    _UI_MODE_STACK.addWidget(batch_scroll)
    _SINGLE_PANEL = single_panel
    _BATCH_PANEL = batch_panel
    _SINGLE_BODY = single_body

    # Place shared widgets into single mode by default.
    single_body_layout.addWidget(batch_widget.native)
    single_body_layout.addWidget(metrics_panel)
    _set_ui_mode("Single", viewer)
    bottom_panel = QWidget()
    bottom_layout = QVBoxLayout(bottom_panel)
    bottom_layout.setContentsMargins(4, 4, 4, 4)
    bottom_layout.setSpacing(6)
    bottom_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    bottom_layout.addWidget(quickbar)
    bottom_layout.addWidget(manualbar)
    global _single_only_panels
    _single_only_panels = [bottom_panel]

    right_container = QWidget()
    right_layout = QVBoxLayout(right_container)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(6)
    right_layout.addWidget(_UI_MODE_STACK, stretch=1)
    right_layout.addWidget(bottom_panel, stretch=0)

    controls_tabs = QTabWidget()
    controls_tabs.setTabPosition(QTabWidget.South)
    controls_tabs.setMovable(False)
    controls_tabs.setTabsClosable(False)
    main_controls_page = QWidget()
    compare_controls_page = QWidget()
    for page in (main_controls_page, compare_controls_page):
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
    controls_tabs.addTab(main_controls_page, "Main Controls")
    main_controls_page.layout().addWidget(right_container)

    def _on_controls_tab(idx: int):
        target = viewer
        if idx == 1:
            target = _ensure_compare_viewer(viewer) or viewer
        _activate_viewer_state(target)
        _sync_magicgui_viewer(target)
        current_parent = right_container.parentWidget()
        target_page = controls_tabs.widget(idx)
        if current_parent is not target_page:
            right_container.setParent(None)
            target_page.layout().addWidget(right_container)

    controls_tabs.currentChanged.connect(_on_controls_tab)

    dock = viewer.window.add_dock_widget(controls_tabs, area="right", name="MuscleQuant")
    global _MAIN_DOCK, _MAIN_WINDOW, _CONTROLS_TABS, _CONTROLS_MAIN_PAGE, _CONTROLS_COMPARE_PAGE
    _MAIN_DOCK = dock
    _MAIN_WINDOW = viewer.window._qt_window
    _CONTROLS_TABS = controls_tabs
    _CONTROLS_MAIN_PAGE = main_controls_page
    _CONTROLS_COMPARE_PAGE = compare_controls_page

    splash.close()

    print(
        """
MuscleQuant tips:
  • Segmentation and quantification are separate: run 'Run Segmentation', then 'Run Quantification' to write CSVs.
  • Background mode: center-based intracellular subtraction.
  • Manual additions: Start (add layer) → paint → Merge additions → Re-quantify & Save (into the same sample folder).
"""
    )

    napari.run()


if __name__ == "__main__":
    launch_viewer()
