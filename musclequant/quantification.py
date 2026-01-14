from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from skimage import measure
from skimage.draw import disk as draw_disk
from skimage.filters import rank
from skimage.morphology import disk as morph_disk
from qtpy.QtWidgets import QFileDialog, QInputDialog

from musclequant.config import DEFAULT_PX_UM, SYNAPSE_RING_PX, SYNAPSE_PERCENTILE

METRIC_DEFAULTS = {"area": True, "ctcf": False, "eccentricity": False, "solidity": False}


def compute_center_masks(mask: np.ndarray) -> dict[int, np.ndarray]:
    """
    Precompute intracellular background regions (center masks) per label.
    This is geometry-only and can be reused across stains.
    """
    center_masks: Dict[int, np.ndarray] = {}
    for prop in measure.regionprops(mask):
        lbl = int(prop.label)
        area_px = max(int(prop.area), 1)
        target_area = max(1, int(round(area_px * 0.33)))
        radius = max(1, int(np.sqrt(target_area / np.pi)))
        rr, cc = draw_disk(prop.centroid, radius, shape=mask.shape)
        inner = np.zeros_like(mask, dtype=bool)
        inner[rr, cc] = True
        inner &= (mask == lbl)
        if not np.any(inner):
            inner = (mask == lbl)
        center_masks[lbl] = inner
    return center_masks


def quantify(
    mask: np.ndarray,
    intensity_raw: np.ndarray,
    px_um: float = DEFAULT_PX_UM,
    protein_name: str = "protein",
    bg_mode: str = "center",
    bg_percentile: float = 5.0,
    local_ring_px: int = 5,
    rolling_ball_radius_px: int = 50,
    synapse_ring_px: int = SYNAPSE_RING_PX,
    synapse_percentile: float = SYNAPSE_PERCENTILE,
    add_texture: bool = True,
    add_spatial: bool = True,
    return_masks: bool = False,
    center_masks: Optional[Dict[int, np.ndarray]] = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-cell quantification with center-based intracellular background."""
    img = intensity_raw.astype(np.float32, copy=False)
    base_bg_mean = float(np.median(img))
    bg_mode_used = "center_fraction"

    if center_masks is None:
        center_masks = compute_center_masks(mask)

    center_means: Dict[int, float] = {}
    for lbl, inner in center_masks.items():
        center_means[lbl] = float(np.mean(img[inner])) if np.any(inner) else base_bg_mean

    props = measure.regionprops_table(
        mask,
        intensity_image=img,
        properties=(
            "label",
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "equivalent_diameter",
            "centroid",
            "mean_intensity",
            "max_intensity",
        ),
    )
    df = pd.DataFrame(props).rename(columns={"area": "area_px"})
    df["px_um"] = float(px_um)
    df["area_um2"] = df["area_px"] * (px_um**2)

    df[f"{protein_name} Mean"] = df.pop("mean_intensity")
    _ = df.pop("max_intensity")
    df[f"{protein_name} Integrated"] = df["area_px"] * df[f"{protein_name} Mean"]

    df[f"{protein_name} Background Mean"] = df["label"].map(lambda l: center_means.get(int(l), base_bg_mean)).astype(np.float32)
    df[f"{protein_name} CTCF"] = df[f"{protein_name} Integrated"] - (df["area_px"] * df[f"{protein_name} Background Mean"])
    # Background-corrected mean fluorescence intensity (MFI) per cell.
    area_safe = df["area_px"].replace(0, np.nan)
    df[f"{protein_name} MFI"] = (df[f"{protein_name} CTCF"] / area_safe).fillna(0.0)

    if add_texture:
        max_val = img.max()
        if max_val > 0:
            img8 = np.clip((img / max_val) * 255.0, 0, 255).astype(np.uint8)
            # Use morphology disk as a footprint for entropy; draw_disk expects a center+radius.
            entropy_map = rank.entropy(img8, morph_disk(4))
            ent_props = measure.regionprops_table(
                mask,
                intensity_image=entropy_map,
                properties=("label", "mean_intensity"),
            )
            ent_df = pd.DataFrame(ent_props).rename(columns={"mean_intensity": f"{protein_name} Texture"})
            df = df.merge(ent_df, on="label", how="left")
        else:
            df[f"{protein_name} Texture"] = 0.0

    if add_spatial:
        # keep nearest-neighbor distance if needed, but we may drop it later in UI export
        centroids = df[["centroid-0", "centroid-1"]].to_numpy(dtype=np.float32)
        if len(centroids) >= 2:
            from scipy.spatial import distance_matrix

            dist = distance_matrix(centroids, centroids)
            np.fill_diagonal(dist, np.inf)
            nn = np.min(dist, axis=1) * float(px_um)
        else:
            nn = np.full(len(df), np.nan, dtype=np.float32)
        df["nn_dist_um"] = nn

    overlays: dict[str, np.ndarray] = {}
    if return_masks and center_masks:
        bg_overlay = np.zeros_like(mask, dtype=np.uint16)
        for lbl, cmask in center_masks.items():
            bg_overlay[cmask] = lbl
        overlays["bg_mask"] = bg_overlay
    if return_masks:
        return df, overlays
    return df


def quantify_geometry(
    mask: np.ndarray,
    px_um: float = DEFAULT_PX_UM,
    add_spatial: bool = True,
) -> pd.DataFrame:
    props = measure.regionprops_table(
        mask,
        properties=(
            "label",
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "equivalent_diameter",
            "centroid",
        ),
    )
    df = pd.DataFrame(props).rename(columns={"area": "area_px"})
    df["px_um"] = float(px_um)
    df["area_um2"] = df["area_px"] * (px_um**2)
    if add_spatial:
        centroids = df[["centroid-0", "centroid-1"]].to_numpy(dtype=np.float32)
        if len(centroids) >= 2:
            from scipy.spatial import distance_matrix

            dist = distance_matrix(centroids, centroids)
            np.fill_diagonal(dist, np.inf)
            nn = np.min(dist, axis=1) * float(px_um)
        else:
            nn = np.full(len(df), np.nan, dtype=np.float32)
        df["nn_dist_um"] = nn
    return df


def _filter_quant_columns(df: pd.DataFrame, metrics: Dict[str, bool], protein_name: str) -> pd.DataFrame:
    prefs = _sanitize_metrics(metrics)
    drops: list[str] = []
    if not prefs.get("area", True):
        drops.extend(
            [
                "area_px",
                "area_um2",
                "membrane_area_px",
                "membrane_area_um2",
                "perimeter",
                "equivalent_diameter",
                "centroid-0",
                "centroid-1",
                "px_um",
            ]
        )
    if not prefs.get("eccentricity", True):
        drops.append("eccentricity")
    if not prefs.get("solidity", True):
        drops.append("solidity")
    if not prefs.get("ctcf", True):
        drops.extend(
            [
                f"mean_{protein_name}",
                f"max_{protein_name}",
                f"integrated_{protein_name}",
                f"bg_mean_{protein_name}",
                f"bg_mode_{protein_name}",
                f"ctcf_{protein_name}",
                f"mfi_{protein_name}",
                f"synaptic_area_px_{protein_name}",
                f"synaptic_mean_{protein_name}",
                f"synaptic_max_{protein_name}",
                f"{protein_name} Synaptic Integrated",
                f"{protein_name} Extrasynaptic Area (px)",
                f"{protein_name} Texture",
                f"{protein_name} Membrane Background Mean",
                f"{protein_name} Membrane Mean",
                f"{protein_name} Membrane CTCF",
                f"{protein_name} Membrane MFI",
            ]
        )
    if drops:
        df = df.drop(columns=[c for c in drops if c in df.columns])
    return df


META_RELATED_COLS = {"px_um", "image_base", "background_stain"}


def _split_metadata_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    drop_cols: list[str] = []
    for col in list(df.columns):
        if col in META_RELATED_COLS or col.startswith("plane_"):
            if col in df.columns and col not in drop_cols:
                meta[col] = df[col].iloc[0]
                drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df, meta


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
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def choose_export_root(default: Path) -> Optional[Path]:
    root = QFileDialog.getExistingDirectory(
        None,
        "Select a directory to save results",
        str(Path(default).expanduser()),
    )
    if root and str(root).strip():
        outdir = Path(root)
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir
    return None


def normalize_bg_mode(bg_mode: str) -> str:
    """Single background mode: center-based."""
    return "center"


def _sanitize_metrics(metrics: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    base = dict(METRIC_DEFAULTS)
    if isinstance(metrics, dict):
        for key in base:
            if key in metrics:
                base[key] = bool(metrics[key])
    if not any(base.values()):
        base["area"] = True
    return base
