from __future__ import annotations

import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
from napari.utils import progress
from napari.utils.notifications import show_info as _show_info, show_warning as _show_warning
from skimage.segmentation import find_boundaries, relabel_sequential
from skimage.transform import resize

try:
    from cellpose.models import CellposeModel
except Exception:  # module not available until env has cellpose installed
    CellposeModel = Any  # type: ignore
from musclequant.config import (
    DEFAULT_MODEL,
    DEFAULT_DIAMETER,
    FLOW_THRESH,
    CELLPROB_THRESH,
    CHUNK_COUNT_DEFAULT,
    TILE_SIZE_PX,
    MIN_TILE_PX,
    TILE_OVERLAP_PX,
)

# ---------------------- LAZY DEVICE / MODEL ----------------------
_DEVICE = None
_MODEL_CACHE: Optional[CellposeModel] = None
_MODEL_NAME: Optional[str] = DEFAULT_MODEL
_SEGMENT_RUNNING = False
_SEGMENT_RUN_COUNTER = 0


def show_info(message: str, *, duration: float = 6.0, **kwargs):
    """Info notification that auto-dismisses after a short duration (best-effort)."""
    try:
        return _show_info(message, duration=duration, **kwargs)
    except TypeError:
        return _show_info(message, **kwargs)


def show_warning(message: str, *, duration: float = 8.0, **kwargs):
    """Warning notification that auto-dismisses after a short duration (best-effort)."""
    try:
        return _show_warning(message, duration=duration, **kwargs)
    except TypeError:
        return _show_warning(message, **kwargs)


def get_device():
    """Pick a device lazily the first time Cellpose is actually used."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    import torch  # heavy import done only once, and only if needed
    import os

    # Prefer MPS, then CUDA, then CPU. Force float32 default to avoid bfloat16 on MPS.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.set_default_dtype(torch.float32)

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
    global _MODEL_CACHE, _MODEL_NAME
    if _MODEL_CACHE is not None and _MODEL_NAME == pretrained:
        return _MODEL_CACHE

    from cellpose import models  # heavy import done only once, on first use

    device = get_device()
    _MODEL_NAME = pretrained
    _MODEL_CACHE = models.CellposeModel(pretrained_model=pretrained, device=device)
    return _MODEL_CACHE


def overlay_boundaries(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    b = find_boundaries(labels, mode="outer")
    rgb = np.dstack([gray, gray, gray])
    rgb[b] = [1.0, 0.0, 0.0]
    return (rgb * 255).astype(np.uint8)


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg


def _console_progress(pct: float, prefix: str = "Cellpose"):
    pct = max(0.0, min(100.0, float(pct)))
    sys.stdout.write(f"\r[{prefix}] {pct:5.1f}%")
    sys.stdout.flush()
    if pct >= 100.0:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _resize_to_target_diameter(img: np.ndarray, current_diameter_px: float, target_diameter_px: float):
    """Rescale image so the estimated diameter matches ``target_diameter_px``.

    Returns the resized image, the applied scale factor, and the diameter fed to Cellpose.
    """
    if current_diameter_px is None or current_diameter_px <= 0:
        return img, 1.0, current_diameter_px
    if target_diameter_px is None or target_diameter_px <= 0:
        return img, 1.0, current_diameter_px

    scale = float(target_diameter_px) / float(current_diameter_px)
    # Avoid extreme resizes that would explode memory or obliterate detail
    scale = max(min(scale, 8.0), 0.1)
    if abs(scale - 1.0) < 0.02:
        return img, 1.0, float(target_diameter_px)

    new_h = max(1, int(round(img.shape[0] * scale)))
    new_w = max(1, int(round(img.shape[1] * scale)))
    out_shape = (new_h, new_w) if img.ndim == 2 else (new_h, new_w, img.shape[2])
    resized = resize(img, out_shape, order=1, preserve_range=True, anti_aliasing=True)
    resized = resized.astype(img.dtype, copy=False)
    return resized, scale, float(target_diameter_px)


def _run_in_worker(fn, *args, **kwargs):
    """Run a function in a background thread while keeping the Qt UI responsive."""
    result: Dict[str, Any] = {}
    error: Dict[str, Exception] = {}

    def _target():
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - pass through to caller
            error["err"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    while t.is_alive():
        if app is not None:
            app.processEvents()
        time.sleep(0.02)
    if error:
        raise error["err"]
    return result.get("value")


def _pbar_update(pbar, target_pct: float):
    """Update napari progress bar using integer values to satisfy Qt."""
    if pbar is None:
        return
    target_i = int(round(target_pct))
    delta = target_i - int(pbar.n)
    if delta > 0:
        pbar.update(delta)


def _tile_and_merge_masks(
    img: np.ndarray,
    evaluator,
    *,
    diameter,
    flow_thr,
    cellprob_thr,
    tile_px: int = TILE_SIZE_PX,
    overlap_px: int = TILE_OVERLAP_PX,
    pbar=None,
    progress_start: float = 0.0,
    progress_end: float = 100.0,
    existing_mask: np.ndarray | None = None,
    existing_owner: np.ndarray | None = None,
    initial_offset: int | None = None,
    tiles: list[tuple[int, int, int, int]] | None = None,
    return_owner: bool = False,
    allowed_replace_mask: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, int]:
    """Run Cellpose on overlapping tiles and merge results.

    When ``existing_mask``/``existing_owner`` are provided, new tiles are merged
    into the pre-populated arrays so we can re-use the same distance-based
    tie-breaking for refinement passes.
    """
    h, w = img.shape[:2]
    merged = np.array(existing_mask, copy=True) if existing_mask is not None else np.zeros((h, w), dtype=np.uint32)
    owner = np.array(existing_owner, copy=True) if existing_owner is not None else np.full((h, w), -1, dtype=np.int64)
    offset = int(initial_offset) if initial_offset is not None else int(merged.max())
    tile_id = 0

    def _run(arr: np.ndarray) -> np.ndarray:
        out = evaluator(arr, diameter=diameter, flow_thr=flow_thr, cellprob_thr=cellprob_thr)
        if isinstance(out, tuple):
            return out[0]
        return out

    step = max(tile_px - overlap_px, 1)
    if tiles is None:
        tiles = []
        for y0 in range(0, h, step):
            y1 = min(h, y0 + tile_px)
            for x0 in range(0, w, step):
                x1 = min(w, x0 + tile_px)
                tiles.append((y0, x0, y1, x1))

    total_tiles = max(len(tiles), 1)
    step_percent = (progress_end - progress_start) / float(total_tiles)

    for y0, x0, y1, x1 in tiles:
        tile = img[y0:y1, x0:x1]
        tmask = _run(tile)
        if tmask is None:
            tile_id += 1
            _pbar_update(pbar, progress_start + tile_id * step_percent)
            _console_progress(progress_start + tile_id * step_percent)
            print(f"[Cellpose] Tile {tile_id}/{total_tiles} ({tile_px}px) skipped (empty)")
            continue
        tmask = tmask.astype(np.uint32, copy=False)
        if tmask.max() > 0:
            tmask[tmask > 0] += offset
            offset += int(tmask.max())

        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = ((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64, copy=False)
        sub_owner = owner[y0:y1, x0:x1]
        replace = (sub_owner < 0) | (dist < np.maximum(sub_owner, 0))
        if allowed_replace_mask is not None:
            replace = replace & allowed_replace_mask[y0:y1, x0:x1]
        sub_mask = merged[y0:y1, x0:x1]
        sub_mask[replace] = tmask[replace]
        sub_owner[replace] = dist[replace].astype(np.int64)
        merged[y0:y1, x0:x1] = sub_mask
        owner[y0:y1, x0:x1] = sub_owner
        tile_id += 1
        _pbar_update(pbar, progress_start + tile_id * step_percent)
        _console_progress(progress_start + tile_id * step_percent)
        print(f"[Cellpose] Tile {tile_id}/{total_tiles} ({tile_px}px) done")
    if offset >= 2**16:
        show_warning(f"Tiled segmentation produced {offset} labels; clipping to uint16.")
    merged = merged.astype(np.uint16)
    if return_owner:
        return merged, owner, offset
    return merged


def _border_tiles(h: int, w: int, *, tile_px: int, overlap_px: int, band_px: int | None = None):
    """Generate thin stripes centered on tile seams for border refinement."""
    if band_px is None:
        band_px = max(1, min(overlap_px, 8))
    step = max(tile_px - overlap_px, 1)
    seams = []
    for y in range(step, h, step):
        y0 = max(0, y - band_px)
        y1 = min(h, y + band_px)
        seams.append((y0, 0, y1, w))
    for x in range(step, w, step):
        x0 = max(0, x - band_px)
        x1 = min(w, x + band_px)
        seams.append((0, x0, h, x1))
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for t in seams:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def _resegment_tile_borders(
    img: np.ndarray,
    evaluator,
    *,
    diameter,
    flow_thr,
    cellprob_thr,
    tile_px: int,
    overlap_px: int,
    base_mask: np.ndarray,
    base_owner: np.ndarray,
    offset: int,
    pbar=None,
    progress_start: float = 0.0,
    progress_end: float = 5.0,
):
    """Refine seams by re-segmenting thin stripes along tile borders."""
    h, w = img.shape[:2]
    seams = _border_tiles(h, w, tile_px=tile_px, overlap_px=overlap_px)
    if not seams:
        return base_mask, base_owner, offset

    seam_mask = np.zeros((h, w), dtype=bool)
    for y0, x0, y1, x1 in seams:
        seam_mask[y0:y1, x0:x1] = True

    # Only replace cells that are cut cleanly along the seam lines (straight edges).
    touched_labels = np.unique(base_mask[seam_mask])
    touched_labels = touched_labels[touched_labels != 0]
    eligible = set()
    for lbl in touched_labels:
        yy, xx = np.nonzero((base_mask == lbl) & seam_mask)
        if yy.size == 0:
            continue
        if np.ptp(xx) <= 1 or np.ptp(yy) <= 1:
            eligible.add(int(lbl))

    eligible_mask = np.isin(base_mask, list(eligible)) if eligible else np.zeros_like(base_mask, dtype=bool)
    allowed_replace_mask = seam_mask & (eligible_mask | (base_mask == 0))

    show_info(f"Refining seams with {len(seams)} narrow border tiles…")
    print(f"[Cellpose] Refining tile borders with {len(seams)} stripes (tile_px={tile_px}, overlap={overlap_px})")
    refined, owner, new_offset = _tile_and_merge_masks(
        img,
        evaluator,
        diameter=diameter,
        flow_thr=flow_thr,
        cellprob_thr=cellprob_thr,
        tile_px=tile_px,
        overlap_px=overlap_px,
        pbar=pbar,
        progress_start=progress_start,
        progress_end=progress_end,
        existing_mask=base_mask,
        existing_owner=base_owner,
        initial_offset=offset,
        tiles=seams,
        return_owner=True,
        allowed_replace_mask=allowed_replace_mask,
    )
    return refined, owner, new_offset


def run_cellpose(
    img: np.ndarray,
    model: CellposeModel,
    diameter=DEFAULT_DIAMETER,
    flow_thr=FLOW_THRESH,
    cellprob_thr=CELLPROB_THRESH,
    chunk_first: bool = False,
    chunk_count: int = CHUNK_COUNT_DEFAULT,
    refine_tile_borders: bool = True,
    auto_resize: bool = True,
    target_diameter_px: float = DEFAULT_DIAMETER,
):
    flow_thr_local = flow_thr
    status_device = getattr(model, "device", None) or "auto"
    show_info(f"Running Cellpose on {status_device} (batch=1). Large images may take a minute…")
    print(f"[Cellpose] Starting on device={status_device}, diameter={diameter}, flow_thr={flow_thr}")
    _console_progress(0.0)

    orig_shape = img.shape
    working_img = img
    resize_scale = 1.0
    effective_diameter = diameter
    target_diam = target_diameter_px if target_diameter_px not in (None, 0) else DEFAULT_DIAMETER
    if auto_resize and diameter not in (None, 0) and target_diam not in (None, 0):
        try:
            working_img, resize_scale, effective_diameter = _resize_to_target_diameter(
                img, float(diameter), float(target_diam)
            )
            if resize_scale != 1.0:
                show_info(
                    f"Resizing for Cellpose: scale ×{resize_scale:.2f} so cells are ~{effective_diameter:.0f}px wide."
                )
                print(f"[Cellpose] Auto-resize ×{resize_scale:.3f} → ~{effective_diameter}px cells")
        except Exception as exc_resize:
            show_warning(f"Auto-resize failed; running at original scale. ({exc_resize})")
            working_img = img
            resize_scale = 1.0
            effective_diameter = diameter

    pbar = progress(total=100, desc="Cellpose segmentation", unit="%")

    def _eval(m: CellposeModel, *, arr: np.ndarray = None, diameter=None, flow_thr=None, cellprob_thr=None):
        return _run_in_worker(
            m.eval,
            working_img if arr is None else arr,
            channels=[0, 0],
            diameter=diameter,
            flow_threshold=flow_thr,
            cellprob_threshold=cellprob_thr,
            batch_size=1,
        )

    def _run_tile_pass(tile_px_local: int, progress_start: float, progress_end: float):
        """Run tiling (and optional seam refinement) as a single step."""
        tile_eval = lambda arr, diameter, flow_thr, cellprob_thr: _eval(
            model, arr=arr, diameter=diameter, flow_thr=flow_thr, cellprob_thr=cellprob_thr
        )
        tile_progress_end = progress_end
        if progress_end - progress_start > 4.0:
            tile_progress_end = progress_end - 4.0  # leave a small buffer for refinement
        tiled, owner, offset = _tile_and_merge_masks(
            working_img,
            tile_eval,
            diameter=effective_diameter,
            flow_thr=flow_thr_local,
            cellprob_thr=cellprob_thr,
            tile_px=tile_px_local,
            overlap_px=TILE_OVERLAP_PX,
            pbar=pbar,
            progress_start=progress_start,
            progress_end=tile_progress_end,
            return_owner=True,
        )
        _pbar_update(pbar, tile_progress_end)
        _console_progress(pbar.n)

        if refine_tile_borders:
            try:
                refine_end = progress_end
                tiled, _, offset = _resegment_tile_borders(
                    working_img,
                    tile_eval,
                    diameter=effective_diameter,
                    flow_thr=flow_thr_local,
                    cellprob_thr=cellprob_thr,
                    tile_px=tile_px_local,
                    overlap_px=TILE_OVERLAP_PX,
                    base_mask=tiled,
                    base_owner=owner,
                    offset=offset,
                    pbar=pbar,
                    progress_start=pbar.n,
                    progress_end=refine_end,
                )
                _pbar_update(pbar, refine_end)
                _console_progress(pbar.n)
            except Exception as exc_refine:
                show_warning(f"Tile border refinement skipped: {exc_refine}")
        return tiled

    with pbar:
        try:
            used_chunking = False
            if chunk_first and int(chunk_count) > 1:
                min_dim = min(working_img.shape[0], working_img.shape[1])
                tile_px = max(MIN_TILE_PX, min(TILE_SIZE_PX, int(min_dim / int(chunk_count))))
                show_info(f"Cellpose: chunk-first tiling at ~{tile_px}px")
                print(f"[Cellpose] Chunk-first tiling tile_px={tile_px}, chunks={chunk_count}")
                try:
                    out = _run_tile_pass(tile_px, pbar.n, 95.0)
                    show_info(f"Tiled GPU segmentation succeeded at {tile_px}px (with seam refinement).")
                    print(f"[Cellpose] Chunk-first tiling succeeded at {tile_px}px")
                    used_chunking = True
                except RuntimeError as exc_tile:
                    if not _is_oom_error(exc_tile):
                        raise
                    print("[Cellpose] Chunk-first tiling OOM; falling back to full-image path")

            if not used_chunking:
                show_info("Cellpose: full-image pass on current device…")
                print("[Cellpose] Full-image pass")
                out = _eval(model, diameter=effective_diameter, flow_thr=flow_thr_local, cellprob_thr=cellprob_thr)
                _pbar_update(pbar, 100)
                _console_progress(100.0)
        except RuntimeError as exc:
            import torch

            if _is_oom_error(exc) and model is not None:
                print("Cellpose OOM on device", getattr(model, "device", None), "→ retrying with GPU tiles")
                tile_px = TILE_SIZE_PX
                while tile_px >= MIN_TILE_PX:
                    show_warning(f"Cellpose OOM; retrying with {tile_px}px GPU tiles (overlap {TILE_OVERLAP_PX}px)…")
                    print(f"[Cellpose] OOM → tiling at {tile_px}px with overlap {TILE_OVERLAP_PX}px")
                    try:
                        out = _run_tile_pass(tile_px, pbar.n, 95.0)
                        show_info(f"Tiled GPU segmentation succeeded at {tile_px}px.")
                        print(f"[Cellpose] Tiled GPU segmentation succeeded at {tile_px}px")
                        break
                    except RuntimeError as exc_tile:
                        if not _is_oom_error(exc_tile):
                            raise
                        tile_px = tile_px // 2
                else:
                    print("Tiled GPU segmentation also OOM at minimum tile size → falling back to CPU")
                    if flow_thr_local and flow_thr_local > 0:
                        flow_thr_local = 0.0
                    show_warning("Cellpose ran out of GPU/MPS memory; retrying on CPU (QC flow disabled for speed).")
                    print("[Cellpose] Tiled GPU still OOM → CPU fallback (flow QC off)")
                    try:
                        from cellpose import models as _models

                        cpu = torch.device("cpu")
                        global _MODEL_CACHE, _DEVICE
                        _DEVICE = cpu
                        if hasattr(torch, "mps"):
                            try:
                                torch.mps.empty_cache()
                            except Exception:
                                pass

                        _MODEL_CACHE = _models.CellposeModel(
                            pretrained_model=getattr(model, "pretrained_model", _MODEL_NAME),
                            device=cpu,
                        )
                        show_info("Cellpose now running on CPU; this can take a few minutes for large slides.")
                        out = _eval(
                            _MODEL_CACHE,
                            diameter=effective_diameter,
                            flow_thr=flow_thr_local,
                            cellprob_thr=cellprob_thr,
                        )
                        _pbar_update(pbar, 95.0)
                        _console_progress(pbar.n)
                    except Exception:
                        raise
            else:
                raise

        if isinstance(out, tuple):
            out = out[0]
        if resize_scale != 1.0 and out is not None:
            out = resize(out, orig_shape[:2], order=0, preserve_range=True, anti_aliasing=False)
            out = out.astype(np.uint16, copy=False)
            print(f"[Cellpose] Resized masks back to original shape {orig_shape} from scale ×{resize_scale:.3f}")
        if pbar.n < 100:
            _pbar_update(pbar, 100)
        _console_progress(100.0)
        return out  # masks
    raise RuntimeError("Unexpected Cellpose eval() return.")


def clean_mask(mask: np.ndarray, min_area_px: int = 0, drop_border: bool = False, edge_buffer_px: int = 0):
    lab = mask.copy().astype(np.int32)
    removed_small = 0
    removed_border = 0

    if drop_border:
        buf = max(0, int(edge_buffer_px))
        slices = [
            lab[0 : 1 + buf, :],
            lab[-1 - buf :, :],
            lab[:, 0 : 1 + buf],
            lab[:, -1 - buf :],
        ]
        border_ids = np.unique(np.concatenate([s.ravel() for s in slices]))
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
