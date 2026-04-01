"""
preprocess_utils.py — Bounding-box crop, resize, PNG export, and patient
processing orchestration.

This module handles:
  - Computing bounding boxes from binary masks
  - Cropping and resizing CT slices around the tumor
  - Saving uint8 arrays as PNG
  - Full per-patient processing pipeline
"""

import os
import logging
from typing import Tuple, Optional, List, Dict

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

def compute_bbox(mask_2d: np.ndarray,
                 margin: int = 30) -> Tuple[int, int, int, int]:
    """
    Compute a bounding box around nonzero pixels in a 2-D mask.

    Parameters
    ----------
    mask_2d : np.ndarray
        2-D boolean or binary mask.
    margin : int
        Pixels to pad around the bounding box.

    Returns
    -------
    (r_min, r_max, c_min, c_max) : tuple of int
        Row and column bounds (inclusive).
    """
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    h, w = mask_2d.shape
    r_min = max(0, r_min - margin)
    r_max = min(h - 1, r_max + margin)
    c_min = max(0, c_min - margin)
    c_max = min(w - 1, c_max + margin)

    return int(r_min), int(r_max), int(c_min), int(c_max)


def crop_and_resize(image: np.ndarray,
                    bbox: Tuple[int, int, int, int],
                    target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Crop a 2-D image using a bounding box and resize to target_size.

    Parameters
    ----------
    image : np.ndarray
        2-D array (H, W), uint8 or float.
    bbox : tuple
        (r_min, r_max, c_min, c_max).
    target_size : tuple
        (height, width) for output.

    Returns
    -------
    np.ndarray
        Cropped and resized image, same dtype as input.
    """
    r_min, r_max, c_min, c_max = bbox
    cropped = image[r_min:r_max + 1, c_min:c_max + 1]
    pil_img = Image.fromarray(cropped)
    resized = pil_img.resize((target_size[1], target_size[0]),
                             Image.BILINEAR if cropped.dtype != np.uint8
                             else Image.LANCZOS)
    return np.array(resized)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_slice_png(array: np.ndarray, path: str) -> None:
    """
    Save a uint8 2-D array as PNG.

    Parameters
    ----------
    array : np.ndarray
        2-D uint8 array.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)


# ---------------------------------------------------------------------------
# Per-patient processing
# ---------------------------------------------------------------------------

def process_patient(patient_id: str,
                    patient_dir: str,
                    output_dir: str,
                    window_center: float = -600.0,
                    window_width: float = 1500.0,
                    crop_margin: int = 30,
                    target_size: Tuple[int, int] = (224, 224)
                    ) -> List[Dict]:
    """
    Full processing pipeline for one patient:
      1. Load CT series
      2. Convert to HU + windowed uint8
      3. Extract tumor mask (RTSTRUCT → SEG fallback)
      4. For slices with tumor: crop around tumor, resize, save
      5. For slices without tumor: save full-size windowed image (resized)
      6. Save masks for tumor slices

    Parameters
    ----------
    patient_id : str
        e.g. 'LUNG1-001'
    patient_dir : str
        Absolute path to patient folder.
    output_dir : str
        Base output directory (processed_data/).
    window_center, window_width : float
        CT windowing parameters.
    crop_margin : int
        Margin in pixels around tumor bbox.
    target_size : tuple
        (H, W) for output images.

    Returns
    -------
    list of dict
        One dict per saved slice with keys:
        patient_id, slice_index, path_image, path_mask, has_tumor
    """
    from .dicom_utils import (find_ct_series_dir, load_ct_series,
                              volume_to_hu, hu_to_uint8)
    from .rtstruct_utils import get_tumor_mask

    records = []
    patient_out = os.path.join(output_dir, patient_id)
    os.makedirs(patient_out, exist_ok=True)

    # 1. Find CT series
    ct_dir = find_ct_series_dir(patient_dir)
    if ct_dir is None:
        logger.error(f"No CT series found for {patient_id}")
        return records

    # 2. Load and convert
    try:
        ct_slices, z_positions = load_ct_series(ct_dir)
    except Exception as e:
        logger.error(f"Failed to load CT for {patient_id}: {e}")
        return records

    hu_volume = volume_to_hu(ct_slices)
    uint8_volume = hu_to_uint8(hu_volume, window_center, window_width)

    # 3. Get tumor mask
    tumor_mask = get_tumor_mask(patient_dir, ct_slices)

    # 4. Process each slice
    for i in range(len(ct_slices)):
        slice_img = uint8_volume[i]
        has_tumor = False
        mask_path = ""
        img_name = f"img_slice_{i:04d}.png"
        mask_name = f"mask_slice_{i:04d}.png"
        img_path = os.path.join(patient_out, img_name)
        mask_path_full = os.path.join(patient_out, mask_name)

        if tumor_mask is not None and tumor_mask[i].any():
            has_tumor = True
            slice_mask = tumor_mask[i].astype(np.uint8) * 255

            # Compute bbox and crop
            try:
                bbox = compute_bbox(tumor_mask[i], margin=crop_margin)
                cropped_img = crop_and_resize(slice_img, bbox, target_size)
                cropped_mask = crop_and_resize(slice_mask, bbox, target_size)
                # Binarize mask after resize
                cropped_mask = (cropped_mask > 127).astype(np.uint8) * 255
            except Exception as e:
                logger.warning(f"{patient_id} slice {i} crop failed: {e}")
                cropped_img = np.array(
                    Image.fromarray(slice_img).resize(
                        (target_size[1], target_size[0]), Image.LANCZOS))
                cropped_mask = np.array(
                    Image.fromarray(slice_mask).resize(
                        (target_size[1], target_size[0]), Image.NEAREST))

            save_slice_png(cropped_img, img_path)
            save_slice_png(cropped_mask, mask_path_full)
            mask_path = os.path.relpath(mask_path_full, output_dir)
        else:
            # No tumor on this slice — save resized full image
            resized = np.array(
                Image.fromarray(slice_img).resize(
                    (target_size[1], target_size[0]), Image.LANCZOS))
            save_slice_png(resized, img_path)
            mask_path = ""

        records.append({
            'patient_id': patient_id,
            'slice_index': i,
            'path_image': os.path.relpath(img_path, output_dir),
            'path_mask': mask_path,
            'has_tumor': has_tumor,
        })

    logger.info(f"{patient_id}: saved {len(records)} slices, "
                f"{sum(r['has_tumor'] for r in records)} with tumor")
    return records
