"""
rtstruct_utils.py — RTSTRUCT / SEG parsing and GTV-1 mask rasterization.

This module handles:
  - Locating RTSTRUCT and DICOM SEG files within a patient directory
  - Extracting GTV-1 ROI contours from RTSTRUCT
  - Rasterizing polygon contours into 3-D binary masks aligned with CT geometry
  - Loading DICOM SEG masks as fallback
"""

import os
import glob
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import pydicom
from pydicom.dataset import FileDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_rtstruct_file(patient_dir: str) -> Optional[str]:
    """
    Locate the RTSTRUCT DICOM file within a patient directory.

    Strategy: look for directories named like '0.000000-NA-*' first,
    then brute-force scan for Modality == 'RTSTRUCT'.

    Returns
    -------
    str or None
    """
    for root, dirs, files in os.walk(patient_dir):
        # Prefer the 0.000000-NA-* convention
        basename = os.path.basename(root)
        if basename.startswith('0.000000'):
            for f in files:
                if f.endswith('.dcm'):
                    fp = os.path.join(root, f)
                    try:
                        ds = pydicom.dcmread(fp, stop_before_pixels=True)
                        if getattr(ds, 'Modality', '') == 'RTSTRUCT':
                            return fp
                    except Exception:
                        continue

    # Fallback: scan everything
    for root, dirs, files in os.walk(patient_dir):
        for f in files:
            if not f.endswith('.dcm'):
                continue
            fp = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True)
                if getattr(ds, 'Modality', '') == 'RTSTRUCT':
                    return fp
            except Exception:
                continue
    return None


def find_seg_file(patient_dir: str) -> Optional[str]:
    """
    Locate a DICOM SEG file within a patient directory.

    Strategy: look for directories named '300.000000-Segmentation-*' first.

    Returns
    -------
    str or None
    """
    for root, dirs, files in os.walk(patient_dir):
        basename = os.path.basename(root)
        if 'Segmentation' in basename or basename.startswith('300.000000'):
            for f in files:
                if f.endswith('.dcm'):
                    fp = os.path.join(root, f)
                    try:
                        ds = pydicom.dcmread(fp, stop_before_pixels=True)
                        if getattr(ds, 'Modality', '') == 'SEG':
                            return fp
                    except Exception:
                        continue
    return None


# ---------------------------------------------------------------------------
# RTSTRUCT ROI extraction
# ---------------------------------------------------------------------------

def get_gtv1_roi_number(rt_ds: FileDataset) -> Optional[int]:
    """
    Find the ROI number for GTV-1 (or fallback containing 'GTV').

    Search order:
      1. Exact match 'GTV-1'
      2. Contains 'GTV' AND '1'
      3. First ROI containing 'GTV'

    Returns
    -------
    int or None
    """
    if not hasattr(rt_ds, 'StructureSetROISequence'):
        return None

    rois = rt_ds.StructureSetROISequence
    # Pass 1: exact
    for roi in rois:
        name = getattr(roi, 'ROIName', '').strip()
        if name.upper() == 'GTV-1':
            return int(roi.ROINumber)
    # Pass 2: contains GTV + 1
    for roi in rois:
        name = getattr(roi, 'ROIName', '').upper()
        if 'GTV' in name and '1' in name:
            return int(roi.ROINumber)
    # Pass 3: any GTV
    for roi in rois:
        name = getattr(roi, 'ROIName', '').upper()
        if 'GTV' in name:
            return int(roi.ROINumber)
    return None


def get_contours_for_roi(rt_ds: FileDataset,
                         roi_number: int) -> Dict[str, np.ndarray]:
    """
    Extract contour data for a specific ROI number.

    Parameters
    ----------
    rt_ds : FileDataset
        RTSTRUCT dataset.
    roi_number : int
        Target ROI number.

    Returns
    -------
    dict
        Maps ReferencedSOPInstanceUID → np.ndarray of shape (N, 3) with
        (x, y, z) coordinates in mm.
    """
    contours: Dict[str, np.ndarray] = {}
    if not hasattr(rt_ds, 'ROIContourSequence'):
        return contours

    for roi_contour in rt_ds.ROIContourSequence:
        if int(roi_contour.ReferencedROINumber) != roi_number:
            continue
        if not hasattr(roi_contour, 'ContourSequence'):
            continue
        for contour in roi_contour.ContourSequence:
            # Each contour belongs to a specific slice
            ref_uid = None
            if hasattr(contour, 'ContourImageSequence'):
                ref_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            data = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
            if ref_uid is None:
                # Use the z-coordinate as a key fallback
                ref_uid = f"z_{data[0, 2]:.4f}"
            if ref_uid in contours:
                # Multiple contours on same slice — store as list of arrays
                if isinstance(contours[ref_uid], list):
                    contours[ref_uid].append(data)
                else:
                    contours[ref_uid] = [contours[ref_uid], data]
            else:
                contours[ref_uid] = data
    return contours


# ---------------------------------------------------------------------------
# Contour rasterization
# ---------------------------------------------------------------------------

def _contour_to_mask(contour_pts: np.ndarray,
                     origin_xy: Tuple[float, float],
                     pixel_spacing: Tuple[float, float],
                     shape: Tuple[int, int]) -> np.ndarray:
    """
    Rasterize a single closed contour polygon into a 2-D binary mask.

    Uses matplotlib.path to test which pixels are inside the polygon.

    Parameters
    ----------
    contour_pts : np.ndarray
        (N, 3) array of (x, y, z) coordinates in mm (z ignored).
    origin_xy : tuple
        (x0, y0) from ImagePositionPatient.
    pixel_spacing : tuple
        (row_spacing, col_spacing) in mm.
    shape : tuple
        (rows, cols) of the CT slice.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (rows, cols).
    """
    from matplotlib.path import Path

    rows, cols = shape
    row_sp, col_sp = pixel_spacing
    x0, y0 = origin_xy

    # Convert mm coordinates to pixel indices
    col_indices = (contour_pts[:, 0] - x0) / col_sp
    row_indices = (contour_pts[:, 1] - y0) / row_sp

    # Build polygon path
    polygon = np.column_stack([col_indices, row_indices])
    path = Path(polygon)

    # Create grid of pixel centers
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.column_stack([cc.ravel(), rr.ravel()])

    mask = path.contains_points(points).reshape(rows, cols)
    return mask


def rasterize_contours(contours: Dict[str, np.ndarray],
                       ct_slices: List[FileDataset]) -> np.ndarray:
    """
    Convert RTSTRUCT polygon contours into a 3-D binary mask aligned
    with the CT volume.

    Parameters
    ----------
    contours : dict
        Output of get_contours_for_roi(). Maps SOP UID (or z key) →
        contour array(s).
    ct_slices : list[FileDataset]
        Sorted CT slices (same order as the volume).

    Returns
    -------
    np.ndarray
        Binary mask of shape (Z, H, W), dtype bool.
    """
    rows = int(ct_slices[0].Rows)
    cols = int(ct_slices[0].Columns)
    n_slices = len(ct_slices)
    mask_volume = np.zeros((n_slices, rows, cols), dtype=bool)

    # Build lookup from SOP UID → slice index
    uid_to_idx: Dict[str, int] = {}
    z_coords: List[float] = []
    for i, s in enumerate(ct_slices):
        uid_to_idx[s.SOPInstanceUID] = i
        z_coords.append(float(s.ImagePositionPatient[2]))
    z_coords_arr = np.array(z_coords)

    for ref_key, contour_data in contours.items():
        # Find slice index
        if ref_key in uid_to_idx:
            idx = uid_to_idx[ref_key]
        elif ref_key.startswith('z_'):
            z_val = float(ref_key[2:])
            idx = int(np.argmin(np.abs(z_coords_arr - z_val)))
        else:
            logger.warning(f"Could not map contour key {ref_key} to a slice")
            continue

        ds = ct_slices[idx]
        ipp = ds.ImagePositionPatient
        origin_xy = (float(ipp[0]), float(ipp[1]))
        ps = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        pixel_spacing = (float(ps[0]), float(ps[1]))
        shape = (rows, cols)

        # Handle single or multiple contours on this slice
        contour_list = contour_data if isinstance(contour_data, list) else [contour_data]
        for cpts in contour_list:
            try:
                slice_mask = _contour_to_mask(cpts, origin_xy, pixel_spacing, shape)
                # XOR to handle donut-shaped contours
                mask_volume[idx] = np.logical_xor(mask_volume[idx], slice_mask)
            except Exception as e:
                logger.warning(f"Rasterization failed for slice {idx}: {e}")

    return mask_volume


# ---------------------------------------------------------------------------
# DICOM SEG fallback
# ---------------------------------------------------------------------------

def load_seg_mask(seg_path: str,
                  ct_slices: List[FileDataset]) -> Optional[np.ndarray]:
    """
    Load a DICOM SEG file and return a 3-D binary mask aligned with the
    CT volume.

    This is a best-effort fallback — SEG parsing can be complex.

    Parameters
    ----------
    seg_path : str
        Path to the SEG .dcm file.
    ct_slices : list[FileDataset]
        Sorted CT slices.

    Returns
    -------
    np.ndarray or None
        Binary mask of shape (Z, H, W), or None if parsing fails.
    """
    try:
        seg_ds = pydicom.dcmread(seg_path)
        pixel_array = seg_ds.pixel_array  # (frames, H, W) or (H, W)

        n_ct = len(ct_slices)
        rows = int(ct_slices[0].Rows)
        cols = int(ct_slices[0].Columns)

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]

        n_frames = pixel_array.shape[0]

        # Try to align frames with CT slices
        mask_volume = np.zeros((n_ct, rows, cols), dtype=bool)

        if n_frames == n_ct:
            # Assume 1:1 correspondence
            for i in range(n_frames):
                frame = pixel_array[i]
                if frame.shape != (rows, cols):
                    # Resize if dimensions differ
                    from PIL import Image
                    frame = np.array(Image.fromarray(frame.astype(np.uint8)).resize(
                        (cols, rows), Image.NEAREST))
                mask_volume[i] = frame > 0
        else:
            # Try using PerFrameFunctionalGroupsSequence for mapping
            if hasattr(seg_ds, 'PerFrameFunctionalGroupsSequence'):
                ct_z = np.array([float(s.ImagePositionPatient[2])
                                 for s in ct_slices])
                for frame_idx, fg in enumerate(
                        seg_ds.PerFrameFunctionalGroupsSequence):
                    try:
                        pos = fg.PlanePositionSequence[0] \
                                .ImagePositionPatient
                        z = float(pos[2])
                        ct_idx = int(np.argmin(np.abs(ct_z - z)))
                        frame = pixel_array[frame_idx]
                        if frame.shape != (rows, cols):
                            from PIL import Image
                            frame = np.array(
                                Image.fromarray(frame.astype(np.uint8)).resize(
                                    (cols, rows), Image.NEAREST))
                        mask_volume[ct_idx] = np.logical_or(
                            mask_volume[ct_idx], frame > 0)
                    except Exception:
                        continue
            else:
                logger.warning("SEG frame count doesn't match CT and no "
                               "PerFrameFunctionalGroupsSequence found.")
                return None

        return mask_volume

    except Exception as e:
        logger.warning(f"Failed to load SEG from {seg_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def get_tumor_mask(patient_dir: str,
                   ct_slices: List[FileDataset]) -> Optional[np.ndarray]:
    """
    Attempt to get a GTV-1 tumor mask for a patient.

    Tries RTSTRUCT first, then DICOM SEG fallback.

    Parameters
    ----------
    patient_dir : str
    ct_slices : list[FileDataset]

    Returns
    -------
    np.ndarray or None
        Binary mask (Z, H, W), or None if nothing found.
    """
    # Try RTSTRUCT
    rt_path = find_rtstruct_file(patient_dir)
    if rt_path is not None:
        try:
            rt_ds = pydicom.dcmread(rt_path)
            roi_num = get_gtv1_roi_number(rt_ds)
            if roi_num is not None:
                contours = get_contours_for_roi(rt_ds, roi_num)
                if contours:
                    mask = rasterize_contours(contours, ct_slices)
                    n_positive = mask.sum()
                    if n_positive > 0:
                        logger.info(f"RTSTRUCT mask: {n_positive} positive "
                                    f"voxels across {(mask.any(axis=(1,2))).sum()} slices")
                        return mask
                    else:
                        logger.warning("RTSTRUCT parsed but mask is empty")
        except Exception as e:
            logger.warning(f"RTSTRUCT parsing failed: {e}")

    # Fallback to SEG
    seg_path = find_seg_file(patient_dir)
    if seg_path is not None:
        logger.info("Falling back to DICOM SEG")
        mask = load_seg_mask(seg_path, ct_slices)
        if mask is not None and mask.sum() > 0:
            return mask

    logger.warning(f"No tumor mask found for {patient_dir}")
    return None
