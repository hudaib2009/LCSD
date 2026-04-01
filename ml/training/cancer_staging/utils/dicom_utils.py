"""
dicom_utils.py — DICOM loading, HU conversion, and windowing utilities.

This module handles:
  - Walking patient directories to locate CT series
  - Reading and sorting DICOM slices by ImagePositionPatient
  - Converting raw pixel values to Hounsfield Units
  - Applying lung/mediastinal windowing and normalization
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
# Directory discovery
# ---------------------------------------------------------------------------

def find_ct_series_dir(patient_dir: str) -> Optional[str]:
    """
    Walk a patient directory tree and return the path to the CT series folder.

    Heuristic: the CT series folder is the one that contains many .dcm files
    and whose first file has Modality == 'CT'. We skip directories whose name
    starts with '0.000000' (RTSTRUCT) or '300.000000' (SEG).

    Parameters
    ----------
    patient_dir : str
        Absolute path to the patient folder (e.g. .../LUNG1-001).

    Returns
    -------
    str or None
        Path to the CT series directory, or None if not found.
    """
    for root, dirs, files in os.walk(patient_dir):
        dcm_files = [f for f in files if f.endswith('.dcm')]
        if len(dcm_files) < 5:
            continue
        # Skip known non-CT directories
        basename = os.path.basename(root)
        if basename.startswith('0.000000') or basename.startswith('300.000000'):
            continue
        # Verify modality on first file
        try:
            sample = pydicom.dcmread(os.path.join(root, dcm_files[0]),
                                     stop_before_pixels=True)
            if getattr(sample, 'Modality', '') == 'CT':
                return root
        except Exception:
            continue
    return None


def load_ct_series(series_dir: str) -> Tuple[List[FileDataset], np.ndarray]:
    """
    Load all DICOM slices from a CT series directory and sort by
    ImagePositionPatient (z-coordinate). Falls back to InstanceNumber.

    Parameters
    ----------
    series_dir : str
        Path to directory containing .dcm CT files.

    Returns
    -------
    slices : list[FileDataset]
        Sorted list of pydicom datasets.
    positions : np.ndarray
        Z-positions for each slice (mm).
    """
    dcm_files = sorted(glob.glob(os.path.join(series_dir, '*.dcm')))
    slices: List[FileDataset] = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp)
            if getattr(ds, 'Modality', '') == 'CT':
                slices.append(ds)
        except Exception as e:
            logger.warning(f"Could not read {fp}: {e}")

    if not slices:
        raise ValueError(f"No CT slices found in {series_dir}")

    # Sort by ImagePositionPatient z-coordinate (preferred)
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, IndexError):
        logger.info("Falling back to InstanceNumber for sorting")
        slices.sort(key=lambda s: int(getattr(s, 'InstanceNumber', 0)))

    positions = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    return slices, positions


# ---------------------------------------------------------------------------
# HU conversion
# ---------------------------------------------------------------------------

def to_hounsfield(ds: FileDataset) -> np.ndarray:
    """
    Convert a single DICOM slice pixel array to Hounsfield Units.

    HU = pixel_value * RescaleSlope + RescaleIntercept

    Parameters
    ----------
    ds : FileDataset
        A pydicom dataset with pixel_array.

    Returns
    -------
    np.ndarray
        2-D float32 array in Hounsfield Units.
    """
    pixel_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    hu = pixel_array * slope + intercept
    return hu


def volume_to_hu(slices: List[FileDataset]) -> np.ndarray:
    """
    Convert an entire sorted slice list to a 3-D HU volume.

    Parameters
    ----------
    slices : list[FileDataset]

    Returns
    -------
    np.ndarray
        Shape (Z, H, W), dtype float32.
    """
    return np.stack([to_hounsfield(s) for s in slices], axis=0)


# ---------------------------------------------------------------------------
# Windowing / normalization
# ---------------------------------------------------------------------------

def apply_window(hu_array: np.ndarray,
                 center: float = -600.0,
                 width: float = 1500.0) -> np.ndarray:
    """
    Apply CT windowing and normalize to [0, 1].

    Parameters
    ----------
    hu_array : np.ndarray
        Array in Hounsfield Units.
    center : float
        Window center (default -600 for lung).
    width : float
        Window width (default 1500 for lung).

    Returns
    -------
    np.ndarray
        Clipped + normalized array, float32 in [0, 1].
    """
    lower = center - width / 2.0
    upper = center + width / 2.0
    clipped = np.clip(hu_array, lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    return normalized.astype(np.float32)


def hu_to_uint8(hu_array: np.ndarray,
                center: float = -600.0,
                width: float = 1500.0) -> np.ndarray:
    """
    Full pipeline: HU → windowed → uint8 for PNG export.

    Parameters
    ----------
    hu_array : np.ndarray
    center, width : float

    Returns
    -------
    np.ndarray
        uint8 array in [0, 255].
    """
    normed = apply_window(hu_array, center, width)
    return (normed * 255.0).astype(np.uint8)


def get_pixel_spacing(ds: FileDataset) -> Tuple[float, float]:
    """
    Extract pixel spacing (row_spacing, col_spacing) in mm.
    """
    ps = getattr(ds, 'PixelSpacing', [1.0, 1.0])
    return float(ps[0]), float(ps[1])


def get_image_position(ds: FileDataset) -> Tuple[float, float, float]:
    """
    Extract ImagePositionPatient (x, y, z) in mm.
    """
    ipp = getattr(ds, 'ImagePositionPatient', [0.0, 0.0, 0.0])
    return float(ipp[0]), float(ipp[1]), float(ipp[2])
