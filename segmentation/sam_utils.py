"""
SAM Utilities
Helper functions for mask processing, box operations, and format conversion
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
from pycocotools import mask as mask_utils


def load_voc_box(xml_path: str) -> tuple:
    """
    Load bounding box from VOC XML annotation file

    Args:
        xml_path: Path to VOC XML file

    Returns:
        Tuple (xmin, ymin, xmax, ymax) or None if not found
    """
    if not os.path.isfile(xml_path):
        return None
    try:
        root = ET.parse(xml_path).getroot()
        obj = root.find("object")
        if obj is None:
            return None
        bndbox = obj.find("bndbox")
        if bndbox is None:
            return None
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        return xmin, ymin, xmax, ymax
    except Exception:
        return None


def apply_clip_to_box(mask_bool: np.ndarray, box: tuple, H: int, W: int) -> np.ndarray:
    """
    Clip mask to bounding box

    Args:
        mask_bool: Boolean mask array
        box: Bounding box (xmin, ymin, xmax, ymax)
        H: Image height
        W: Image width

    Returns:
        Clipped mask
    """
    if box is None:
        return mask_bool

    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(W - 1, xmin))
    xmax = max(0, min(W - 1, xmax))
    ymin = max(0, min(H - 1, ymin))
    ymax = max(0, min(H - 1, ymax))

    box_mask = np.zeros((H, W), dtype=bool)
    box_mask[ymin : ymax + 1, xmin : xmax + 1] = True
    return mask_bool & box_mask


def mask_to_rle(mask_bool: np.ndarray) -> dict:
    """
    Convert boolean mask to RLE (Run-Length Encoding) format

    Args:
        mask_bool: Boolean mask array

    Returns:
        RLE dictionary with 'size' and 'counts'
    """
    rle = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def rle_to_mask(rle: dict) -> np.ndarray:
    """
    Convert RLE to boolean mask

    Args:
        rle: RLE dictionary with 'size' and 'counts'

    Returns:
        Boolean mask array
    """
    counts = rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]
    mask = mask_utils.decode({"size": rle["size"], "counts": counts})
    return mask.astype(bool)
