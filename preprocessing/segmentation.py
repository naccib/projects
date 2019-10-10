"""
This file defines methods for calculating and
segmenting bounding boxes encompassing ROIs.
"""

import nibabel as nib
import pandas as pd
import numpy as np

from skimage.transform import resize

from constants import SEG_MAP_FOLDER, FSF_COLOR_LUT, RAW_SHAPE, fix_segmentation_dimensions
from typing import NewType
from pathlib import Path

BoundingBox = NewType('BoundingBox', ((int, int), (int, int), (int, int)))
LOG = lambda s: print(f"[SEG] {s}")

# Load and resize segmentation map

SEG_MAP = 'aparc.DKTatlas+aseg'

LOG(f"Loading segmentation map {SEG_MAP}...")
SEG_MAP = nib.load(str(SEG_MAP_FOLDER.joinpath(f"{SEG_MAP}.mgz"))).get_data()

LOG(f"Map shape is {SEG_MAP.shape}, resizing to (193, 229, 193)...")
SEG_MAP = resize(SEG_MAP, RAW_SHAPE, order=0, anti_aliasing=False, preserve_range=True)

LOG(f"Swapping dimensions to match images...")
SEG_MAP = np.transpose(SEG_MAP, (2, 1, 0))

# Load the lookup table

LOG(f'Loading LUT from {FSF_COLOR_LUT}...')

LUT = np.loadtxt(FSF_COLOR_LUT, dtype='i4,S32,i2,i2,i2,i2')
LUT = pd.DataFrame(LUT)

LUT.columns=['No', 'Label', 'R', 'G', 'B', 'A']
LUT.set_index('No', drop=True, inplace=True)


def find_bounding_box(struct_name: str) -> (BoundingBox, np.ndarray):
    """
    Returns the 3D bounding box of a structure.
    The bounding box is formatted as (x0, x1), (y0, y1), (z0, z1)
    """

    value = struct_value(struct_name)

    LOG(f"Finding bounding box for struct {struct_name} ({value})...")

    x_min = SEG_MAP.shape[0]
    y_min = SEG_MAP.shape[1]
    z_min = SEG_MAP.shape[2]

    x_max = 0.0
    y_max = 0.0
    z_max = 0.0

    for x in range(0, SEG_MAP.shape[0]):
        for y in range(0, SEG_MAP.shape[1]):
            for z in range(0, SEG_MAP.shape[2]):
                if SEG_MAP[x, y, z] == value:
                    if x > x_max:
                        x_max = x
                    elif x < x_min:
                        x_min = x

                    if y > y_max:
                        y_max = y
                    elif y < y_min:
                        y_min = y

                    if z > z_max:
                        z_max = z
                    elif z < z_min:
                        z_min = z

    mask = np.zeros(SEG_MAP.shape, dtype=bool)
    mask[x_min : x_max, y_min : y_max, z_min : z_max] = True

    return (((x_min, x_max), (y_min, y_max), (z_min, z_max)), mask)


def view_bounding_mask(mri: np.ndarray, mask: np.ndarray, box_color: tuple = (1.0, 0.0, 0.0)) -> np.ndarray:
    """
    Blends the bounding box on a given MRI and returns the resultant RGB image.
    """

    for i in range(2):
        assert mri.shape[i] == mask.shape[i]

    mri = normalize_view(mri)
    output = np.stack((mri,)*3, axis=-1)

    for x in range(0, mri.shape[0]):
        for y in range(0, mri.shape[1]):
            for z in range(0, mri.shape[2]):
                if mask[x, y, z]:
                    output[x, y, z] = blend_pixels(output[x, y, z], box_color, 0.8) 

    return output


def normalize_view(image: np.ndarray) -> np.ndarray:
    """
    Normalizes an image to fit in the (0..1) interval.
    """

    min = image.min()
    max = image.max()

    m1 = (image - min)
    m2 = 1 / (max - min)

    return m1 * m2         


def struct_value(struct_name: str, lut: pd.DataFrame = None) -> int:
    """
    Returns the integer value of a structure.
    """

    if lut == None:
        lut = LUT

    index = lut.index[lut['Label'] == struct_name.encode('utf-8')]
    return index.to_numpy()[0]


def blend_pixels(a: tuple, b: tuple, e: float) -> tuple:
    """
    Blends two RGB pixels with coefficient `e`.
    """

    r = a[0] * e + b[0] * (1 - e)
    g = a[1] * e + b[1] * (1 - e)
    b = a[2] * e + b[2] * (1 - e)

    return (r, g, b)