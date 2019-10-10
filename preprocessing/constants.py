"""
Defines project-wide constants.
"""

import numpy as np

from pathlib import Path

DATA_FOLDER      = Path(f'{Path.home()}/data/data/ds000030_R1.0.5/derivatives/fmriprep')
METADATA_FILE    = Path(f'{Path.home()}/data/data/ds000030_R1.0.5/participants.tsv')

SEG_MAP_FOLDER   = Path(f"{Path.home()}/data/data/ds000030_R1.0.5/derivatives/fmriprep/map/")
FSF_COLOR_LUT    = Path(f"{Path.home()}/data/data/ds000030_R1.0.5/FreeSurferColorLUT.txt")

OUT_FOLDER       = Path(f'{Path.home()}/.cache/out')

# Bounding box stuff

RAW_SHAPE = (193, 229, 193)


def build_mask(X, Y, Z) -> np.ndarray:
    mask = np.zeros(RAW_SHAPE, dtype=bool)
    mask[X[0] : X[1], Y[0] : Y[1], Z[0] : Z[1]] = True

    return mask


def fix_segmentation_dimensions(seg_map: np.ndarray) -> np.ndarray:
    result = np.ndarray(seg_map.shape)

    for x in range(seg_map.shape[0]):
        for y in range(seg_map.shape[1]):
            for z in range(seg_map.shape[2]):
                result[x, y, z] = seg_map[z, y, x]

    return result


BoundingMasks = {
    'Left-Thalamus': build_mask((97, 117), (113, 136), (82, 111))
}