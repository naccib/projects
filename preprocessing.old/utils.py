"""
Diverse utils functions.
"""

import numpy as np
from nibabel import Nifti1Image, Nifti2Image


def flatten_subjects(subjs) -> np.ndarray:
    """
    Flattens the subjects to a single
    `np.ndarray`.
    """

    shape = subjs[0].shape
    output = np.ndarray((len(subjs), shape[0], shape[1], shape[2]))

    for i in range(0, len(subjs)):
        if type(subjs[i]) in (Nifti1Image, Nifti2Image):
            subjs[i] = subjs[i].get_data()

    return subjs
