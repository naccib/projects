"""
Whitens and spatially normalizes the images
"""

import numpy as np
import pandas as pd

from nibabel import Nifti2Image
from typing import List


def normalize(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Performs a (0..1) normalization.
    """

    return [one_zero_normalize(img) for img in images]


def one_zero_normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalizes an image to fit in the (0..1) interval.
    """

    min = image.min()
    max = image.max()

    m1 = (image - min)
    m2 = 1 / (max - min)

    return m1 * m2 


def whiten(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Performs a whitening normalization.
    """

    (mean, std) = calculate_whitening_stats(images)
    whiten_function = lambda image: (image - mean) / std

    return [whiten_function(img) for img in images]


def calculate_whitening_stats(images: List[np.ndarray]) -> (float, float):
    """
    Given a `List` of `np.ndarray`, calculates the global mean and
    the mean standard deviance.
    """

    global_mean = 0.0
    global_dev = 0.0

    for image in images:
        mean = image.mean()
        dev = image.std()

        global_mean = np.average([global_mean, mean])
        global_dev  = np.average([global_dev, dev])

    return (global_mean, global_dev)




    