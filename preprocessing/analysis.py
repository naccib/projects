"""
Contains an array of analysis functions.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imshow, imshow_collection

from subject import LA5C_SUBJECTS

from random import choices

subjects = LA5C_SUBJECTS#.select(lambda s: s['diagnosis'] in ['CONTROL', 'SCHZ'])

def view_slices(mri: str = 'brain', count: int = 12):
    subjs = choices(subjects.subjects, k=count)
    subjs = [sub.load_mri(mri)[:, 128, :] for sub in subjs]

    imshow_collection(subjs, cmap='gray')
    plt.show()

