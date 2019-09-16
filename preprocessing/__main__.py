"""
Preprocessing for the T1-MRI images consists of 4 steps:
 1. Loading from disk (`subject.py`)
 2. Align and normalize (`normalization.py`)
 3. Data augmentation (`augment.py`) *
 4. K-Fold Cross validation (`cross_validation.py`) *

 * = Exports NumPy compressed (`.npz`) files.
"""

import SimpleITK as sitk

from subject import LA5C_SUBJECTS

subjs = LA5C_SUBJECTS.select(lambda s: s['diagnosis'] in ['CONTROL', 'SCHZ'])

print(f"Found {len(subjs)} subjects.")

brain = subjs[0].load_babel('brain')

image = sitk.Image([256, 256, 256], sitk.sitkFloat32, 1)