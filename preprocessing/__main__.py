"""
Preprocessing for the T1-MRI images consists of 4 steps:
 1. Loading from disk (`subject.py`)
 2. Align and normalize (`normalization.py`)
 3. Data augmentation (`augment.py`) *
 4. K-Fold Cross validation (`cross_validation.py`) *

 * = Exports NumPy compressed (`.npz`) files.
"""

from subject import LA5C_SUBJECTS
from normalization import whiten, normalize

from matplotlib import pyplot as plt
from skimage.io import imshow, imshow_collection

subjs = LA5C_SUBJECTS.select(lambda s: s['diagnosis'] in ['CONTROL', 'SCHZ'])

print(f"Found {len(subjs)} subjects.")

# select a small sample
subjs = subjs[0:20]
subjs = [subj.load_anat('space-MNI152NLin2009cAsym_preproc').get_data() for subj in subjs]

for subj in whiten(subjs):
    print(f"{subj.mean()}, {subj.std()}")