"""
Preprocessing for the T1-MRI images consists of 4 steps:
 1. Loading from disk (`subject.py`) [DONE]
 2. Data augmentation (`augment.py`) *
 3. Data segmentation (`segment.py`) *
 4. Align and normalize (`normalization.py`) * [DONE]
 5. K-Fold Cross validation (`cross_validation.py`) *

 * = Exports NumPy compressed (`.npz`) files.
"""

from subject import LA5C_SUBJECTS

from augment import augment
from normalization import whiten, normalize

from matplotlib import pyplot as plt
from skimage.io import imshow, imshow_collection

subjs = LA5C_SUBJECTS.select(lambda s: s['diagnosis'] in ['CONTROL', 'SCHZ'])
sample = subjs.subjects[42].load_anat('space-MNI152NLin2009cAsym_preproc').get_data()

print(f"Found {len(subjs)} subjects.")

subjs = augment(subjs)



# subjs = [subj.load_anat('space-MNI152NLin2009cAsym_preproc').get_data() for subj in subjs]