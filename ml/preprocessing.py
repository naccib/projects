import nibabel as nib

from pathlib import Path
from typing import List


ROOT_DIR = Path(f"{Path.home()}/data/ds000030_R1.0.5/derivatives/freesurfer/")


class Subject:
    def __init__(self, path: str):
        self.path: Path = ROOT_DIR.joinpath(path)

    def load_mri(self, identifier: str) -> nib.freesurfer.mghformat.MGHImage:
        mri_path = self.path.joinpath(f'mri/{identifier}.mgz')

        return nib.load(str(mri_path))

def load_subjects(root: Path = ROOT_DIR) -> List[Subject]:
    return [Subject(item.name) for item in root.iterdir() if item.name.startswith('sub')]


