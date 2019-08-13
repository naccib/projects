import nibabel as nib
import pandas as pd

from pathlib import Path
from typing import List


ROOT_DIR = Path(f"{Path.home()}/data/ds000030_R1.0.5/derivatives/freesurfer/")
PARTICIPANTS_INFO = Path(f"{Path.home()}/data/ds000030_R1.0.5/participants.tsv")

class Subject:
    def __init__(self, path: str, diagnosis: str):
        self.path: Path = ROOT_DIR.joinpath(path)
        self.diagnosis: str = diagnosis

    def load_mri(self, identifier: str) -> nib.freesurfer.mghformat.MGHImage:
        mri_path = self.path.joinpath(f'mri/{identifier}.mgz')

        return nib.load(str(mri_path))

def load_subjects(root: Path = ROOT_DIR, group_filter: list=None) -> List[Subject]:
    df = pd.read_csv(PARTICIPANTS_INFO, sep='\t')
    df = df.dropna(subset=['T1w']) # drop all patients that do not have T1w images
    df.index = df['participant_id']

    diagnosis = lambda name: df.loc[name]['diagnosis']
    desired_diagnosis = lambda name: (diagnosis(name) in group_filter) if group_filter != None else True  

    return [Subject(item.name, diagnosis(item.name)) for item in root.iterdir() if item.name.startswith('sub') and desired_diagnosis(item.name)]