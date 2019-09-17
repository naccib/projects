"""
Defines the `Subject` class.
"""

import nibabel

import pandas as pd
import numpy as np

from pathlib import Path
from typing import List

TEST_FOLDER = Path(f"{Path.home()}/data/data/ds000030_R1.0.5/derivatives/fmriprep/")
TEST_METADATA = PARTICIPANTS_INFO = Path(f"{Path.home()}/data/data/ds000030_R1.0.5/participants.tsv")

class Subject:
    """
    Represents a single subject in a LA5c study.
    """

    def __init__(self, path: Path):
        """
        Creates a new `Subject` object.
        `path` is the `Path` of the given subject.
        """

        if not path.exists():
            raise ValueError(f'Path {str(path)} doesn\'t exist')

        self.path: Path = path
        self.id: str = path.name

    
    def load_anat(self, image: str) -> nibabel.Nifti2Image:
        """
        Loads the `nibabel.Nifti2Image` corresponding the `image` string.
        `image` represents the type of image, such as: `aparc`, `aparc+aseg`, `brain`...
        """

        image_path = self.path.joinpath(f'anat/{self.id}_T1w_{image}.nii.gz')

        if not image_path.exists():
            raise ValueError(f'Image {image} does not exist in {str(self.path)}/anat/')

        return nibabel.load(str(image_path))


    def load_mri(self, image: str) -> np.ndarray:
        """
        Loads the `np.ndarray` from the `image` specified.
        `image` represents the type of image, such as: `aparc`, `aparc+aseg`, `brain`...
        """

        return self.load_anat(image).get_data()


    def load_map(self, image: str) -> np.ndarray:
        """
        Loads the `np.ndarray` from the `image` specified.
        `image` represents the type of map, such as `aparc`, `aparc+aseg`, `brain`...
        """

        image_path = self.path.joinpath('map/{image}.mgz')

        if not image_path.exists():
            raise ValueError(f'Image {image} does not exist in {str(self.path)}/anat/')

        return nibabel.load(str(image_path)).get_data()


class SubjectCollection:
    """
    Represents a collection of `Subject`s.
    This class is able to filter them.
    """

    def __init__(self, folder: Path, metadata_file: Path):
        self.subjects: List[Subject] = self._load_folder(folder)
        self.metadata: pd.DataFrame  = self._load_metadata(metadata_file)

        
    def subject_metadata(self, id: str) -> pd.Series:
        """
        Returns the metadata of the `Subject` with ID `id`.
        """

        return self.metadata.loc[id]

    
    def select(self, f) -> List[Subject]:
        """
        Filters all subjects with `f(subject) == True`.
        `f` takes the `pd.Series` containing the `Subject`'s metadata.
        """ 

        return [subj for subj in self.subjects if f(self.subject_metadata(subj.id))]


    def _load_folder(self, folder: Path) -> List[Subject]:
        """
        Returns a `list` of all `Subject`s within a `folder`.
        """

        return [Subject(item) for item in folder.glob('*') if item.is_dir() and item.name.startswith('sub')]

    
    def _load_metadata(self, metadata_file: Path) -> pd.DataFrame:
        """
        Returns a `pd.DataFrame` containing subject info.
        """

        metadata  = pd.read_csv(str(metadata_file), sep='\t')
        metadata = metadata.dropna(subset=['T1w']) # drop all patients that do not have T1w images
        metadata.index = metadata['participant_id']

        return metadata

    def __len__(self):
        return self.subjects.__len__()


LA5C_SUBJECTS = SubjectCollection(TEST_FOLDER, TEST_METADATA)

if __name__ == '__main__':
    pass
        

    