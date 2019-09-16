"""
This scripts intends to read automatic segmentation status
for different FreeSurfer outputs.
"""

import nibabel as nib

import numpy as np
from numpy import ndarray

import pandas as pd

from pathlib import Path

FSF_DATA_DIR = Path(f"{Path.home()}/data/ds000030_R1.0.5/derivatives/freesurfer/")
RAW_DATA_DIR = Path(f"{Path.home()}/data/ds000030_R1.0.5")

class Subject:
    def __init__(self, subj_dir: Path, participants: pd.DataFrame):
        self.stats: pd.DataFrame = self.load_aseg(subj_dir)

        self.aparc_lh: pd.DataFrame = self.load_aparc(subj_dir, 'lh')
        self.aparc_rh: pd.DataFrame = self.load_aparc(subj_dir, 'rh')

        self.ID: str = subj_dir.name

        subj_row = participants.loc[participants['participant_id'] == self.ID]
        self.diagnosis: int = ['CONTROL', 'SCHZ', 'BIPOLAR', 'ADHD'].index(subj_row['diagnosis'].array[0])


    def load_aseg(self, subj_dir: Path) -> pd.DataFrame:
        if not subj_dir.exists():
            raise f"Subject folder doesn't exist: {str(subj_dir)}"

        stats_filepath = str(subj_dir.joinpath('stats/aseg.stats'))
        raw_data = np.loadtxt(str(stats_filepath), dtype="i1,i1,i4,f4,S32,f4,f4,f4,f4,f4")

        col_names = ['Index', 'SegId', 'NVoxels', 
                     'Volume_mm3', 'StructName', 'normMean', 
                     'normStdDev', 'normMin', 'normMax', 
                     'normRange']

        stats = pd.DataFrame(raw_data)
        stats.columns = col_names
        stats = stats.drop(columns=['Index'])

        return stats


    def load_aparc(self, subj_dir: Path, hemisphere: str) -> pd.DataFrame:
        if not subj_dir.exists():
           raise f"Subject folder doesn't exist: {str(subj_dir)}"
        
        stats_filepath = str(subj_dir.joinpath(f'stats/{hemisphere}.aparc.DKTatlas.stats'))
        raw_data = np.loadtxt(str(stats_filepath), dtype="S32,i4,i4,i4,f4,f4,f4,f4,i4,f4")

        col_names = ['StructName', 'NumVert', 'SurfArea',
                     'GrayVol', 'ThickAvg', 'ThickStd',
                     'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']

        stats = pd.DataFrame(raw_data)
        stats.columns = col_names

        return stats


    def get_aseg_anatomical_info(self, segmentation_indexes: list = [0, 1, 8, 9, 18, 19],
                                 data_fields: list = ['Volume_mm3']) -> pd.Series:
        """
        Gets all automatic segmentation anatomical information of this subject
        and returns it as a `Series`.

        `segmentation_indexes` are the indexes of the anatomical areas that will be gathered.
        `data_fields` are the specific fields that will be gathered of each anatomical area.

        `data_fields` is a list containing the label of all fields that should be gathered.
        Possible items are: ColHeaders, Index, SegId, NVoxels, Volume_mm3
        StructName, normMean, normStdDev, normMin, normMax, normRange.

        By default, it only uses 'Volume_mm3'.
        """
        
        relevant_data = self.stats.iloc[segmentation_indexes].copy()
        relevant_data = relevant_data[data_fields + ['StructName']]

        formated_df = pd.DataFrame()

        for _, row in relevant_data.iterrows():
            struct = row['StructName']

            for field in data_fields:
                value  = row[field]
                column_name = f"{struct.decode('utf-8')}_{field}"

                formated_df[column_name] = [value]

        formated_df.insert(column='participant_id', value=self.ID, loc=0)
        formated_df.set_index('participant_id', inplace=True)

        # We want to return it as a Series not a DataFrame,
        # so use `iloc[0]` to get the first (and only) row of
        # `formated_df`.
        return formated_df.iloc[0]

    
    def get_aparc_anatomical_info(self, hemisphere: str, filter_indexes: list = None,
                                  filter_fields = None) -> pd.Series:
        """
        Gets all automatic parcellation anatomical information of this subject
        and returns it as a `Series`.

        `filter_indexes` are the indexes of the anatomical areas that will be gathered.

        All indexes will be gathered if this is `None`.

        `filter_fields` is a list of the specific fields that will be gathered of each anatomical area.
        Possible items are: ColHeaders, StructName, NumVert, SurfArea, GrayVol,
        ThickAvg, ThickStd, MeanCurv, GausCurv, FoldInd, CurvInd

        All fields will be gathered if this is `None`.
        """

        relevant_data = pd.DataFrame()

        if hemisphere == 'rh':
            relevant_data = self.aparc_rh.copy()
        elif hemisphere == 'lh':
            relevant_data = self.aparc_lh.copy()
        else:
            raise(f"Hemisphere must either be 'lh' or 'rh'.")

        if filter_indexes != None:
            relevant_data = relevant_data.iloc[filter_indexes]

        if filter_fields != None:
            relevant_data = relevant_data[filter_fields + ['StructName']]

        formated_df = pd.DataFrame()

        for _, row in relevant_data.iterrows():
            struct = row['StructName']

            # If `filter_fields` is None, all fields must
            # be in the filter.
            # This basically sets `filter_fields` to all column names
            # except StructName.
            if filter_fields == None:
                filter_fields = list(relevant_data)
                filter_fields.remove('StructName')

            for field in filter_fields:
                value  = row[field]
                column_name = f"{hemisphere}_{struct.decode('utf-8')}_{field}"

                formated_df[column_name] = [value]

        formated_df.insert(column='participant_id', value=self.ID, loc=0)
        formated_df.set_index('participant_id', inplace=True)

        # We want to return it as a Series not a DataFrame,
        # so use `iloc[0]` to get the first (and only) row of
        # `formated_df`.
        return formated_df.iloc[0]

        
def make_aseg_table(subjects: list, anatomical_fields: list) -> pd.DataFrame:
    """
    Joins all automatic segmentation `Series` generated by each subject 
    of `subjects` into a big `DataFrame` and returns it.
    """

    df = pd.DataFrame()

    for subj in subjects:
        aseg = subj.get_aseg_anatomical_info(data_fields=anatomical_fields)
        df = df.append(aseg)
    
    return df


def make_aparc_table(subjects: list, anatomical_fields: list = None, cortical_indexes: list = None) -> pd.DataFrame:
    """
    Joins all automatic parcellation `Series` generated by each subject 
    of `subjects` into a big `DataFrame` and returns it.
    """

    df = pd.DataFrame()

    for subj in subjects:
        aparc_lh = subj.get_aparc_anatomical_info('lh', filter_fields=anatomical_fields, filter_indexes=cortical_indexes)
        aparc_rh = subj.get_aparc_anatomical_info('rh', filter_fields=anatomical_fields, filter_indexes=cortical_indexes)

        aparc = pd.concat([aparc_lh, aparc_rh], axis=0, sort=False)

        df = df.append(aparc)
    
    return df


def load_all_stats(fsf_root: Path, participants: pd.DataFrame) -> ndarray:
    stats = [Subject(item, participants) for item in fsf_root.iterdir() if item.name.startswith('sub')]

    return np.asarray(stats)
    
def get_participants_data(data_fields=['Volume_mm3', 'normStdDev'], verbose=False) -> (pd.DataFrame, ndarray):
    log = lambda s: print(s) if verbose else None

    log("Reading participants.csv...")

    participants = pd.read_csv(RAW_DATA_DIR.joinpath('participants.tsv'), sep='\t')
    participants = participants.dropna(subset=['T1w']) # drop all patients that do not have T1w images
    participants.index = participants['participant_id']

    log("Making subject list...")

    data = load_all_stats(FSF_DATA_DIR, participants)

    log('Transforming automatic segmentation values...')
    aseg_table = make_aseg_table(data, data_fields)

    log('Transforming automatic parcellation values...')
    aparc_table = make_aparc_table(data, anatomical_fields=['ThickAvg'], cortical_indexes=[27, 12])

    log('Done')

    return participants.join(aseg_table).join(aparc_table), data
