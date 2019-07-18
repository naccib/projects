from load_aseg_stats import get_participants_data

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


sns.set(style="white", palette="muted", color_codes=True)

OUTPUT_DIR = Path(f"{Path.home()}/Documents/UFS/plots")
VENTRICLES = ['3rd-Ventricle', '4th-Ventricle', 'Left-Inf-Lat-Vent',
              'Left-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-Lateral-Ventricle']

RELEVANT_FIELDS = ['Volume_mm3', 'normStdDev']


participants, _ = get_participants_data()


def cortical_thickness_table(df: pd.DataFrame) -> pd.DataFrame:
    lh_columns = [col for col in df if col.startswith('lh') and col.endswith('ThickAvg')]
    rh_columns = [col for col in df if col.startswith('rh') and col.endswith('ThickAvg')]

    thickness_df = pd.DataFrame(columns=['participant_id', 'diagnosis', 'ThickAvg', 'lh_ThickAvg', 'rh_ThickAvg'])

    for idx, row in df.iterrows():
        lh_sum = 0.0
        rh_sum = 0.0

        for col in lh_columns:
            lh_sum += row[col]
        
        for col in rh_columns:
            rh_sum += row[col]

        thickness_df.at[idx, 'participant_id'] = row['participant_id']
        thickness_df.at[idx, 'diagnosis'] = row['diagnosis']

        thickness_df.at[idx, 'ThickAvg'] = (lh_sum + rh_sum) / (len(lh_columns) + len(rh_columns))
        thickness_df.at[idx, 'lh_ThickAvg'] = lh_sum / len(lh_columns)
        thickness_df.at[idx, 'rh_ThickAvg'] = rh_sum / len(rh_columns)

    thickness_df.set_index('participant_id', inplace=True)
    return thickness_df


def ventricule_distribution(df: pd.DataFrame, area: str, 
                     stat='Volume_mm3', 
                     title='Gaussian Distribution', 
                     include_adhd=False,
                     save=False):

    column_name = f"{area}_{stat}"

    df = df[['diagnosis', column_name]].copy()
    get_volume = lambda diagnosis: df.loc[df['diagnosis'] == diagnosis][column_name]

    kde_kws= { 'shade': True }
    
    sns.distplot(get_volume('CONTROL'), hist=False, label='Controls', 
    kde=True, color='b', norm_hist=True, kde_kws=kde_kws)

    sns.distplot(get_volume('SCHZ'), hist=False, label='Schizophrenia', 
    kde=True, color='g', norm_hist=True, kde_kws=kde_kws)

    if include_adhd:
        sns.distplot(get_volume('ADHD'), hist=False, label='ADHD', 
        kde=True, color='r', norm_hist=True, kde_kws=kde_kws)


    plt.title(title)
    plt.xlabel('Volume (mm3)' if stat == 'Volume_mm3' else '')
    plt.ylabel('Gaussian Coefficient')

    if save:
        file_path = OUTPUT_DIR.joinpath(f"{column_name}.png")
        plt.savefig(str(file_path))
        plt.clf()
    else:
        plt.show()


def cort_thickness_distribution(df: pd.DataFrame, hemisphere: str = None, save: bool = False):
    cort_table = cortical_thickness_table(df)
    column_name = 'ThickAvg' if hemisphere is None else f"{hemisphere}_ThickAvg"

    get_thickness = lambda diagnosis: cort_table.loc[cort_table['diagnosis'] == diagnosis][column_name]

    kde_kws= { 'shade': True }
    
    sns.distplot(get_thickness('CONTROL'), hist=False, label='Controls', 
    kde=True, color='b', norm_hist=True, kde_kws=kde_kws)

    sns.distplot(get_thickness('SCHZ'), hist=False, label='Schizophrenia', 
    kde=True, color='g', norm_hist=True, kde_kws=kde_kws)
 
    if hemisphere is None:
        plt.title('Average Cortical Thickness Distribution')
    else:
        plt.title(f"{'Right' if hemisphere == 'rh' else 'Left'} Hemisphere Average Cortical Thickness Distribution")

    plt.xlabel('Thickness (mm)')
    plt.ylabel('Gaussian Coefficient')

    if save:
        file_path = OUTPUT_DIR.joinpath(f"cortical_thickness/{column_name}.png")
        plt.savefig(str(file_path))
        plt.clf()
    else:
        plt.show()


def ventricule_cort_thickness_scatter(df: pd.DataFrame, area: str):
    cort_table = cortical_thickness_table(df)

    vol_column_name = f"{area}_Volume_mm3"
    thick_column_name = "ThickAvg"

    df = df[[vol_column_name]].copy()
    df = pd.concat([df, cort_table], axis=1, sort=False)

    # drop adhd and bipolar patients

    adhd_index = df[df['diagnosis'] == 'ADHD'].index
    bipolar_index = df[df['diagnosis'] == 'BIPOLAR'].index

    df.drop(adhd_index, inplace=True)
    df.drop(bipolar_index, inplace=True)
     
    sns.scatterplot(x=thick_column_name, y=vol_column_name, hue='diagnosis', data=df)
    plt.show()


ventricule_cort_thickness_scatter(participants, '3rd-Ventricle')