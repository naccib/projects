from load_aseg_stats import get_participants_data

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


sns.set(style="white", palette="muted", color_codes=True)

OUTPUT_DIR = Path(f"{Path.home()}/Documents/UFS/plots")
participants, _ = get_participants_data()


def temporal_thickness_distribution(df: pd.DataFrame, thickness_column: str, title: str, save: bool=False):
    df = df[['diagnosis', thickness_column]]

    controls = df.loc[df['diagnosis'] == 'CONTROL']
    schizo   = df.loc[df['diagnosis'] == 'SCHZ']

    kde_kws= { 'shade': True }

    sns.distplot(controls[thickness_column], hist=False, label='Controls', 
    kde=True, color='b', norm_hist=True, kde_kws=kde_kws)

    sns.distplot(schizo[thickness_column], hist=False, label='Schizophrenia', 
    kde=True, color='g', norm_hist=True, kde_kws=kde_kws)

    plt.title(title)
    plt.xlabel('Thickness (mm)')
    plt.ylabel('Gaussian Coefficient')

    if save:
        file_path = OUTPUT_DIR.joinpath(f"temporal_thickness/{thickness_column}.png")
        plt.savefig(str(file_path))
        plt.clf()
    else:
        plt.show()