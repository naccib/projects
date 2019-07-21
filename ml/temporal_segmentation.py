from skimage.io import imshow, imshow_collection

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd

from preprocessing import load_subjects
from pathlib import Path


ROOT_DIR = Path(f"{Path.home()}/data/ds000030_R1.0.5/derivatives/freesurfer/")
LUT_PATH = ROOT_DIR.joinpath('FreeSurferColorLUT.txt')


class FreeSurferLUT:
    """
    Creates MatPlotLib's Colormaps from the FreeSurfer Color Lookup Table.
    """

    def __init__(self, path: Path):
        """
        Creates a `FreeSurferLUT` instance.
        The source for the lookup table is `path`.
        """

        self.path = LUT_PATH if path is None else path
        self._lookup_df: pd.DataFrame = None

        self._load_lut()

    
    def label_index(self, label: str) -> int:
        """
        Returns the index of a anatomical area named `label`.
        """
        byte_str = bytes(label, encoding='utf-8')
        index = self._lookup_df.index[self._lookup_df['Label'] == byte_str]

        return index.to_numpy()[0]

 
    def apply(self, image: np.ndarray, areas: list = None) -> np.ndarray:
        """
        Applies the colormap to a image.

        `areas` should be a list of label names (`StructName`) to be filtered in.
        """

        if areas != None:
            areas = [self.label_index(label) for label in areas]

        shape = (image.shape[0], image.shape[1], 3)
        output = np.empty(shape)

        for x in range(0, output.shape[0]):
            for y in range(0, output.shape[1]):
                pixel = image[x, y]

                series = self._lookup_df.loc[pixel]

                if areas == None:
                    pixel = (series[1] / 255.0, series[2] / 255.0, series[3] / 255.0)
                else:
                    if pixel in areas:
                        pixel = (series[1] / 255.0, series[2] / 255.0, series[3] / 255.0)
                    else:
                        pixel = (0., 0., 0.)

                output[x, y] = pixel

        return output


    def mask(self, image: np.ndarray, segmented: np.ndarray, areas: list) -> np.ndarray:
        """
        Masks a number of anatomical areas (`areas`) on a MRI image (`image`) and returns it.
        The source used for masking is a segmented version of the same image (`segmented`).
        """

        output = np.empty(image.shape)
        areas = [self.label_index(label) for label in areas]

        for x in range(0, output.shape[0]):
            for y in range(0, output.shape[1]):
                map_pixel = segmented[x, y]

                if map_pixel in areas:
                    output[x, y] = image[x, y] / 255.0
                else:
                    output[x, y] = 0

        return output


    def _load_lut(self) -> np.ndarray:
        packed_lut            = np.loadtxt(self.path, dtype="i4,S32,i2,i2,i2,i2", unpack=False)
        column_names = ['No', 'Label', 'R', 'G', 'B', 'A']

        self._lookup_df = pd.DataFrame(packed_lut)
        self._lookup_df.columns = column_names

        self._lookup_df.set_index(keys=['No'], drop=True, inplace=True)


subjs = load_subjects()

brain = subjs[0].load_mri('brain')
aseg  = subjs[0].load_mri('aseg')
aparc = subjs[0].load_mri('aparc.DKTatlas+aseg')

lut = FreeSurferLUT('/usr/local/freesurfer/FreeSurferColorLUT.txt')

aparc = aparc.get_data()[:, 128, :]
brain = brain.get_data()[:, 128, :]

thalamus_areas = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']

mapped_image = lut.apply(aparc, areas=thalamus_areas)
masked_image = lut.mask(brain, aparc, thalamus_areas)

imshow(mapped_image)
plt.show()

imshow(masked_image, cmap='gray')
plt.show()