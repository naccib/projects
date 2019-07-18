from skimage.io import imshow, imshow_collection

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

from preprocessing import load_subjects
from pathlib import Path


ROOT_DIR = Path(f"{Path.home()}/data/ds000030_R1.0.5/derivatives/freesurfer/")
LUT_PATH = ROOT_DIR.joinpath('FreeSurferColorLUT.txt')


class FreeSurferLUT:
    """
    Creates MatPlotLib's Colormaps from the FreeSurfer Color Lookup Table.
    """

    def __init__(self, path: Path):
        self.path = LUT_PATH if path is None else path

        self._color_array: np.ndarray = None
        self._labels: list            = None

        self._load_lut()

    
    def label_index(self, label: str) -> int:
        return self._labels.index(label)


    def get_colormap(self, areas: list = None) -> ListedColormap:
        """
        Gets the `ListedColormap` from FreeSurfer's lookup table.

        The `areas` optional argument should contain label names (`StructName`) to be filtered in.
        This is specially useful when visualizing a few key areas in the brain.
        """

        if areas is None:
            return ListedColormap(self._color_array, name='FreeSurferColorLUT')

        areas = [self.label_index(label) for label in areas]

        BLACK = (0., 0., 0.)
        color_filter = lambda x: x[1] if x[0] in areas else BLACK

        filtered_colors = [color_filter(item) for item in enumerate(self._color_array)]

        return ListedColormap(filtered_colors, name='FilteredFreeSurferColorLUT')


    def _load_lut(self) -> np.ndarray:
        _, labels, r, g, b, _ = np.loadtxt(self.path, dtype="i4,S32,i2,i2,i2,i2", unpack=True)
        assert(r.shape == g.shape == b.shape)

        lut = []
        for i in range(0, len(r)):
            color = (r[i] / 255, g[i] / 255, b[i] / 255)
            lut.append(color)

        self._color_array = np.asarray(lut)
        self._labels      = [label.decode('utf-8') for label in labels]


subjs = load_subjects()

brain = subjs[0].load_mri('brain')
aseg  = subjs[0].load_mri('aseg')
aparc = subjs[0].load_mri('aparc.DKTatlas+aseg')

thalamus_areas = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']

lut = FreeSurferLUT('/usr/local/freesurfer/FreeSurferColorLUT.txt')

normal_map = lut.get_colormap()
thalamus_map = lut.get_colormap(areas=thalamus_areas)

data = aparc.get_data()
slice = data[:, 128, :]

imshow(slice, cmap=normal_map)
plt.show()

imshow(slice, cmap=thalamus_map)

plt.show()