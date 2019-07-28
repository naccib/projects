from skimage import img_as_float, img_as_bool
from skimage.draw import rectangle_perimeter
from skimage.io import imshow, imshow_collection
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.morphology import convex_hull_image

from nilearn.plotting import plot_anat

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd

from preprocessing import load_subjects
from mri_viewer import MRIViewer

from pathlib import Path
from typing import NewType


ROOT_DIR = Path(f"{Path.home()}/data/ds000030_R1.0.5/derivatives/freesurfer/")
LUT_PATH = ROOT_DIR.joinpath('FreeSurferColorLUT.txt')

BoudingBox2D = NewType('BoundingBox2D', ((float, float), (float, float)))
BoudingBox3D = NewType('BoundingBox3D', ((float, float, float), (float, float, float)))


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

 
    def apply_to_slice(self, image: np.ndarray, areas: list = None) -> np.ndarray:
        """
        Applies this colormap colormap to a MRI image's slice.

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


    def mask_slice(self, image: np.ndarray, segmented: np.ndarray, areas: list) -> np.ndarray:
        """
        Masks a number of anatomical areas (`areas`) on a MRI image slice (`image`) and returns it.
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
        packed_lut = np.loadtxt(self.path, dtype="i4,S32,i2,i2,i2,i2", unpack=False)
        column_names = ['No', 'Label', 'R', 'G', 'B', 'A']

        self._lookup_df = pd.DataFrame(packed_lut)
        self._lookup_df.columns = column_names

        self._lookup_df.set_index(keys=['No'], drop=True, inplace=True)


    def bounding_box(self, segmentation: np.ndarray, struct_name: str) -> BoudingBox3D:
        struct_color = self.label_index(struct_name)

        x_min = segmentation.shape[0]
        x_max = 0.0

        y_min = segmentation.shape[1]
        y_max = 0.0

        z_min = segmentation.shape[2]
        z_max = 0.0

        for x in range(0, segmentation.shape[0]):
            for y in range(0, segmentation.shape[1]):
                for z in range(0, segmentation.shape[2]):
                    if segmentation[x, y, z] == struct_color:
                        if y > y_max:
                            y_max = y
                        elif y < y_min:
                            y_min = y

                        if x > x_max:
                            x_max = x
                        elif x < x_min:
                            x_min = x

                        if z > z_max:
                            z_max = z
                        elif x < x_min:
                            z_min = z

        return ((x_min, y_min, z_min), (x_max, y_max, z_max))


    def slice_bounding_box(self, segmentation: np.ndarray, struct_name: str) -> BoudingBox2D:
        """
        Given the `segmentation` map and the structure label (`struct_name`),
        this function returns the bounding box of said struct in the image.

        The bouding box will be returned in this format:
        (x_min, y_min), (x_max, y_max)
        """

        struct_color = self.label_index(struct_name)

        x_min = segmentation.shape[0]
        x_max = 0.0

        y_min = segmentation.shape[1]
        y_max = 0.0

        for x in range(0, segmentation.shape[0]):
            for y in range(0, segmentation.shape[1]):
                if segmentation[x, y] == struct_color:
                    if y > y_max:
                        y_max = y
                    elif y < y_min:
                        y_min = y

                    if x > x_max:
                        x_max = x
                    elif x < x_min:
                        x_min = x

        return ((x_min, y_min), (x_max, y_max))



subjs = load_subjects()

lut = FreeSurferLUT('/usr/local/freesurfer/FreeSurferColorLUT.txt')

brain = subjs[0].load_mri('brain').get_data()
aparc = subjs[0].load_mri('aparc.DKTatlas+aseg').get_data()

thalamus_areas = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']
minima, maxima = lut.bounding_box(aparc, thalamus_areas[0])

#rr, cc = rectangle_perimeter(start, end=end, shape=brain.shape)
#brain_one[rr, cc] = 1

viewer = MRIViewer(brain)