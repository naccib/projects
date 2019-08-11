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

CTX_SUP_FRONTAL = 'ctx-lh-superiorfrontal'
LEFT_THALAMUS = 'Left-Thalamus-Proper'

GLOBAL_BOUNDING_BOX = ((102, 30, 30), (173, 172, 221))

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


    def view_bounding_box(self, original: np.ndarray, segmentation: np.ndarray, struct_name: str) -> np.ndarray:
        """
        Applies a red bounding box containing the given `struct_name` on the original MRI image.
        """

        assert(brain.shape == segmentation.shape)

        # simple function to check if `x` is in a range (without creating a `Range` object)
        is_in = lambda x, a, b: a <= x <= b
        
        # Make output colored
        new_shape = (brain.shape[0], brain.shape[1], brain.shape[2], 3)
        new_image = np.empty(new_shape, dtype=float)

        (x_min, y_min, z_min), (x_max, y_max, z_max) = self.find_bounding_box(segmentation, struct_name)

        for x in range(0, original.shape[0]):
            for y in range(0, original.shape[1]):
                for z in range(0, original.shape[2]):
                    if is_in(x, x_min, x_max) and is_in(y, y_min, y_max) and is_in(z, z_min, z_max):
                        old_pixel = original[x, y, z] / 255.0
                        old_pixel = [old_pixel, old_pixel, old_pixel]

                        red_pixel = np.array([1.0, 0.0, 0.0])

                        new_image[x, y, z] = self._blend_pixels(old_pixel, red_pixel, alpha=0.3)

                    else:
                        old_pixel = original[x, y, z] / 255.0
                        new_image[x, y, z] = [old_pixel, old_pixel, old_pixel]

        return new_image


    def crop_bounding_box(self, original: np.ndarray, segmentation: np.ndarray, struct_name: str) -> np.ndarray:
        assert(brain.shape == segmentation.shape)

        (x_min, y_min, z_min), (x_max, y_max, z_max) = self.find_bounding_box(segmentation, struct_name)
        new_image = original[x_min : x_max, y_min : y_max, z_min : z_max]

        return new_image


    def find_bounding_box(self, segmentation: np.ndarray, struct_name: str) -> BoudingBox3D:
        """
        Finds the bounding box (the smallest rectangular region) containing `struct_name`.
        """

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
                        elif z < z_min:
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


    def _load_lut(self) -> np.ndarray:
        packed_lut = np.loadtxt(self.path, dtype="i4,S32,i2,i2,i2,i2", unpack=False)
        column_names = ['No', 'Label', 'R', 'G', 'B', 'A']

        self._lookup_df = pd.DataFrame(packed_lut)
        self._lookup_df.columns = column_names

        self._lookup_df.set_index(keys=['No'], drop=True, inplace=True)
    

    def _blend_pixels(self, p1: np.ndarray, p2: np.ndarray, alpha=0.5) -> np.ndarray:
        """
        Blends two pixels with a fixed alpha.
        Returns the blended pixel.
        """

        return np.array([
            p1[0] * (1 - alpha) + p2[0] * alpha,
            p1[1] * (1 - alpha) + p2[1] * alpha,
            p1[2] * (1 - alpha) + p2[2] * alpha
        ])


def bounding_box_statistics(subjects: list, lut: FreeSurferLUT, struct_name: str, verbose=True) -> BoudingBox3D:
    """
    Given the subject list, the lookup table and
    the name of the desired structure, extracts data
    about possible bounding boxes.

    Returns a Pandas DataFrame and a BoundingBox3D representing the biggest bounding box.
    """

    log = lambda s: print(s) if verbose else None
    log(f'Calculating statistics for structure {struct_name} in {len(subjects)} subjects...')

    # The maximum values needs to be adjusted 
    # when dealing with bigger images

    df = pd.DataFrame(columns=['x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'edge_x', 'edge_y', 'edge_z', 'volume'])

    x_min = 256
    x_max = 0

    y_min = 256
    y_max = 0

    z_min = 256
    z_max = 0

    for (i, subj) in enumerate(subjects):
        log(f'[{i + 1}] Loading subject at {subj.path}...')
        aparc = subj.load_mri('aparc.DKTatlas+aseg').get_data()

        log(f'[{i + 1}] Calculating bounding box...')

        (x0, y0, z0), (x1, y1, z1) = lut.find_bounding_box(aparc, struct_name)

        edge_x = x1 - x0
        edge_y = y1 - y0
        edge_z = z1 - z0

        vol = edge_x * edge_y * edge_z


        df = df.append({
            'x0': x0,
            'y0': y0,
            'z0': z0,
            'x1': x1,
            'y1': y1,
            'z1': z1,
            'edge_x': edge_x,
            'edge_y': edge_y,
            'edge_z': edge_z,
            'volume': vol
        }, ignore_index=True)

        x_min = min(x0, x_min)
        x_max = max(x1, x_max)

        y_min = min(y0, y_min)
        y_max = max(y1, y_max)

        z_min = min(z0, z_min)
        z_max = max(z1, z_max)

        if i % 10 == 0:
            log(f'[{i + 1}] Current Bounding Box: ({x_min}, {y_min}, {z_min}), ({x_max}, {y_max}, {z_max})')

    return ((x_min, y_min, z_min), (x_max, y_max, z_max)), df
        
        



subjs = load_subjects(group_filter=['CONTROL', 'SCHZ'])

lut = FreeSurferLUT('/usr/local/freesurfer/FreeSurferColorLUT.txt')

"""
brain = subjs[0].load_mri('brain').get_data()
aparc = subjs[0].load_mri('aparc.DKTatlas+aseg').get_data()
print(subjs[0].path)

boxed = lut.view_bounding_box(brain, aparc, CTX_SUP_FRONTAL)
viewer = MRIViewer(boxed, title='Superior Temporal Cortex Segmentation')
"""

gbb, data = bounding_box_statistics(subjs, lut, CTX_SUP_FRONTAL)
data.to_csv(f'bounding_box_{CTX_SUP_FRONTAL}.csv', index=False, header=True)