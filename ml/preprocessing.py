from skimage import img_as_float, img_as_bool
from skimage.draw import rectangle_perimeter
from skimage.io import imshow, imshow_collection
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.morphology import convex_hull_image

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd
import seaborn as sns

from subject import Subject, load_subjects
from pathlib import Path
from typing import List, NewType
import datetime

ROOT_DIR = Path(f"{Path.home()}/data/ds000030_R1.0.5/derivatives/freesurfer/")
LUT_PATH = ROOT_DIR.joinpath('FreeSurferColorLUT.txt')

CTX_SUP_FRONTAL = 'ctx-lh-superiorfrontal'
LEFT_THALAMUS = 'Left-Thalamus-Proper'

BoudingBox2D = NewType('BoundingBox2D', ((float, float), (float, float)))
BoudingBox3D = NewType('BoundingBox3D', ((float, float, float), (float, float, float)))

# This bounding box is the biggest bounding box
# that encompesses the structures below in all subjects.
# Se `bouding_box_statistics` for details.

BOUNDING_BOXES = {
    'ctx-lh-superiorfrontal': ((102, 30, 70), (173, 172, 221)),
    'Left-Amygdala': ((134, 100, 107), (167, 196, 160))
}

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

        assert(original.shape == segmentation.shape)

        # simple function to check if `x` is in a range (without creating a `Range` object)
        is_in = lambda x, a, b: a <= x <= b
        
        # Make output colored
        new_shape = (original.shape[0], original.shape[1], original.shape[2], 3)
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


    def crop_bounding_box(self, original: np.ndarray, bounding_box: BoudingBox3D) -> np.ndarray:
        (x_min, y_min, z_min), (x_max, y_max, z_max) = bounding_box
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


def bounding_box_statistics(subjects: list, struct_name: str, verbose=True) -> BoudingBox3D:
    """
    Given the subject list, the lookup table and
    the name of the desired structure, extracts data
    about possible bounding boxes.

    Returns a Pandas DataFrame and a BoundingBox3D representing the biggest bounding box.
    """

    lut = FreeSurferLUT('/usr/local/freesurfer/FreeSurferColorLUT.txt')

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

    df.to_csv(f"bounding_box_{struct_name}.csv")

    return ((x_min, y_min, z_min), (x_max, y_max, z_max)), df


def bounding_box_graphics(stats: pd.DataFrame):
    kde_kws= { 'shade': True }

    for point in ('x0', 'x1', 'y0', 'y1', 'z0', 'z1'):
        sns.distplot(stats[point], hist=True, label=point, 
        kde=False, norm_hist=False, kde_kws=kde_kws)

    plt.title('Bounding Box Point Coordinate Distribution')

    plt.xlabel('Length (voxels)')
    plt.ylabel('Gaussian Coefficient')

    plt.show()


def preprocess(subjects: List[Subject], bounding_box: BoudingBox3D, struct_name: str = 'unknown', verbose: bool = True) -> (np.ndarray, np.ndarray):
    """
    Preprocesses the images and labels.
    If `use_file` is `True`, the program will load the dataset from the last cached file.
    Optionally, `struct_name` will be added to the output file. 

    Returns two `np.ndarray`s containing the labels and images.
    """

    lut = FreeSurferLUT('/usr/local/freesurfer/FreeSurferColorLUT.txt')
    log = lambda s: print(s) if verbose else None

    images_shape = (len(subjects), 
                    bounding_box[1][0] - bounding_box[0][0],
                    bounding_box[1][1] - bounding_box[0][1],
                    bounding_box[1][2] - bounding_box[0][2],
                    1)

    images = np.empty(images_shape)
    labels = np.empty((len(subjects),))

    log(f"Created output array with shape {images_shape}")

    log(f'Cropping {len(subjects)} images with bounding box {bounding_box}...')

    for (i, subj) in enumerate(subjects):
        log(f'[{i + 1}] Cropping image...')

        brain = subj.load_mri('brain').get_data()
        cropped = lut.crop_bounding_box(brain, bounding_box) / 255.0

        images[i] = np.expand_dims(cropped, axis=-1)
        labels[i] = 0 if subj.diagnosis == 'CONTROL' else 1

    log('Saving...')
    
    time_str = datetime.datetime.now().strftime("%m-%d")

    images_file_path = Path(f"{struct_name}_{time_str}")
    labels_file_path = Path(f"labels_{struct_name}_{time_str}")
    
    np.savez_compressed(images_file_path, images)
    np.savez_compressed(labels_file_path, labels)

    log('Done')
    return images, labels

subjects = load_subjects(ROOT_DIR, group_filter=['CONTROL', 'SCHZ'])