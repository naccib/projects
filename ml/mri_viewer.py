import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backend_bases import MouseEvent

from subject import load_subjects
from scipy.ndimage import rotate


STEP = 5


class MRIViewer(object):
    def __init__(self, data: np.ndarray, title: str = 'MRI Viewer'):
        self.data: np.ndarray = data
        self.index: list = [x // 2 for x in self.data.shape]

        fig, axes = plt.subplots(ncols=3, nrows=1)

        self.fig = fig
        self.axes = axes

        self.fig.suptitle(title, fontsize=16)

        self._update_axes()
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)


    def show(self):
        plt.show()


    def _on_scroll(self, event: MouseEvent):
        if event.inaxes is None:
            return

        for i in [0, 1, 2]:
            if event.inaxes == self.axes[i]:
                new_val = self.index[i] + (STEP if event.button == 'up' else -STEP)
                new_val = max(0, min(new_val, self.data.shape[i]))

                self.index[i] = new_val

        self._update_axes()


    def _update_axes(self):
        for i in [0, 1, 2]:
            slice = self._slice_mri(self.index[i], i)

            view_name = ('Sagittal', 'Axial', 'Coronal')[i]
            unit_name = ('x', 'y', 'z')[i]

            self.axes[i].imshow(slice, cmap='gray')
            self.axes[i].set_title(f'{view_name} ({unit_name}={self.index[i]})')

        self.fig.canvas.draw()


    def _slice_mri(self, index: int, cut: int) -> np.ndarray:
        """
        Returns the 2D grayscale image representing the `index`
        of an anatomical plane (`cut`).
        """

        slice = None
        
        if cut == 0:
            slice = self.data[index, :, :]
        elif cut == 1:
            slice = self.data[:, index, :]
            slice = rotate(slice, -90)
        elif cut == 2:
            slice = self.data[:, :, index]
            slice = rotate(slice, -90)
        else:
            raise Exception('cut must be either 0, 1 or 2.')

        return slice

        

        


if __name__ == '__main__':
    subjs = load_subjects()
    t1w = subjs[82].load_mri('brain').get_data()

    viewer = MRIViewer(t1w)
    viewer.show()

    