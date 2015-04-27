__author__ = 'Sherwin'

import numpy as np
from sr_util import sr_image_util
from sklearn.neighbors import NearestNeighbors
DEFAULT_PYRAMID_LEVEL = 3
DEFAULT_DOWNGRADE_RATIO = 1.25

class SRDataSet(object):

    def __init__(self, low_res_patches, high_res_patches):
        self._low_res_patches = low_res_patches
        self._high_res_patches = high_res_patches
        self._nearest_neighbor = None
        self._need_update = True

    @classmethod
    def from_sr_image(cls, sr_image):
        """Create a SRDataset object from a SRImage object.

        @param sr_image:
        @type sr_image: L{sr_image.SRImage}
        @return: SRDataset object
        @rtype: L{sr_dataset.SRDataset}
        """
        high_res_patches = sr_image_util.get_patches_without_dc(sr_image)
        sr_dataset = SRDataSet(high_res_patches, high_res_patches)
        for downgraded_sr_image in sr_image.get_pyramid(DEFAULT_PYRAMID_LEVEL, DEFAULT_DOWNGRADE_RATIO):
            low_res_patches = sr_image_util.get_patches_without_dc(downgraded_sr_image)
            sr_dataset.add(low_res_patches, high_res_patches)
        return sr_dataset

    @property
    def low_res_patches(self):
        return self._low_res_patches

    @property
    def high_res_patches(self):
        return self._high_res_patches

    def _update(self):
        self._nearest_neighbor = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(self._low_res_patches)
        self._need_update = False

    def add(self, low_res_patches, high_res_patches):
        """Add low_res_patches -> high_res_patches mapping to the dataset.

        @param low_res_patches: low resolution patches
        @type low_res_patches: L{numpy.array}
        @param high_res_patches: high resolution patches
        @type high_res_patches: L{numpy.array}
        """
        self._low_res_patches = np.concatenate((self._low_res_patches, low_res_patches))
        self._high_res_patches = np.concatenate((self._high_res_patches, high_res_patches))
        self._need_update = True

    def merge(self, sr_dataset):
        """Merge with the given dataset.

        @param sr_dataset: an instance of SRDataset
        @type sr_dataset: L{sr_dataset.SRDataset}
        """
        low_res_patches = sr_dataset.low_res_patches
        high_res_patches = sr_dataset.high_res_patches
        self.add(low_res_patches, high_res_patches)

    def query(self, low_res_patches, neighbors=1, eps=0.0):
        """Query the high resolution patches for the given low resolution patches.

        @param low_res_patches: low resolution patches
        @type low_res_patches: L{numpy.array}
        @param neighbors: number of neighbors to query for
        @type neighbors: int
        @return: high resolution patches for the given low resolution patches
        @rtype: L{numpy.array}
        """
        if self._need_update:
            self._update()
        distances, indices = self._nearest_neighbor.kneighbors(low_res_patches,
                                                              n_neighbors=neighbors)
        neighbor_patches = self.high_res_patches[indices]
        return self._merge_high_res_patches(neighbor_patches, distances) if neighbors > 1 else neighbor_patches

    def _merge_high_res_patches(self, neighbor_patches, distances):
        """Get the high resolution patches by merging the neighboring patches with the given distance as weight.

        @param neighbor_patches: neighboring high resolution patches
        @type neighbor_patches: L{numpy.array}
        @param distances: distance vector associate with the neighboring patches
        @type distances: L{numpy.array}
        @return: high resolution patches by merging the neighboring patches
        @rtype: L{numpy.array}
        """
        patch_number, neighbor_number, patch_dimension = np.shape(neighbor_patches)
        weights = sr_image_util.normalize(np.exp(distances))
        weights = weights[:, np.newaxis].reshape(patch_number, neighbor_number, 1)
        high_res_patches = np.sum(neighbor_patches*weights, axis=1)
        return high_res_patches
