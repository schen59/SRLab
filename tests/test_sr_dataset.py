__author__ = 'Sherwin'

import numpy as np
from sr_dataset import SRDataSet
from sr_factory import sr_image_factory
import unittest

class TestSRDataset(unittest.TestCase):

    def setUp(self):
        self.sr_dataset = SRDataSet(np.array([[1, 1], [2, 2]]), np.array([[2, 2], [3, 3]]),
                                    neighbors=1)

    def test_add(self):
        self.sr_dataset.add(np.array([[3, 3], [4, 4]]), np.array([[5, 5], [6, 6]]))
        expected_low_res_patches = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        expected_high_res_patches = np.array([[2, 2], [3, 3], [5, 5], [6, 6]])
        self.assertTrue(np.array_equal(expected_low_res_patches, self.sr_dataset.low_res_patches))
        self.assertTrue(np.array_equal(expected_high_res_patches, self.sr_dataset.high_res_patches))

    def test_query(self):
        low_res_patches = np.array([[1, 1]])
        high_res_patches = self.sr_dataset.query(low_res_patches)
        expected_high_res_patches = np.array([[[2, 2]]])
        self.assertTrue(np.array_equal(expected_high_res_patches, high_res_patches))

    def test_from_sr_image(self):
        sr_image = sr_image_factory.SRImageFactory.create_sr_image_from("test_data/babyface_4.png")
        sr_dataset = SRDataSet.from_sr_image(sr_image)
        low_res_patches = sr_dataset.low_res_patches
        high_res_patches = sr_dataset.high_res_patches
        self.assertEqual(np.shape(low_res_patches), np.shape(high_res_patches))

if __name__ == '__main__':
    unittest.main()
