__author__ = 'Sherwin'

import unittest
import numpy as np
from sr_util import sr_image_util
from sr_exception.sr_exception import SRException


class TestSRImageUtil(unittest.TestCase):

    def test_create_size(self):
        self.assertEqual([1, 1], sr_image_util.create_size([1, 1], 1.0))
        self.assertEqual([125, 125], sr_image_util.create_size([100, 100], 1.25))
        self.assertEqual([50, 50], sr_image_util.create_size([100, 100], 0.5))

    def test_gaussian_kernel(self):
        gaussian_kernel = sr_image_util.create_gaussian_kernel(2, 0.1)
        expected_size = (5, 5)
        self.assertEqual(expected_size, np.shape(gaussian_kernel))
        normalized = (np.sum(gaussian_kernel)>0.99 and np.sum(gaussian_kernel)<=1.01)
        self.assertTrue(normalized)

    def test_patchify(self):
        array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        patches = sr_image_util.patchify(array, [3, 3], 1)
        expected_patches = np.array([[6,5,6,2,1,2,6,5,6],
                                     [5,6,7,1,2,3,5,6,7],
                                     [6,7,8,2,3,4,6,7,8],
                                     [7,8,7,3,4,3,7,8,7],
                                     [2,1,2,6,5,6,10,9,10],
                                     [1,2,3,5,6,7,9,10,11],
                                     [2,3,4,6,7,8,10,11,12],
                                     [3,4,3,7,8,7,11,12,11],
                                     [6,5,6,10,9,10,14,13,14],
                                     [5,6,7,9,10,11,13,14,15],
                                     [6,7,8,10,11,12,14,15,16],
                                     [7,8,7,11,12,11,15,16,15],
                                     [10,9,10,14,13,14,10,9,10],
                                     [9,10,11,13,14,15,9,10,11],
                                     [10,11,12,14,15,16,10,11,12],
                                     [11,12,11,15,16,15,11,12,11]])
        self.assertTrue(np.array_equal(expected_patches, patches))
        with self.assertRaises(SRException):
            sr_image_util.patchify(array, [3, 4])

if __name__ == "__main__":
    unittest.main()
