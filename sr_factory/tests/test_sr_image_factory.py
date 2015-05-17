__author__ = 'Sherwin'

import unittest
from PIL import Image
from sr_factory.sr_image_factory import SRImageFactory

class TestSRImageFactory(unittest.TestCase):

    def test_create_sr_image_from(self):
        sr_image = SRImageFactory.create_sr_image_from("test_data/babyface_4.png")
        self.assertIsNotNone(sr_image.get_data())

    def test_create_sr_image(self):
        image = Image.open("test_data/babyface_4.png")
        sr_image = SRImageFactory.create_sr_image(image)
        self.assertIsNotNone(sr_image.get_data())

if __name__ == "__main__":
    unittest.main()