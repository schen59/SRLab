__author__ = 'Sherwin'

from PIL import Image
from sr_image import SRImage

class SRImageFactory(object):

    @classmethod
    def create_sr_image_from(cls, image_path):
        """Create a super resolution(sr) image from a file path.

        @param image_path: file path for the image
        @type image_path: str
        @return: an instance of SRImage
        @rtype: L{sr_image.SRImage}
        """
        image = Image.open(image_path)
        return SRImageFactory.create_sr_image(image)

    @classmethod
    def create_sr_image(cls, image):
        """Create a SR image from a PIL image.

        @param image: an instance of PIL image.
        @type image:
        @return: an instance of SRImage
        @rtype: L{sr_image.SRImage}
        """
        return SRImage(image.convert("L"))
