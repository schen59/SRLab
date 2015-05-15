__author__ = 'shaofeng'

from PIL import Image
from sr_factory.sr_image_factory import SRImageFactory

if __name__ == '__main__':
    image = Image.open("../test_data/babyface_4.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(4, 'iccv09')
    reconstructed_sr_image.save("../test_data/babyface_4x.png", "png")