"""
Changing the contrast and brightness of an image!

http://docs.opencv.org/2.4/doc/tutorials/core/basic_linear_transform/basic_linear_transform.html#basic-linear-transform

"""

import sys

import numpy
from PIL import Image

from _opencv import lib
from opencv_cffi.core import Matrix
from opencv_cffi.gui import Window


image = numpy.array(Image.open(sys.argv[1]))
alpha, beta = float(sys.argv[2]), int(sys.argv[3])
new_image = alpha * image + beta
print len(new_image)

with Window(name="Example") as window:
    window.show_until_keypress(new_image)
