"""
Now you're symmetric! Maybe. Sort of.

Usage:

    $ pypy example.py ~/Desktop/haarcascade_frontalface_default.xml

"""

from time import time
import sys

from bp.filepath import FilePath

from opencv_cffi.core import Color, invert
from opencv_cffi.imaging import Camera
from opencv_cffi.gui import ESCAPE, Window
from opencv_cffi.object_detection import HaarClassifier


cascade_filepath = FilePath(sys.argv[1])
classifier = HaarClassifier.from_path(cascade_filepath, canny_pruning=True)


def uglify(frame, facetangle):
    with frame.region_of_interest(facetangle.right_half):
        invert(frame)


def prettify(frame, facetangle):
    with frame.region_of_interest(facetangle.right_half):
        prettified = frame.flipped_vertical()

    with frame.region_of_interest(facetangle.left_half):
        prettified.write_into(frame)


def untransformed(frame, facetangle):
    pass


def debug(transform):
    def _debug(frame, facetangle):
        facetangle.draw_onto(frame, color=Color.GREEN)
        transform(frame=frame, facetangle=facetangle)
    return _debug


class Symmetrizer(object):

    transform = staticmethod(prettify)

    def handle_input(self, key):
        if key == ESCAPE:
            sys.exit()
        elif key == "0":
            self.transform = untransformed
        elif key == "p":
            self.transform = prettify
        elif key == "u":
            self.transform = uglify
        elif key == "D":
            self.transform = debug(self.transform)

    def symmetrized(self, camera=None):
        if camera is None:
            camera = Camera()
        for frame in camera.frames():
            for facetangle in classifier.detect_objects(inside=frame):
                self.transform(frame=frame, facetangle=facetangle)
            yield frame


with Window(name="Front") as front_window:
    symmetrizer = Symmetrizer()
    front_window.loop_over(
        symmetrizer.symmetrized(), handle_input=symmetrizer.handle_input,
    )
