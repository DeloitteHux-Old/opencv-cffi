"""
Now you're symmetric! Maybe. Sort of.

Usage:

    $ pypy example.py ~/Desktop/haarcascade_frontalface_default.xml

"""

from time import time
import itertools
import sys

from bp.filepath import FilePath

from opencv_cffi.core import invert
from opencv_cffi.imaging import Camera
from opencv_cffi.gui import ESCAPE, Window, key_pressed
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
        facetangle.draw_onto(frame)
        transform(frame=frame, facetangle=facetangle)
    return _debug



with Window(name="Front") as front_window:

    front = Camera(index=0)
    transform = prettify
    this_second, first_frame_this_second = time(), 0

    for count, frame in enumerate(front.frames()):

        now = time()
        if now - this_second >= 1:
            print "~{0} fps".format(count - first_frame_this_second)
            this_second, first_frame_this_second = now, count

        pressed = key_pressed()
        if pressed == ESCAPE:
            break
        elif pressed == "0":
            transform = untransformed
        elif pressed == "p":
            transform = prettify
        elif pressed == "u":
            transform = uglify
        elif pressed == "D":
            transform = debug(transform)

        for rectangle in classifier.detect_objects(inside=frame):
            transform(frame=frame, facetangle=rectangle)

        front_window.show(frame)
