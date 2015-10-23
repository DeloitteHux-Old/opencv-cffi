import sys

from bp.filepath import FilePath

from _opencv import ffi, lib
from opencv_cffi.imaging import Camera
from opencv_cffi.gui import Window
from opencv_cffi.object_detection import HaarClassifier


ESCAPE = ord("\x1b")

classifier = HaarClassifier.from_path(FilePath(sys.argv[1]))
camera = Camera(index=0)


def escape_is_pressed(milliseconds=1):
    return lib.cvWaitKey(milliseconds) == ESCAPE


def uglify(frame, facetangle):
    lib.cvSetImageROI(frame, facetangle.right_half._cv_rect)
    lib.cvNot(frame, frame)
    lib.cvResetImageROI(frame)


def prettify(frame, facetangle):
    facetangle.draw_onto(frame)


transform = uglify if sys.argv[2] == "uglify" else prettify


with Window(name="Example") as window:
    for frame in camera.frames():
        if escape_is_pressed():
            break

        for rectangle in classifier.detect_objects(inside=frame):
            transform(frame=frame, facetangle=rectangle)

        window.show(frame)
