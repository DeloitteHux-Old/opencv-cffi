import itertools
import sys

from bp.filepath import FilePath

from _opencv import ffi, lib
from opencv_cffi.imaging import Camera
from opencv_cffi.gui import ESCAPE, Window, key_pressed
from opencv_cffi.object_detection import HaarClassifier


cascade_filepath = FilePath(sys.argv[1])
classifier = HaarClassifier.from_path(cascade_filepath, canny_pruning=True)


def uglify(frame, facetangle):
    lib.cvSetImageROI(frame, facetangle.right_half._cv_rect)
    lib.cvNot(frame, frame)
    lib.cvResetImageROI(frame)


def prettify(frame, facetangle):
    facetangle.draw_onto(frame)

    right_half = facetangle.right_half
    lib.cvSetImageROI(frame, right_half._cv_rect)
    prettified = lib.cvCreateImage(
        lib.cvGetSize(frame), frame.depth, frame.nChannels,
    )

    lib.cvCopy(frame, prettified, ffi.NULL)
    lib.cvFlip(prettified, prettified, 1)

    shift_left = lib.cvCreateMat(2, 3, 6)
    lib.cvSetZero(shift_left)
    lib.cvmSet(shift_left, 0, 0, 1)
    lib.cvmSet(shift_left, 1, 1, 1)
    lib.cvmSet(shift_left, 0, 2, right_half.width)

    lib.cvWarpAffine(
        prettified,
        prettified,
        shift_left,
        lib.CV_INTER_LINEAR + lib.CV_WARP_FILL_OUTLIERS,
        lib.cvScalarAll(0.0),
    )

    lib.cvCopy(prettified, frame, ffi.NULL)
    lib.cvResetImageROI(frame)


transform = uglify if sys.argv[2] == "uglify" else prettify


with Window(name="Front") as front_window:

    front = Camera(index=0)

    for window, frames in itertools.cycle(
        [
            (front_window, front.frames()),
        ],
    ):
        frame = next(frames)
        if key_pressed() == ESCAPE:
            break

        for rectangle in classifier.detect_objects(inside=frame):
            transform(frame=frame, facetangle=rectangle)

        window.show(frame)
