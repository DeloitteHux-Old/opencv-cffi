import sys

from bp.filepath import FilePath

from _opencv import ffi, lib
from opencv_cffi.imaging import Camera
from opencv_cffi.gui import ESCAPE, Window, key_pressed
from opencv_cffi.object_detection import HaarClassifier


classifier = HaarClassifier.from_path(FilePath(sys.argv[1]))
camera = Camera(index=0)


def uglify(frame, facetangle):
    lib.cvSetImageROI(frame, facetangle.right_half._cv_rect)
    lib.cvNot(frame, frame)
    lib.cvResetImageROI(frame)
    return frame


def prettify(frame, facetangle):
    facetangle.draw_onto(frame)
    right_half = facetangle.right_half
    lib.cvSetImageROI(frame, right_half._cv_rect)
    prettified = lib.cvCreateImage(
        lib.cvGetSize(frame), frame.depth, frame.nChannels,
    )
    lib.cvFlip(frame, frame, 1)

    shift_left = lib.cvCreateMat(2, 3, 1)
    lib.cvZero(shift_left)
    lib.cvmSet(mat, 0, 0, 1)
    lib.cvmSet(mat, 1, 1, 1)
    lib.cvmSet(mat, 0, 2, right_half.width)

    lib.cvWarpAffine(
        frame,
        frame,
        shift_left,
        lib.CV_INTER_LINEAR + lib.CV_WARP_FILL_OUTLIERS,
        lib.cvScalarAll(0.0),
    )
    lib.cvCopy(frame, prettified, ffi.NULL)
    lib.cvResetImageROI(frame)
    return frame


transform = uglify if sys.argv[2] == "uglify" else prettify


with Window(name="Example") as window:
    for frame in camera.frames():
        if key_pressed(ESCAPE):
            break

        for rectangle in classifier.detect_objects(inside=frame):
            frame = transform(frame=frame, facetangle=rectangle)

        window.show(frame)
