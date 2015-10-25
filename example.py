import itertools
import sys

from bp.filepath import FilePath

from _opencv import ffi, lib
from opencv_cffi.core import Matrix, copy, invert
from opencv_cffi.imaging import Camera
from opencv_cffi.gui import ESCAPE, Window, key_pressed
from opencv_cffi.object_detection import HaarClassifier


cascade_filepath = FilePath(sys.argv[1])
classifier = HaarClassifier.from_path(cascade_filepath, canny_pruning=True)


def uglify(frame, facetangle):
    with frame.region_of_interest(facetangle.right_half):
        invert(frame)


def prettify(frame, facetangle):
    facetangle.draw_onto(frame)

    with frame.region_of_interest(facetangle.right_half) as right_half:
        prettified = frame.flipped_horizontal()
        shift_left = Matrix.from_data(
            [1, 0, -right_half.width],
            [0, 1, 0],
        )

        lib.cvWarpAffine(
            prettified._ipl_image,
            prettified._ipl_image,
            shift_left._cv_mat,
            lib.CV_INTER_LINEAR + lib.CV_WARP_FILL_OUTLIERS,
            lib.cvScalarAll(0.0),
        )

        copy(array=prettified._ipl_image, into=frame._ipl_image)


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
