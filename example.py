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


original_transform = lambda frame, facetangle : None
transform = uglify if sys.argv[2] == "uglify" else prettify


with Window(name="Front") as front_window:

    front = Camera(index=0)

    for window, frames in itertools.cycle(
        [
            (front_window, front.frames()),
        ],
    ):
        frame = next(frames)
        pressed = key_pressed()
        if pressed == ESCAPE:
            break
        elif pressed == "\t":
            original_transform, transform = transform, original_transform

        for rectangle in classifier.detect_objects(inside=frame):
            transform(frame=frame, facetangle=rectangle)

        window.show(frame)
