from bp.filepath import FilePath

from _opencv import ffi, lib
from opencv_cffi.io import Camera
from opencv_cffi.gui import Window
from opencv_cffi.object_detection import HaarClassifier


ESCAPE = ord("\x1b")


camera = Camera(index=0)


def escape_is_pressed(milliseconds=20):
    return lib.cvWaitKey(milliseconds) == ESCAPE



filepath = FilePath(
    "/Users/Julian/Desktop/haarcascades/haarcascade_frontalface_default.xml",
)
classifier = HaarClassifier.from_path(filepath)


with Window(name="Example") as window:
    for frame in camera.frames():
        if escape_is_pressed():
            break

        for rectangle in classifier.detect_objects(inside=frame):
            rectangle.draw_onto(frame)

        window.show(frame)
