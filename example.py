from _opencv import ffi, lib
from opencv_cffi.io import Camera
from opencv_cffi.gui import Window
from opencv_cffi.utils import _OpenCVSequence


ESCAPE = ord("\x1b")


camera = Camera(index=0)


def escape_is_pressed(milliseconds=20):
    return lib.cvWaitKey(milliseconds) == ESCAPE


cascade = lib.cvLoadHaarClassifierCascade(
    "/Users/Julian/Desktop/haarcascades/haarcascade_frontalface_default.xml",
    lib.cvSize(1, 1),
)


with Window(name="Example") as window:
    for frame in camera.frames():
        if escape_is_pressed():
            break

        raw_objects = lib.cvHaarDetectObjects(
            frame,
            cascade,
            lib.cvCreateMemStorage(0),
            1.1,
            4,
            0,
            lib.cvSize(50, 50),
            lib.cvSize(0, 0),
        )
        objects = _OpenCVSequence(cv_seq=raw_objects, contents_type="CvRect *")
        for i in xrange(len(objects)):
            rectangle = objects[i]
            width, height = rectangle.width, rectangle.height

            lib.cvRectangle(
                frame,
                [rectangle.x, rectangle.y],
                [rectangle.x + width, rectangle.y + height],
                lib.cvScalar(255, 0, 0, 0),
                1,
                8,
                0,
            )

        window.show(frame)
