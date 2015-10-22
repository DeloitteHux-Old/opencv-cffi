from _opencv import ffi, lib
from opencv_cffi.gui import Window


ESCAPE = ord("\x1b")


capture = lib.cvCreateCameraCapture(0)
window = Window(name="Example")


def escape_is_not_pressed(milliseconds=20):
    return lib.cvWaitKey(milliseconds) != ESCAPE


cascade = lib.cvLoadHaarClassifierCascade(
    "/Users/Julian/Desktop/haarcascades/haarcascade_frontalface_default.xml",
    lib.cvSize(1, 1),
)


with Window(name="Example") as window:
    while escape_is_not_pressed():
        frame = lib.cvQueryFrame(capture)
        objects = lib.cvHaarDetectObjects(
            frame,
            cascade,
            lib.cvCreateMemStorage(0),
            1.1,
            4,
            0,
            lib.cvSize(50, 50),
            lib.cvSize(0, 0),
        )
        for i in xrange(objects.total):
            rectangle = ffi.cast("CvRect*", lib.cvGetSeqElem(objects, i))
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

        # lib.cvWriteFrame(writer, frame)
        window.show(frame)
