from _opencv import ffi, lib


ESCAPE = ord("\x1b")


def fourcc((a, b, c, d)):
    """
    Calculate a FourCC integer from the four characters.

    http://www.fourcc.org/

    """

    return (((((ord(d) << 8) | ord(c)) << 8) | ord(b)) << 8) | ord(a);


capture = lib.cvCreateCameraCapture(0)
size = lib.cvSize(
    int(lib.cvGetCaptureProperty(capture, lib.CV_CAP_PROP_FRAME_WIDTH)),
    int(lib.cvGetCaptureProperty(capture, lib.CV_CAP_PROP_FRAME_HEIGHT)),
)

# writer = lib.cvCreateVideoWriter(
#     "/Users/Julian/Desktop/testing123.mpeg",
#     fourcc("PIM1"),
#     30,
#     size,
#     1,
# )


lib.cvNamedWindow("Example", lib.CV_WINDOW_AUTOSIZE)


def check_for_escape(milliseconds=20):
    return lib.cvWaitKey(WAIT_MILLISECONDS) != ESCAPE


for i in xrange(50):
    frame = lib.cvQueryFrame(capture)
    # lib.cvWriteFrame(writer, frame)
    lib.cvShowImage("Example", frame)
    lib.cvHaarDetectObjects()


lib.cvDestroyWindow("Example");
