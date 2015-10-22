from _opencv import ffi, lib


ESCAPE = ord("\x1b")


def fourcc(code):
    assert len(code) == 4
    return sum(
        (ord(letter) & 255) << (8 * index)
        for index, letter in enumerate(code)
    )


capture = lib.cvCreateCameraCapture(0)
size = lib.cvSize(
    int(lib.cvGetCaptureProperty(capture, lib.CV_CAP_PROP_FRAME_WIDTH)),
    int(lib.cvGetCaptureProperty(capture, lib.CV_CAP_PROP_FRAME_HEIGHT)),
)

writer = lib.cvCreateVideoWriter(
    "testing123",
    fourcc("AVC1"),
    30,
    size,
    True,
)


lib.cvNamedWindow("Example", lib.CV_WINDOW_AUTOSIZE)


WAIT_MILLISECONDS = 20

while lib.cvWaitKey(WAIT_MILLISECONDS) != ESCAPE:
    frame = lib.cvQueryFrame(capture)
    lib.cvWriteFrame(writer, frame)
    lib.cvShowImage("Example", frame)

lib.cvDestroyWindow("Example");
