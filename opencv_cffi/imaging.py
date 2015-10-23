from functools import partial

from characteristic import Attribute, attributes

from _opencv import lib


@attributes(
    [
        Attribute(name="index"),
    ],
)
class Camera(object):
    """
    A camera device.

    """

    def frames(self):
        capture = lib.cvCreateCameraCapture(self.index)
        next_frame = partial(lib.cvQueryFrame, capture)
        for frame in iter(next_frame, None):
            yield frame
