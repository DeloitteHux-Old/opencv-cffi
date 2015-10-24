from functools import partial

from characteristic import Attribute, attributes

from _opencv import ffi, lib


class InitializationError(Exception):
    pass


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
        if capture == ffi.NULL:
            raise InitializationError(self)
        next_frame = partial(lib.cvQueryFrame, capture)
        for frame in iter(next_frame, None):
            yield frame
