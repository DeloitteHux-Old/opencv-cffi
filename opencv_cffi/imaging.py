from functools import partial

from characteristic import Attribute, attributes

from _opencv import ffi, lib
from opencv_cffi.core import Image


class InitializationError(Exception):
    pass


@attributes(
    [
        Attribute(name="index", default_value=lib.CV_CAP_ANY),
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
            yield Image(cv_arr=frame)
