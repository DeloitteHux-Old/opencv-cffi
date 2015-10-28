from characteristic import Attribute, attributes

from _opencv import ffi, lib
from opencv_cffi._types import Rectangle, Sequence


class _CannotLoadClassifierCascade(Exception):
    pass


@attributes(
    [
        Attribute(name="_cascade"),
        Attribute(name="_min_neighbors", default_value=3),
        Attribute(name="_scale_factor", default_value=1.1),
        Attribute(name="_min_size", exclude_from_init=True),
        Attribute(name="_max_size", exclude_from_init=True),
        Attribute(name="canny_pruning", default_value=False),
    ],
)
class HaarClassifier(object):
    def __init__(self, min_size=(100, 100), max_size=None):
        storage = lib.cvCreateMemStorage(0)
        assert storage is not None
        self._storage = storage

        if min_size is None:
            min_width = min_height = 0
        else:
            min_width, min_height = min_size
        if max_size is None:
            max_width = max_height = 0
        else:
            max_width, max_height = max_size

        self._min_size = lib.cvSize(min_width, min_height)
        self._max_size = lib.cvSize(max_width, max_height)

    @classmethod
    def from_path(cls, path, **kwargs):
        cascade = lib.cvLoadHaarClassifierCascade(
            path.path,
            lib.cvSize(1, 1),
        )
        if cascade == ffi.NULL:
            raise _CannotLoadClassifierCascade(path, cascade)
        return cls(cascade=cascade, **kwargs)

    def detect_objects(self, inside):
        objects = lib.cvHaarDetectObjects(
            inside._ipl_image,
            self._cascade,
            self._storage,
            self._scale_factor,
            self._min_neighbors,
            lib.CV_HAAR_DO_CANNY_PRUNING if self.canny_pruning else 0,
            self._min_size,
            self._max_size,
        )
        return Sequence(cv_seq=objects, type=Rectangle)
