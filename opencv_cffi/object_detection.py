from characteristic import Attribute, attributes

from _opencv import lib
from opencv_cffi._types import Rectangle, Sequence


@attributes(
    [
        Attribute(name="_cascade"),
    ],
)
class HaarClassifier(object):
    def __init__(self):
        storage = lib.cvCreateMemStorage(0)
        assert storage is not None
        self._storage = storage

    @classmethod
    def from_path(cls, path, **kwargs):
        cascade = lib.cvLoadHaarClassifierCascade(
            path.path,
            lib.cvSize(1, 1),
        )
        assert cascade is not None
        return cls(cascade=cascade, **kwargs)

    def detect_objects(self, inside):
        objects = lib.cvHaarDetectObjects(
            inside,
            self._cascade,
            self._storage,
            1.1,
            4,
            0,
            lib.cvSize(100, 100),
            lib.cvSize(0, 0),
        )
        return Sequence(cv_seq=objects, type=Rectangle)
