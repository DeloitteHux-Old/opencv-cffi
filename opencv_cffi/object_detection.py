from characteristic import Attribute, attributes

from _opencv import lib
from opencv_cffi._types import Rectangle, Sequence


@attributes(
    [
        Attribute(name="cascade"),
    ],
)
class HaarClassifier(object):
    @classmethod
    def from_path(cls, path, **kwargs):
        cascade = lib.cvLoadHaarClassifierCascade(
            path.path,
            lib.cvSize(1, 1),
        )
        return cls(cascade=cascade, **kwargs)

    def detect_objects(self, inside):
        objects = lib.cvHaarDetectObjects(
            inside,
            self.cascade,
            lib.cvCreateMemStorage(0),
            1.1,
            4,
            0,
            lib.cvSize(50, 50),
            lib.cvSize(0, 0),
        )

        return Sequence(cv_seq=objects, type=Rectangle)
