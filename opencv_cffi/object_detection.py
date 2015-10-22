from characteristic import Attribute, attributes

from _opencv import lib
from opencv_cffi.utils import _OpenCVSequence


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
        raw_objects = lib.cvHaarDetectObjects(
            inside,
            self.cascade,
            lib.cvCreateMemStorage(0),
            1.1,
            4,
            0,
            lib.cvSize(50, 50),
            lib.cvSize(0, 0),
        )

        return _OpenCVSequence(cv_seq=raw_objects, contents_type="CvRect *")
