from characteristic import Attribute, attributes

from _opencv import ffi,lib


@attributes(
    [
        Attribute(name="_cv_seq"),
        Attribute(name="_contents_type"),
    ],
)
class _OpenCVSequence(object):
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        return self._casted(lib.cvGetSeqElem(self._cv_seq, index))

    def __len__(self):
        return self._cv_seq.total

    def _casted(self, element):
        return ffi.cast(self._contents_type, element)


def fourcc((a, b, c, d)):
    """
    Calculate a FourCC integer from the four characters.

    http://www.fourcc.org/

    """

    return (((((ord(d) << 8) | ord(c)) << 8) | ord(b)) << 8) | ord(a)
