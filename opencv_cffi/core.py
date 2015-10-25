from contextlib import contextmanager

from characteristic import Attribute, attributes

from _opencv import ffi, lib


@attributes(
    [
        Attribute(name="_ipl_image")
    ],
)
class Image(object):
    @property
    def depth(self):
        return self._ipl_image.depth

    @property
    def channels(self):
        return self._ipl_image.nChannels

    @contextmanager
    def region_of_interest(self, rectangle):
        lib.cvSetImageROI(self._ipl_image, rectangle._cv_rect)
        yield rectangle
        lib.cvResetImageROI(self._ipl_image)

    def copy(self):
        copied = lib.cvCreateImage(
            lib.cvGetSize(self._ipl_image),
            self.depth,
            self.channels,
        )
        lib.cvCopy(self._ipl_image, copied, ffi.NULL)
        return self.__class__(ipl_image=copied)

    def flipped_horizontal(self):
        copied = self.copy()
        flip_horizontal(copied._ipl_image)
        return copied

    def flipped_vertical(self):
        copied = self.copy()
        flip_horizontal(copied._ipl_image)
        return copied

    def flipped(self):
        copied = self.copy()
        flip(copied._ipl_image)
        return copied


@attributes(
    [
        Attribute(name="_cv_mat")
    ],
)
class Matrix(object):
    @classmethod
    def from_data(cls, *rows, **kwargs):
        if not rows:
            raise ValueError("Cannot create an empty matrix.")
        matrix = cls.of_dimensions(
            rows=len(rows),
            columns=len(rows[0]),
            **kwargs
        )
        for row, elements in enumerate(rows):
            for column, element in enumerate(elements):
                matrix[row, column] = element
        return matrix

    @classmethod
    def of_dimensions(cls, rows, columns, **kwargs):
        return cls(cv_mat=lib.cvCreateMat(rows, columns, 6), **kwargs)

    @classmethod
    def zeros(cls, **kwargs):
        matrix = cls.of_dimensions(**kwargs)
        lib.cvSetZero(matrix)
        return matrix

    def __setitem__(self, (row, column), element):
        lib.cvmSet(self._cv_mat, row, column, element)


def invert(image, into=None):
    if into is None:
        into = image
    lib.cvNot(image._ipl_image, into._ipl_image)


def flip_horizontal(array, into=None):
    if into is None:
        into = array
    lib.cvFlip(array, into, 0)


def flip_vertical(array, into=None):
    if into is None:
        into = array
    lib.cvFlip(array, into, 1)


def flip(array, into=None):
    if into is None:
        into = array
    lib.cvFlip(array, into, -1)
