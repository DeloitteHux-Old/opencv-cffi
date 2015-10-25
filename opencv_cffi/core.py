from contextlib import contextmanager

from characteristic import Attribute, attributes

from _opencv import ffi, lib


@attributes(
    [
        Attribute(name="_ipl_image")
    ],
)
class Image(object):
    def __del__(self):
        lib.cvReleaseImage(self._ipl_image)

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
        copy(self._ipl_image, copied)
        return self.__class__(ipl_image=copied)

    def write_into(self, image):
        copy(array=self._ipl_image, into=image._ipl_image)

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
    def __del__(self):
        lib.cvReleaseData(self._cv_mat)

    def __setitem__(self, (row, column), element):
        lib.cvmSet(self._cv_mat, row, column, element)

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
    def translation(cls, x=0, y=0, **kwargs):
        """
        Return an (affine) translation matrix of the given x and y amounts.

        """

        return cls.from_data(
            [1, 0, x],
            [0, 1, y],
        )

    @classmethod
    def zeros(cls, **kwargs):
        matrix = cls.of_dimensions(**kwargs)
        lib.cvSetZero(matrix)
        return matrix

    @property
    def rows(self):
        return self._cv_mat.rows

    @property
    def columns(self):
        return self._cv_mat.cols

    def warp_affine(self, image):
        if self.rows != 2 or self.columns != 3:
            raise TypeError("Affine transformations are 2x3 matrices.")
        warp_affine(matrix=self._cv_mat, array=image._ipl_image)


def copy(array, into=None):
    if into is None:
        into = array
    lib.cvCopy(array, into, ffi.NULL)


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


def warp_affine(matrix, array, into=None):
    if into is None:
        into = array
    lib.cvWarpAffine(
        array,
        into,
        matrix,
        lib.CV_INTER_LINEAR + lib.CV_WARP_FILL_OUTLIERS,
        lib.cvScalarAll(0.0),
    )
