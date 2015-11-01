from contextlib import contextmanager

from characteristic import Attribute, attributes

from _opencv import ffi, lib


@attributes(
    [
        Attribute(name="red", default_value=0),
        Attribute(name="blue", default_value=0),
        Attribute(name="green", default_value=0),
        Attribute(name="alpha", default_value=0),
    ],
)
class Color(object):
    pass


Color.RED = Color(red=255)
Color.BLUE = Color(blue=255)
Color.GREEN = Color(green=255)


@attributes(
    [
        Attribute(name="_cv_arr")
    ],
)
class Image(object):
    @classmethod
    def from_path(cls, path):
        ipl_image = lib.cvLoadImage(path.path, lib.CV_LOAD_IMAGE_COLOR)
        return cls(cv_arr=ipl_image)

    @property
    def depth(self):
        return self._cv_arr.depth

    @property
    def channels(self):
        return self._cv_arr.nChannels

    @contextmanager
    def region_of_interest(self, rectangle):
        lib.cvSetImageROI(self._cv_arr, rectangle._cv_rect)
        yield rectangle
        lib.cvResetImageROI(self._cv_arr)

    def copy(self):
        copied = lib.cvCreateImage(
            lib.cvGetSize(self._cv_arr),
            self.depth,
            self.channels,
        )
        ffi.gc(copied, lib.cvReleaseImage)
        copy(self._cv_arr, copied)
        return self.__class__(cv_arr=copied)

    def to_matrix(self):
        cv_mat = ffi.new("CvMat *")
        lib.cvGetMat(self._cv_arr, cv_mat, ffi.NULL, 0)
        return Matrix(cv_arr=cv_mat)

    def write_into(self, image):
        copy(array=self._cv_arr, into=image._cv_arr)

    def flipped_horizontal(self):
        copied = self.copy()
        flip_horizontal(copied._cv_arr)
        return copied

    def flipped_vertical(self):
        copied = self.copy()
        flip_vertical(copied._cv_arr)
        return copied

    def flipped(self):
        copied = self.copy()
        flip(copied._cv_arr)
        return copied


@attributes(
    [
        Attribute(name="_cv_arr")
    ],
)
class Matrix(object):
    def __setitem__(self, (row, column), element):
        lib.cvmSet(self._cv_arr, row, column, element)

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
        cv_mat = lib.cvCreateMat(rows, columns, lib.CV_64FC1)
        return cls(cv_arr=cv_mat, **kwargs)

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
        lib.cvSetZero(matrix._cv_arr)
        return matrix

    @property
    def rows(self):
        return self._cv_arr.rows

    @property
    def columns(self):
        return self._cv_arr.cols

    def warp_affine(self, image):
        if self.rows != 2 or self.columns != 3:
            raise TypeError("Affine transformations are 2x3 matrices.")
        warp_affine(matrix=self._cv_arr, array=image._cv_arr)


def copy(array, into=None):
    if into is None:
        into = array
    lib.cvCopy(array, into, ffi.NULL)


def invert(image, into=None):
    if into is None:
        into = image
    lib.cvNot(image._cv_arr, into._cv_arr)


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
