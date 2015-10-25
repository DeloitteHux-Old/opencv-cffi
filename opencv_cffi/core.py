from characteristic import Attribute, attributes

from _opencv import lib


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
