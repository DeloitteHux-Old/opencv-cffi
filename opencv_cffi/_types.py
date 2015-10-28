from characteristic import Attribute, attributes

from _opencv import ffi, lib


@attributes(
    [
        Attribute(name="_cv_seq"),
        Attribute(name="type"),
    ],
)
class Sequence(object):
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        return self.type.from_chars(lib.cvGetSeqElem(self._cv_seq, index))

    def __len__(self):
        return self._cv_seq.total


@attributes(
    [
        Attribute(name="_cv_rect"),
    ],
)
class Rectangle(object):
    @classmethod
    def from_chars(cls, chars):
        return cls(cv_rect=ffi.cast("CvRect *", chars))

    def draw_onto(
        self,
        frame,
        color,
        line_thickness=1,
        line_type=8,
        shift=0,
    ):
        lib.cvRectangle(
            frame._ipl_image,
            self.top_left,
            self.bottom_right,
            [[color.blue, color.green, color.red, color.alpha]],
            line_thickness,
            line_type,
            shift,
        )

    @property
    def top_left(self):
        return self.x, self.y

    @property
    def bottom_right(self):
        return self.x + self.width, self.y + self.height

    @property
    def width(self):
        return self._cv_rect.width

    @property
    def height(self):
        return self._cv_rect.height

    @property
    def x(self):
        return self._cv_rect.x

    @property
    def y(self):
        return self._cv_rect.y

    @property
    def left_half(self):
        new_width = self.width // 2
        cv_rect = lib.cvRect(
            self.x,
            self.y,
            new_width,
            self.height,
        )
        return self.__class__(cv_rect=cv_rect)

    @property
    def right_half(self):
        new_width = self.width // 2
        cv_rect = lib.cvRect(
            self.x + new_width,
            self.y,
            new_width,
            self.height,
        )
        return self.__class__(cv_rect=cv_rect)
