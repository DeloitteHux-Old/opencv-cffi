from characteristic import Attribute, attributes

from _opencv import lib


ESCAPE = "\x1b"
UP_ARROW = 63232
DOWN_ARROW = 63233
LEFT_ARROW = 63234
RIGHT_ARROW = 63235


@attributes(
    [
        Attribute(name="name"),
        # TODO: Support sizing modes
    ],
)
class Window(object):
    """
    A GUI window.

    """

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        lib.cvDestroyWindow(self.name)

    def show(self, image):
        lib.cvShowImage(self.name, image._ipl_image)


def key_pressed(milliseconds=1):
    pressed = lib.cvWaitKey(milliseconds)
    if 0 <= pressed <= 255:
        return chr(pressed)
    return pressed
