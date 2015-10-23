from characteristic import Attribute, attributes

from _opencv import lib


ESCAPE = ord("\x1b")


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
        lib.cvShowImage(self.name, image)


def key_pressed(keycode=ESCAPE, milliseconds=1):
    return lib.cvWaitKey(milliseconds) == keycode
