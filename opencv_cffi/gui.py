from characteristic import Attribute, attributes

from _opencv import lib


@attributes(
    [
        Attribute(name="name"),
    ],
)
class Window(object):
    """
    A GUI window.

    """

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        lib.cvDestroyWindow(self.name)

    def show(self, image):
        lib.cvShowImage(self.name, image)

# lib.cvNamedWindow("Example", lib.CV_WINDOW_AUTOSIZE)
