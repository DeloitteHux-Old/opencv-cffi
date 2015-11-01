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

    def _show(self, image):
        lib.cvShowImage(self.name, image._cv_arr)

    def loop_over(self, images, handle_input=lambda key : None, delay=25):
        for image in images:
            self._show(image)

            keycode = lib.cvWaitKey(delay)
            nothing_was_pressed = keycode == -1
            if nothing_was_pressed:
                continue

            if 0 <= keycode <= 255:
                key = chr(keycode)
            else:
                key = keycode

            handle_input(key=key)
