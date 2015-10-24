===========
opencv-cffi
===========

Requirements
------------

You'll need a copy of OpenCV 3.0.x to link against, and the Haar Cascade
files from OpenCV 2.4.10 to do recognition, since the newer versions
changed the format but the C API seems to not have been updated to be
able to read them.

You can grab both versions from:

    http://opencv.org/downloads.html

Build the 3.0 version (via ``cmake . && make``). Install it if you
wish, otherwise follow below to just tell ``opencv-cffi`` where you've
downloaded it.

On OS X, you can also get OpenCV 3.0 by running::

    $ brew edit homebrew/science/opencv

remove the ``, branch => '2.4'``, and then

    $ brew install --HEAD --without-python homebrew/science/opencv

(We do not need the Python bindings, you're lookin' at Python bindings).

Building
--------

E.g.::

    $ opencv=~/Development/opencv-3.0.0 \
      LD_LIBRARY_PATH=$opencv/lib/ \
      DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH \
      C_INCLUDE_PATH=$opencv/include/:$opencv/modules/core/include/:$opencv/modules/hal/include:$opencv/modules/imgproc/include:$opencv/modules/photo/include:$opencv/modules/video/include:$opencv/modules/objdetect/include:$opencv/modules/videoio/include/:$opencv/modules/highgui/include/:$opencv/modules/imgcodecs/include \
      LDFLAGS=-L$opencv/lib\
      pypy opencv_cffi/build.py
