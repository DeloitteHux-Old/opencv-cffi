===========
opencv-cffi
===========

Building
--------

E.g.::

    opencv=~/Development/LookHowPretty/opencv-3.0.0 LD_LIBRARY_PATH=$opencv/lib/ DYLD_LIBRARY_PATH=$opencv/lib/ C_INCLUDE_PATH=$opencv/include/:$opencv/modules/core/include/:$opencv/modules/hal/include:$opencv/modules/imgproc/include:$opencv/modules/photo/include:$opencv/modules/video/include:$opencv/modules/objdetect/include:$opencv/modules/videoio/include/:$opencv/modules/highgui/include/:$opencv/modules/imgcodecs/include LDFLAGS=-L$opencv/lib pypy opencv_cffi/build.py
