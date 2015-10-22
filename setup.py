import os

from setuptools import find_packages, setup


with open(os.path.join(os.path.dirname(__file__), "README.rst")) as readme:
    long_description = readme.read()

classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy"
]

setup(
    name="opencv_cffi",
    packages=find_packages(),
    py_modules=(),
    setup_requires=["vcversioner"],
    vcversioner={"version_module_paths": ["opencv_cffi/_version.py"]},
    author="Magnetic Engineering",
    author_email="Engineering@Magnetic.com",
    classifiers=classifiers,
    description="A random subset of OpenCV's functionality, wrapped via CFFI",
    license="MIT",
    long_description=long_description,
    url="https://github.com/Magnetic/opencv-cffi",
)
