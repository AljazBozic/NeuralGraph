# -*- encoding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from setuptools import setup

from setuptools.extension import Extension


class lazy_cythonize(list):
    """
    Lazy evaluate extension definition, to allow correct requirements install.
    """

    def __init__(self, callback):
        super(lazy_cythonize, self).__init__()
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()

        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():

    from Cython.Build import cythonize
    import numpy

    numpy_include_dir = numpy.get_include()
    eigen_include_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eigen3'))

    marching_cubes_module = Extension(
        "marching_cubes._mcubes",
        [
            "marching_cubes/src/_mcubes.pyx",
            "marching_cubes/src/pywrapper.cpp",
            "marching_cubes/src/marchingcubes.cpp"
        ],
        language="c++",
        extra_compile_args=['-std=c++11', '-Wall'],
        include_dirs=[numpy_include_dir, eigen_include_dir],
        depends=[
            "marching_cubes/src/marchingcubes.h",
            "marching_cubes/src/pyarray_symbol.h",
            "marching_cubes/src/pyarraymodule.h",
            "marching_cubes/src/pywrapper.h"
        ],
    )

    return cythonize([marching_cubes_module])

setup(
    name="PyMarchingCubes",
    version="0.0.2",
    description="Marching cubes for Python",
    author="Justus Thies (PyMCubes: Pablo Márquez Neila)",
    url="https://github.com/JustusThies/PyMarchingCubes",
    license="BSD 3-clause",
    long_description="""
    Marching cubes for Python
    """,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    packages=["marching_cubes"],
    ext_modules=lazy_cythonize(extensions),
    requires=['numpy', 'Cython', 'PyCollada'],
    setup_requires=['numpy', 'Cython']
)
