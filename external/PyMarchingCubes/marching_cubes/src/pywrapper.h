
#ifndef _PYWRAPPER_H
#define _PYWRAPPER_H

#include <Python.h>
#include "pyarraymodule.h"

#include <vector>

PyObject* marching_cubes(PyArrayObject* arr, double isovalue);
PyObject* marching_cubes_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* f, double isovalue);

    
PyObject* marching_cubes_color(PyArrayObject* arr_sdf, PyArrayObject* arr_color, double isovalue);
PyObject* marching_cubes_color_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* f_sdf, PyObject* f_color_r, PyObject* f_color_g, PyObject* f_color_b, double isovalue);


PyObject* marching_cubes_super_sampling(PyArrayObject* arrX, PyArrayObject* arrY, PyArrayObject* arrZ, double isovalue);

#endif // _PYWRAPPER_H
