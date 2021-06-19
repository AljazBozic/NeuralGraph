
#include "pywrapper.h"

#include "marchingcubes.h"

#include <stdexcept>
#include <array>


PyObject* marching_cubes_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* pyfunc, double isovalue)
{
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
    // Copy the lower and upper coordinates to a C array.
    std::array<double,3> lower_;
    std::array<double,3> upper_;
    for(int i=0; i<3; ++i)
    {
        PyObject* l = PySequence_GetItem(lower, i);
        if(l == NULL)
            throw std::runtime_error("len(lower) < 3");
        PyObject* u = PySequence_GetItem(upper, i);
        if(u == NULL)
        {
            Py_DECREF(l);
            throw std::runtime_error("len(upper) < 3");
        }
        
        lower_[i] = PyFloat_AsDouble(l);
        upper_[i] = PyFloat_AsDouble(u);
        
        Py_DECREF(l);
        Py_DECREF(u);
        if(lower_[i]==-1.0 || upper_[i]==-1.0)
        {
            if(PyErr_Occurred())
                throw std::runtime_error("unknown error");
        }
    }

    auto pyfunc_to_cfunc = [&](double x, double y, double z) -> double {
        PyObject* res = PyObject_CallFunction(pyfunc, "(d,d,d)", x, y, z);
        if(res == NULL)
            return 0.0;
        
        double result = PyFloat_AsDouble(res);
        Py_DECREF(res);
        return result;
    };
    
    // Marching cubes.
    mc::marching_cubes(lower_, upper_, numx, numy, numz, pyfunc_to_cfunc, isovalue, vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    return res;
}


PyObject* marching_cubes(PyArrayObject* arr, double isovalue)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");
    
    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    std::array<long, 3> lower{0, 0, 0};
    std::array<long, 3> upper{shape[0]-1, shape[1]-1, shape[2]-1};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
    auto pyarray_to_cfunc = [&](long x, long y, long z) -> double {
        const npy_intp c[3] = {x, y, z};
        return PyArray_SafeGet<double>(arr, c);
    };

    // Marching cubes.
    mc::marching_cubes(lower, upper, numx, numy, numz, pyarray_to_cfunc, isovalue,
                        vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    
    return res;
}






PyObject* marching_cubes_color_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* pyfunc_sdf, PyObject* pyfunc_r, PyObject* pyfunc_g, PyObject* pyfunc_b, double isovalue)
{
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
    // Copy the lower and upper coordinates to a C array.
    std::array<double,3> lower_;
    std::array<double,3> upper_;
    for(int i=0; i<3; ++i)
    {
        PyObject* l = PySequence_GetItem(lower, i);
        if(l == NULL)
            throw std::runtime_error("len(lower) < 3");
        PyObject* u = PySequence_GetItem(upper, i);
        if(u == NULL)
        {
            Py_DECREF(l);
            throw std::runtime_error("len(upper) < 3");
        }
        
        lower_[i] = PyFloat_AsDouble(l);
        upper_[i] = PyFloat_AsDouble(u);
        
        Py_DECREF(l);
        Py_DECREF(u);
        if(lower_[i]==-1.0 || upper_[i]==-1.0)
        {
            if(PyErr_Occurred())
                throw std::runtime_error("unknown error");
        }
    }

    auto pyfunc_to_cfunc_sdf = [&](double x, double y, double z) -> double {
        PyObject* res = PyObject_CallFunction(pyfunc_sdf, "(d,d,d)", x, y, z);
        if(res == NULL)
            return 0.0;
        
        double result = PyFloat_AsDouble(res);
        Py_DECREF(res);
        return result;
    };

    auto pyfunc_to_cfunc_color = [&](double x, double y, double z) -> Eigen::Vector3d {
        Eigen::Vector3d col;
        { // red
            PyObject* res = PyObject_CallFunction(pyfunc_r, "(d,d,d)", x, y, z);
            if(res == NULL) return Eigen::Vector3d::Zero();
            col.x() = PyFloat_AsDouble(res);
            Py_DECREF(res);
        }
        { // green
            PyObject* res = PyObject_CallFunction(pyfunc_g, "(d,d,d)", x, y, z);
            if(res == NULL) return Eigen::Vector3d::Zero();
            col.y() = PyFloat_AsDouble(res);
            Py_DECREF(res);
        }
        { // blue
            PyObject* res = PyObject_CallFunction(pyfunc_b, "(d,d,d)", x, y, z);
            if(res == NULL) return Eigen::Vector3d::Zero();
            col.z() = PyFloat_AsDouble(res);
            Py_DECREF(res);
        }


        return col;
    };

    
    // Marching cubes.
    mc::marching_cubes_color(lower_, upper_, numx, numy, numz, pyfunc_to_cfunc_sdf, pyfunc_to_cfunc_color, isovalue, vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    return res;
}


PyObject* marching_cubes_color(PyArrayObject* arr_sdf, PyArrayObject* arr_color, double isovalue)
{
    if(PyArray_NDIM(arr_sdf) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported (SDF).");
    if(PyArray_NDIM(arr_color) != 4)
        throw std::runtime_error("Only four-dimensional arrays are supported (RGB).");
    
    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr_sdf);
    npy_intp* shape_color = PyArray_DIMS(arr_color);
    
    if(shape[0] != shape_color[0] || shape[1] != shape_color[1] || shape[2] != shape_color[2])
        throw std::runtime_error("SDF and RGB volumes do not match in size.");
        
    if(shape_color[3] != 3)
        throw std::runtime_error("Only RGB colors are supported.");

    std::array<long, 3> lower{0, 0, 0};
    std::array<long, 3> upper{shape[0]-1, shape[1]-1, shape[2]-1};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
    auto pyarray_to_cfunc_sdf = [&](long x, long y, long z) -> double {
        const npy_intp c[3] = {x, y, z};
        return PyArray_SafeGet<double>(arr_sdf, c);
    };

    auto pyarray_to_cfunc_color = [&](long x, long y, long z) -> Eigen::Vector3d {        
        Eigen::Vector3d color;
        npy_intp c_r[4] = {x, y, z, 0};
        npy_intp c_g[4] = {x, y, z, 1};
        npy_intp c_b[4] = {x, y, z, 2};
        color.x() = PyArray_SafeGet<double>(arr_color, c_r);
        color.y() = PyArray_SafeGet<double>(arr_color, c_g);
        color.z() = PyArray_SafeGet<double>(arr_color, c_b);
        return color;
    };

    // Marching cubes.
    mc::marching_cubes_color(lower, upper, numx, numy, numz, pyarray_to_cfunc_sdf, pyarray_to_cfunc_color, isovalue, vertices, polygons);

    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    
    return res;
}




PyObject* marching_cubes_super_sampling(PyArrayObject* arrX, PyArrayObject* arrY, PyArrayObject* arrZ, double isovalue)
{
    if(PyArray_NDIM(arrX) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");
    if(PyArray_NDIM(arrY) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");
    if(PyArray_NDIM(arrZ) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");
    
    // Prepare data.
    npy_intp* shapeX = PyArray_DIMS(arrX);
    npy_intp* shapeY = PyArray_DIMS(arrY);
    npy_intp* shapeZ = PyArray_DIMS(arrZ);

    npy_intp shape[3] = {shapeY[0], shapeX[1], shapeX[2]};
    npy_intp supersamples[3] = {    (shapeX[0]-shape[0]) / (shape[0]-1), // samples along the edge (between two voxel nodes)
                                    (shapeY[1]-shape[1]) / (shape[1]-1),
                                    (shapeZ[2]-shape[2]) / (shape[2]-1)};

    if(shapeX[2] != shapeY[2] || shapeX[1] != shapeZ[1] || shapeY[0] != shapeZ[0])
        throw std::runtime_error("X,Y,Z supersampled sdf arrays need to be compatible.");

    if(     shapeX[0] != shape[0] + (shape[0]-1)*supersamples[0] 
        ||  shapeY[1] != shape[1] + (shape[1]-1)*supersamples[1]
        ||  shapeZ[2] != shape[2] + (shape[2]-1)*supersamples[2])
        throw std::runtime_error("X,Y,Z supersampled sdf arrays need to be compatible (must be dim + supersamples*(dim-1) !).");

    std::array<long, 3> lower{0, 0, 0};
    std::array<long, 3> upper{shape[0]-1, shape[1]-1, shape[2]-1};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
    auto pyarray_to_cfunc = [&](long x, long y, long z) -> double {
        const npy_intp c[3] = {x * (supersamples[0]+1), y, z};
        return PyArray_SafeGet<double>(arrX, c);
    };

    auto pyarray_to_cfuncX = [&](long x, long y, long z) -> double {
        const npy_intp c[3] = {x, y, z};
        return PyArray_SafeGet<double>(arrX, c);
    };

    auto pyarray_to_cfuncY = [&](long x, long y, long z) -> double {
        const npy_intp c[3] = {x, y, z};
        return PyArray_SafeGet<double>(arrY, c);
    };

    auto pyarray_to_cfuncZ = [&](long x, long y, long z) -> double {
        const npy_intp c[3] = {x, y, z};
        return PyArray_SafeGet<double>(arrZ, c);
    };

    // Marching cubes.
    mc::marching_cubes_super_sampling(lower, upper, numx, numy, numz, supersamples[0], supersamples[1], supersamples[2], pyarray_to_cfunc, pyarray_to_cfuncX, pyarray_to_cfuncY, pyarray_to_cfuncZ, isovalue,
                        vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    
    return res;
}