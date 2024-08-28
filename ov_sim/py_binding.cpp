#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "BsplineSE3.h"

namespace py = pybind11;

PYBIND11_MODULE(ov_sim, m)
{
    py::class_<ov_core::BsplineSE3>(m, "BsplineSE3")
        .def(py::init<>())
        .def("feed_trajectory", &ov_core::BsplineSE3::feed_trajectory)
        .def("get_start_time", &ov_core::BsplineSE3::get_start_time)
        .def("get_motion", &ov_core::BsplineSE3::get_motion);
}