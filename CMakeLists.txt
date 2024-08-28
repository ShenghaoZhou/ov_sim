cmake_minimum_required(VERSION 3.12)
project(ov_sim_bindings LANGUAGES CXX)

# Find pybind11
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Add the module using pybind11_add_module
pybind11_add_module(ov_sim ov_sim/py_binding.cpp ov_sim/BsplineSE3.h ov_sim/BsplineSE3.cpp
ov_sim/utils/colors.h ov_sim/utils/print.cpp ov_sim/utils/print.h ov_sim/utils/quat_ops.h 
)
target_link_libraries(ov_sim PRIVATE Eigen3::Eigen)
install(TARGETS ov_sim DESTINATION ov_sim)
