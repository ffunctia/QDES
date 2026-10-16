cmake_minimum_required(VERSION 3.18)
project(qdes LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(pybind11 CONFIG REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

pybind11_add_module(utils_cside MODULE csrc/utils_cside_new.cpp)
target_include_directories(utils_cside PRIVATE "${NUMPY_INCLUDE_DIR}")
target_compile_features(utils_cside PRIVATE cxx_std_14)
target_compile_options(utils_cside PRIVATE -O3)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(utils_cside PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(utils_cside PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/qdes"
)
