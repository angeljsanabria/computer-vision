cmake_minimum_required(VERSION 3.18)
project(my_rga LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 CONFIG REQUIRED PATHS "${pybind11_DIR}")
    else()
        message(FATAL_ERROR "pybind11 not found. Install: pip install pybind11")
    endif()
endif()

set(RGA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../rknn-toolkit2/rknpu2/examples/3rdparty/rga/include"
    CACHE PATH "Path to librga IM2D headers")

if(NOT EXISTS "${RGA_INCLUDE_DIR}/im2d.h")
    set(RGA_INCLUDE_DIR "/usr/include/rga" CACHE PATH "Path to librga IM2D headers" FORCE)
endif()

if(NOT RGA_LIBRARY)
    find_library(RGA_LIBRARY rga)
endif()
if(NOT RGA_LIBRARY)
    message(WARNING "librga not found at configure time; link may fail on build")
    set(RGA_LIBRARY rga CACHE FILEPATH "Path to librga.so")
endif()

if(CMAKE_CROSSCOMPILING)
    if(NOT Python3_INCLUDE_DIR OR NOT Python3_LIBRARY)
        message(FATAL_ERROR "Cross compile requires -DPython3_INCLUDE_DIR and -DPython3_LIBRARY")
    endif()
    if(NOT PY_SYSROOT)
        get_filename_component(PY_SYSROOT "${Python3_INCLUDE_DIR}" DIRECTORY)
        get_filename_component(PY_SYSROOT "${PY_SYSROOT}" DIRECTORY)
    endif()

    add_library(my_rga MODULE src/rga_bindings.cpp src/image_utils_port.c)
    set_target_properties(my_rga PROPERTIES PREFIX "" SUFFIX ".so")
    target_link_libraries(my_rga PRIVATE pybind11::headers "${Python3_LIBRARY}" "${RGA_LIBRARY}")
    target_include_directories(my_rga BEFORE PRIVATE
        "${PY_SYSROOT}/usr/include"
        "${Python3_INCLUDE_DIR}"
    )
    target_include_directories(my_rga PRIVATE src "${RGA_INCLUDE_DIR}")
else()
    find_package(Python REQUIRED COMPONENTS Development.Module)
    pybind11_add_module(my_rga src/rga_bindings.cpp src/image_utils_port.c)
    target_include_directories(my_rga PRIVATE src "${RGA_INCLUDE_DIR}")
    target_link_libraries(my_rga PRIVATE "${RGA_LIBRARY}")
endif()

install(TARGETS my_rga LIBRARY DESTINATION . COMPONENT my_rga)
