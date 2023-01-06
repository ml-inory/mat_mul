cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mm)

set(CMAKE_CXX_FLAGS "-O3")
add_executable(mm mm.cpp)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
install(TARGETS mm
        RUNTIME DESTINATION ./)