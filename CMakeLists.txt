cmake_minimum_required (VERSION 2.6)
project (ellipse_fit)
subdirs(src)
enable_testing()
find_package(GTest REQUIRED)
#May need to export GTEST_ROOT=googletest/build
#and put the include directory in the build also. This is a hack.
#For Armadillo, you may need to export CMAKE_PREFIX_PATH=$YOUR_ARMADILLO_ROOT
find_package(Threads)
include_directories(${GTEST_INCLUDE_DIRS})
SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
