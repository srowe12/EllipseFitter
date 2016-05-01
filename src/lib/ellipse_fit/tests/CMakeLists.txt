set(sources ellipse_fit_tests.cpp )
set(test_name ${lib_name}_tests)

add_executable(${test_name} ${sources})
#include_directories( ${CMAKE_LIB_DIR}/math_utils/ ${math_hdr} ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
target_link_libraries(${test_name} ${lib_name} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

add_test(ellipse_fit_test ${test_name})

