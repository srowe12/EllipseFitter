set(sources ellipse_fit_tests.cpp )
set(test_name ${lib_name}_tests)

add_executable(${test_name} ${sources})
target_link_libraries(${test_name} ${lib_name} ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

add_test(ellipse_fit_test ${test_name})


# Add performance test

set(sources_speed ellipse_fit_speed_test.cpp)
add_executable(${test_name}_speed ${sources_speed})
target_link_libraries(${test_name}_speed ${lib_name} ${ARMADILLO_LIBRARIES})


