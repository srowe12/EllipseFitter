set(lib_name ellipse_fit)
set(sources ellipse_fit.cpp)

add_library(${lib_name} SHARED ${sources})
target_link_libraries(${lib_name} armadillo)

add_subdirectory(tests)
