include_directories(${PROJECT_SOURCE_DIR})

add_executable(ga ga_test.cc)
add_executable(brkga brkga_test.cc)
add_executable(brkga_init brkga_init_test.cc)
add_executable(rcga_init rcga_init_test.cc)

add_executable(pso pso_test.cc)
