add_library(gtest ./gtest/gtest-all.cc)
include_directories(.)
target_link_libraries(gtest ${CMAKE_THREAD_LIBS_INIT})

set(test_src
  run_all.cc
  loss_function_test.cc
  optimizer_test.cc
  activation_test.cc
  ann_test.cc
)

add_executable(ann_tests ${test_src})

target_link_libraries(ann_tests
  gtest
  ann
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)

add_test(
  NAME tests
  COMMAND ann_tests
)
