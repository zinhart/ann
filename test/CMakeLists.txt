# includes
include_directories(.)

# add google test library
add_library(gtest_ann ./gtest/gtest-all.cc)

# link google test library
target_link_libraries(gtest_ann ${CMAKE_THREAD_LIBS_INIT})

# cpu & gpu unit tests
if(BuildCuda)
  # gpu unit tests
  set(test_lib
	  gpu_unit_tests/run_all.cc
	  gpu_unit_tests/loss_function_test.cc
	  gpu_unit_tests/optimizer_test.cc
	  gpu_unit_tests/activation_test.cc
	  gpu_unit_tests/ffn_test.cc
	  gpu_unit_tests/ann_test.cc
	 )

add_executable(ann_tests ${test_lib})

target_link_libraries(ann_tests
  gtest_ann
  ann
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)
else()
  # cpu unit tests
  set(test_lib
	  cpu_unit_tests/run_all.cc
	  cpu_unit_tests/loss_function_test.cc
	  cpu_unit_tests/optimizer_test.cc
	  cpu_unit_tests/activation_test.cc
	  cpu_unit_tests/multi_layer_perceptron_test.cc
	  cpu_unit_tests/ann_test.cc
	 )
  add_executable(ann_tests ${test_lib})
  
  target_link_libraries(ann_tests
						gtest_ann
						ann
				   	    ${MKL_LIBRARIES}
						${CMAKE_THREAD_LIBS_INIT}
						${MATH_LIBS}
					   )
endif()


add_test(
  NAME tests
  COMMAND ann_tests
)
