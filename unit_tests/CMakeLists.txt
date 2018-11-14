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
  gtest_main
  ann
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)
else()
  # cpu unit tests
  set(test_lib
	  cpu_unit_tests/run_all.cc
	  cpu_unit_tests/loss_function_test.cc
	  cpu_unit_tests/activation_test.cc
	  cpu_unit_tests/layer_test.cc
	  cpu_unit_tests/optimizer_test.cc
	  cpu_unit_tests/multi_layer_perceptron_test.cc
	  cpu_unit_tests/ann_mlp_test.cc
	 )
  add_executable(ann_unit_tests ${test_lib})
  
  target_link_libraries(ann_unit_tests
						gtest
	#	gmock_main
						ann
				   	    ${MKL_LIBRARIES}
						${CMAKE_THREAD_LIBS_INIT}
						${MATH_LIBS}
					   )
endif()


add_test(
  NAME ann_unit_tests
  COMMAND ann_unit_tests
)
