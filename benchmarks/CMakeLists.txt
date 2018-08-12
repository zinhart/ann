# cpu & gpu benchmarks
if(BuildCuda)
  #to do
# cpu only unit tests
else()
  set (concurrent_routines_benchmarks_src dense_mlp_benchmarks.cc)
  add_executable(concurrent_routines_benchmarks ${concurrent_routines_benchmarks_src})
  target_link_libraries(concurrent_routines_benchmarks 
	                    benchmark
						concurrent_routines
						ann
						${MKL_LIBRARIES}
						${CMAKE_THREAD_LIBS_INIT}
					   )
endif()

