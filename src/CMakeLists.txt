if(BuildCuda)
  set(ann_lib
	  gpu/loss_function.cu  
	  gpu/optimizer.cu
	  gpu/activation.cu  
	  gpu/ann.cu	
	 )
  cuda_add_library(ann ${LIB_TYPE} ${ann_lib})

  target_link_libraries(ann
						concurrent_routines
						${CUDA_CUBLAS_LIBRARIES}
						${CMAKE_THREAD_LIBS_INIT}
						${MATH_LIBS}
					  )
  if(BuildShared)
	#install  libconcurrent_routines.so to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS ann LIBRARY DESTINATION lib LIBRARY DESTINATION lib)
  else()
	# install  libconcurrent_routines.a to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS ann LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  endif()
  #to do
  install(DIRECTORY ../include/ann DESTINATION include)

else()
  set(ann_lib 
	  cpu/optimizer.cc
	  cpu/activation.cc
	 )

   add_library(ann ${ann_lib})

   target_link_libraries(ann
						concurrent_routines
						${MKL_LIBRARIES}
						${CMAKE_THREAD_LIBS_INIT}
						${MATH_LIBS}
					  )
  if(BuildShared)
	#install  libconcurrent_routines.so to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS ann LIBRARY DESTINATION lib LIBRARY DESTINATION lib)
  else()
	# install  libconcurrent_routines.a to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS ann LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  endif()

  #to do
  install(DIRECTORY ../include/ann DESTINATION include)
endif()

