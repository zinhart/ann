set(ann_src
  loss_function.cu  
  optimizer.cu
  activation.cu  
  ann.cu	
)

cuda_add_library(ann ${LIB_TYPE} ${ann_src})

target_link_libraries(ann
  concurrent_routines
  ${CUDA_CUBLAS_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)

install(TARGETS ann LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ../include/ann DESTINATION include)
