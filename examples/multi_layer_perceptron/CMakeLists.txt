#include_directories("${ANN_INCLUDE_DIR}" "${MULTI_CORE_INCLUDE_DIR}" "${BMP_INCLUDE_DIR}" "${PROJECT_BINARY_DIR}")
set(HEADERS 
  #"${CMAKE_SOURCE_DIR}/${ANN_INCLUDE_DIR}"
  "${CMAKE_SOURCE_DIR}/${MULTI_CORE_INCLUDE_DIR}" 
  "${CMAKE_SOURCE_DIR}/${BMP_INCLUDE_DIR}" 
  "${PROJECT_BINARY_DIR}")
set(logic_gates_src logic_gates/logic_gates.cc)
set(quadrants_src quadrants/quadrants.cc)

add_executable(logic_gates ${logic_gates_src} )
add_executable(quadrants ${quadrants_src})

target_include_directories(logic_gates PUBLIC ann bmp multi_core)

target_link_libraries(logic_gates ann ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(quadrants ann ${CMAKE_THREAD_LIBS_INIT})
