set(logic_gates_src logic_gates/logic_gates.cc)
set(quadrants_src quadrants/quadrants.cc)

add_executable(logic_gates ${logic_gates_src} )
add_executable(quadrants ${quadrants_src})

target_include_directories(logic_gates PUBLIC ann bmp multi_core)

target_link_libraries(logic_gates ann ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(quadrants ann ${CMAKE_THREAD_LIBS_INIT})
