cmake_minimum_required(VERSION 3.1)
project(ann)

# Set build type to RELEASE by default:
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to RELEASE.")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif ()

# for unit tests using google test
option(BuildTest "BuildTest" ON)
# for cuda when this is off will use stl threads and lapack instead of cuda and cublas
option(BuildCuda "BuildCuda" ON)
# library type
option(BuildShared "BuildShared" OFF)
# sanitizer support
option(Sanitize "Sanitize" OFF)

#threading options 
option(PTHREADS "PTHREADS" ON)
option(WIN32_THREADS "WIN32_THREADS" OFF)


# set lib type
if(BuildShared)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

#set module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# set lib type
if(PTHREADS)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
elseif(WIN32_THREADS)
  set(THREADS_PREFER_WIN32_FLAG ON)
elseif(PTHREADS AND WIN32_THREADS)
  message(FATAL_ERROR "Choose 1 threading library")
endif()

# threading library
find_package(Threads REQUIRED)


# find cuda package and set compute achitecture
if(BuildCuda)
  find_package(CUDA REQUIRED)
  set(CUDA_ARCH "-arch=sm_61")

  # check to see if CUDA_ENABLED was defined by another library using this one
  string(REGEX MATCH "CUDA_ENABLED" result "${COMPILE_DEFINITIONS}" )
  if(${result} MATCHES "CUDA_ENABLED")
    # do nothing
  else()
	# set macro to conditionaly include headers define functions etc w.r.t gpu code
	add_definitions(-DCUDA_ENABLED=true)
  endif()
else()
  string(REGEX MATCH "CUDA_ENABLED" result "${COMPILE_DEFINITIONS}")
  if(${result} MATCHES "CUDA_ENABLED")
  else()
	add_definitions(-DCUDA_ENABLED=false)
  endif()
  find_package(LAPACK)
endif()

# set include paths
include_directories(. include ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} concurrent_routines/include)

# set cxx standard
set(CMAKE_CXX_STANDARD 11)
# force cxx standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set cxx flags
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 ${CMAKE_CXX_FLAGS}")
# set cxx release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# set cxx debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -pedantic -fno-omit-frame-pointer -fno-inline-functions")

#only if Building gpu support
if(BuildCuda)
  # set nvcc flags for debugging
  if("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -lineinfo;-G;-Wno-deprecated-gpu-targets;")
  # release flags # to do
  else()
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -Xptxas -O3")
  endif()
endif()


# sanitize flags
if (Sanitize)
  set(CMAKE_CXX_FLAGS "-fsantize=address -fsanitize=undefined ${CMAKE_CXX_FLAGS}")
endif()

# add concurrent_routines directory
add_subdirectory(concurrent_routines)

# add src directory
add_subdirectory(src)

#enable testing and add test directory
if (BuildTest)
  enable_testing()
  add_subdirectory(test)
endif ()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR       : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID           : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library type          : ${LIB_TYPE}")
message(STATUS "  Build tests           : ${BuildTest}")
message(STATUS "  Sanitize flags        : ${Sanitize}")
if(BuildCuda)
  message(STATUS "  Found CUDA          : ${CUDA_FOUND}")
  message(STATUS "  NVCC_FLAGS          : ${CUDA_NVCC_FLAGS}")
else()
  message(STATUS "  FOUND THREADS       : ${THREADS_FOUND}")
  message(STATUS "  FOUND LAPACKE         : ${LAPACK_FOUND}")
endif()
message(STATUS "  CXX_FLAGS             : ${CMAKE_CXX_FLAGS}")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  message(STATUS "  CXX_FLAGS_RELEASE    : ${CMAKE_CXX_FLAGS_RELEASE}")
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  message(STATUS "  CXX_FLAGS_DEBUG      : ${CMAKE_CXX_FLAGS_DEBUG}")
endif ()
message(STATUS "")
