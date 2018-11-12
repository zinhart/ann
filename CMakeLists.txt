cmake_minimum_required(VERSION 3.1)
project(ann)

# Set build type to RELEASE by default:
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to RELEASE.")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif ()


# for other libraries 
set(ANN_INCLUDE_DIR ann/include CACHE STRING "ANN INCLUDE DIRS") 

# for unit tests using google test
option(BuildUnitTest "BuildUnitTest" ON)
# for benchmarking using google bench
option(BuildBenchmarks "BuildBenchmarks" OFF)
# for cuda when this is off will use stl threads and lapack instead of cuda and cublas
option(BuildCuda "BuildCuda" ON)
# library type
option(BuildShared "BuildShared" OFF)
# nesting protection
option(AnnUseCodeCoverage "AnnUseCodeCoverage" ON)
#set(ConcurrentRoutinesUseCodeCoverage OFF CACHE BOOL "Ignore concurrent_routines in local CodeCoverage" FORCE)
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
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

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
# mkl blas
set(MKL_USE_SINGLE_DYNAMIC_LIBRARY OFF CACHE BOOL "Use single dynamic library interface")
set(MKL_MULTI_THREADED OFF CACHE BOOL "Use multi-threading(intel tbb, omp etc)")
find_package(MKL REQUIRED)

# find cuda package and set compute achitecture
if(BuildCuda)
  find_package(CUDA REQUIRED)

# check to see if CUDA_ENABLED was defined by another library using this one
  string(REGEX MATCH "CUDA_ENABLED" result "${COMPILE_DEFINITIONS}" )
  if(${result} MATCHES "CUDA_ENABLED")
    # do nothing
  else()
	# set macro to conditionaly include headers define functions etc w.r.t gpu code
	add_definitions(-DCUDA_ENABLED=true)
  endif()

# set nvcc flags for debugging
  if("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
	# check if nvcc flags have been set already
	if(CUDA_NVCC_FLAGS STREQUAL "")
	  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_61 -G -Wno-deprecated-gpu-targets;")
	endif()
# otherwise set release flags
  else()
	if(CUDA_NVCC_FLAGS STREQUAL "")
	  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_61 -Xptxas -O3 -lineinfo")
	endif()
  endif()
  
  # set include paths
  include_directories(. include ${MKL_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} multi_core/include)

else()
  string(REGEX MATCH "CUDA_ENABLED" result "${COMPILE_DEFINITIONS}")
  if(${result} MATCHES "CUDA_ENABLED")
  else()
	add_definitions(-DCUDA_ENABLED=false)
  endif()
  # set include paths
  include_directories(. include "${MULTI_CORE_INCLUDE_DIR}" ${MKL_INCLUDE_DIR} )
endif()


# set cxx standard
set(CMAKE_CXX_STANDARD 11)
# force cxx standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set cxx flags
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 ${CMAKE_CXX_FLAGS}")
# set cxx release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fPIC")
# set cxx debug flags
if(AnnUseCodeCoverage)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -pedantic -fno-omit-frame-pointer -fno-inline-functions -fPIC --coverage")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -pedantic -fno-omit-frame-pointer -fno-inline-functions -fPIC")
endif()

# sanitize flags
if (Sanitize)
  set(CMAKE_CXX_FLAGS "-fsantize=address -fsanitize=undefined ${CMAKE_CXX_FLAGS}")
endif()

# add multi_core directory
#add_subdirectory(multi_core)

# add src directory
add_subdirectory(src)

#enable testing and add test directory
if(BuildUnitTest)
  enable_testing()
  find_package(GTest)

  if(GTEST_FOUND)
	# add include directories
	include_directories(${GTEST_INCLUDE_DIRS})
  else()
  	# Download and unpack googletest at configure time
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExternalProjects/CMakeLists.txt.in.gtest googletest-download/CMakeLists.txt)
	execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
	endif()
	execute_process(COMMAND ${CMAKE_COMMAND} --build .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "Build step for googletest failed: ${result}")
	endif()

	# Prevent overriding the parent project's compiler/linker
	# settings on Windows
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

	# Conditionally Add googletest directly to our build, if this is not a sub project. This defines
	# the gtest and gtest_main targets.
	string(REGEX MATCH "${CMAKE_CURRENT_SOURCE_DIR}" result "${CMAKE_SOURCE_DIR}")
	if(${result} MATCHES "${CMAKE_CURRENT_SOURCE_DIR}")
	  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
  				     ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
  				     EXCLUDE_FROM_ALL 
				    )
	endif()
  endif()

  add_subdirectory(unit_tests)

endif()

if(BuildBenchmarks)
  #  set(CMAKE_ARGS "${CMAKE_ARGS} -DBENCHMARK_ENABLE_TESTING${STRING}=OFF") would like to config this with cmake args
  #  set(CMAKE_CACHE_ARGS "${CMAKE_CACHE_ARGS} -DBENCHMARK_ENABLE_TESTING${STRING}=OFF")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
  if(BuildUnitTest)
	set(GTEST_ROOT_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/googletest-src/googletest" CACHE PATH "Path to externally added gtest")
	#$  list(APPEND CMAKE_ARGS "-D${GTEST_ROOT}:${string}=\"${GTEST_ROOT_LOCATION}\"")
  endif()

  # Download and unpack googlebench at configure time
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExternalProjects/CMakeLists.txt.in.gbench googlebench-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
	RESULT_VARIABLE result
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlebench-download )
  if(result)
	message(FATAL_ERROR "CMake step for googlebench failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
	RESULT_VARIABLE result
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlebench-download )
  if(result)
	message(FATAL_ERROR "Build step for googlebench failed: ${result}")
  endif()

  # Add googlebench directly to our build. This defines
  # the gtest and gtest_main targets.
  string(REGEX MATCH "${CMAKE_CURRENT_SOURCE_DIR}" result "${CMAKE_SOURCE_DIR}")
  if(${result} MATCHES "${CMAKE_CURRENT_SOURCE_DIR}")
   	add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googlebench-src
					 ${CMAKE_CURRENT_BINARY_DIR}/googlebench-build
				     EXCLUDE_FROM_ALL
					)
  endif()
  add_subdirectory(benchmarks)
endif()

# code coverage
if(AnnUseCodeCoverage)
  if(CMAKE_COMPILER_IS_GNUCXX)
	  include(CodeCoverage)
	  # ignore extraneous files
	  set(COVERAGE_LCOV_EXCLUDES '/usr/include/*' '${PROJECT_BINARY_DIR}/googlebench-src/*' '${PROJECT_BINARY_DIR}/googletest-src/*' '${PROJECT_BINARY_DIR}/multi_core/*')
	  SETUP_TARGET_FOR_COVERAGE_LCOV(NAME         ${PROJECT_NAME}_coverage 
									 EXECUTABLE   ./unit_tests/ann_unit_tests 
									 DEPENDENCIES ${PROJECT_NAME})
  endif()
endif()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR       : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID           : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library type          : ${LIB_TYPE}")
if(BuildUnitTest)
message(STATUS " Build Tests			  : ${BuildUnitTest}")
endif()
if(BuildBenchmarks)
message(STATUS " Build Benchmarks			  : ${BuildBenchmarks}")
endif()
message(STATUS "  Sanitize flags        : ${Sanitize}")
if(BuildCuda)
  message(STATUS " Found CUDA			  : ${CUDA_FOUND}")
  message(STATUS " CUDA_NVCC_FLAGS               : ${CUDA_NVCC_FLAGS}")
else()
  message(STATUS "  FOUND THREADS       : ${THREADS_FOUND}")
  message(STATUS " Found MKL include             : ${MKL_INCLUDE_DIR}")
  message(STATUS " Found MKL libraries           : ${MKL_LIBRARIES}")
endif()
message(STATUS "  CXX_FLAGS             : ${CMAKE_CXX_FLAGS}")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  message(STATUS "  CXX_FLAGS_RELEASE    : ${CMAKE_CXX_FLAGS_RELEASE}")
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  message(STATUS "  CXX_FLAGS_DEBUG      : ${CMAKE_CXX_FLAGS_DEBUG}")
endif ()
message(STATUS "")
