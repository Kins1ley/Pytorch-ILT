cmake_minimum_required(VERSION 3.9)

include(CMakeLists.txt.config)

if(WITH_GPU)
  add_definitions(-DWITH_GPU)
  project(cuilt LANGUAGES CXX CUDA)
else()
  project(cuilt LANGUAGES CXX)
endif(WITH_GPU)

# Cmake cxx standard is set to c++-11 by default. Change the number for some
# additional requirements (14, 17) if everything still works.
set(CMAKE_CXX_STANDARD 11)
if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  # Suppress warnings about braces when clang is chosen.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-missing-braces -O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -D_DEBUG")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(ExternalProject)
include(GNUInstallDirs)

set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(PROJECT_CMAKE_DIR ${PROJECT}/cmake)
set(PROJECT_SOURCES "" CACHE STRING "target sources")
set(PROJECT_INCLUDE_DIR ${PROJECT_ROOT_DIR})

include("cmake/gflags.cmake")
list(APPEND PROJECT_INCLUDE_DIR ${GFLAGS_INCLUDE_DIRS})
list(APPEND PROJECT_LINKER_LIB PUBLIC ${GFLAGS_LIBRARIES})

include("cmake/glog.cmake")
list(APPEND PROJECT_INCLUDE_DIR PUBLIC ${GLOG_INCLUDE_DIRS})
list(APPEND PROJECT_LINKER_LIB PUBLIC ${GLOG_LIBRARIES})

include("cmake/ProtoBuf.cmake")
set(proto_gen_folder "${PROJECT_SOURCE_DIR}/proto")
include_directories("${PROJECT_SOURCE_DIR}/proto")
set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)

add_subdirectory(ilt)
add_subdirectory(shapes)
add_subdirectory(eval)
add_subdirectory(utils)
add_subdirectory(io)
add_subdirectory(lithosim)
add_subdirectory(lithosim/pvbandsim)
if(WITH_LEVEL_SET)
    add_subdirectory(levelSet)
endif()

if(WITH_GPU)
  list(APPEND PROJECT_INCLUDE_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  add_subdirectory(lithosim/pvbandsim_gpu)
  list(APPEND PROJECT_LINKER_LIB PUBLIC pvsim_gpu)
endif(WITH_GPU)

# generate protobuf sources
file(GLOB proto_files proto/*.proto)
protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})
# include python files either to force generation
add_library(iltproto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
if (DEFINED external_project_dependencies)
    add_dependencies(iltproto ${external_project_dependencies})
endif()
target_link_libraries(iltproto PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(iltproto PUBLIC ${PROTOBUF_INCLUDE_DIR})

list(INSERT PROJECT_LINKER_LIB 0 PUBLIC iltproto) # note, crucial to prepend!

add_library(libcuilt SHARED ${PROJECT_SOURCES})
if (DEFINED external_project_dependencies)
    add_dependencies(libcuilt ${external_project_dependencies})
endif()

include_directories(${PROJECT_INCLUDE_DIR})
target_link_libraries(libcuilt ${PROJECT_LINKER_LIB})

add_executable(cuilt ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(cuilt libcuilt)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
install(TARGETS cuilt DESTINATION bin)
install(DIRECTORY ${PROJECT_INCLUDE_DIR}/ DESTINATION include)
install(TARGETS libcuilt DESTINATION lib)
install(FILES ${proto_hdrs} DESTINATION ${PROJECT_SOURCE_DIR}/proto)
install(TARGETS iltproto EXPORT CaffeTargets DESTINATION lib)

message(STATUS "Host system: " ${CMAKE_HOST_SYSTEM})
message(STATUS "System processor: " ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS "Cmake CXX compiler: " ${CMAKE_CXX_COMPILER})
message(STATUS "Cmake CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "Cmake CXX compiler flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "WITH_GPU is: " ${WITH_GPU})
message(STATUS "Installation directory: " ${CMAKE_INSTALL_PREFIX})
