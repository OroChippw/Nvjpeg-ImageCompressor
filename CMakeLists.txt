cmake_minimum_required(VERSION 3.0)
project(Nvjpeg-ImageCompressor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    # Set the MSVC input code encoding format to utf-8
    set(CMAKE_C_FLAGS /source-charset:utf-8)
endif()

message(STATUS "Project: Nvjpeg-ImageCompressor")
message(STATUS "Root Path: ${CMAKE_SOURCE_DIR}")

include(cmake/platform.cmake)  # checking platform

# <- * - ***** Link Lib ***** - * - >
# link third-party library nvjpeg and opencv4
set(NVJPEG_DIR  ${CMAKE_SOURCE_DIR}/third_party/nvjpeg)
include_directories(${NVJPEG_DIR}/include)
link_directories(${NVJPEG_DIR}/lib/x64)
set(NVJPEG_LIB nvjpeg cublas cuda cudadevrt cudart_static)

set(OPENCV_DIR  ${CMAKE_SOURCE_DIR}/third_party/opencv4)
include_directories(${OPENCV_DIR}/include)
link_directories("${OPENCV_DIR}/lib")
# set(OPENCV_LIB opencv_world470d)
set(OPENCV_LIB opencv_world470)

# <- * - ***** Build Exe ***** - * - >

include_directories("src/ImageCompressorDll")

# compile own file
file(GLOB SRC_LIST 
    ${CMAKE_SOURCE_DIR}/src/ImageCompressor/*.cpp
    ${CMAKE_SOURCE_DIR}/src/ImageCompressorDll/*.cpp
)
add_executable(demo  ${SRC_LIST})

target_link_libraries(demo ${OPENCV_LIB} ${NVJPEG_LIB})

# <- * - ***** Build Lib ***** - * - >

# Set dynamic library/static library generation path
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(
    nvjpeg_compressor_dynamic SHARED
    ${CMAKE_SOURCE_DIR}/src/ImageCompressorDll/ImageCompressor.h
    ${CMAKE_SOURCE_DIR}/src/ImageCompressorDll/ImageCompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/ImageCompressorDll/ImageCompressorImpl.h
    ${CMAKE_SOURCE_DIR}/src/ImageCompressorDll/ImageCompressorlmpl.cpp
)

target_link_libraries(
    nvjpeg_compressor_dynamic PRIVATE ${OPENCV_LIB} ${NVJPEG_LIB}
)

# 设置头文件目录
target_include_directories(nvjpeg_compressor_dynamic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(
    nvjpeg_compressor_dynamic PROPERTIES 
    OUTPUT_NAME "nvjpeg_compressor"
    VERSION 1.0 SOVERSION 1 
)

target_link_libraries(nvjpeg_compressor_dynamic PRIVATE
   ${NVJPEG_LIB}
)

target_compile_definitions(nvjpeg_compressor_dynamic PRIVATE NVJPEG_COMPRESS_RUNNER_EXPORTS)

# <- * - ***** Build Pdb ***** - * - >
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(nvjpeg_compressor_dynamic PRIVATE /Zi)
    set_target_properties(nvjpeg_compressor_dynamic PROPERTIES LINK_FLAGS "/DEBUG")
endif()

# ---------------------------------- >

# # build static library [*.lib/*.a]
# add_library(nvjpeg_compressor_static ${SRC_LIST})

# set_target_properties(
#     nvjpeg_compressor_static PROPERTIES 
#     OUTPUT_NAME "nvjpeg_compressor"
#     VERSION 1.0 SOVERSION 1 
#     DEFINE_SYMBOL NVJPEG_COMPRESS_RUNNER_EXPORTS
# )

# target_include_directories(
#     nvjpeg_compressor_static PUBLIC
#     $<BUILD_INTERFACE:${OPENCV_DIR}/include>
#     $<BUILD_INTERFACE:${NVJPEG_DIR}/include>
# )

# target_link_libraries(
#     nvjpeg_compressor_static PUBLIC
#     ${OPENCV_LIB} ${NVJPEG_LIB}
# )

# # build dynamic library [*.dll/*.so]
# add_library(nvjpeg_compressor_dynamic SHARED ${SRC_LIST})

# set_target_properties(
#     nvjpeg_compressor_dynamic PROPERTIES 
#     OUTPUT_NAME "nvjpeg_compressor"
#     VERSION 1.0 SOVERSION 1 # dll version and api version
#     DEFINE_SYMBOL NVJPEG_COMPRESS_RUNNER_EXPORTS
# )

# target_include_directories(
#     nvjpeg_compressor_dynamic PUBLIC
#     $<BUILD_INTERFACE:${OPENCV_DIR}/include>
#     $<BUILD_INTERFACE:${NVJPEG_DIR}/include>
# )

# target_link_libraries(
#     nvjpeg_compressor_dynamic PUBLIC
#     ${OPENCV_LIB} ${NVJPEG_LIB}
# )
