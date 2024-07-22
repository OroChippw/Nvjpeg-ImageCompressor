cmake_minimum_required(VERSION 3.5)
project(Nvjpeg-ImageSynchronizeCompressor CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
# Add the unsupported compiler flag for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

#
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/source-charset:utf-8>")

# link third-party library nvjpeg and opencv4
set(NVJPEG_DIR  ${CMAKE_SOURCE_DIR}/third_party/nvjpeg)
include_directories(${NVJPEG_DIR}/include)
link_directories(${NVJPEG_DIR}/lib/x64)
set(NVJPEG_LIB nvjpeg cublas cuda cudadevrt cudart_static)
link_libraries(${NVJPEG_LIB})


set(OPENCV_DIR  ${CMAKE_SOURCE_DIR}/third_party/opencv4)
include_directories(${OPENCV_DIR}/include)
link_directories("${OPENCV_DIR}/lib")
# set(OPENCV_LIB opencv_world470d)
set(OPENCV_LIB opencv_world470)

include_directories("src/ImageCompressorDll")

# compile own file
file(GLOB SRC_LIST 
    ${CMAKE_SOURCE_DIR}/src/ImageCompressor/*.cpp
    ${CMAKE_SOURCE_DIR}/src/ImageCompressorDll/*.cpp
    ${CMAKE_SOURCE_DIR}/src/ImageCompressorDll/*.cu
    ${CMAKE_SOURCE_DIR}/src/ImageCompressorDll/*.cuh
)
add_executable(demo ${SRC_LIST})

target_link_libraries(demo ${OPENCV_LIB} ${NVJPEG_LIB})

# Set dynamic library/static library generation path
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# build static library [*.lib/*.a]
add_library(nvjpeg_compressor_static ${SRC_LIST})

set_target_properties(
    nvjpeg_compressor_static PROPERTIES 
    OUTPUT_NAME "nvjpeg_compressor"
    VERSION 1.0 SOVERSION 1 
    DEFINE_SYMBOL NVJPEG_COMPRESS_RUNNER_EXPORTS
)

target_include_directories(
    nvjpeg_compressor_static PUBLIC
    $<BUILD_INTERFACE:${OPENCV_DIR}/include>
    $<BUILD_INTERFACE:${NVJPEG_DIR}/include>
)

target_link_libraries(
    nvjpeg_compressor_static PUBLIC
    ${OPENCV_LIB} ${NVJPEG_LIB}
)

# build dynamic library [*.dll/*.so]
add_library(nvjpeg_compressor_dynamic SHARED ${SRC_LIST})

set_target_properties(
    nvjpeg_compressor_dynamic PROPERTIES 
    OUTPUT_NAME "nvjpeg_compressor"
    VERSION 1.0 SOVERSION 1 # dll version and api version
    DEFINE_SYMBOL NVJPEG_COMPRESS_RUNNER_EXPORTS
)

target_include_directories(
    nvjpeg_compressor_dynamic PUBLIC
    $<BUILD_INTERFACE:${OPENCV_DIR}/include>
    $<BUILD_INTERFACE:${NVJPEG_DIR}/include>
)

target_link_libraries(
    nvjpeg_compressor_dynamic PUBLIC
    ${OPENCV_LIB} ${NVJPEG_LIB}
)

