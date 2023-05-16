cmake_minimum_required(VERSION 3.0)
project(NvjpegImageCompressor)

# 增加标准std=C++11
set(CMAKE_CXX_STANDARD 11)

# 链接第三方库
# nvjpeg
set(NVJPEG_DIR  ${CMAKE_SOURCE_DIR}/third_party/nvjpeg)
include_directories(${NVJPEG_DIR}/include)
link_directories(${NVJPEG_DIR}/lib)
set(NVJPEG_LIB nvjpeg cublas cuda cudadevert cudart cudart_static cufft cufftw curand cusolver cusparse)

# opencv4
set(OPENCV_DIR  ${CMAKE_SOURCE_DIR}/third_party/opencv4)
include_directories(${OPENCV_DIR}/include)
link_directories("${OPENCV_DIR}/lib")
set(OPENCV_LIB opencv_world470d)


# 包含静态库路径
# link_directories(${PROJECT_SOURCE_DIR}/lib)
# link_libraries(NvjpegImageCompressor)

add_executable(demo src/ImageCompressor/main.cpp)
target_link_libraries(demo ${OPENCV_LIB})
# 指定要链接的动态库
# target_link_libraries(opencv_world470 opencv_world470d demo)
# file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# # => 制作静态库lib
# # 设置动态库/静态库生成路径
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# add_library(nvjpeg_image_compressor STATIC ${SRC_LIST})

# # => 制作动态库dll
# add_library(nvjpeg_image_compressor SHARED ${SRC_LIST})