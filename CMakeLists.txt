cmake_minimum_required(VERSION 3.0)
project(NvjpegImageCompressor)

# 增加标准std=C++11
set(CMAKE_CXX_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# => 制作静态库lib
# 设置动态库/静态库生成路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(nvjpeg_image_compressor STATIC ${SRC_LIST})

# => 制作动态库dll
add_library(nvjpeg_image_compressor SHARED ${SRC_LIST})
