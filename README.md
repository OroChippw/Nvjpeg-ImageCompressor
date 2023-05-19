# Nvjpeg-ImageCompressor
## Introduction
Nvjpeg ImageCompressor is to create a C++ interface that enables direct invocation of nvJPEG libraries.The nvjpeg library is a high-performance GPU-accelerated library for decoding, encoding and transcoding JPEG format images. This repository is used to record the ablation experiment data that use nvjpeg for image compression.The code repository contains a C++ library with a test program to facilitate easy integration of the interface into other projects.

Currently, the interface only supports GPU execution.The specific experimental data and equipment used are shown below. And the inferface is only supported on Windows and may encounter issues when running on Linux.

## Quick Start

### Requirements
``` 
# nvjpeg 3rdparty
the nvjpeg.h usually placed in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\include . You can create a folder called nvjpeg and copy include and lib into it
# opencv 3rdparty
This repository use opencv 4.x
# CXX_STANDARD 17
```

### How to build and run
```
# Open the terminal, enter the source code directory where CMakeLists.txt is located, and create a new build folder
mkdir build
# enter the build folder and run CMake to configure the project
cd build
cmake ..
# Use the build system to compile/link this project
cmake --build .
# If the specified compilation mode is debug or release, it is as follows
# cmake --build . --config Debug
# cmkae --build . --config Release
```

## Ablation Experiment Record
### Different SamplingFactors
| SamplingFactors | meanCostTime | meanPSNR | meanPSNR | compression Ratio |
| :--------------:| :------------: | :---------: | :--------: | :-------: |
| NVJPEG_CSS_444 | 266.32 | 32.31 | 0.28 |
| NVJPEG_CSS_422 | 266.32 | 32.31 | 0.28 |
| NVJPEG_CSS_440 | 266.32 | 32.31 | 0.28 |
| NVJPEG_CSS_420 | 266.32 | 32.31 | 0.28 |
| NVJPEG_CSS_411 | 266.32 | 32.31 | 0.28 |

### Different Encode quality

### use_optimizedHuffman

### License
