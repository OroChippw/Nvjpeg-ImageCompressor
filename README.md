# Nvjpeg-ImageCompressor
## Introduction
Nvjpeg ImageCompressor is to create a C++ interface that enables direct invocation of nvJPEG libraries.The nvjpeg library is a high-performance GPU-accelerated library for decoding, encoding and transcoding JPEG format images. This repository is used to record the ablation experiment data that use nvjpeg for image compression.The code repository contains a C++ library with a test program to facilitate easy integration of the interface into other projects.

Currently, the interface only supports GPU execution.The specific experimental data and equipment used are shown below. And the inferface is only supported on Windows and may encounter issues when running on Linux.

## Feature👏👋
* **Support secondary compression.** Support to calculate difference map and perform secondary compression on difference map

## Development Enviroments
>  - Windows 10 Professional 
>  - CUDA v11.3
>  - cmake version 3.26.2

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
# Enter the source code directory where CMakeLists.txt is located, and create a new build folder
mkdir build
# Enter the build folder and run CMake to configure the project
cd build
cmake ..
# Use the build system to compile/link this project
cmake --build .
# If the specified compilation mode is debug or release, it is as follows
# cmake --build . --config Debug
# cmkae --build . --config Release
```

## Ablation Experiment Record
Environment Device : i5-13600KF + NVIDIA GeForce RTX 3060（12GB）
Input Image : [sample A] : 1.png 525MB ; [sample B] : 6.bmp 952MB ; [sample C] : 11.png 583MB
all image resolution is 8320*40000
### ① Different SamplingFactors
| SamplingFactors | meanCostTime(ms) | meanPSNR(dB) | compression Ratio(%) |
| :--------------:| :------------: | :------------: | :----------------: |
| NVJPEG_CSS_444 | 266.32 | 32.31 | A : 28 ; B : 15 ; C : 33 |
| NVJPEG_CSS_422 :star2: | 201.45 | 30.33 | A : 19 ; B : 11 ; C : 23 |
| NVJPEG_CSS_440 | 201.92 | 30.80 | A : 19 ; B : 11 ; C : 23 |
| NVJPEG_CSS_420 | 165.18 | 30.20 | A : 16 ; B : 9.5 ; C : 19 |
| NVJPEG_CSS_411 | 164.11 | 29.98 | A : 16 ; B : 9.5 ; C : 19 |

### ② Different Encode quality
| EncodeQuality | meanCostTime(ms) | meanPSNR(dB) | compression Ratio(%) |
| :--------------:| :------------: | :------------: | :----------------: |
| 100 | 266.32 | 32.31 | A : 56 ; B : 31 ; C : 58 |
| 95 :star2: | 266.32 | 32.31 | A : 19 ; B : 11 ; C : 23 |
| 85 | 266.32 | 32.31 | A : 6.9 ; B : 4.7 ; C : 10 |
| 75 | 266.32 | 32.31 | A : 4 ; B : 3.1 ; C : 6.6 |

### ③ use_optimizedHuffman
| use_optimizedHuffman | meanCostTime(ms) | meanPSNR(dB) | compression Ratio(%) |
| :--------------:| :------------: | :------------: | :----------------: |
| ✔ :star2: | 201.45 | 30.33 | A : 19 ; B : 11 ; C : 23 |
| / | 128.05 | 29.90 | A : 10 ; B : 6 ; C : 13 |

### License
