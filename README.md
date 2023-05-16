# Nvjpeg-ImageCompressor

## Quick Start
### How to build and run
```
# 打开终端，进入CMakeLists.txt所在的源代码目录，新建一个build文件夹
mkdir build
# 进入build文件夹，运行CMake来配置项目
cd build
cmake ..
# 调用build构建系统来编译/链接这个项目
cmake --build .
# 若指定编译模式为debug或者release，则如下
# cmake --build . --config Debug
# cmkae --build . --config Release
```

## Experiment Record
### Different SamplingFactors
| SamplingFactors | meanCostTime | meanPSNR | meanPSNR | compression Ratio |
| :----:| :----: | :----: | :----: | :----: |
| NVJPEG_CSS_444 | 266.32 | 32.31 | 0.28 |
| NVJPEG_CSS_422 | 266.32 | 32.31 | 0.28 |
| NVJPEG_CSS_440 | 266.32 | 32.31 | 0.28 |
| NVJPEG_CSS_420 | 266.32 | 32.31 | 0.28 |
| NVJPEG_CSS_411 | 266.32 | 32.31 | 0.28 |

### Different Encode quality

### use_optimizedHuffman
