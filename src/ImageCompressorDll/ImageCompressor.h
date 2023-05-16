#pragma once

#include <iostream>
#include "CompressConfig.h"

#ifdef NVJPEGCOMPRESSRUNNER_EXPORTS
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllexport)
#else
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllimport)
#endif

class NvjpegCompressRunnerImpl

class NVJPEG_COMPRESS_RUNNER_API NvjpegCompressRunner{
private:
    NvjpegCompressRunnerImpl* compressor;

public:
    NvjpegCompressRunner();
    ~NvjpegCompressRunner();
    void compress(Configuration cfg);
};

#pragma warning(pop)