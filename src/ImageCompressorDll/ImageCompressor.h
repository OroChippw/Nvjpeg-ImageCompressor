#pragma once
#include <iostream>
#include "CompressConfig.h"

#ifdef NVJPEGCOMPRESSRUNNER_EXPORTS
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllexport)
#else
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllimport)
#endif


#pragma warning(push)

class NvjpegCompressRunnerImpl;

class NvjpegCompressRunner{
private:
    NvjpegCompressRunnerImpl* compressor;

public:
    NvjpegCompressRunner();
    ~NvjpegCompressRunner();
    void compress(CompressConfiguration cfg);
};

#pragma warning(pop)