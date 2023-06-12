#pragma once
#include <iostream>
#include "CompressConfig.h"

#ifdef NVJPEG_COMPRESS_RUNNER_EXPORTS
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllexport)
#else
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllimport)
#endif


#pragma warning(push)

class NvjpegCompressRunnerImpl;

class NVJPEG_COMPRESS_RUNNER_API NvjpegCompressRunner{
private:
    NvjpegCompressRunnerImpl* compressor;

public:
    NvjpegCompressRunner();
    ~NvjpegCompressRunner();
    void compress(CompressConfiguration cfg);
    void reconstruct(CompressConfiguration cfg , std::string ImageDirPath);
};

#pragma warning(pop)