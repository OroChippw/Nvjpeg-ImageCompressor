#include <iostream>

#include "ImageCompressor.h"
#include "ImageCompressorImpl.h"

NvjpegCompressRunner::NvjpegCompressRunner()
{
    
    compressor = new NvjpegCompressRunnerImpl();
    std::cout << "=> Build NvjpegCompressRunnerImpl successfully ..." << std::endl;
}

NvjpegCompressRunner::~NvjpegCompressRunner()
{
    delete compressor;
    std::cout << "=> Delete NvjpegCompressRunnerImpl successfully ..." << std::endl;
}

void NvjpegCompressRunner::compress(CompressConfiguration cfg)
{
    compressor->CompressSingleImage(cfg);
}