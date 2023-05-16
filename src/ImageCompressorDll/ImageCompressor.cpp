#include "ImageCompressor.h"
#include "ImageCompressorImpl.h"

NvjpegCompressRunner::NvjpegCompressRunner()
{
    compressor = new NvjpegCompressRunnerImpl();
}

NvjpegCompressRunner::~NvjpegCompressRunner()
{
    delete compressor;
}

NvjpegCompressRunner::compress(Configuration cfg)
{
    compressor.->CompressSingleImage(cfg);
}