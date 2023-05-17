#include <iostream>
#include <opencv2/core.hpp>

#include "../ImageCompressorDll/CompressConfig.h"
#include "../ImageCompressorDll/ImageCompressor.h"

int main()
{
    std::string inputFilePath = "";
    std::string outputFilePath = "";

    CompressConfiguration cfg;
    cfg.input_dir = inputFilePath;
    cfg.output_dir = outputFilePath;
    cfg.encode_quality = 95;

    NvjpegCompressRunner* compressor = new NvjpegCompressRunner();
    // compressor->compress(cfg);
    

    

    std::cout << "hello world" << std::endl;
    return 0;
}