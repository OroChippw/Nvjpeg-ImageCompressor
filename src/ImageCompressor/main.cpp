#include <iostream>
#include <opencv2/core.hpp>

#include "../ImageCompressorDll/CompressConfig.h"
#include "../ImageCompressorDll/ImageCompressor.h"

int main()
{
    std::string inputFilePath = "..//data//org";
    std::string outputFilePath = "..//data//result";

    CompressConfiguration cfg;
    cfg.input_dir = inputFilePath;
    cfg.output_dir = outputFilePath;
    cfg.encode_quality = 95;
    cfg.use_optimizedHuffman = true;
    cfg.multi_stage = true;
    cfg.show_diff_info = false;

    NvjpegCompressRunner* compressor = new NvjpegCompressRunner();
    compressor->compress(cfg);
    
    return 0;
}
