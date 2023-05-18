#include <iostream>
#include <opencv2/core.hpp>

#include "../ImageCompressorDll/CompressConfig.h"
#include "../ImageCompressorDll/ImageCompressor.h"

int main()
{
    std::string inputFilePath = "..//data//org";
    std::string CompressOutputFilePath = "..//data//compress_result";
    std::string ReconstructedFilePath = "..//data//reconstruct_result";


    CompressConfiguration cfg;
    cfg.input_dir = inputFilePath;
    cfg.output_dir = CompressOutputFilePath;
    cfg.rebuild_dir = ReconstructedFilePath;
    cfg.encode_quality = 95;
    cfg.use_optimizedHuffman = true;
    cfg.multi_stage = true;
    cfg.show_diff_info = false;
    cfg.save_mat = false;
    cfg.save_binary = true;
    cfg.do_val = false;

    if (!(cfg.save_mat || cfg.save_binary))
    {
        std::cout << "Choice save as mat or binary file." << std::endl;
        return EXIT_FAILURE;
    }
    /* Compress Samples */
    NvjpegCompressRunner* compressor = new NvjpegCompressRunner();
    compressor->compress(cfg);
    
    /* Reconstruct Samples */
    std::string path1 = "..//data//compress_result//2-2//B.bin";
    std::string path2 = "..//data//compress_result//2-2//D.bin";

    compressor->reconstruct(cfg , path1 , path2);
    

    return EXIT_SUCCESS;
}
