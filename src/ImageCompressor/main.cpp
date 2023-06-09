#include <iostream>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#include "../ImageCompressorDll/CompressConfig.h"
#include "../ImageCompressorDll/ImageCompressor.h"



int main()
{
    std::string inputFilePath = "..//data//test";
    std::string CompressOutputFilePath = "..//data//compress_result";
    std::string ReconstructedFilePath = "..//data//reconstruct_result";

    CompressConfiguration cfg;
    cfg.input_dir = inputFilePath;
    cfg.output_dir = CompressOutputFilePath;
    cfg.rebuild_dir = ReconstructedFilePath;

    cfg.width = 8320;
    cfg.height = 40000;
    
    cfg.encode_quality = 95;
    cfg.use_optimizedHuffman = true;
    cfg.multi_stage = false;
    cfg.show_diff_info = false;
    cfg.save_mat = true;
    cfg.save_binary = true;
    cfg.do_val = false;

    cfg.do_crop = true;
    cfg.crop_ratio = 4;
    
    cfg.use_roi = false;
    if (cfg.use_roi)
    {
        cfg.roi_w = 30;
        cfg.roi_h = 30;
        cfg.roi_rect = cv::Rect( 6648 , 22230 , cfg.roi_w , cfg.roi_h);
    }
    
    if (!(cfg.save_mat || cfg.save_binary))
    {
        std::cout << "Choice save as mat or binary file." << std::endl;
        return EXIT_FAILURE;
    }

    if (cfg.do_crop)
    {
        if (!((cfg.width % cfg.crop_ratio == 0) || (cfg.height % cfg.crop_ratio == 0)))
        {
            std::cout << "The width and height of the image must be divisible by the number of blocks" << std::endl;
            return EXIT_FAILURE;
        }
    }


    /* Init compressor*/    
    NvjpegCompressRunner* compressor = new NvjpegCompressRunner();

    /* Compress Samples */
    compressor->compress(cfg);

    /* Reconstruct Samples */
    std::string reconstruct_path = "..//data//compress_result//9-3";

    compressor->reconstruct(cfg , reconstruct_path);
    

    return EXIT_SUCCESS;
}
