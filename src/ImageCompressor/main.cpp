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

    cfg.width = 8432;
    cfg.height = 40000;
    
    cfg.encode_quality = 95;
    cfg.use_optimizedHuffman = true;
    cfg.multi_stage = false;
    cfg.show_diff_info = false;
    cfg.save_mat = true;
    cfg.save_binary = true;
    cfg.do_val = false;

    cfg.do_crop = true;
    cfg.crop_ratio = 2;
    

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

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        std::cout << "The current PC does not have a graphics card hardware device that supports CUDA " << std::endl;
    }

    size_t gpu_total_size;
    size_t gpu_free_size;

    cudaError_t cuda_status = cudaMemGetInfo(&gpu_free_size , & gpu_total_size);

    if (cuda_status != cudaSuccess)
    {
        std::cout << "Error: cudaMemGetInfo fails : " << cudaGetErrorString(cuda_status) << std::endl;
    }

    double total_memory = double(gpu_total_size) / (1024.0 * 1024.0);
    double free_memory = double(gpu_total_size) / (1024.0 * 1024.0);
    double use_memory = total_memory - free_memory;    

    std::cout << "The total video memory of the current graphics card :" << total_memory << "m" << std::endl;
    std::cout << "Memory have used : " << use_memory << "m" << std::endl;
    std::cout << "Remaining memory : " << free_memory << "m" << std::endl;


    /* Compress Samples */
    NvjpegCompressRunner* compressor = new NvjpegCompressRunner();
    compressor->compress(cfg);

    /* Reconstruct Samples */
    // std::string path1 = "..//data//compress_result//2-2//B.bin";
    // std::string path2 = "..//data//compress_result//2-2//D.bin";

    // compressor->reconstruct(cfg , path1 , path2);
    

    return EXIT_SUCCESS;
}
