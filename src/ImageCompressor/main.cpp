/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.08
    Description: Call Nvjpeg image compression demonstration
*********************************/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "../ImageCompressorDll/ImageCompressor.h"

int main(int argc , char* argv[])
{
    std::string input_dirPath = "D:\\OroChiLab\\Nvjpeg-ImageCompressor\\data\\test";
    // std::string CompressOutputFilePath = "..//data//compress_result";
    // std::string reconstruct_dirPath = "..//data//compress_result//9-3";

    int width = 8320; // require value
    int height = 40000; // required value
    int encode_quality = 95; // default value : true
    bool use_optimizedHuffman = true; // default value : true
    
    std::vector<cv::Mat> image_matlist;
    std::vector<cv::String> image_filelist;
    cv::glob(input_dirPath , image_filelist);

    for (const auto& file : image_filelist)
    {
        std::cout << "File : " << file << std::endl;
        image_matlist.emplace_back(cv::imread(file , cv::IMREAD_COLOR));
    }
    std::cout << "Image list size : " << image_matlist.size() << std::endl;

    /* Init compressor properties */ 
    NvjpegCompressRunner* compressor = new NvjpegCompressRunner();
    compressor->init(width , height , encode_quality , use_optimizedHuffman);

    /* Compress Samples */
    std::vector<std::vector<unsigned char>> obuffer_lists = compressor->compress(image_matlist);
    std::cout << "Compress buffer list size : " << obuffer_lists.size() << std::endl;

    /* Reconstruct Samples */
    std::vector<cv::Mat> reconstructImageList = compressor->reconstruct(obuffer_lists);
    std::cout << "Reconstruct image list size : " << reconstructImageList.size() << std::endl;

    std::string ReconstructedFilePath = "D:\\OroChiLab\\Nvjpeg-ImageCompressor\\data\\reconstruct_result";
    for (unsigned int index = 0 ; index < reconstructImageList.size() ; index++)
    {
        std::string savePath = ReconstructedFilePath + "//" + std::to_string(index) + ".png";
        cv::imwrite(savePath , reconstructImageList[index]);
    }

    return EXIT_SUCCESS;
}
