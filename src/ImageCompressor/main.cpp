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
    int encode_quality = 95; // default value : true
    bool use_optimizedHuffman = true; // default value : true

    /* Init compressor properties */ 
    NvjpegCompressRunner* compressor = new NvjpegCompressRunner();
    compressor->init(encode_quality , use_optimizedHuffman);
    
    /* ************************** */
    /* Single image Samples */
    /* ************************** */
    /* Compress Samples */
    std::string input_imagePath = "${your compress file}.png";
    cv::Mat image = cv::imread(input_imagePath , cv::IMREAD_COLOR);
    std::vector<unsigned char> obuffer = compressor->compress(image);

    /* Save Samples */
    std::string save_mat = "${your compress result save path}.jpeg";
    compressor->save(save_mat , obuffer);

    /* Reconstruct From obuffer Samples */
    // cv::Mat reconstructResult = compressor->reconstruct(obuffer);

    // std::string reconstructSavePath = "D:\\OroChiLab\\Nvjpeg-ImageCompressor\\data\\reconstruct_result\\reconstruct.png";
    // cv::imwrite(reconstructSavePath , reconstructResult);

    /* Decode From obuffer Samples */
    cv::Mat decodeResult = compressor->decode(obuffer);
    std::string decodeSavePath = "${your decode result save path}.png";
    cv::imwrite(decodeSavePath , decodeResult);

    /* Decode From cv::Mat Samples */
    cv::Mat decode_mat_result = compressor->decode(save_mat); // 传入的是地址
    std::string decode_save_mat = "${your decode result save path}.png";
    cv::imwrite(decode_save_mat , decode_mat_result);

    /* ************************** */
    /* ** Batch image Samples *** */
    /* ************************** */

    // std::string input_dirPath = "D:\\OroChiLab\\Nvjpeg-ImageCompressor\\data\\test\\4K";

    // std::vector<cv::Mat> image_matlist;
    // std::vector<cv::String> image_filelist;
    // cv::glob(input_dirPath , image_filelist);

    // for (const auto& file : image_filelist)
    // {
    //     std::cout << "File : " << file << std::endl;
    //     image_matlist.emplace_back(cv::imread(file , cv::IMREAD_COLOR));
    // }
    // std::cout << "Image list size : " << image_matlist.size() << std::endl;


    // /* Compress Samples */
    // std::vector<std::vector<unsigned char>> obuffer_lists = compressor->compress(image_matlist);
    // std::cout << "Compress buffer list size : " << obuffer_lists.size() << std::endl;

    // /* Reconstruct Samples */
    // std::vector<cv::Mat> reconstructImageList = compressor->reconstruct(obuffer_lists);
    // std::cout << "Reconstruct image list size : " << reconstructImageList.size() << std::endl;

    // std::string ReconstructedFilePath = "D:\\OroChiLab\\Nvjpeg-ImageCompressor\\data\\reconstruct_result";
    // for (unsigned int index = 0 ; index < reconstructImageList.size() ; index++)
    // {
    //     std::string savePath = ReconstructedFilePath + "//" + std::to_string(index) + ".png";
    //     cv::imwrite(savePath , reconstructImageList[index]);
    // }

    return EXIT_SUCCESS;
}
