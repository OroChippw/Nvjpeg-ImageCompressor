#include <iostream>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#include "../ImageCompressorDll/ImageCompressor.h"

int main(int argc , char* argv[])
{
    int image_width = 8320;
    int image_height= 40000;
    int encode_quality = 95; // default value : 95
    bool use_optimizedHuffman = true; // default value : true

    /* Init compressor properties */ 
    NvjpegCompressRunner* compressor = new NvjpegCompressRunner();
    /*
    NvjpegCompressRunner* compressor = new NvjpegCompressRunner( 
        image_width , image_height , encode_quality , use_optimizedHuffman);
    */
   compressor->buildCompressEnv();
    
    /* Compress Samples */
    std::string input_imagePath1 = "E:\\OroChiLab\\NvjpegSynchronize\\data\\9-3.png";
    cv::Mat image1 = cv::imread(input_imagePath1 , cv::IMREAD_COLOR);
    std::vector<unsigned char> obuffer1 = compressor->compress(image1);
    std::string input_imagePath2 = "E:\\OroChiLab\\NvjpegSynchronize\\data\\5-1.png";
    cv::Mat image2 = cv::imread(input_imagePath2 , cv::IMREAD_COLOR);
    std::vector<unsigned char> obuffer2 = compressor->compress(image2);

   compressor->deleteCompressEnv();

    // std::string input_imagePath3 = "E:\\OroChiLab\\NvjpegSynchronize\\data\\1-2.png";
    // cv::Mat image3 = cv::imread(input_imagePath3 , cv::IMREAD_COLOR);
    // std::vector<unsigned char> obuffer3 = compressor->compress(image3);
    
    compressor->buildDecodeEnv();

    /* Save Samples */
    std::string save_mat1 = "E:\\OroChiLab\\NvjpegSynchronize\\data\\9-3.jpeg";
    compressor->save(save_mat1 , obuffer1);

    std::string save_mat2 = "E:\\OroChiLab\\NvjpegSynchronize\\data\\5-1.jpeg";
    compressor->save(save_mat2 , obuffer2);

    // std::string save_mat3 = "E:\\OroChiLab\\NvjpegSynchronize\\data\\1-2.jpeg";
    // compressor->save(save_mat3 , obuffer3);

    /* Decode From cv::Mat Samples */
    cv::Mat decode_mat_result = compressor->decode(save_mat1); // 传入的是地址
    std::string decode_save_mat = "E:\\OroChiLab\\NvjpegSynchronize\\data\\9-3_decode.png";
    cv::imwrite(decode_save_mat , decode_mat_result);
    cv::Mat decode_mat_result2 = compressor->decode(save_mat2); // 传入的是地址
    std::string decode_save_mat2 = "E:\\OroChiLab\\NvjpegSynchronize\\data\\5-1_decode.png";
    cv::imwrite(decode_save_mat2 , decode_mat_result2);

    compressor->deleteDecodeEnv();

    delete compressor;

    return EXIT_SUCCESS;
}