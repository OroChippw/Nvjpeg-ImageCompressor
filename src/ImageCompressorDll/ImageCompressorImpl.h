#pragma once
#pragma warning (disable:4819)

#include <iostream>
#include <opencv2/core.hpp>
#include <nvjpeg.h>

#include "dirent.h"
#include "CompressConfig.h"

#define CHECK_CUDA(call)                                                    \
{                                                                           \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess)                                                  \
    {                                                                       \
        std::cout << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                            \
    }                                                                       \
}

#define CHECK_NVJPEG(call)                                                  \
{                                                                           \
    nvjpegStatus_t _e = (call);                                             \
    if (_e != NVJPEG_STATUS_SUCCESS)                                        \
    {                                                                       \
        std::cout << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                            \
    }                                                                       \
}

class NvjpegCompressRunnerImpl
{
private:
    cv::Mat compress_image;
    std::vector<std::string> files_list;

    double time_total;
    double psnr_val_score; 

private:
    /* 通过和原图对比计算均方误差和峰值信噪比以评估图像质量 */
    double CalculatePSNR(cv::Mat srcImage , cv::Mat compImage);
    void CalculateGrayAvgStdDev(cv::Mat&src , double& avg , double &stddev);
    std::vector<unsigned char> CompressWorker(CompressConfiguration cfg , const cv::Mat Image);
    std::vector<cv::Mat> CropImage(const cv::Mat Image , int crop_ratio);

public:
    NvjpegCompressRunnerImpl() {};
    ~NvjpegCompressRunnerImpl() {};

public:
    int ReadInput(const std::string input_path);
    int Compress(CompressConfiguration cfg);
    /* 计算原图和压缩图的差异图 */
    cv::Mat CalculateDiffmap(CompressConfiguration cfg , const cv::Mat srcImage , const std::string compImagePath);
    /* 通过压缩图还原原图 */
    cv::Mat Reconstructed(cv::Mat Image1 , cv::Mat Image2);
    cv::Mat Binaryfile2Mat(CompressConfiguration cfg , const std::string ImagePath);

public:
    int CompressImage(CompressConfiguration cfg);
    double CalculateDiffImagePSNR(const cv::Mat image1 , const std::string ImagePath2);
    int ReconstructedImage(CompressConfiguration cfg , std::string ImagePath1 , std::string ImagePath2);

};
