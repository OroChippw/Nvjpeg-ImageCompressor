/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.08
    Description: The specific implementation 
of NvjpegCompressRunnerImpl each module
*********************************/
#pragma once
#pragma warning (disable:4819)

#include <iostream>
#include <opencv2/core.hpp>
#include <nvjpeg.h>

#include "dirent.h"

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

class ReconstructException : public std::exception {
    public:
        const char* what() const noexcept override {
            return "ReconstructException";
        }
};

class NvjpegCompressRunnerImpl
{
private:
    cv::Mat compress_image;
    std::vector<std::string> files_list;

    /* Configuration */
    int encode_quality = 95;
    bool use_optimizedHuffman = true;
    bool multi_stage = false;
    bool show_diff_info = false;
    bool save_mat = false;
    bool save_binary = true;
    bool do_val = false;
    bool in_memory = true;
    bool do_crop = false;
    int crop_ratio = 100;

    /* Image Properties */
    int compress_image_width = 8320;
    int compress_image_height = 40000;

    /* Verifier */
    double time_total = 0.0;
    double psnr_val_score; 

    /* Compress Buffer List */
    std::vector<std::vector<unsigned char>> obuffer_lists;
    std::vector<cv::Mat> result_lists;

public:
    NvjpegCompressRunnerImpl() {};
    ~NvjpegCompressRunnerImpl() {};

public:
    std::vector<unsigned char> Compress(cv::Mat image);
    int CompressImage(std::vector<cv::Mat> image_matlist);

    cv::Mat Reconstructed(std::vector<unsigned char> obuffer);
    int ReconstructedImage(std::vector<std::vector<unsigned char>> obuffer_lists);

public:
    void setImageProperties(int width , int height);
    void setEncodeQuality(int quality);
    int getEncodeQuality();
    void setOptimizedHuffman(bool optimize);
    bool isOptimizedHuffman();
    std::vector<std::vector<unsigned char>> getObufferList();
    std::vector<cv::Mat> getResultList();

private:
    /* Read file list and image pre-processing */
    int ReadInput(const std::string input_path);
    std::vector<cv::Mat> CropImage(const cv::Mat Image , int crop_ratio);
    int CalculateGreatestFactor(int m , int n);

    /* Components that perform compression */
    int Compress(std::vector<cv::Mat> image_list);
    std::vector<unsigned char> CompressWorker(const cv::Mat Image);

    /* Calculate the MSE and PSNR to evaluate the image quality */
    double CalculatePSNR(cv::Mat srcImage , cv::Mat compImage);
    /* Calculate the difference map */
    cv::Mat CalculateDiffmap(const cv::Mat srcImage , const std::string compImagePath);
    
    /* Components that perform refactoring */
    int Reconstructed(std::vector<std::vector<unsigned char>> obuffer_lists);
    cv::Mat ReconstructWorker(const std::vector<unsigned char> obuffer);
    
    /* Other */
    bool cmp(const std::string& str1, const std::string& str2);
    cv::Mat addImage(cv::Mat image_1 , cv::Mat image_2);
    cv::Mat Binaryfile2Mat(std::string ImagePath);
};
