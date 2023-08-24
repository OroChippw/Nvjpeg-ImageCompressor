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
#include <cuda_runtime_api.h>

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
    double compress_time_total = 0.0;
    double decode_time_total = 0.0;
    double psnr_val_score; 

    /* Compress Buffer List */
    std::vector<std::vector<unsigned char>> obuffer_lists;
    std::vector<cv::Mat> reconstruct_result_lists;
    std::vector<cv::Mat> decode_result_lists;


public:
    NvjpegCompressRunnerImpl() {};
    ~NvjpegCompressRunnerImpl() {};

public:
    std::vector<unsigned char> Compress(cv::Mat image);
    int CompressImage(std::vector<cv::Mat> image_matlist);

    cv::Mat Reconstructed(std::vector<unsigned char> obuffer);
    int ReconstructedImage(std::vector<std::vector<unsigned char>> obuffer_lists);

    cv::Mat Decode(std::vector<unsigned char> obuffer);
    int DecodeImage(std::vector<std::vector<unsigned char>> obuffer_lists);

public:
    void setImageProperties(int width , int height);
    void setEncodeQuality(int quality);
    int getEncodeQuality();
    void setOptimizedHuffman(bool optimize);
    bool isOptimizedHuffman();
    std::vector<std::vector<unsigned char>> getObufferList();
    std::vector<cv::Mat> getReconstructResultList();
    std::vector<cv::Mat> getDecodeResultList();


private:
    /* Image pre-processing */
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
    /* ******* Use opencv imdecode to decode directly ******* */
    int Reconstructed(std::vector<std::vector<unsigned char>> obuffer_lists);
    cv::Mat ReconstructWorker(const std::vector<unsigned char> obuffer);
    /* ******* Use NvjpegDecoder to decode ******* */
    // int dev_malloc(void **p, size_t s);
    // int dev_free(void *p);
    // int host_malloc(void** p, size_t s, unsigned int f);
    // int host_free(void* p);
    int Decode(std::vector<std::vector<unsigned char>> obuffer_lists);
    cv::Mat DecodeWorker(const std::vector<unsigned char> obuffer);

    /* Other */
    bool cmp(const std::string& str1, const std::string& str2);
    cv::Mat addImage(cv::Mat image_1 , cv::Mat image_2);
    cv::Mat Binaryfile2Mat(std::string ImagePath);
    // std::string nvjpegStatusToString(nvjpegStatus_t status);
    
    int writeBMP(const char *filename, const unsigned char *d_chanR, int pitchR,
                const unsigned char *d_chanG, int pitchG,
                const unsigned char *d_chanB, int pitchB, int width, int height);
    
    int writeBMPi(const char *filename, const unsigned char *d_RGB, int pitch,
                int width, int height);
    
    cv::Mat getCVImage(const unsigned char *d_chanB, int pitchB, \
                const unsigned char *d_chanG, int pitchG, \
                const unsigned char *d_chanR, int pitchR, \
                int width, int height);

};
