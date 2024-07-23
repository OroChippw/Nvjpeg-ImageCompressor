/*
    Copyright: OroChippw
    Author: OroChippw
    Date: 2024.07.17
    Description: Head file of the external wrapper class NvjpegCompressRunner
*/
#ifndef IMAGECOMPRESSOR_H_
#define IMAGECOMPRESSOR_H_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#ifdef NVJPEG_COMPRESS_RUNNER_EXPORTS
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllexport)
#else
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllimport)
#endif

class NvjpegCompressRunnerImpl;

class NVJPEG_COMPRESS_RUNNER_API NvjpegCompressRunner{
private:
    NvjpegCompressRunnerImpl* compressor;

public:
    NvjpegCompressRunner(int width=8320 , int height=40000 , int quality=95 , bool optimize=true);
    ~NvjpegCompressRunner();

    // Disable copy constructor and copy assignment operator
    NvjpegCompressRunner(const NvjpegCompressRunner&) = delete;
    NvjpegCompressRunner& operator=(const NvjpegCompressRunner&) = delete;

    std::vector<unsigned char> compress(cv::Mat image , int* run_state);
    cv::Mat decode(std::string image_path , int* run_state);
    void save(std::string save_path , std::vector<unsigned char> obuffer);

    void buildCompressEnv();
    void buildDecodeEnv();
    void deleteCompressEnv();
    void deleteDecodeEnv();
};

#endif // IMAGECOMPRESSOR_H_