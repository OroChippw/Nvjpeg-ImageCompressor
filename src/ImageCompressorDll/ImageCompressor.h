/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.08
    Description: Define each module of NvjpegCompressRunner
*********************************/
#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


#ifdef NVJPEG_COMPRESS_RUNNER_EXPORTS
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllexport)
#else
#define NVJPEG_COMPRESS_RUNNER_API __declspec(dllimport)
#endif

#pragma warning(push)

class NvjpegCompressRunnerImpl;

class NVJPEG_COMPRESS_RUNNER_API NvjpegCompressRunner{
private:
    NvjpegCompressRunnerImpl* compressor;

public:
    NvjpegCompressRunner();
    ~NvjpegCompressRunner();
    void init(int quality , bool optimize);

    std::vector<unsigned char> compress(cv::Mat image);
    std::vector<std::vector<unsigned char>> compress(std::vector<cv::Mat> image_matlist);

    cv::Mat reconstruct(std::vector<unsigned char> obuffer);
    std::vector<cv::Mat> reconstruct(std::vector<std::vector<unsigned char>> obuffer_lists);
};

#pragma warning(pop)