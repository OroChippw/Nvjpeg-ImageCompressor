/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.08
    Description: Implementation of the external wrapper class NvjpegCompressRunner
*********************************/
#include <iostream>
#include <chrono>

#include "ImageCompressor.h"
#include "ImageCompressorImpl.h"

NvjpegCompressRunner::NvjpegCompressRunner()
{
    compressor = new NvjpegCompressRunnerImpl();
    std::cout << "=> Build NvjpegCompressRunnerImpl Successfully ..." << std::endl;
}

NvjpegCompressRunner::~NvjpegCompressRunner()
{
    delete compressor;
    std::cout << "=> Delete NvjpegCompressRunnerImpl Successfully ..." << std::endl;
}

void NvjpegCompressRunner::init(int quality=95 , bool optimize=true)
{
    compressor->setEncodeQuality(quality);
    // compressor->setImageProperties(width , height);
    compressor->setOptimizedHuffman(optimize);
    std::cout << "=> Initial NvjpegCompressRunnerImpl Properties Successfully ..." << std::endl;
}

std::vector<std::vector<unsigned char>> NvjpegCompressRunner::compress(std::vector<cv::Mat> image_matlist)
{
    auto startTime = std::chrono::steady_clock::now();

    std::string run_state = compressor->CompressImage(image_matlist) ? "Failure" : "Finish";

    auto endTime = std::chrono::steady_clock::now();
    
    std::cout << "=> Compress Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "NvjpegCompressRunner Compress Func Cost Time : " << elapsedTime << " ms" << std::endl;

    return compressor->getObufferList();
}

std::vector<unsigned char> NvjpegCompressRunner::compress(cv::Mat image)
{
    auto obuffer = compressor->Compress(image);
    return obuffer;
}


std::vector<cv::Mat> NvjpegCompressRunner::reconstruct(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    auto startTime = std::chrono::steady_clock::now();

    std::string run_state = compressor->ReconstructedImage(obuffer_lists) ? "Failure" : "Finish";

    auto endTime = std::chrono::steady_clock::now();
    
    std::cout << "=> Reconstructed Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "NvjpegCompressRunner Reconstruct Func Cost Time : " << elapsedTime << " ms" << std::endl;

    return compressor->getResultList();
}

cv::Mat NvjpegCompressRunner::reconstruct(std::vector<unsigned char> obuffer)
{
    cv::Mat image = compressor->Reconstructed(obuffer);
    return image;
}