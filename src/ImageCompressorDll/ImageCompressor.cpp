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
    std::cout << "[INFO] Build NvjpegCompressRunnerImpl Successfully ..." << std::endl;
}

NvjpegCompressRunner::~NvjpegCompressRunner()
{
    delete compressor;
    std::cout << "[INFO] Delete NvjpegCompressRunnerImpl Successfully ..." << std::endl;
}

void NvjpegCompressRunner::init(int quality=95 , bool optimize=true)
{
    compressor->setEncodeQuality(quality);
    // compressor->setImageProperties(width , height);
    compressor->setOptimizedHuffman(optimize);
    std::cout << "[INFO] Initial NvjpegCompressRunnerImpl Properties Successfully ..." << std::endl;
}

std::vector<std::vector<unsigned char>> NvjpegCompressRunner::compress(std::vector<cv::Mat> image_matlist)
{
    auto startTime = std::chrono::steady_clock::now();
    std::string run_state = compressor->CompressImage(image_matlist) ? "Failure" : "Finish";
    auto endTime = std::chrono::steady_clock::now();
    
    std::cout << "[INFO] Compress Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Compress Func Cost Time : " << elapsedTime << " ms" << std::endl;

    return compressor->getObufferList();
}

std::vector<unsigned char> NvjpegCompressRunner::compress(cv::Mat image)
{
    auto startTime = std::chrono::steady_clock::now();
    auto obuffer = compressor->Compress(image);
    auto endTime = std::chrono::steady_clock::now();

    std::string run_state = obuffer.empty() ? "Failure" : "Finish";
    std::cout << "[INFO] Compress Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Compress Func Cost Time : " << elapsedTime << " ms" << std::endl;
    
    return obuffer;
}

cv::Mat NvjpegCompressRunner::reconstruct(std::vector<unsigned char> obuffer)
{
    auto startTime = std::chrono::steady_clock::now();
    cv::Mat image = compressor->Reconstructed(obuffer);
    auto endTime = std::chrono::steady_clock::now();

    std::string run_state = obuffer.empty() ? "Failure" : "Finish";
    std::cout << "[INFO] Reconstruct Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Reconstruct Func Cost Time : " << elapsedTime << " ms" << std::endl;
    
    return image;
}

std::vector<cv::Mat> NvjpegCompressRunner::reconstruct(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    auto startTime = std::chrono::steady_clock::now();
    std::string run_state = compressor->ReconstructedImage(obuffer_lists) ? "Failure" : "Finish";
    auto endTime = std::chrono::steady_clock::now();
    
    std::cout << "[INFO] Reconstructed Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Reconstruct Func Cost Time : " << elapsedTime << " ms" << std::endl;

    return compressor->getResultList();
}

cv::Mat NvjpegCompressRunner::decode(std::vector<unsigned char> obuffer)
{
    auto startTime = std::chrono::steady_clock::now();
    cv::Mat image = compressor->Decode(obuffer);
    auto endTime = std::chrono::steady_clock::now();

    std::string run_state = image.empty() ? "Failure" : "Finish";
    std::cout << "[INFO] Reconstruct Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Decode Func Cost Time : " << elapsedTime << " ms" << std::endl;

    return image;
}

std::vector<cv::Mat> NvjpegCompressRunner::decode(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    auto startTime = std::chrono::steady_clock::now();

    auto endTime = std::chrono::steady_clock::now();

    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Decode Func Cost Time : " << elapsedTime << " ms" << std::endl;
    
    return compressor->getResultList();
}