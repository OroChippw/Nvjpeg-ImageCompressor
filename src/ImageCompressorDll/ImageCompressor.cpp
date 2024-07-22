/*
    Copyright: OroChippw
    Author: OroChippw
    Date: 2024.07.17
    Description: Implementation of the external wrapper class NvjpegCompressRunner
*/
#include <iostream>
#include <fstream>
#include <chrono>

#include "ImageCompressor.h"
#include "ImageCompressorImpl.cuh"

NvjpegCompressRunner::NvjpegCompressRunner(int width , int height , int quality , bool optimize)
{
    compressor = new NvjpegCompressRunnerImpl(width , height , quality , optimize);
}

void NvjpegCompressRunner::buildCompressEnv()
{
    compressor->initCompressEnv();
}

void NvjpegCompressRunner::buildDecodeEnv()
{
    compressor->initDecodeEnv();
}

void NvjpegCompressRunner::deleteCompressEnv()
{
    compressor->destoryCompressEnv();
}

void NvjpegCompressRunner::deleteDecodeEnv()
{
    compressor->destoryDecodeEnv();
}

NvjpegCompressRunner::~NvjpegCompressRunner()
{
    delete compressor;
    std::cout << "[INFO] Delete NvjpegCompressRunnerImpl Successfully ..." << std::endl;
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

cv::Mat NvjpegCompressRunner::decode(std::string image_path)
{
    FILE *jpeg_file;
    if (fopen_s(&jpeg_file, image_path.c_str(), "rb") != 0) {
        std::cerr << "Failed to open JPEG file." << std::endl;
        return cv::Mat();
    }
    auto startTime = std::chrono::steady_clock::now();
    cv::Mat result = compressor->Decode(jpeg_file);
    auto endTime = std::chrono::steady_clock::now();

    std::string run_state = result.empty() ? "Failure" : "Finish";
    std::cout << "[INFO] Decode Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Decode Func Cost Time : " << elapsedTime << " ms" << std::endl;

    return result;
}

void NvjpegCompressRunner::save(std::string save_path , std::vector<unsigned char> obuffer)
{
    try 
    {
        std::ofstream outputFile(save_path, std::ios::out | std::ios::binary);
        outputFile.write(reinterpret_cast<const char*>(obuffer.data()), static_cast<int>(obuffer.size()));
        outputFile.close();
    } catch(const std::exception& e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
}