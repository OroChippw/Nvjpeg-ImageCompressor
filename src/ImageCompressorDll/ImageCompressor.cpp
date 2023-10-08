/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.08
    Description: Implementation of the external wrapper class NvjpegCompressRunner
*********************************/
#include <iostream>
#include <fstream>
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

    return compressor->getReconstructResultList();
}

cv::Mat NvjpegCompressRunner::decode(std::vector<unsigned char> obuffer)
{
    auto startTime = std::chrono::steady_clock::now();
    cv::Mat image = compressor->Decode(obuffer);
    auto endTime = std::chrono::steady_clock::now();

    std::string run_state = image.empty() ? "Failure" : "Finish";
    std::cout << "[INFO] Decode Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Decode Func Cost Time : " << elapsedTime << " ms" << std::endl;

    return image;
}

cv::Mat NvjpegCompressRunner::decode(cv::Mat image)
{
    auto startTime = std::chrono::steady_clock::now();
    cv::Mat result = compressor->Decode(image);
    auto endTime = std::chrono::steady_clock::now();

    std::string run_state = image.empty() ? "Failure" : "Finish";
    std::cout << "[INFO] Decode Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Decode Func Cost Time : " << elapsedTime << " ms" << std::endl;

    return result;
}

cv::Mat NvjpegCompressRunner::decode(std::string image_path)
{
    FILE *jpeg_file = fopen(image_path.c_str() , "rb");
    if (!jpeg_file) {
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

std::vector<cv::Mat> NvjpegCompressRunner::decode(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    auto startTime = std::chrono::steady_clock::now();
    std::string run_state = compressor->DecodeImage(obuffer_lists) ? "Failure" : "Finish";
    auto endTime = std::chrono::steady_clock::now();
    
    std::cout << "[INFO] Decode Result : " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] NvjpegCompressRunner Decode Func Cost Time : " << elapsedTime << " ms" << std::endl;
    
    return compressor->getDecodeResultList();
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