#include <iostream>
#include <chrono>

#include "ImageCompressor.h"
#include "ImageCompressorImpl.h"

NvjpegCompressRunner::NvjpegCompressRunner()
{
    compressor = new NvjpegCompressRunnerImpl();
    std::cout << "=> Build NvjpegCompressRunnerImpl successfully ..." << std::endl;
}

NvjpegCompressRunner::~NvjpegCompressRunner()
{
    delete compressor;
    std::cout << "=> Delete NvjpegCompressRunnerImpl successfully ..." << std::endl;
}

void NvjpegCompressRunner::compress(CompressConfiguration cfg)
{
    auto startTime = std::chrono::steady_clock::now();
    std::string run_state = compressor->CompressImage(cfg) ? "Failure" : "Finish";
    auto endTime = std::chrono::steady_clock::now();
    std::cout << "=> Compress " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Compress func cost time: " << elapsedTime << " ms" << std::endl;
}

cv::Mat NvjpegCompressRunner::reconstruct(CompressConfiguration cfg , std::string ImageDirPath)
{
    struct stat buffer;
    auto startTime = std::chrono::steady_clock::now();
    cv::Mat resultImage = compressor->ReconstructedImage(cfg , ImageDirPath);
    std::string run_state = resultImage.empty() ? "Failure" : "Finish";
    auto endTime = std::chrono::steady_clock::now();
    std::cout << "=> Reconstructed " << run_state << std::endl;
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Reconstruct func cost time: " << elapsedTime << " ms" << std::endl;
    return resultImage;
}