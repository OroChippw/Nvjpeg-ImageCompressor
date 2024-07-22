/*
    Copyright: OroChippw
    Author: OroChippw
    Date: 2024.07.17
    Description: Define each module of NvjpegCompressRunnerImpl
*/

#include <chrono>
#include <vector>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "ImageCompressorImpl.cuh"


void NvjpegCompressRunnerImpl::initCompressEnv()
{
    auto startTime = std::chrono::steady_clock::now();
    CHECK_NVJPEG(nvjpegCreate(backend, nullptr, &nvjpeg_handle_compress));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle_compress, &encoder_params, NULL));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nvjpeg_handle_compress, &encoder_state, NULL));
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_end));

    nvjpegEncoderParamsSetEncoding(encoder_params, nvjpegJpegEncoding_t::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, NULL);
    nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, use_optimizedHuffman , NULL); // 是否使用优化的Huffman编码
    nvjpegEncoderParamsSetQuality(encoder_params, encode_quality , NULL); // 设置质量参数
    nvjpegEncoderParamsSetSamplingFactors(encoder_params, nvjpegChromaSubsampling_t::NVJPEG_CSS_444, NULL); // 设置色度子采样参数

    channel_size = image_width * image_height;
    for (int i = 0; i < 3; ++i) {
        input.pitch[i] = image_width;
        CHECK_CUDA(cudaMalloc((void**)&(input.channel[i]), channel_size));
    }

    compress_init = true;

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "[INFO] Init Compress Env Successfully Cost Time : " << elapsedTime << " ms" << std::endl;
}

void NvjpegCompressRunnerImpl::destoryCompressEnv()
{
    auto startTime = std::chrono::steady_clock::now();
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(encoder_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(encoder_state));
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle_compress));
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_end));

    for (int i = 0; i < 3; ++i) {
        cudaFree(input.channel[i]);
    }

    compress_init = false;

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] Destory Compress Env Successfully Cost Time : " << elapsedTime << " ms" << std::endl;
}

void NvjpegCompressRunnerImpl::initDecodeEnv()
{
    auto startTime = std::chrono::steady_clock::now();

    CHECK_NVJPEG(nvjpegCreate(backend, nullptr, &nvjpeg_handle_decode));
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle_decode, &nvjpeg_state));
    CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle_decode, backend, &nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle_decode, nvjpeg_decoder, &nvjpeg_decoupled_state));
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle_decode, &nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle_decode, nullptr, &pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle_decode, nullptr, &pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle_decode, &jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle_decode, &jpeg_streams[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(nvjpeg_handle_decode, nullptr, &device_buffer));
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params, format));
    CHECK_CUDA(cudaEventCreate(&decode_ev_start));
    CHECK_CUDA(cudaEventCreate(&decode_ev_end));

    for (int c = 0; c < 3; c++) {
        nvjpeg_image.channel[c] = NULL;
        nvjpeg_image.pitch[c] = 0;
        nvjpeg_output_size.pitch[c] = 0;
    }

    decode_init = true;
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] Init Decode Env Successfully Cost Time : " << elapsedTime << " ms" << std::endl;
}

void NvjpegCompressRunnerImpl::destoryDecodeEnv()
{
    auto startTime = std::chrono::steady_clock::now();

    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));
    CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams[1]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer));
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle_decode));
    CHECK_CUDA(cudaEventDestroy(decode_ev_start));
    CHECK_CUDA(cudaEventDestroy(decode_ev_end));

    decode_init = false;
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "[INFO] Destory Destory Env Successfully Cost Time : " << elapsedTime << " ms" << std::endl;
}

NvjpegCompressRunnerImpl::NvjpegCompressRunnerImpl(int width , int height , int quality , bool optimize)
{
    this->setImageProperties(width , height);
    this->setEncodeQuality(quality);
    this->setOptimizedHuffman(optimize);
}

NvjpegCompressRunnerImpl::~NvjpegCompressRunnerImpl()
{
    try
    {
        if (compress_init)
        {
            destoryCompressEnv();
        }
        if (decode_init)
        {
            destoryDecodeEnv();
        } 
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

void NvjpegCompressRunnerImpl::setImageProperties(int width , int height)
{
    image_width = width;
    image_height = height;
}

void NvjpegCompressRunnerImpl::setEncodeQuality(int quality)
{
    encode_quality = quality;
}

int NvjpegCompressRunnerImpl::getEncodeQuality()
{
    return encode_quality;
}

void NvjpegCompressRunnerImpl::setOptimizedHuffman(bool optimize)
{
    use_optimizedHuffman = optimize;
}

bool NvjpegCompressRunnerImpl::isOptimizedHuffman()
{
    return use_optimizedHuffman;
}

__global__ void combineChannels(unsigned char* d_dst, int width, int height, int pitch,
                                unsigned char* d_chanR, unsigned char* d_chanG, unsigned char* d_chanB) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        d_dst[y * pitch + 3 * x] = d_chanB[idx];
        d_dst[y * pitch + 3 * x + 1] = d_chanG[idx];
        d_dst[y * pitch + 3 * x + 2] = d_chanR[idx];
    }
}

cv::Mat NvjpegCompressRunnerImpl::getCVImageOnCPU(const unsigned char *d_chanB, int pitchB, \
                                             const unsigned char *d_chanG, int pitchG, \
                                             const unsigned char *d_chanR, int pitchR, \
                                             int width, int height) 
{
    cudaEvent_t start, end;
    float milliseconds = 0.0;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(start));

    cv::Mat cvImage(height, width, CV_8UC3); //BGR
    std::vector<unsigned char> vchanR(height * width);
    std::vector<unsigned char> vchanG(height * width);
    std::vector<unsigned char> vchanB(height * width);
    unsigned char *chanR = vchanR.data();
    unsigned char *chanG = vchanG.data();
    unsigned char *chanB = vchanB.data();

    CHECK_CUDA(cudaMemcpy2D(chanR, (size_t)width, d_chanR, (size_t)pitchR, \
                    width, height, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy2D(chanG, (size_t)width, d_chanG, (size_t)pitchR, \
                    width, height, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy2D(chanB, (size_t)width, d_chanB, (size_t)pitchR, \
                    width, height, cudaMemcpyDeviceToHost));

    unsigned char* imgPtr = cvImage.ptr<unsigned char>();
    int imgStep = cvImage.step;

    for (int y = 0; y < height; y++) {
        unsigned char* rowPtr = imgPtr + y * imgStep;
        for (int x = 0; x < width; x++) {
            rowPtr[x * 3] = chanB[y * width + x];       // Blue channel
            rowPtr[x * 3 + 1] = chanG[y * width + x];   // Green channel
            rowPtr[x * 3 + 2] = chanR[y * width + x];   // Red channel
        }
    }

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, end));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));

    std::cout << "=> getCVImage execution time: " << milliseconds << " ms" << std::endl;

    return cvImage;
}


cv::Mat NvjpegCompressRunnerImpl::getCVImageOnGPU(const unsigned char *d_chanB, int pitchB,
                                                  const unsigned char *d_chanG, int pitchG,
                                                  const unsigned char *d_chanR, int pitchR,
                                                  int width, int height) {
    float milliseconds = 0.0;
    cv::Mat cvImage(height, width, CV_8UC3);

    // 分配设备内存
    unsigned char* d_dst;
    size_t pitch;
    CHECK_CUDA(cudaMallocPitch((void**)&d_dst, &pitch, width * 3 * sizeof(unsigned char), height));

    // 设置CUDA核函数的执行配置
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 调用CUDA核函数
    combineChannels<<<gridSize, blockSize>>>(d_dst, width, height, pitch, (unsigned char*)d_chanR, (unsigned char*)d_chanG, (unsigned char*)d_chanB);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 分配主机内存
    std::vector<unsigned char> h_dst(width * height * 3);
    unsigned char* h_dst_ptr = h_dst.data();

    CHECK_CUDA(cudaMemcpy2D(h_dst_ptr, width * 3 * sizeof(unsigned char), d_dst, pitch,
                            width * 3 * sizeof(unsigned char), height, cudaMemcpyDeviceToHost));

    std::memcpy(cvImage.data, h_dst_ptr, width * height * 3 * sizeof(unsigned char));

    std::cout << "=> getCVImageOnGPU execution time: " << milliseconds << " ms" << std::endl;
    cudaFree(d_dst);

    return cvImage;
}

std::vector<unsigned char> NvjpegCompressRunnerImpl::CompressWorker(const cv::Mat Image)
{
    float ms = 0.0;
    std::vector<cv::Mat> channels_list;
    cv::split(Image, channels_list);
    for (int i = 0; i < channels_list.size(); i++) {
        size_t inputSize = channels_list[i].total() * channels_list[i].elemSize();
        CHECK_CUDA(cudaMemcpy((void*)input.channel[i], channels_list[i].ptr(0), inputSize, cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaEventRecord(ev_start));
    CHECK_NVJPEG(nvjpegEncodeImage(nvjpeg_handle_compress, encoder_state, encoder_params, &input, input_format, image_width, image_height, NULL));
    CHECK_CUDA(cudaEventRecord(ev_end));

    std::vector<unsigned char> obuffer;
    size_t length;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nvjpeg_handle_compress, encoder_state, NULL, &length, NULL));
    obuffer.resize(length);
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nvjpeg_handle_compress, encoder_state, obuffer.data(), &length, NULL));

    cudaEventSynchronize(ev_end);
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    std::cout << "=> Compress Cost time : " << ms << "ms" << std::endl;

    return obuffer;
}

std::vector<unsigned char> NvjpegCompressRunnerImpl::Compress(cv::Mat image)
{
    std::vector<unsigned char>obuffer;
    try
    {
        obuffer = CompressWorker(image); 
    }
    catch(const std::exception& e)
    {
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
    }

    return obuffer;
}

cv::Mat NvjpegCompressRunnerImpl::DecodeWorker(FILE *jpeg_file)
{
    fseek(jpeg_file, 0, SEEK_END);
    size_t jpeg_data_size = ftell(jpeg_file);
    rewind(jpeg_file);

    std::vector<unsigned char> jpeg_data(jpeg_data_size);
    if (jpeg_data.empty()) {
        fclose(jpeg_file);
        return cv::Mat();
    }

    size_t bytes_read = fread(jpeg_data.data(), 1, jpeg_data_size, jpeg_file);
    fclose(jpeg_file);

    if (bytes_read != jpeg_data_size) {
        std::cerr << "[INFO] Failed to read the entire JPEG data." << std::endl;
        return cv::Mat();
    }

    int nComponent = 0;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle_decode, jpeg_data.data(), jpeg_data_size, &nComponent, &subsampling, widths, heights));

    int mul = (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI) ? 3 : 1;
    if (format == NVJPEG_OUTPUT_RGB || format == NVJPEG_OUTPUT_BGR) {
        widths[1] = widths[2] = widths[0];
        heights[1] = heights[2] = heights[0];
    }

    for (int c = 0; c < nComponent; c++) {
        int aw = mul * widths[c];
        int ah = heights[c];
        int sz = aw * ah;
        nvjpeg_image.pitch[c] = aw;
        if (sz > nvjpeg_output_size.pitch[c]) {
            if (nvjpeg_image.channel[c]) {
                CHECK_CUDA(cudaFree(nvjpeg_image.channel[c]));
            }
            CHECK_CUDA(cudaMalloc((void**)&nvjpeg_image.channel[c], sz));
            nvjpeg_output_size.pitch[c] = sz;
        }
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaEventRecord(decode_ev_start , stream));

    CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
    CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle_decode, jpeg_data.data(), jpeg_data_size, 0, 0, jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegDecodeJpegHost(nvjpeg_handle_decode, nvjpeg_decoder, nvjpeg_decoupled_state, nvjpeg_decode_params, jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_decode, nvjpeg_decoder, nvjpeg_decoupled_state, jpeg_streams[0], stream));
    CHECK_NVJPEG(nvjpegDecodeJpegDevice(nvjpeg_handle_decode, nvjpeg_decoder, nvjpeg_decoupled_state, &nvjpeg_image, stream));

    CHECK_CUDA(cudaEventRecord(decode_ev_end , stream));
    CHECK_CUDA(cudaEventSynchronize(decode_ev_end));

    float ms = 0.0;
    cudaEventElapsedTime(&ms, decode_ev_start, decode_ev_end);
    std::cout << "=> Decode Cost time : " << ms << "ms" << std::endl;

    cv::Mat resultImage;
    if (nvjpeg_image.channel[0] != nullptr) {
        resultImage = getCVImageOnCPU(nvjpeg_image.channel[0], nvjpeg_image.pitch[0],
                                               nvjpeg_image.channel[1], nvjpeg_image.pitch[1],
                                               nvjpeg_image.channel[2], nvjpeg_image.pitch[2], widths[0], heights[0]);
    } else {
        std::cerr << "[ERROR] JPEG decode failed: Output image channel is null." << std::endl;
    }

    return resultImage;
}

cv::Mat NvjpegCompressRunnerImpl::Decode(FILE *jpeg_file)
{
    cv::Mat result;
    try
    {
        result = DecodeWorker(jpeg_file);
    }
    catch(const std::exception& e)
    {
        std::cerr << "[ERROR] Exception caught : " << e.what() << '\n';
    }

    return result;
}
