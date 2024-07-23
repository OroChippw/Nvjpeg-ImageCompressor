/*
    Copyright: OroChippw
    Author: OroChippw
    Date: 2024.07.17
    Description: Head file of NvjpegCompressRunnerImpl
*/

#ifndef IMAGECOMPRESSORIMPL_CUH_
#define IMAGECOMPRESSORIMPL_CUH_

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
        /* Configuration */
        int encode_quality = 95;
        bool use_optimizedHuffman = true;
    
        /* Image Properties */
        int image_width = 8320;
        int image_height = 40000;
        
    private:
        /* tag */
        bool compress_init = false;
        bool decode_init = false;

        /* Init Event Pointer */
        cudaEvent_t ev_start, ev_end;
        cudaEvent_t decode_ev_start , decode_ev_end;
        nvjpegBackend_t backend = NVJPEG_BACKEND_GPU_HYBRID; // 或者 NVJPEG_BACKEND_HARDWARE

        /* Init Compress Nvjpeg Pointer */
        nvjpegHandle_t nvjpeg_handle_compress;
        nvjpegEncoderState_t encoder_state;
        nvjpegEncoderParams_t encoder_params;
        nvjpegImage_t input; // 输入图像数据指针为显式指针，每个颜色分量分别存储
        nvjpegInputFormat_t input_format = NVJPEG_INPUT_BGR;
        unsigned int channel_size;

        /* Init Decode Nvjpeg Pointer */
        nvjpegHandle_t nvjpeg_handle_decode;
        nvjpegJpegState_t nvjpeg_state;
        nvjpegBufferDevice_t device_buffer;
        nvjpegOutputFormat_t format = NVJPEG_OUTPUT_BGR;
        nvjpegChromaSubsampling_t subsampling;
        nvjpegJpegState_t nvjpeg_decoupled_state;
        nvjpegDecodeParams_t nvjpeg_decode_params;
        nvjpegJpegDecoder_t nvjpeg_decoder;
        nvjpegBufferPinned_t pinned_buffers[2];
        nvjpegJpegStream_t jpeg_streams[2];
        nvjpegImage_t nvjpeg_image; // output buffers
        nvjpegImage_t nvjpeg_output_size; // output buffer sizes, for convenience

    public:
        NvjpegCompressRunnerImpl(int width , int height , int quality , bool optimize);
        ~NvjpegCompressRunnerImpl();

        NvjpegCompressRunnerImpl(const NvjpegCompressRunnerImpl&) = delete;
        NvjpegCompressRunnerImpl& operator=(const NvjpegCompressRunnerImpl&) = delete;

        // 移动构造函数和赋值运算符
        NvjpegCompressRunnerImpl(NvjpegCompressRunnerImpl&&) = default;
        NvjpegCompressRunnerImpl& operator=(NvjpegCompressRunnerImpl&&) = default;
    
    public:
        void setImageProperties(int width , int height);
        void setEncodeQuality(int quality);
        int getEncodeQuality();
        void setOptimizedHuffman(bool optimize);
        bool isOptimizedHuffman();

        void initCompressEnv();
        void initDecodeEnv();
        void destoryCompressEnv();
        void destoryDecodeEnv();

        std::vector<unsigned char> Compress(cv::Mat image);
        cv::Mat Decode(FILE *jpeg_file);

    private:
        /* Components that perform compression */
        std::vector<unsigned char> CompressWorker(const cv::Mat Image);

        /* ******* Use NvjpegDecoder to decode ******* */
        cv::Mat DecodeWorker(FILE *jpeg_file);

        cv::Mat getCVImageOnCPU(const unsigned char *d_chanB, int pitchB, \
                                             const unsigned char *d_chanG, int pitchG, \
                                             const unsigned char *d_chanR, int pitchR, \
                                             int width, int height);

        cv::Mat getCVImageOnGPU(const unsigned char *d_chanB, int pitchB, \
                                const unsigned char *d_chanG, int pitchG, \
                                const unsigned char *d_chanR, int pitchR, \
                                int width, int height);
};


#endif // IMAGECOMPRESSORIMPL_CUH_