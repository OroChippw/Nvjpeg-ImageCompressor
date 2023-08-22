/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.08
    Description: Define each module of NvjpegCompressRunnerImpl
*********************************/
# pragma warning (disable:4819)
# pragma warning (disable:4996)

#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "ImageCompressorImpl.h"

/* Setting & Getting Function */

void NvjpegCompressRunnerImpl::setImageProperties(int width , int height)
{
    compress_image_width = width;
    compress_image_height = height;
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

std::vector<std::vector<unsigned char>> NvjpegCompressRunnerImpl::getObufferList()
{
    return obuffer_lists;
}

std::vector<cv::Mat> NvjpegCompressRunnerImpl::getResultList()
{
    return result_lists;
}

/* Preprocess Function */

std::vector<cv::Mat> NvjpegCompressRunnerImpl::CropImage(const cv::Mat Image, int crop_ratio)
{
    std::vector<cv::Mat> crop_list;
    unsigned int index_x = 0, index_y = 0;
    unsigned int after_crop_width = Image.cols / crop_ratio;
    unsigned int after_crop_height = Image.rows / crop_ratio;

    // 列优先切分大图为crop_ratio平方块小图
    for (int i = 1; i <= crop_ratio; i++)
    {
        for (int j = 1; j <= crop_ratio; j++)
        {
            cv::Rect rect(index_x, index_y, after_crop_width, after_crop_height);
            crop_list.emplace_back(Image(rect));
            index_y += (after_crop_height);
        }
        index_x += (after_crop_width);
        index_y = 0;
    }

    std::cout << "=》 Finish cropping and Crop_list size is " << crop_list.size() << std::endl;
    return crop_list;
}

cv::Mat NvjpegCompressRunnerImpl::Binaryfile2Mat(std::string ImagePath)
{
    cv::Mat Image;
    FILE* pfile = fopen(ImagePath.c_str(), "rb");
    if (pfile == NULL)
        return Image;

    fseek(pfile, 0, SEEK_END);
    const unsigned int length = ftell(pfile);
    fseek(pfile, 0, SEEK_SET);
    if (length <= 0)
    {
        fclose(pfile);
        return Image;
    }
    unsigned char* pre_image = new unsigned char[length];
    size_t data = fread(pre_image, 1, length, pfile);
    fclose(pfile);

    std::vector<unsigned char> buffer(pre_image, pre_image + data);
    Image = imdecode(buffer, cv::IMREAD_ANYCOLOR);

    delete[]pre_image;

    return Image;
}

bool NvjpegCompressRunnerImpl::cmp(const std::string& str1, const std::string& str2)
{
    std::string::size_type iPos = str1.find_last_of('/') + 1;
    std::string filename = str1.substr(iPos, str1.length() - iPos);
    std::string num1 = filename.substr(0, filename.find("."));
    int num1_ = std::stoi(num1);

    std::string::size_type iPos2 = str2.find_last_of('/') + 1;
    std::string filename2 = str2.substr(iPos2, str2.length() - iPos2);
    std::string num2 = filename2.substr(0, filename2.find("."));
    int num2_ = std::stoi(num2);

    return (num1_ < num2_);
}

/* Compress Function */

std::vector<unsigned char> NvjpegCompressRunnerImpl::CompressWorker(const cv::Mat Image)
{
    const unsigned int image_width = Image.cols;
    const unsigned int image_height = Image.rows;
    const unsigned int channel_size = image_width * image_height;
    std::cout << "[COMPRESS IMAGE INFO] width : " << image_width << " height : " << image_height << std::endl;

    /* nvjpeg handles init */
    nvjpegHandle_t nvjpeg_handle;
    nvjpegEncoderState_t encoder_state; // 存储用于压缩中间缓冲区和变量的结构体
    nvjpegEncoderParams_t encoder_params; // 存储用于JPEG压缩参数的结构体

    nvjpegInputFormat_t input_format = NVJPEG_INPUT_BGR; // 指定用户提供何种类型的输入进行编码,输入是BGR，编码前会被转换成YCbCr

    // nvjpegBackend_t用来选择运行后端，使用GPU解码baseline JPEG或者使用CPU进行Huffman解码
    nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;

    cudaEvent_t ev_start = NULL, ev_end = NULL;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_end));

    // nvjpegCreate函数已被弃用。 使用 nvjpegCreateSimple() 或 nvjpegCreateEx() 函数创建库句柄。
    CHECK_NVJPEG(nvjpegCreate(backend, nullptr, &nvjpeg_handle));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle, &encoder_params, NULL)); // 创建保存压缩参数的结构
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL)); // 创建存储压缩中使用的中间缓冲区的编码器状态

    /* 设置编码压缩参数 */
    nvjpegEncoderParamsSetEncoding(encoder_params, nvjpegJpegEncoding_t::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, NULL);
    /*
        nvjpegEncoderParamsSetOptimizedHuffman设置是否使用优化的Huffman编码，
        使用优化的Huffman生成更小的JPEG比特流，质量相同但性能较慢,第二个参数值默认为0不使用优化Huffman编码
    */
    nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, use_optimizedHuffman, NULL);
    nvjpegEncoderParamsSetQuality(encoder_params, encode_quality, NULL); // 设置质量参数
    nvjpegEncoderParamsSetSamplingFactors(encoder_params, nvjpegChromaSubsampling_t::NVJPEG_CSS_422, NULL); // 设置用于JPEG压缩的色度子采样参数，官方默认为NVJPEG_CSS_444

    nvjpegImage_t input; // 输入图像数据指针为显式指针，每个颜色分量分别存储

    std::vector<cv::Mat> channels_list;
    cv::split(Image, channels_list);
    for (int i = 0; i < channels_list.size(); i++)
    {
        input.pitch[i] = image_width;
        CHECK_CUDA(cudaMalloc((void**)&(input.channel[i]), channel_size));
        size_t inputSize = channels_list[i].total() * channels_list[i].elemSize();
        CHECK_CUDA(cudaMemcpy((void*)input.channel[i], channels_list[i].ptr(0), inputSize, cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaEventRecord(ev_start));
    CHECK_NVJPEG(nvjpegEncodeImage(nvjpeg_handle, encoder_state, encoder_params, &input, input_format, image_width, image_height, NULL));
    CHECK_CUDA(cudaEventRecord(ev_end));

    std::vector<unsigned char> obuffer;
    size_t length;

    /* 从先前在其中一个编码器功能中使用的编码器状态中检索压缩流，如果数据参数data为NULL，则编码器将在长度参数中返回压缩流大小 */
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, NULL, &length, NULL));
    /* 先返回压缩流的长度 再用压缩流长度大小的buffer接受压缩流 */

    obuffer.resize(length);
    // std::cout << "Resize outbuffer length : " << length << std::endl;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, obuffer.data(), &length, NULL));

    cudaEventSynchronize(ev_end);

    for (int i = 0; i < channels_list.size(); i++)
    {
        cudaFree(input.channel[i]);
    }

    float ms = 0.0;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    std::cout << "=> Compress Cost time : " << ms << "ms" << std::endl;
    compress_time_total += ms;
    
    /* nvjpeg handles destory */
    nvjpegEncoderParamsDestroy(encoder_params);
    nvjpegEncoderStateDestroy(encoder_state);
    nvjpegDestroy(nvjpeg_handle);

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
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    return obuffer;
}

int NvjpegCompressRunnerImpl::Compress(std::vector<cv::Mat> image_list)
{
    std::cout << "# ----------------------------------------------- #" << std::endl;
    compress_time_total = 0.0;     
    try
    {
        for (unsigned int index = 0; index < image_list.size(); index++)
        {
            obuffer_lists.push_back(CompressWorker(image_list[index]));
        }
    }  
    catch (const std::exception& e) 
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        
        return EXIT_FAILURE;
    }

    std::cout << "# ----------------------------------------------- #" << std::endl;
    std::cout << image_list.size() << " Images mean Cost time : " << compress_time_total / image_list.size() << "ms" << std::endl;
    
    return EXIT_SUCCESS;
}

int NvjpegCompressRunnerImpl::CompressImage(std::vector<cv::Mat> image_matlist)
{
    std::vector<cv::Mat> image_lists;
    std::cout << "=> Start image compression ... " << std::endl;
    if (do_crop)
    {
        if (!((compress_image_width % crop_ratio == 0) && (compress_image_width % crop_ratio == 0)))
        {
            std::cout << "The width and height of the image must be divisible by the number of blocks" << std::endl;
            std::cout << "=> Calculate the greatest common factor of width and height" << std::endl;
            crop_ratio = CalculateGreatestFactor(compress_image_width , compress_image_width);
        }
        for (unsigned int index = 0 ; index < image_matlist.size() ; index++)
        {
            image_lists = CropImage(image_matlist[index] , crop_ratio);
        }
    }else{
        image_lists = image_matlist;
    }
    auto only_compress_startTime = std::chrono::steady_clock::now();

    if (Compress(image_lists))
    {
        return EXIT_FAILURE;
    }

    auto only_compress_endTime = std::chrono::steady_clock::now();
    auto only_compress_elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(only_compress_endTime - only_compress_startTime).count();
    std::cout << "Only Compress func cost time: " << only_compress_elapsedTime << " ms" << std::endl;
    
    return EXIT_SUCCESS;
}

/* Reconstruct Function */

cv::Mat NvjpegCompressRunnerImpl::ReconstructWorker(const std::vector<unsigned char> obuffer)
{    
    cv::Mat resultImage = cv::imdecode(obuffer , cv::IMREAD_ANYCOLOR);

    return resultImage;
}

cv::Mat NvjpegCompressRunnerImpl::Reconstructed(std::vector<unsigned char> obuffer)
{
    cv::Mat image;
    try
    {
        image = ReconstructWorker(obuffer);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Exception caught : " << e.what() << std::endl;
    }
    
    return image;
}

int NvjpegCompressRunnerImpl::Reconstructed(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    try
    {
        for (unsigned int index = 0; index < obuffer_lists.size(); index++)
        {
            result_lists.emplace_back(ReconstructWorker(obuffer_lists[index]));
        }
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

int NvjpegCompressRunnerImpl::ReconstructedImage(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    std::cout << "=> Start image reconstruction ... " << std::endl;   
    if (Reconstructed(obuffer_lists))
    {
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

/* Decode Function */

int NvjpegCompressRunnerImpl::dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

int NvjpegCompressRunnerImpl::dev_free(void *p) { return (int)cudaFree(p); }

int NvjpegCompressRunnerImpl::host_malloc(void** p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }

int NvjpegCompressRunnerImpl::host_free(void* p) { return (int)cudaFreeHost(p); }

cv::Mat NvjpegCompressRunnerImpl::DecodeWorker(const std::vector<unsigned char> obuffer)
{   
    // /* 为RGBi输出格式准备缓冲区，NVJPEG_MAX_COMPONENT为nvjpeg解码器支持的最大通道数，默认为4 */
    // int widths[NVJPEG_MAX_COMPONENT];
    // int heights[NVJPEG_MAX_COMPONENT];
    // int channels;
    // int batch_size = 1;
    // float ms = 0.0;
    // size_t file_len;
    // bool hw_decode_available;

    // /* nvjpeg handles init */
    // nvjpegHandle_t nvjpeg_handle;
    // nvjpegJpegState_t nvjpeg_state;
    // cudaStream_t stream;

    // nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGB;
    // nvjpegChromaSubsampling_t subsampling;
    // nvjpegJpegState_t nvjpeg_decoupled_state;
    // nvjpegDecodeParams_t nvjpeg_decode_params;
    // nvjpegJpegDecoder_t nvjpeg_decoder;
    // nvjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
    // nvjpegBufferDevice_t device_buffer;
    // nvjpegJpegStream_t  jpeg_streams[2]; //  2 streams for pipelining
    // std::vector<nvjpegImage_t> iout(batch_size); // output buffers
    // std::vector<nvjpegImage_t> isz(batch_size);  // output buffer sizes, for convenience
    // /* nvjpegBackend_t用来选择运行后端，使用GPU解码baseline JPEG或者使用CPU进行Huffman解码 */
    // nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
    // for (int i = 0; i < iout.size(); i++) {
    //     for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
    //         iout[i].channel[c] = NULL;
    //         iout[i].pitch[c] = 0;
    //         isz[i].pitch[c] = 0;
    //     }
    // }
    // /* stream for decoding */
    // CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // for (int i = 0 ; i < 1 ; i++)
    // {
    //     CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle , (unsigned char *)obuffer.data() , file_len , &channels , &subsampling , widths , heights));
    //     int img_width = widths[0];
    //     int img_height = heights[0];
    //     std::cout << "Image is " << channels << " channels." << std::endl;
    //     switch (subsampling) {
    //     case NVJPEG_CSS_444:
    //         std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
    //         break;
    //     case NVJPEG_CSS_440:
    //         std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
    //         break;
    //     case NVJPEG_CSS_422:
    //         std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
    //         break;
    //     case NVJPEG_CSS_420:
    //         std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
    //         break;
    //     case NVJPEG_CSS_411:
    //         std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
    //         break;
    //     case NVJPEG_CSS_410:
    //         std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
    //         break;
    //     case NVJPEG_CSS_GRAY:
    //         std::cout << "Grayscale JPEG " << std::endl;
    //         break;
    //     case NVJPEG_CSS_UNKNOWN:
    //         std::cout << "Unknown chroma subsampling" << std::endl;
    //         return ;
    //     }

    //     int mul = 1;
    //     // in the case of interleaved RGB output, write only to single channel, but
    //     // 3 samples at once
    //     if (fmt == NVJPEG_OUTPUT_RGBI || fmt == NVJPEG_OUTPUT_BGRI) 
    //     {
    //         channels = 1;
    //         mul = 3;
    //     }
    //     // in the case of rgb create 3 buffers with sizes of original image
    //     else if (fmt == NVJPEG_OUTPUT_RGB || fmt == NVJPEG_OUTPUT_BGR) 
    //     {
    //         channels = 3;
    //         widths[1] = widths[2] = widths[0];
    //         heights[1] = heights[2] = heights[0];
    //     }

    //     for (int c = 0; c < channels; c++) {
    //     int aw = mul * widths[c];
    //     int ah = heights[c];
    //     int sz = aw * ah;
    //     iout[i].pitch[c] = aw;
    //     if (sz > isz[i].pitch[c]) {
    //         if (iout[i].channel[c]) {
    //             CHECK_CUDA(cudaFree(iout[i].channel[c]));
    //         }
    //         // CHECK_CUDA(cudaMalloc(&iout.channel[c], sz));
    //         isz[i].pitch[c] = sz;
    //     }
    //     }

    // }
    
    // CHECK_CUDA(cudaStreamSynchronize(stream));
    // cudaEvent_t ev_start = NULL, ev_end = NULL;
    // CHECK_CUDA(cudaEventCreate(&ev_start));
    // CHECK_CUDA(cudaEventCreate(&ev_end));

    // nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr , nullptr , NVJPEG_FLAGS_DEFAULT , &nvjpeg_handle);
    // hw_decode_available = true;
    // if( status == NVJPEG_STATUS_ARCH_MISMATCH) 
    // {
    //     std::cout << "Hardware Decoder not supported. Falling back to default backend" << std::endl;
    //     CHECK_NVJPEG(nvjpegCreateEx(backend, nullptr, nullptr, NVJPEG_FLAGS_DEFAULT, &nvjpeg_handle));
    //     hw_decode_available = false;
    // }else
    // {
    //     CHECK_NVJPEG(status);
    // }
    // CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
    // /* Create decoupled api handles */
    // CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder));
    // CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state));   

    // CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[0]));
    // CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[1]));
    // CHECK_NVJPEG(nvjpegBufferDeviceCreate(nvjpeg_handle, NULL, &device_buffer));
    
    // CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[0]));
    // CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[1]));
    // CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));

    // std::vector<const unsigned char*> batched_bitstreams;
    // std::vector<size_t> batched_bitstreams_size;
    // std::vector<nvjpegImage_t>  batched_output;

    // // bit-streams that batched decode cannot handle
    // std::vector<const unsigned char*> otherdecode_bitstreams;
    // std::vector<size_t> otherdecode_bitstreams_size;
    // std::vector<nvjpegImage_t> otherdecode_output;
    
    // if(hw_decode_available)
    // {
    //     for(int i = 0 ; i < batch_size ; i++)
    //     {
    //         // extract bitstream meta data to figure out whether a bit-stream can be decoded
    //         nvjpegJpegStreamParseHeader(nvjpeg_handle, (const unsigned char *)obuffer.data(), file_len , jpeg_streams[0]);
    //         int isSupported = -1;
    //         nvjpegDecodeBatchedSupported(nvjpeg_handle, jpeg_streams[0], &isSupported);
    //         if(isSupported == 0)
    //         {
    //             batched_bitstreams.push_back((const unsigned char *)obuffer.data());
    //             batched_bitstreams_size.push_back(file_len);
    //             batched_output.push_back(iout[i]);
    //         }else
    //         {
    //             otherdecode_bitstreams.push_back((const unsigned char *)obuffer.data());
    //             otherdecode_bitstreams_size.push_back(file_len);
    //             otherdecode_output.push_back(iout[i]);
    //         }
    //     }
    // }else
    // {
    //     for(int i = 0; i < batch_size; i++) {
    //         otherdecode_bitstreams.push_back((const unsigned char *)obuffer.data());
    //         otherdecode_bitstreams_size.push_back(file_len);
    //         otherdecode_output.push_back(iout[i]);
    //     }
    // }
    
    // CHECK_CUDA(cudaEventRecord(ev_start , stream));

    // if(batched_bitstreams.size() > 0)
    // {
    //     CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(nvjpeg_handle, nvjpeg_state, \
    //                         batched_bitstreams.size(), 1, fmt));
    //     CHECK_NVJPEG(nvjpegDecodeBatched(nvjpeg_handle, nvjpeg_state, batched_bitstreams.data(), \
    //          batched_bitstreams_size.data(), batched_output.data(), stream));
    // }
    // if(otherdecode_bitstreams.size() > 0)
    // {
    //     CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
    //     int buffer_index = 0;
    //     CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params, fmt));
    //     for (int i = 0; i < batch_size; i++) {
    //           CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle, otherdecode_bitstreams[i], \
    //           otherdecode_bitstreams_size[i], 0, 0, jpeg_streams[buffer_index]));
    //           CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, \
    //               pinned_buffers[buffer_index]));
    //           CHECK_NVJPEG(nvjpegDecodeJpegHost(nvjpeg_handle, nvjpeg_decoder, nvjpeg_decoupled_state, \
    //               nvjpeg_decode_params, jpeg_streams[buffer_index]));
    //           CHECK_CUDA(cudaStreamSynchronize(stream));
    //           CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle, nvjpeg_decoder, nvjpeg_decoupled_state, \
    //                 jpeg_streams[buffer_index], stream));

    //           buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

    //           CHECK_NVJPEG(nvjpegDecodeJpegDevice(nvjpeg_handle, nvjpeg_decoder, nvjpeg_decoupled_state, \
    //               &otherdecode_output[i], stream));
    //     }
    // }

    // CHECK_CUDA(cudaEventRecord(ev_end, stream));
    // CHECK_CUDA(cudaEventSynchronize(ev_end));
    // CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_end)); 
    // std::cout << "=> Decode Cost time : " << ms << "ms" << std::endl;
    // decode_time_total += ms;

    // /* write images */
    // for (int i = 0 ; i < batch_size ; i++)
    // {
    //     int err;
    //     if (err)
    //     {
    //         std::cout << "[ERROR] Cannot write output file: " << std::endl;
    //         return ;
    //     }
    //     std::cout << "Done writing decoded image to file: " << std::endl;
    // }

    // /* nvjpeg handles destory */
    // CHECK_CUDA(cudaStreamDestroy(stream));
    // CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
    // CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams[0]));
    // CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams[1]));
    // CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers[0]));
    // CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers[1]));
    // CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer));
    // CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));  
    // CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_decoder));
    // CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));
    // CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));

    // /* release buffers */
    // for (int i = 0 ; i < iout.size() ; i++)
    // {
    //     for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
    //     {
    //         if (iout[i].channel[c])
    //         {
    //             CHECK_CUDA(cudaFree(iout[i].channel[c]));
    //         } 
    //     }
    // }

    int width , height;

    width = 3840;
    height = 2160;
    nvjpegImage_t nvjpeg_image;
    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t nvjpeg_state;
    nvjpegOutputFormat_t format = NVJPEG_OUTPUT_RGB;

    nvjpeg_image.channel[0] = nullptr;
    nvjpeg_image.channel[1] = nullptr;
    nvjpeg_image.channel[2] = nullptr;
    nvjpeg_image.pitch[0] = 0;
    nvjpeg_image.pitch[1] = 0;
    nvjpeg_image.pitch[2] = 0;

    CHECK_NVJPEG(nvjpegCreateSimple(&nvjpeg_handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
    // CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle , (unsigned char *)obuffer.data() , file_len , &channels , &subsampling , widths , heights));
    nvjpegStatus_t status = nvjpegDecode(nvjpeg_handle, nvjpeg_state , obuffer.data(), obuffer.size(), \
                        NVJPEG_OUTPUT_BGR , &nvjpeg_image, nullptr);
    std::cout << "0" << std::endl;
    if (status != NVJPEG_STATUS_SUCCESS)
    {
        std::cerr << "Failed to decode JPEG." << std::endl;
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));
        CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
        return cv::Mat();  // Return an empty cv::Mat
    }
    std::cout << "1" << std::endl;

    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
    std::cout << "2" << std::endl;

    cv::Mat decodedImage(height, width, CV_8UC3);
    for (int c = 0; c < 3; ++c) {
        std::cout << c << std::endl;

        std::memcpy(decodedImage.data + c, nvjpeg_image.channel[c], height * width);
    }

    // Clone the data to ensure it's correctly managed
    cv::Mat resultImage = decodedImage.clone();

    return resultImage;
}

cv::Mat NvjpegCompressRunnerImpl::Decode(std::vector<unsigned char> obuffer)
{
    cv::Mat image;
    try
    {
        image = DecodeWorker(obuffer);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Exception caught : " << e.what() << '\n';
    }
    
    return image;
}

int NvjpegCompressRunnerImpl::DecodeImage(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    return EXIT_SUCCESS;
}

/* Calculate indicator function */

cv::Mat NvjpegCompressRunnerImpl::CalculateDiffmap(const cv::Mat srcImage, const std::string compImagePath)
{
    cv::Mat compressImage = cv::imread(compImagePath, cv::IMREAD_ANYCOLOR);
    cv::Mat diffMap = srcImage - compressImage;

    if (show_diff_info)
    {
        double minVal, maxVal;
        cv::Point minIdx, maxIdx;
        std::vector<cv::Mat> img_channels;
        cv::split(diffMap, img_channels);

        /* 计算差异图像素最大值、最小值、均值以及标准差 */
        for (int index = 0; index < img_channels.size(); index++) {
            cv::minMaxLoc(img_channels[index], &minVal, &maxVal, &minIdx, &maxIdx);
            std::cout << "diffMap[" << index << "] minVal : " << minVal << " , minIdx : " << minIdx << std::endl;
            std::cout << "diffMap[" << index << "] maxVal : " << maxVal << " , maxIdx : " << maxIdx << std::endl;
        }

        cv::Scalar channelsMean;
        channelsMean = mean(diffMap);
        cv::Mat meanMat, stddevMat;
        cv::meanStdDev(diffMap, meanMat, stddevMat);
        std::cout << "diffMap MeanMat : " << meanMat << std::endl;
        std::cout << "diffMap stddevMat : " << stddevMat << std::endl;
    }

    return diffMap;
}

double NvjpegCompressRunnerImpl::CalculatePSNR(cv::Mat srcImage, cv::Mat compImage)
{
    const unsigned int max = 255;
    cv::Mat subImage;
    cv::absdiff(srcImage, compImage, subImage);
    subImage = subImage.mul(subImage);
    cv::Scalar sumScalar = sum(subImage);
    double sse = sumScalar.val[0] + sumScalar[1] + sumScalar[2];
    if (sse <= 1e-10)
    {
        return 0;
    }
    else
    {
        double mse = sse / srcImage.rows / srcImage.cols;
        double psnr = 10 * log10(pow(max, 2) / mse);
        std::cout << "[VAL->MSE] : " << mse << " [VAL->PSNR] : " << psnr << std::endl;
        return psnr;
    }
}

int NvjpegCompressRunnerImpl::CalculateGreatestFactor(const int width , const int height)
{
    std::vector<int> width_factor , height_factor;
    int ceil = 3 , factor = 1;
    for(int i = 1 ; i <= width ; i++){
        if (width % i==0) {
            width_factor.emplace_back(i);
        }
        if (i > ceil) {break;}
    }
    for(int i = 1 ; i <= height ; i++){
        if (height % i==0) {
            height_factor.emplace_back(i);
        }
        if (i > ceil) {break;}
    }
    for (int index = 0 ; index < std::min(width_factor.size() , height_factor.size()); index++)
    {
        if (width_factor[index] == height_factor[index])
        {
            factor = width_factor[index];
            continue;
        }else{
            factor = width_factor[index - 1];
            break;
        } 
    }
    
	return factor;
}

cv::Mat NvjpegCompressRunnerImpl::addImage(cv::Mat image_1, cv::Mat image_2)
{
    return image_1 + image_2;
}