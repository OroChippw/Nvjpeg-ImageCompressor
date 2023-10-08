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

std::vector<cv::Mat> NvjpegCompressRunnerImpl::getReconstructResultList()
{
    return reconstruct_result_lists;
}

std::vector<cv::Mat> NvjpegCompressRunnerImpl::getDecodeResultList()
{
    return decode_result_lists;
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
    nvjpegEncoderParamsSetSamplingFactors(encoder_params, nvjpegChromaSubsampling_t::NVJPEG_CSS_444, NULL); // 设置用于JPEG压缩的色度子采样参数，官方默认为NVJPEG_CSS_444

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
        std::cerr << "[ERROR] Exception caught : " << e.what() << std::endl;
    }
    
    return image;
}

int NvjpegCompressRunnerImpl::Reconstructed(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    try
    {
        for (unsigned int index = 0; index < obuffer_lists.size(); index++)
        {
            reconstruct_result_lists.emplace_back(ReconstructWorker(obuffer_lists[index]));
        }
    }
    catch (const std::exception& e) 
    {
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
        
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

cv::Mat NvjpegCompressRunnerImpl::DecodeWorker(const std::vector<unsigned char> obuffer)
{   
    if (obuffer.empty())
    {
        std::cerr << "[ERROR] OBuffer is empty" << std::endl;
        return cv::Mat();
    }

    bool is_interleaved = false;
    bool hw_decode_available = false;
    int batch_size = 1;
    bool saveBMP = false;

    nvjpegImage_t nvjpeg_image; // output buffers
    nvjpegImage_t nvjpeg_output_size;  // output buffer sizes, for convenience

    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t nvjpeg_state;
    nvjpegBufferDevice_t device_buffer;

    nvjpegOutputFormat_t format = NVJPEG_OUTPUT_BGR;
    nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegJpegState_t nvjpeg_decoupled_state;
    nvjpegDecodeParams_t nvjpeg_decode_params;
    nvjpegJpegDecoder_t nvjpeg_decoder;

    nvjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
    nvjpegJpegStream_t  jpeg_streams[2]; //  2 streams for pipelining

    // nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    
    /* Retrieve the componenet and size info. */
    int nComponent = 0;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    cudaEvent_t ev_start = NULL, ev_end = NULL;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_end));

    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
        nvjpeg_image.channel[c] = NULL;
        nvjpeg_image.pitch[c] = 0;
        nvjpeg_output_size.pitch[c] = 0;
    }

    CHECK_NVJPEG(nvjpegCreate(backend , nullptr , &nvjpeg_handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));

    CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle, backend, &nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state));   
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));

    CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(nvjpeg_handle, NULL, &device_buffer));
    std::cout << "obuffer size : " << obuffer.size() << std::endl;
    CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle , (const unsigned char *)obuffer.data() , obuffer.size() , &nComponent , &subsampling , widths , heights));
    std::cout << "[INFO] NvjpegGetImageInfo Image Channels is " << nComponent << std::endl;
    for (int c = 0; c < nComponent; c++) {
        std::cout << "[INFO] Channel #" << c << " size: " << widths[c] << " x " << heights[c] << std::endl;
    }

    // in the case of interleaved RGB output, write only to single channel, but 3 samples at once
    int mul = 1;
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
    {
        nComponent = 1;
        mul = 3;
        is_interleaved = true;
    }else if (format == NVJPEG_OUTPUT_RGB || format == NVJPEG_OUTPUT_BGR) 
    {
        widths[1] = widths[2] = widths[0];
        heights[1] = heights[2] = heights[0];
        is_interleaved = false;
    }

    // realloc output buffer if required
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

    std::vector<const unsigned char*> batched_bitstreams;
    std::vector<size_t> batched_bitstreams_size;
    std::vector<nvjpegImage_t>  batched_output;

    // bit-streams that batched decode cannot handle
    std::vector<const unsigned char*> otherdecode_bitstreams;
    std::vector<size_t> otherdecode_bitstreams_size;
    std::vector<nvjpegImage_t> otherdecode_output;

    if (hw_decode_available)
    {
        std::cout << "[INFO] UnSupported hw_decode_available mode" << std::endl;
    }else
    {
        otherdecode_bitstreams.push_back((const unsigned char *)obuffer.data());
        otherdecode_bitstreams_size.push_back(obuffer.size());
        otherdecode_output.push_back(nvjpeg_image);
    }

    CHECK_CUDA(cudaEventRecord(ev_start));

    if(batched_bitstreams.size() > 0)
    {
        CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(nvjpeg_handle, nvjpeg_state, \
                            batched_bitstreams.size(), 1, format));

        CHECK_NVJPEG(nvjpegDecodeBatched(nvjpeg_handle, nvjpeg_state, batched_bitstreams.data(),
            batched_bitstreams_size.data(), batched_output.data(), NULL));
    }
    if(otherdecode_bitstreams.size() > 0)
    {
        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
        int buffer_index = 0;
        CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params, format));
        for (int i = 0; i < batch_size; i++) {
            CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle, otherdecode_bitstreams[i], \
                            otherdecode_bitstreams_size[i], 0, 0, jpeg_streams[buffer_index]));
            CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, \
                            pinned_buffers[buffer_index]));
            CHECK_NVJPEG(nvjpegDecodeJpegHost(nvjpeg_handle, nvjpeg_decoder, nvjpeg_decoupled_state, \
                            nvjpeg_decode_params, jpeg_streams[buffer_index]));
            // CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle, nvjpeg_decoder, \
                            nvjpeg_decoupled_state , jpeg_streams[buffer_index], NULL));
            // switch pinned buffer in pipeline mode to avoid an extra sync
            buffer_index = 1 - buffer_index; 

            CHECK_NVJPEG(nvjpegDecodeJpegDevice(nvjpeg_handle, nvjpeg_decoder, nvjpeg_decoupled_state, \
                            &otherdecode_output[i], NULL));
        }
    }

    CHECK_CUDA(cudaEventRecord(ev_end));

    float ms = 0.0;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    std::cout << "=> Decode Cost time : " << ms << "ms" << std::endl;
    if (saveBMP)
    {
        for (int i = 0; i < batch_size; i++) {
            std::string filenames = "D:\\OroChiLab\\Nvjpeg-ImageCompressor\\data\\test\\4K\\8-3.png";
            size_t position = filenames.rfind("\\");
            std::string sFileName = (std::string::npos == position)
                    ? filenames
                    : filenames.substr(position + 1, filenames.size());
            position = sFileName.rfind(".");
            sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(0, position);
            
            std::string output_dir = "D:\\OroChiLab\\Nvjpeg-ImageCompressor\\data\\reconstruct_result";
            
            std::string fname(output_dir + "\\" + sFileName + ".bmp");

            int err;
            if (format == NVJPEG_OUTPUT_RGB || format == NVJPEG_OUTPUT_BGR) {
                err = writeBMP(fname.c_str(), nvjpeg_image.channel[0], nvjpeg_image.pitch[0],
                                nvjpeg_image.channel[1], nvjpeg_image.pitch[1], nvjpeg_image.channel[2],
                                nvjpeg_image.pitch[2], widths[i], heights[i]);
            } else if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI) {
                // Write BMP from interleaved data
                err = writeBMPi(fname.c_str(), nvjpeg_image.channel[0], nvjpeg_image.pitch[0],
                                widths[i], heights[i]);
            }
            if (err) {
                std::cerr << "[ERROR] Cannot write output file: " << fname << std::endl;
            }
            std::cout << "Done writing decoded image to file: " << fname << std::endl;
        }
    }
    
    cv::Mat resultImage;
    for (int i = 0; i < batch_size; i++) {
        if (nvjpeg_image.channel[0] != nullptr)
        {
            /*
                nvjpeg_image.channel数组的通道顺序与OpenCV相反，它采用的是RG（红绿蓝）的顺序
            */
            cv::Mat decodedImage = getCVImage(nvjpeg_image.channel[0], nvjpeg_image.pitch[0], \
                                            nvjpeg_image.channel[1], nvjpeg_image.pitch[1], \
                                            nvjpeg_image.channel[2], nvjpeg_image.pitch[2], widths[i], heights[i]);
            if (decodedImage.empty())
            {
                std::cout << "[ERROR] DecodedImage is empty" << std::endl;
                break;
            }
            resultImage = decodedImage.clone();

        }else
        {
            std::cerr << "[ERROR] JPEG decode failed: Output image channel is null." << std::endl;
        }
    }

    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));  
    CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));

    CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams[1]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer));

    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
    {
      if (nvjpeg_image.channel[c]) {CHECK_CUDA(cudaFree(nvjpeg_image.channel[c]))};
    }

    // CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
    
    return resultImage;
}

cv::Mat NvjpegCompressRunnerImpl::DecodeWorker(FILE *jpeg_file)
{

    fseek(jpeg_file, 0, SEEK_END);
    size_t jpeg_data_size = ftell(jpeg_file);
    rewind(jpeg_file);
    std::cout << "[INFO] JPEG DATA SIZE : " << jpeg_data_size << std::endl;

    unsigned char *jpeg_data = new unsigned char[jpeg_data_size];
    if (!jpeg_data) {
        std::cerr << "[INFO] Memory allocation error." << std::endl;
        fclose(jpeg_file);
        return cv::Mat();
    }

    size_t bytes_read = fread(jpeg_data, 1, jpeg_data_size, jpeg_file);
    fclose(jpeg_file);

    if (bytes_read != jpeg_data_size) {
        std::cerr << "Failed to read JPEG data." << std::endl;
        delete[] jpeg_data;
        return cv::Mat();
    }

    bool is_interleaved = false;
    bool hw_decode_available = false;
    int batch_size = 1;
    bool saveBMP = false;

    nvjpegImage_t nvjpeg_image; // output buffers
    nvjpegImage_t nvjpeg_output_size;  // output buffer sizes, for convenience

    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t nvjpeg_state;
    nvjpegBufferDevice_t device_buffer;

    nvjpegOutputFormat_t format = NVJPEG_OUTPUT_BGR;
    nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegJpegState_t nvjpeg_decoupled_state;
    nvjpegDecodeParams_t nvjpeg_decode_params;
    nvjpegJpegDecoder_t nvjpeg_decoder;

    nvjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
    nvjpegJpegStream_t  jpeg_streams[2]; //  2 streams for pipelining
    int nComponent = 0;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    cudaEvent_t ev_start = NULL, ev_end = NULL;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_end));

    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
        nvjpeg_image.channel[c] = NULL;
        nvjpeg_image.pitch[c] = 0;
        nvjpeg_output_size.pitch[c] = 0;
    }

    CHECK_NVJPEG(nvjpegCreate(backend , nullptr , &nvjpeg_handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));

    CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle, backend, &nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state));   
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));

    CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(nvjpeg_handle, NULL, &device_buffer));

    CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle , jpeg_data , jpeg_data_size , &nComponent , &subsampling , widths , heights));
    std::cout << "[INFO] NvjpegGetImageInfo Image Channels is " << nComponent << std::endl;
    for (int c = 0; c < nComponent; c++) {
        std::cout << "[INFO] Channel #" << c << " size: " << widths[c] << " x " << heights[c] << std::endl;
    }
    // in the case of interleaved RGB output, write only to single channel, but 3 samples at once
    int mul = 1;
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
    {
        nComponent = 1;
        mul = 3;
        is_interleaved = true;
    }else if (format == NVJPEG_OUTPUT_RGB || format == NVJPEG_OUTPUT_BGR) 
    {
        widths[1] = widths[2] = widths[0];
        heights[1] = heights[2] = heights[0];
        is_interleaved = false;
    }

    // realloc output buffer if required
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

    std::vector<const unsigned char*> batched_bitstreams;
    std::vector<size_t> batched_bitstreams_size;
    std::vector<nvjpegImage_t>  batched_output;

    // bit-streams that batched decode cannot handle
    std::vector<const unsigned char*> otherdecode_bitstreams;
    std::vector<size_t> otherdecode_bitstreams_size;
    std::vector<nvjpegImage_t> otherdecode_output;

    if (hw_decode_available)
    {
        std::cout << "[INFO] UnSupported hw_decode_available mode" << std::endl;
    }else
    {
        otherdecode_bitstreams.push_back(jpeg_data);
        otherdecode_bitstreams_size.push_back(jpeg_data_size);
        otherdecode_output.push_back(nvjpeg_image);
    }

    CHECK_CUDA(cudaEventRecord(ev_start));

    if(batched_bitstreams.size() > 0)
    {
        CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(nvjpeg_handle, nvjpeg_state, \
                            batched_bitstreams.size(), 1, format));

        CHECK_NVJPEG(nvjpegDecodeBatched(nvjpeg_handle, nvjpeg_state, batched_bitstreams.data(),
            batched_bitstreams_size.data(), batched_output.data(), NULL));
    }
    if(otherdecode_bitstreams.size() > 0)
    {
        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
        int buffer_index = 0;
        CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params, format));
        for (int i = 0; i < batch_size; i++) {
            CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle, otherdecode_bitstreams[i], \
                            otherdecode_bitstreams_size[i], 0, 0, jpeg_streams[buffer_index]));
            CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, \
                            pinned_buffers[buffer_index]));
            CHECK_NVJPEG(nvjpegDecodeJpegHost(nvjpeg_handle, nvjpeg_decoder, nvjpeg_decoupled_state, \
                            nvjpeg_decode_params, jpeg_streams[buffer_index]));
            // CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle, nvjpeg_decoder, \
                            nvjpeg_decoupled_state , jpeg_streams[buffer_index], NULL));
            // switch pinned buffer in pipeline mode to avoid an extra sync
            buffer_index = 1 - buffer_index; 

            CHECK_NVJPEG(nvjpegDecodeJpegDevice(nvjpeg_handle, nvjpeg_decoder, nvjpeg_decoupled_state, \
                            &otherdecode_output[i], NULL));
        }
    }

    CHECK_CUDA(cudaEventRecord(ev_end));

    float ms = 0.0;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    std::cout << "=> Decode Cost time : " << ms << "ms" << std::endl;
    cv::Mat resultImage;
    for (int i = 0; i < batch_size; i++) {
        if (nvjpeg_image.channel[0] != nullptr)
        {
            /*
                nvjpeg_image.channel数组的通道顺序与OpenCV相反，它采用的是RG（红绿蓝）的顺序
            */
            cv::Mat decodedImage = getCVImage(nvjpeg_image.channel[0], nvjpeg_image.pitch[0], \
                                            nvjpeg_image.channel[1], nvjpeg_image.pitch[1], \
                                            nvjpeg_image.channel[2], nvjpeg_image.pitch[2], widths[i], heights[i]);
            if (decodedImage.empty())
            {
                std::cout << "[ERROR] DecodedImage is empty" << std::endl;
                break;
            }
            resultImage = decodedImage.clone();

        }else
        {
            std::cerr << "[ERROR] JPEG decode failed: Output image channel is null." << std::endl;
        }
    }

    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));  
    CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));

    CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams[1]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer));

    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
    {
      if (nvjpeg_image.channel[c]) {CHECK_CUDA(cudaFree(nvjpeg_image.channel[c]))};
    }

    // CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
    
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
        std::cerr << "[ERROR] Exception caught : " << e.what() << '\n';
    }
    
    return image;
}

cv::Mat NvjpegCompressRunnerImpl::Decode(cv::Mat image)
{
    cv::Mat result;
    try
    {
        result = DecodeWorker(image);
    }
    catch(const std::exception& e)
    {
        std::cerr << "[ERROR] Exception caught : " << e.what() << '\n';
    }

    return result;
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

int NvjpegCompressRunnerImpl::Decode(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    try
    {
        for (unsigned int index = 0; index < obuffer_lists.size(); index++)
        {
            decode_result_lists.emplace_back(DecodeWorker(obuffer_lists[index]));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
        
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

int NvjpegCompressRunnerImpl::DecodeImage(std::vector<std::vector<unsigned char>> obuffer_lists)
{
    std::cout << "=> Start image decode ... " << std::endl;   
    if (Decode(obuffer_lists))
    {
        return EXIT_FAILURE;
    }
    
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

/* Debug */

// std::string NvjpegCompressRunnerImpl::nvjpegStatusToString(nvjpegStatus_t status) {
//     switch (status) {
//         case NVJPEG_STATUS_SUCCESS:
//             return "NVJPEG_STATUS_SUCCESS";
//         case NVJPEG_STATUS_NOT_INITIALIZED:
//             return "NVJPEG_STATUS_NOT_INITIALIZED";
//         case NVJPEG_STATUS_INVALID_PARAMETER:
//             return "NVJPEG_STATUS_INVALID_PARAMETER";
//         case NVJPEG_STATUS_BAD_JPEG:
//             return "NVJPEG_STATUS_BAD_JPEG";
//         case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
//             return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
//         case NVJPEG_STATUS_ALLOCATOR_FAILURE:
//             return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
//         case NVJPEG_STATUS_EXECUTION_FAILED:
//             return "NVJPEG_STATUS_EXECUTION_FAILED";
//         case NVJPEG_STATUS_ARCH_MISMATCH:
//             return "NVJPEG_STATUS_ARCH_MISMATCH";
//         case NVJPEG_STATUS_INTERNAL_ERROR:
//             return "NVJPEG_STATUS_INTERNAL_ERROR";
//         case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
//             return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
//         case NVJPEG_STATUS_INCOMPLETE_BITSTREAM:
//             return "NVJPEG_STATUS_INCOMPLETE_BITSTREAM";
//         default:
//             return "Unknown error";
//     }
// }

/* Write Decode */

int NvjpegCompressRunnerImpl::writeBMP(const char *filename, const unsigned char *d_chanR, int pitchR, \
                                       const unsigned char *d_chanG, int pitchG , \
                                       const unsigned char *d_chanB, int pitchB, \
                                       int width, int height) {
    /* 以RGB的格式写入BMP文件 */
    unsigned int headers[13]; // 存储 BMP 文件头信息的数组
    FILE *outfile;
    int extrabytes;  // BMP文件中行的字节数需要是4的倍数，extrabytes用来记录补充的字节数
    int paddedsize; // 调整后的图像数据大小
    int x , y , n; // 循环计数器 
    int red, green, blue; // 存储像素值

    // 分别存储RGB通道的图像数据并将GPU上的图像数据拷贝到主机上的缓冲区中
    std::vector<unsigned char> vchanR(height * width);
    std::vector<unsigned char> vchanG(height * width);
    std::vector<unsigned char> vchanB(height * width);
    unsigned char *chanR = vchanR.data();
    unsigned char *chanG = vchanG.data();
    unsigned char *chanB = vchanB.data();
    CHECK_CUDA(cudaMemcpy2D(chanR, (size_t)width, d_chanR, (size_t)pitchR , \
                    width, height, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy2D(chanG, (size_t)width, d_chanG, (size_t)pitchR , \
                    width, height, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy2D(chanB, (size_t)width, d_chanB, (size_t)pitchR, \
                    width, height, cudaMemcpyDeviceToHost));
    // 计算需要补充的字节数及调整后的图像数据大小
    extrabytes = 4 - ((width * 3) % 4);
    if (extrabytes == 4) extrabytes = 0;
    paddedsize = ((width * 3) + extrabytes) * height;
    std::cout << "[INFO] Extrabytes : " << extrabytes << " Paddedsize : " << paddedsize << std::endl;

    // 设置BMP文件头部信息
    headers[0] = paddedsize + 54;  // bfSize (whole file size)
    headers[1] = 0;                // bfReserved (both)
    headers[2] = 54;               // bfOffbits
    headers[3] = 40;               // biSize
    headers[4] = width;            // biWidth
    headers[5] = height;           // biHeight
    headers[7] = 0;           // biCompression
    headers[8] = paddedsize;  // biSizeImage
    headers[9] = 0;           // biXPelsPerMeter
    headers[10] = 0;          // biYPelsPerMeter
    headers[11] = 0;          // biClrUsed
    headers[12] = 0;          // biClrImportant

    if (!(outfile = fopen(filename, "wb"))) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        return EXIT_FAILURE;
    }

    // 写入 BMP 信息头
    fprintf(outfile, "BM");
    for (n = 0; n <= 5; n++) {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }
    fprintf(outfile, "%c", 1);
    fprintf(outfile, "%c", 0);
    fprintf(outfile, "%c", 24);
    fprintf(outfile, "%c", 0);
    for (n = 7; n <= 12; n++) {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    // 写入像素数据，BMP图像格式从下到上写入
    for (y = height - 1; y >= 0; y--) 
    {
        for (x = 0; x <= width - 1; x++) 
        {
            red = chanR[y * width + x];
            green = chanG[y * width + x];
            blue = chanB[y * width + x];
            // 确保像素值在合法范围内，并以BGR格式写入像素数据
            if (red > 255) red = 255;
            if (red < 0) red = 0;
            if (green > 255) green = 255;
            if (green < 0) green = 0;
            if (blue > 255) blue = 255;
            if (blue < 0) blue = 0;
            fprintf(outfile, "%c", blue);
            fprintf(outfile, "%c", green);
            fprintf(outfile, "%c", red);
        }
        if (extrabytes)  // 写入补充字节
        {
            for (n = 1; n <= extrabytes; n++) {
                fprintf(outfile, "%c", 0);
            }
        }
    }

    fclose(outfile);
    return EXIT_SUCCESS;
}

int NvjpegCompressRunnerImpl::writeBMPi(const char *filename, const unsigned char *d_RGB, int pitch, \
                                        int width, int height) 
{
    /* 以RGB的格式写入BMP文件，更详细注释见writeBMP */
    unsigned int headers[13];
    FILE *outfile;
    int extrabytes , paddedsize;
    int x , y , n;
    int red, green, blue;

    std::vector<unsigned char> vchanRGB(height * width * 3);
    unsigned char *chanRGB = vchanRGB.data();
    CHECK_CUDA(cudaMemcpy2D(chanRGB, (size_t)width * 3, d_RGB, (size_t)pitch,
                                width * 3, height, cudaMemcpyDeviceToHost));

    extrabytes = 4 - ((width * 3) % 4);
    if (extrabytes == 4) extrabytes = 0;
    paddedsize = ((width * 3) + extrabytes) * height;
    std::cout << "[INFO] Extrabytes : " << extrabytes << " Paddedsize : " << paddedsize << std::endl;

    headers[0] = paddedsize + 54;  // bfSize (whole file size)
    headers[1] = 0;                // bfReserved (both)
    headers[2] = 54;               // bfOffbits
    headers[3] = 40;               // biSize
    headers[4] = width;            // biWidth
    headers[5] = height;           // biHeight
    headers[7] = 0;           // biCompression
    headers[8] = paddedsize;  // biSizeImage
    headers[9] = 0;           // biXPelsPerMeter
    headers[10] = 0;          // biYPelsPerMeter
    headers[11] = 0;          // biClrUsed
    headers[12] = 0;          // biClrImportant

    if (!(outfile = fopen(filename, "wb"))) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        return EXIT_FAILURE;
    }

    fprintf(outfile, "BM");
    for (n = 0; n <= 5; n++) {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }
    fprintf(outfile, "%c", 1);
    fprintf(outfile, "%c", 0);
    fprintf(outfile, "%c", 24);
    fprintf(outfile, "%c", 0);
    for (n = 7; n <= 12; n++) {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    
    for (y = height - 1; y >= 0;y--)
    {
        for (x = 0; x <= width - 1; x++) {
            red = chanRGB[(y * width + x) * 3];
            green = chanRGB[(y * width + x) * 3 + 1];
            blue = chanRGB[(y * width + x) * 3 + 2];

            if (red > 255) red = 255;
            if (red < 0) red = 0;
            if (green > 255) green = 255;
            if (green < 0) green = 0;
            if (blue > 255) blue = 255;
            if (blue < 0) blue = 0;
            fprintf(outfile, "%c", blue);
            fprintf(outfile, "%c", green);
            fprintf(outfile, "%c", red);
        }
        if (extrabytes)
        {
            for (n = 1; n <= extrabytes; n++) {
                fprintf(outfile, "%c", 0);
            }
        }
    }

    fclose(outfile);
    return 0;
}

cv::Mat NvjpegCompressRunnerImpl::getCVImage(const unsigned char *d_chanB, int pitchB, \
                                             const unsigned char *d_chanG, int pitchG, \
                                             const unsigned char *d_chanR, int pitchR, \
                                             int width, int height) 
{
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

    for (int y = 0; y < height; y++) 
    {
        for (int x = 0; x < width; x++) 
        {
            cvImage.at<cv::Vec3b>(y, x) = cv::Vec3b(chanB[y * width + x], chanG[y * width + x], chanR[y * width + x]);
        }
    }

    return cvImage;
}