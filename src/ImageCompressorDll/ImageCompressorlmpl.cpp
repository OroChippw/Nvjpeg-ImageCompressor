#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "ImageCompressorImpl.h"
#include "CompressConfig.h"


int NvjpegCompressRunnerImpl::ReadInput(const std::string input_path)
{   
    std::cout << "=> Start ReadInput and build file lists ..." << std::endl;
    struct stat s;
    int error_code = 1;
    if(stat(input_path.c_str() , &s) == 0)
    {
        if(s.st_mode & S_IFREG)
        {
            files_list.emplace_back(input_path.c_str());
        }else if(s.st_mode & S_IFDIR)
        {
            struct dirent* dir;
            DIR* dir_handle = opendir(input_path.c_str());
            if(dir_handle)
            {
                error_code = 0;
                while((dir = readdir(dir_handle)) != NULL)
                {
                    if(dir->d_type == DT_REG)
                    {
                        std::string filename = input_path + "\\" + dir->d_name;
                        files_list.emplace_back(filename);
                    }else if(dir->d_type == DT_DIR)
                    {
                        std::string sname = dir->d_name;
                        if(sname != "." && sname != "..")
                        {
                            ReadInput(input_path + sname + "\\");
                        }
                    }
                }
                closedir(dir_handle);
            }else
            {
                std::cout << "Can not open input directory : " << input_path << std::endl;
                return EXIT_FAILURE;
            }
        }else
        {
            std::cout << "Cannot find input path " << input_path << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Build file lists successfully ..." << std::endl;
    std::cout << "Files list num : " << files_list.size() << std::endl;
    return EXIT_SUCCESS;
}


int NvjpegCompressRunnerImpl::Compress(CompressConfiguration cfg)
{
    nvjpegHandle_t nvjpeg_handle;
    nvjpegEncoderState_t encoder_state; // 存储用于压缩中间缓冲区和变量的结构体
    nvjpegEncoderParams_t encoder_params; // 存储用于JPEG压缩参数的结构体

    cudaEvent_t ev_start = NULL, ev_end = NULL;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_end));

    nvjpegImage_t input; // 输入图像数据指针为显式指针，每个颜色分量分别存储
    nvjpegInputFormat_t input_format = NVJPEG_INPUT_BGR; // 指定用户提供何种类型的输入进行编码,输入是BGR，编码前会被转换成YCbCr

    // nvjpegBackend_t用来选择运行后端，使用GPU解码baseline JPEG或者使用CPU进行Huffman解码
    nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
    CHECK_NVJPEG(nvjpegCreate(backend, nullptr, &nvjpeg_handle));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle, &encoder_params, NULL)); // 创建保存压缩参数的结构
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL)); // 创建存储压缩中使用的中间缓冲区的编码器状态

    /* 设置编码压缩参数 */
    nvjpegEncoderParamsSetEncoding(encoder_params, nvjpegJpegEncoding_t::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, NULL);
    /*
        nvjpegEncoderParamsSetOptimizedHuffman设置是否使用优化的Huffman编码，
        使用优化的Huffman生成更小的JPEG比特流，质量相同但性能较慢,第二个参数值默认为0不使用优化Huffman编码
    */
    nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, cfg.use_optimizedHuffman , NULL);
    nvjpegEncoderParamsSetQuality(encoder_params, cfg.encode_quality , NULL); // 设置质量参数
    nvjpegEncoderParamsSetSamplingFactors(encoder_params, nvjpegChromaSubsampling_t::NVJPEG_CSS_422 , NULL); // 设置用于JPEG压缩的色度子采样参数，官方默认为NVJPEG_CSS_444

    double time_total = 0.0;
    double psnr_val_score = 0.0; 

    int stage_num = cfg.multi_stage ? 2 : 1;

    std::cout << "# ----------------------------------------------- #" << std::endl;
    for(unsigned int index = 0 ; index < files_list.size() ; index++)
    {
        std::cout << "=> Processing: " << files_list[index] << std::endl;
        std::string stage1_diffmap_path;
        for(unsigned int stage_index = 0 ; stage_index < stage_num ; stage_index++)
        {
            std::string image_path = files_list[index];
            if (stage_index == 1)
            {
                image_path = stage1_diffmap_path;
                std::cout << "=> Enter secondary compression stage 2 ..." << std::endl;
            }
            cv::Mat srcImage = cv::imread(image_path , cv::IMREAD_COLOR);
            const unsigned int image_width = srcImage.cols;
            const unsigned int image_height = srcImage.rows;
            const unsigned int channel_size = image_width * image_height;
            std::cout << "[IMAGE INFO] width : "<< image_width << " height : " << image_height << std::endl;

            std::vector<cv::Mat> channels_list;
            cv::split(srcImage, channels_list);
            for (int i = 0; i < channels_list.size(); i++) 
            {
                input.pitch[i] = image_width;
                CHECK_CUDA(cudaMalloc((void**)&(input.channel[i]), channel_size));
                size_t inputSize = channels_list[i].total() * channels_list[i].elemSize();
                CHECK_CUDA(cudaMemcpy((void*)input.channel[i], channels_list[i].ptr(0), inputSize, cudaMemcpyHostToDevice));
            }

            CHECK_CUDA(cudaEventRecord(ev_start));
            CHECK_NVJPEG(nvjpegEncodeImage(nvjpeg_handle, encoder_state, encoder_params, &input, input_format, 
                image_width, image_height, NULL));
            CHECK_CUDA(cudaEventRecord(ev_end));

            std::vector<unsigned char> obuffer;
            size_t length;
            /* 从先前在其中一个编码器功能中使用的编码器状态中检索压缩流，如果数据参数data为NULL，则编码器将在长度参数中返回压缩流大小 */
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, NULL, &length, NULL));
            /* 先返回压缩流的长度 再用压缩流长度大小的buffer接受压缩流 */

            obuffer.resize(length);
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, obuffer.data(), &length, NULL));

            cudaEventSynchronize(ev_end);

            for (int i = 0; i < channels_list.size(); i++) 
            {
                cudaFree(input.channel[i]);
            }

            float ms = 0.0;
            cudaEventElapsedTime(&ms, ev_start, ev_end);
            std::cout << "=> Cost time : " << ms << "ms" << std::endl;
            time_total += ms;

            std::string::size_type iPos = files_list[index].find_last_of('\\') + 1;

            std::string filename = files_list[index].substr(iPos, files_list[index].length() - iPos);
            std::string name = filename.substr(0, filename.find("."));
            std::string savedir = cfg.output_dir + "\\" + name;
            if (!std::filesystem::exists(savedir))
            {
                std::filesystem::create_directories(savedir);
            }
            std::string compress_result_name = (stage_index != 0) ? "D.png" : "B.png";
            std::string output_result_path = savedir + "\\" + compress_result_name ;
            std::ofstream outputFile(output_result_path, std::ios::out | std::ios::binary);
            outputFile.write(reinterpret_cast<const char*>(obuffer.data()), static_cast<int>(length));
            outputFile.close();
            std::cout << "Save compress result as : " << output_result_path << std::endl;

            if (stage_index != 0) {continue;} // 当进行二次压缩时不再需要计算差异图

            cv::Mat compressImage = cv::imread(output_result_path , cv::IMREAD_ANYCOLOR);
            /* 计算输入图和压缩图的峰值信噪比 */
            psnr_val_score += CalculateDiffImagePSNR(files_list[index], output_result_path);
            cv::Mat diffmap = CalculateDiffmap(files_list[index] ,output_result_path , cfg.show_diff_info);
            std::string output_diffmap_path = savedir + "\\" + "C.png" ;
            cv::imwrite(output_diffmap_path , diffmap);
            stage1_diffmap_path = output_diffmap_path;
            std::cout << "Save diffmap result as : " << output_diffmap_path << std::endl;
        }
    }

    std::cout << "# ----------------------------------------------- #" << std::endl;
    std::cout << files_list.size() << " Images mean Cost time : " << time_total / files_list.size() << "ms" << std::endl;
    std::cout << files_list.size() << " Images mean PSNR  : " << psnr_val_score / files_list.size() << "dB" << std::endl;

    nvjpegEncoderParamsDestroy(encoder_params);
    nvjpegEncoderStateDestroy(encoder_state);
    nvjpegDestroy(nvjpeg_handle);
    
    return EXIT_SUCCESS;
}

cv::Mat NvjpegCompressRunnerImpl::CalculateDiffmap(const std::string srcImagePath , const std::string compImagePath , bool showinfo)
{
    cv::Mat srcImage = cv::imread(srcImagePath, cv::IMREAD_ANYCOLOR);
    cv::Mat compressImage = cv::imread(compImagePath, cv::IMREAD_ANYCOLOR);
    cv::Mat diffMap = srcImage - compressImage;

    if(showinfo)
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
        std::cout << "diffMap channelsMean[0] : " << channelsMean[0] << std::endl; 
        std::cout << "diffMap channelsMean[1] : " << channelsMean[1] << std::endl;
        std::cout << "diffMap channelsMean[2] : " << channelsMean[2] << std::endl;

        cv::Mat meanMat, stddevMat;
        cv::meanStdDev(diffMap, meanMat, stddevMat);
        std::cout << "diffMap MeanMat : " << meanMat << std::endl;
        std::cout << "diffMap stddevMat : " << stddevMat << std::endl;
    }
    
    return diffMap;
}

double NvjpegCompressRunnerImpl::CalculatePSNR(cv::Mat srcImage , cv::Mat compImage)
{
    const unsigned int w = srcImage.cols;
    const unsigned int h = srcImage.rows;
    const unsigned int max = 255;
    cv::Mat subImage;
    cv::absdiff(srcImage , compImage , subImage);
    subImage = subImage.mul(subImage);
    cv::Scalar sumScalar = sum(subImage);
    double sse = sumScalar.val[0] + sumScalar[1] + sumScalar[2];
    if(sse <= 1e-10)
    {
        return 0;
    }else
    {
        double mse = sse / h / w;
        double psnr = 10 * log10(pow(max , 2) / mse);
        std::cout << "[VAL->MSE] : " << mse << " [VAL->PSNR] : " << psnr << std::endl;
        return psnr;
    }
}

cv::Mat ReconstructedImage(const cv::Mat& Image1 , const cv::Mat& Image2)
{
    cv::Mat ConstructedImage = Image1 + Image2;
    return ConstructedImage;
}

int NvjpegCompressRunnerImpl::CompressImage(CompressConfiguration cfg)
{
    int read_state = ReadInput(cfg.input_dir);
    std::cout << "=> Start image compression ... " <<std::endl;
    if(Compress(cfg))
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

double NvjpegCompressRunnerImpl::CalculateDiffImagePSNR(const std::string ImagePath1 , const std::string ImagePath2)
{
    cv::Mat image1 = cv::imread(ImagePath1 , cv::IMREAD_ANYCOLOR);
    cv::Mat image2 = cv::imread(ImagePath2 , cv::IMREAD_ANYCOLOR);

    double psnr = CalculatePSNR(image1 , image2);

    return psnr;
}