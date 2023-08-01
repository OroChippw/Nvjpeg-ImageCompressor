# pragma warning (disable:4819)
# pragma warning (disable:4996)

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
    if (stat(input_path.c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFREG)
        {
            files_list.emplace_back(input_path.c_str());
        }
        else if (s.st_mode & S_IFDIR)
        {
            struct dirent* dir;
            DIR* dir_handle = opendir(input_path.c_str());
            if (dir_handle)
            {
                error_code = 0;
                while ((dir = readdir(dir_handle)) != NULL)
                {
                    if (dir->d_type == DT_REG)
                    {
                        std::string filename = input_path + "\\" + dir->d_name;
                        files_list.emplace_back(filename);
                    }
                    else if (dir->d_type == DT_DIR)
                    {
                        std::string sname = dir->d_name;
                        if (sname != "." && sname != "..")
                        {
                            ReadInput(input_path + sname + "\\");
                        }
                    }
                }
                closedir(dir_handle);
            }
            else
            {
                std::cout << "Can not open input directory : " << input_path << std::endl;
                return EXIT_FAILURE;
            }
        }
        else
        {
            std::cout << "Cannot find input path " << input_path << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Build file lists successfully ..." << std::endl;
    std::cout << "Files list num : " << files_list.size() << std::endl;
    return EXIT_SUCCESS;
}

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

std::vector<unsigned char> NvjpegCompressRunnerImpl::CompressWorker(CompressConfiguration cfg, const cv::Mat Image)
{
    const unsigned int image_width = Image.cols;
    const unsigned int image_height = Image.rows;
    const unsigned int channel_size = image_width * image_height;
    std::cout << "[COMPRESS IMAGE INFO] width : " << image_width << " height : " << image_height << std::endl;

    /* nvjpeg init*/
    nvjpegHandle_t nvjpeg_handle;
    nvjpegEncoderState_t encoder_state; // 存储用于压缩中间缓冲区和变量的结构体
    nvjpegEncoderParams_t encoder_params; // 存储用于JPEG压缩参数的结构体

    nvjpegInputFormat_t input_format = NVJPEG_INPUT_BGR; // 指定用户提供何种类型的输入进行编码,输入是BGR，编码前会被转换成YCbCr

    // nvjpegBackend_t用来选择运行后端，使用GPU解码baseline JPEG或者使用CPU进行Huffman解码
    nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;

    cudaEvent_t ev_start = NULL, ev_end = NULL;

    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_end));

    CHECK_NVJPEG(nvjpegCreate(backend, nullptr, &nvjpeg_handle));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle, &encoder_params, NULL)); // 创建保存压缩参数的结构
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL)); // 创建存储压缩中使用的中间缓冲区的编码器状态

    /* 设置编码压缩参数 */
    nvjpegEncoderParamsSetEncoding(encoder_params, nvjpegJpegEncoding_t::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, NULL);
    /*
        nvjpegEncoderParamsSetOptimizedHuffman设置是否使用优化的Huffman编码，
        使用优化的Huffman生成更小的JPEG比特流，质量相同但性能较慢,第二个参数值默认为0不使用优化Huffman编码
    */
    nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, cfg.use_optimizedHuffman, NULL);
    nvjpegEncoderParamsSetQuality(encoder_params, cfg.encode_quality, NULL); // 设置质量参数
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
    CHECK_NVJPEG(nvjpegEncodeImage(nvjpeg_handle, encoder_state, encoder_params, &input, input_format,
        image_width, image_height, NULL));
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
    std::cout << "=> Cost time : " << ms << "ms" << std::endl;
    time_total += ms;

    nvjpegEncoderParamsDestroy(encoder_params);
    nvjpegEncoderStateDestroy(encoder_state);
    nvjpegDestroy(nvjpeg_handle);

    return obuffer;
}

int NvjpegCompressRunnerImpl::Compress(CompressConfiguration cfg)
{
    int stage_num = cfg.multi_stage ? 2 : 1;
    std::cout << "# ----------------------------------------------- #" << std::endl;
    for (unsigned int index = 0; index < files_list.size(); index++)
    {
        std::cout << "=> Processing: " << files_list[index] << std::endl;
        std::string::size_type iPos = files_list[index].find_last_of('\\') + 1;
        std::string filename = files_list[index].substr(iPos, files_list[index].length() - iPos);
        std::string name = filename.substr(0, filename.find("."));
        std::string savedir = cfg.output_dir + "\\" + name;
        if (!std::filesystem::exists(savedir) && !cfg.in_memory)
        {
            std::filesystem::create_directories(savedir);
        }

        cv::Mat diffmap;
        time_total = 0.0;
        psnr_val_score = 0.0;

        for (int stage_index = 0; stage_index < stage_num; stage_index++)
        {
            cv::Mat srcImage;
            if (stage_index == 0)
            {
                srcImage = cv::imread(files_list[index], cv::IMREAD_COLOR);
                std::cout << "=> Enter secondary compression stage 1 ..." << std::endl;
            }
            else
            {
                srcImage = diffmap;
                std::cout << "=> Enter secondary compression stage 2 ..." << std::endl;
            }

            std::vector<cv::Mat> image_lists;
            if (cfg.do_crop)
            {
                image_lists = CropImage(srcImage, cfg.crop_ratio);
            }
            else
            {
                image_lists.emplace_back(srcImage);
            }

            std::vector<std::vector<unsigned char>> obuffer_lists;
            for (unsigned int index = 0; index < image_lists.size(); index++)
            {
                obuffer_lists.push_back(CompressWorker(cfg, image_lists[index]));
            }

            if (cfg.in_memory)
            {
                std::cout << "Return Obuffer lists" << std::endl;
                return EXIT_SUCCESS;
            }

            std::string output_result_path;
            std::vector<std::string> result_path_lists;

            for (unsigned int index = 0; index < obuffer_lists.size(); index++)
            {
                if (cfg.save_mat)
                {
                    output_result_path = savedir + "\\" + std::to_string(index) + ".png";
                    std::ofstream outputFile(output_result_path, std::ios::out | std::ios::binary);
                    outputFile.write(reinterpret_cast<const char*>(obuffer_lists[index].data()), static_cast<int>(obuffer_lists[index].size()));
                    outputFile.close();
                    // std::cout << "Save compress mat result as : " << output_result_path << std::endl;
                    result_path_lists.emplace_back(output_result_path);
                }
                
                if (cfg.save_binary)
                {
                    std::string output_result_bin_path = savedir + "\\" + std::to_string(index) + ".bin";
                    std::ofstream outputBinFile(output_result_bin_path, std::ios::out | std::ios::binary);
                    outputBinFile.write(reinterpret_cast<const char*>(obuffer_lists[index].data()), static_cast<int>(obuffer_lists[index].size()));
                    outputBinFile.close();
                    // std::cout << "Save compress bin result as : " << output_result_bin_path << std::endl;
                }
            }

            if (stage_index != 0)
            {
                if (!cfg.save_mat)
                {   
                    std::cout << "cfg.save_mat : " <<  cfg.save_mat << std::endl;
                    for (auto file : result_path_lists)
                    {
                        std::string remove_status = (remove(file.c_str()) == 0) ? "Successfully" : "Failure" ;
                        std::cout << "Delete file " << file << " " << remove_status << std::endl;
                    }
                }
                continue; /* 当进行二次压缩时不再需要计算差异图 */
            }

            /* 计算输入图和压缩图的峰值信噪比 */
            if (cfg.do_val)
            {
                for (unsigned int index = 0; index < result_path_lists.size(); index++)
                {
                    psnr_val_score += CalculateDiffImagePSNR(image_lists[index], result_path_lists[index]);
                }
            }

            // for (unsigned int index = 0 ; index < result_path_lists.size() ; index++)
            // {
            //     diffmap = CalculateDiffmap(cfg , files_list[index] , output_result_path);
            // }

           if (!cfg.save_mat)
            { 
                // std::cout << "cfg.save_mat : " <<  cfg.save_mat << std::endl;
                for (auto file : result_path_lists)
                {
                    std::string remove_status = (remove(file.c_str()) == 0) ? "Successfully" : "Failure" ;
                    std::cout << "Delete file " << file << " " << remove_status << std::endl;
                }
            }
        }
    }

    std::cout << "# ----------------------------------------------- #" << std::endl;
    std::cout << files_list.size() << " Images mean Cost time : " << time_total / files_list.size() << "ms" << std::endl;
    std::cout << files_list.size() << " Images mean PSNR  : " << psnr_val_score / files_list.size() << "dB" << std::endl;

    return EXIT_SUCCESS;
}

cv::Mat NvjpegCompressRunnerImpl::CalculateDiffmap(CompressConfiguration cfg, const cv::Mat srcImage, const std::string compImagePath)
{
    cv::Mat compressImage = cv::imread(compImagePath, cv::IMREAD_ANYCOLOR);
    cv::Mat diffMap = srcImage - compressImage;

    if (cfg.show_diff_info)
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

    if (cfg.use_roi)
    {
        double avgGray_crop = 0.0;
        double stddevGray_crop = 0.0;
        double avgGray_compress = 0.0;
        double stddevGray_compress = 0.0;
        cv::Mat srcImage_roi = srcImage(cfg.roi_rect);
        cv::Mat compressImage_roi = compressImage(cfg.roi_rect);

        CalculateGrayAvgStdDev(srcImage_roi, avgGray_crop, stddevGray_crop);
        std::cout << "avgGray_crop : " << avgGray_crop << std::endl;
        std::cout << "stddevGray_crop : " << stddevGray_crop << std::endl;

        CalculateGrayAvgStdDev(compressImage_roi, avgGray_compress, stddevGray_compress);
        std::cout << "avgGray_compress : " << avgGray_compress << std::endl;
        std::cout << "stddevGray_compress : " << stddevGray_compress << std::endl;

        std::cout << "avgGray_diff : " << abs(avgGray_compress - avgGray_crop) << std::endl;
        std::cout << "stddevGray_diff : " << abs(stddevGray_compress - stddevGray_crop) << std::endl;

        double cropPSNR = CalculatePSNR(srcImage_roi, compressImage_roi);
        std::cout << "cropPSNR : " << cropPSNR << std::endl;
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

void NvjpegCompressRunnerImpl::CalculateGrayAvgStdDev(cv::Mat& src, double& avg, double& stddev)
{
    cv::Mat img, mean, stdDev;
    if (src.channels() == 3)
        cv::cvtColor(src, img, cv::COLOR_BGR2GRAY);
    else
        img = src;
    cv::mean(src);
    cv::meanStdDev(img, mean, stdDev);

    avg = mean.ptr<double>(0)[0];
    stddev = stdDev.ptr<double>(0)[0];
}

cv::Mat NvjpegCompressRunnerImpl::Reconstructed(cv::Mat Image1, cv::Mat Image2)
{
    return Image1 + Image2;
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
    for (int index = 0 ; index < min(width_factor.size() , height_factor.size()); index++)
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

int NvjpegCompressRunnerImpl::CompressImage(CompressConfiguration cfg)
{
    int read_state = ReadInput(cfg.input_dir);
    std::cout << "=> Start image compression ... " << std::endl;
    if (cfg.do_crop)
    {
        if (!((cfg.width % cfg.crop_ratio == 0) && (cfg.height % cfg.crop_ratio == 0)))
        {
            std::cout << "The width and height of the image must be divisible by the number of blocks" << std::endl;
            std::cout << "=> Calculate the greatest common factor of width and height" << std::endl;
            cfg.crop_ratio = CalculateGreatestFactor(cfg.width , cfg.height);
        }
    }
    auto only_compress_startTime = std::chrono::steady_clock::now();
    if (Compress(cfg))
    {
        return EXIT_FAILURE;
    }
    auto only_compress_endTime = std::chrono::steady_clock::now();
    auto only_compress_elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(only_compress_endTime - only_compress_startTime).count();
    if (cfg.in_memory)
    {
        std::cout << "=> Work in memory stream" << std::endl;
    }
    std::cout << "only Compress func cost time: " << only_compress_elapsedTime << " ms" << std::endl;

    return EXIT_SUCCESS;
}

double NvjpegCompressRunnerImpl::CalculateDiffImagePSNR(const cv::Mat image1, const std::string ImagePath2)
{
    cv::Mat image2 = cv::imread(ImagePath2, cv::IMREAD_ANYCOLOR);
    double psnr = CalculatePSNR(image1, image2);

    return psnr;
}

cv::Mat NvjpegCompressRunnerImpl::Binaryfile2Mat(CompressConfiguration cfg, std::string ImagePath)
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


bool cmp(const std::string& str1, const std::string& str2)
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

cv::Mat NvjpegCompressRunnerImpl::MergeBinImage(CompressConfiguration cfg, std::vector<std::string> bin_files)
{
    unsigned int after_crop_width = cfg.width / cfg.crop_ratio;
    unsigned int after_crop_height = cfg.height / cfg.crop_ratio;

    std::sort(bin_files.begin(), bin_files.end(), cmp);
    std::vector<cv::Mat> image_list;
    // auto startTime = std::chrono::steady_clock::now();
    for (int i = 0; i < bin_files.size(); i++)
    {
        image_list.emplace_back(Binaryfile2Mat(cfg, bin_files[i]));
    }
    // auto endTime = std::chrono::steady_clock::now();
    // auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    // std::cout << "Binaryfile2Mat func cost time: " << elapsedTime << " s" << std::endl;

    int sum_num = cfg.crop_ratio * cfg.crop_ratio;
    int index = 0, index_x = 0, index_y = 0;
    cv::Mat resultImage = cv::Mat::zeros(cfg.height, cfg.width, image_list[0].type());

    for (int i = 1; i <= cfg.crop_ratio; i++)
    {
        for (int j = 1; j <= cfg.crop_ratio; j++)
        {
            image_list[index].copyTo(resultImage(cv::Rect(index_x, index_y, image_list[index].cols, image_list[index].rows)));
            index_y += (after_crop_height);
            index += 1;
        }
        index_x += (after_crop_width);
        index_y = 0;
    }
   
    return resultImage;
}


cv::Mat NvjpegCompressRunnerImpl::ReconstructedImage(CompressConfiguration cfg, std::string ImageDirPath)
{
    std::cout << "=> Start image reconstruction ... " << std::endl;
    cv::Mat resultImage;
    struct stat buffer;
    if (!((stat(ImageDirPath.c_str(), &buffer) == 0)))
        return resultImage;
    if (cfg.do_crop)
    {
        if (!((cfg.width % cfg.crop_ratio == 0) && (cfg.height % cfg.crop_ratio == 0)))
        {
            std::cout << "The width and height of the image must be divisible by the number of blocks" << std::endl;
            std::cout << "=> Calculate the greatest common factor of width and height" << std::endl;
            cfg.crop_ratio = CalculateGreatestFactor(cfg.width , cfg.height);
        }
    }
    std::vector<std::string> bin_files;
    std::vector<cv::String> images_files;
    // std::string image_jpg_path = ImageDirPath + "//*.jpg";
    std::string image_bin_path = ImageDirPath + "//*.bin";
    // cv::glob(image_jpg_path, images_files);
    cv::glob(image_bin_path, bin_files);

    // auto startTime = std::chrono::steady_clock::now();

    if (bin_files.size() != 0)
    {
        resultImage = MergeBinImage(cfg, bin_files);
    }
    // auto endTime = std::chrono::steady_clock::now();
    // auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    // std::cout << "MergeBinImage func cost time: " << elapsedTime << " s" << std::endl;


    std::string output_result_path = cfg.rebuild_dir + "\\E.png";
    // auto startTime_1 = std::chrono::steady_clock::now();
    // cv::imwrite(output_result_path, resultImage);
    // auto endTime_1 = std::chrono::steady_clock::now();
    // auto elapsedTime_1 = std::chrono::duration_cast<std::chrono::seconds>(endTime_1 - startTime_1).count();
    // std::cout << "imwrite func cost time: " << elapsedTime_1 << " s" << std::endl;

    std::cout << "Save reconstructed mat result as : " << output_result_path << std::endl;

    return resultImage;
}