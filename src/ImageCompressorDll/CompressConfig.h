#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

struct CompressConfiguration {
    std::string input_dir;
    std::string output_dir;
    std::string rebuild_dir;

    int width = 8320;
    int height = 40000;

    int encode_quality = 95; 
    bool use_optimizedHuffman = true; 
    bool multi_stage = false;
    bool show_diff_info = false;
    bool save_mat = false;
    bool save_binary = true;
    bool do_val = false;

    bool use_roi = false;
    int roi_w = 10;
    int roi_h = 10;

    bool do_crop = false;
    int crop_ratio = 0;

    cv::Rect roi_rect;
    
};