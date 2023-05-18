#pragma once
#include <iostream>

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
};