#pragma once
#include <iostream>

struct CompressConfiguration {
    std::string input_dir;
    std::string output_dir;
    int width = 8320;
    int hegiht = 40000;

    int encode_quality = 95; 
    bool use_optimizedHuffman = true; 
    bool multi_stage = false;
};