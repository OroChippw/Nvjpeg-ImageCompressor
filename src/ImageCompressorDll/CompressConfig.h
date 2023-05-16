#include <iostream>

enum COMPRESS_RESULT_CODE : int
{

};

typedef enum
{
    NVJPEG_CSS_444 = 0,
    NVJPEG_CSS_422 = 1,
    NVJPEG_CSS_420 = 2,
    NVJPEG_CSS_440 = 3,
    NVJPEG_CSS_411 = 4,
    NVJPEG_CSS_410 = 5,
    NVJPEG_CSS_GRAY = 6,
    NVJPEG_CSS_UNKNOWN = -1
}nvjpegChromaSubsampling_t;

struct Configuration {
    std::string input_dir;
    std::string output_dir;
    int width = 8320;
    int hegiht = 40000;

    int encode_quality = 95; // 编码器质量参数
    bool use_optimizedHuffman = true; // 是否使用优化Huffman编码
    bool multi_stage = bool; // 是否启用二次压缩
};