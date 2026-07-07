// main_nv12 - 2.cc 完整版

// D-Robotics *.bin 模型路径
#define MODEL_PATH "/home/sunrise/lqx/my_yolo11/YOLO11-Detect_NCHWRGB/cpp/LIME.bin"

// 视频文件路径
#define VIDEO_PATH "/home/sunrise/lqx/my_yolo11/YOLO11-Detect_NCHWRGB/cpp/IMG_6005_RESIZE_final.mp4"

// 推理结果保存视频路径
#define OUTPUT_VIDEO_PATH "test_LIME.avi"

// 前处理方式选择
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 模型的类别数量
#define CLASSES_NUM 1

// NMS的阈值
#define NMS_THRESHOLD 0.1

// 分数阈值
#define SCORE_THRESHOLD 0.5

// NMS选取的前K个框数
#define NMS_TOP_K 300

// 控制回归部分离散化程度的超参数
#define REG 16

// 绘制标签的字体尺寸
#define FONT_SIZE 1.0

// 绘制标签的字体粗细
#define FONT_THICKNESS 1.0

// 绘制矩形框的线宽
#define LINE_SIZE 2.0

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// Third Party Librarys
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

#define RDK_CHECK_SUCCESS(value, errmsg) \
    do { \
        auto ret_code = value; \
        if (ret_code != 0) { \
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cout << errmsg << ", error code:" << ret_code << std::endl; \
            return ret_code; \
        } \
    } while (0);

// COCO Names
std::vector<std::string> object_names = {
    "rebar", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

int main() {
    // 0. 初始化视频输入输出
    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    // 获取视频属性
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 初始化视频写入器
    cv::VideoWriter video_writer(OUTPUT_VIDEO_PATH, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(frame_width, frame_height));

    // 0. 加载bin模型
    auto model_load_begin = std::chrono::system_clock::now();
    hbPackedDNNHandle_t packed_dnn_handle;
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1), "hbDNNInitializeFromFiles failed");

    // 打印相关版本信息
    std::cout << "[INFO] OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "[INFO] MODEL_PATH: " << MODEL_PATH << std::endl;
    std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
    std::cout << "[INFO] NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
    std::cout << "[INFO] SCORE_THRESHOLD: " << SCORE_THRESHOLD << std::endl;

    // 模型信息检查
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle), "hbDNNGetModelNameList failed");

    const char *model_name = model_name_list[0];
    std::cout << "[model name]: " << model_name << std::endl;

    hbDNNHandle_t dnn_handle;
    RDK_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name), "hbDNNGetModelHandle failed");

    // 模型输入检查
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "hbDNNGetInputCount failed");

    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0), "hbDNNGetInputTensorProperties failed");

    if (input_count > 1) {
        std::cout << "Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    if (input_properties.tensorType != HB_DNN_IMG_TYPE_NV12) {
        std::cout << "input tensor type is not HB_DNN_IMG_TYPE_NV12, please check!" << std::endl;
        return -1;
    }

    if (input_properties.tensorLayout != HB_DNN_LAYOUT_NCHW) {
        std::cout << "input tensor layout is not HB_DNN_LAYOUT_NCHW, please check!" << std::endl;
        return -1;
    }

    int32_t input_H = input_properties.validShape.dimensionSize[2];
    int32_t input_W = input_properties.validShape.dimensionSize[3];
    std::cout << "input tensor valid shape: (" << input_properties.validShape.dimensionSize[0] << ", "
              << input_properties.validShape.dimensionSize[1] << ", " << input_H << ", " << input_W << ")" << std::endl;

    // 模型输出检查
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle), "hbDNNGetOutputCount failed");

    if (output_count != 6) {
        std::cout << "Your Model's outputs num is not 6, please check!" << std::endl;
        return -1;
    }

    int order[6] = {0, 1, 2, 3, 4, 5};
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    int32_t order_we_want[6][3] = {
        {H_8, W_8, CLASSES_NUM}, {H_8, W_8, 64}, 
        {H_16, W_16, CLASSES_NUM}, {H_16, W_16, 64}, 
        {H_32, W_32, CLASSES_NUM}, {H_32, W_32, 64}
    };
    
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, j), "hbDNNGetOutputTensorProperties failed");
            int32_t h = output_properties.validShape.dimensionSize[1];
            int32_t w = output_properties.validShape.dimensionSize[2];
            int32_t c = output_properties.validShape.dimensionSize[3];
            if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2]) {
                order[i] = j;
                break;
            }
        }
    }

    if (order[0] + order[1] + order[2] + order[3] + order[4] + order[5] != 0 + 1 + 2 + 3 + 4 + 5) {
        std::cout << "Outputs order check FAILED, use default" << std::endl;
        for (int i = 0; i < 6; i++)
            order[i] = i;
    }

    hbDNNTensor *output = new hbDNNTensor[output_count];
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysMem &mem = output[i].sysMem[0];
        hbSysAllocCachedMem(&mem, out_aligned_size);
    }

    std::cout << "Model loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - model_load_begin).count() << " ms" << std::endl;

    // 主循环处理视频帧
    cv::Mat frame;
    while (true) {
        auto frame_begin = std::chrono::system_clock::now();

        // 读取视频帧
        if (!cap.read(frame)) {
            std::cout << "End of video stream" << std::endl;
            break;
        }

        // ------------------- 预处理 -------------------
        float y_scale = 1.0, x_scale = 1.0;
        int x_shift = 0, y_shift = 0;
        cv::Mat resize_img;

        if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
            x_scale = std::min(1.0 * input_H / frame.rows, 1.0 * input_W / frame.cols);
            y_scale = x_scale;
            if (x_scale <= 0 || y_scale <= 0) {
                throw std::runtime_error("Invalid scale factor.");
            }

            int new_w = frame.cols * x_scale;
            x_shift = (input_W - new_w) / 2;
            int x_other = input_W - new_w - x_shift;

            int new_h = frame.rows * y_scale;
            y_shift = (input_H - new_h) / 2;
            int y_other = input_H - new_h - y_shift;

            cv::resize(frame, resize_img, cv::Size(new_w, new_h));
            cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
        } else if (PREPROCESS_TYPE == RESIZE_TYPE) {
            cv::resize(frame, resize_img, cv::Size(input_W, input_H));
            y_scale = 1.0 * input_H / frame.rows;
            x_scale = 1.0 * input_W / frame.cols;
            y_shift = 0;
            x_shift = 0;
        }

        // BGR转NV12
        cv::Mat img_nv12;
        cv::Mat yuv_mat;
        cv::cvtColor(resize_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
        uint8_t *yuv = yuv_mat.ptr<uint8_t>();
        img_nv12 = cv::Mat(input_H * 3 / 2, input_W, CV_8UC1);
        uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
        int uv_height = input_H / 2;
        int uv_width = input_W / 2;
        int y_size = input_H * input_W;
        memcpy(ynv12, yuv, y_size);
        uint8_t *nv12 = ynv12 + y_size;
        uint8_t *u_data = yuv + y_size;
        uint8_t *v_data = u_data + uv_height * uv_width;
        for (int i = 0; i < uv_width * uv_height; i++) {
            *nv12++ = *u_data++;
            *nv12++ = *v_data++;
        }

        // ------------------- 推理 -------------------
        hbDNNTensor input;
        input.properties = input_properties;
        hbSysAllocCachedMem(&input.sysMem[0], int(3 * input_H * input_W / 2));
        memcpy(input.sysMem[0].virAddr, ynv12, int(3 * input_H * input_W / 2));
        hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

        hbDNNTaskHandle_t task_handle = nullptr;
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
        hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);
        hbDNNWaitTaskDone(task_handle, 0);
        hbDNNReleaseTask(task_handle);

        // ------------------- 后处理 -------------------
        std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM);
        std::vector<std::vector<float>> scores(CLASSES_NUM);

        // 小目标特征图处理
        if (output[order[0]].properties.quantiType != NONE || output[order[1]].properties.quantiType != SCALE) {
            std::cout << "Quantization type check failed for small object feature map!" << std::endl;
            return -1;
        }

        hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        auto *s_cls_raw = reinterpret_cast<float *>(output[order[0]].sysMem[0].virAddr);
        auto *s_bbox_raw = reinterpret_cast<int32_t *>(output[order[1]].sysMem[0].virAddr);
        auto *s_bbox_scale = reinterpret_cast<float *>(output[order[1]].properties.scale.scaleData);

        for (int h = 0; h < H_8; h++) {
            for (int w = 0; w < W_8; w++) {
                float *cur_s_cls_raw = s_cls_raw;
                int32_t *cur_s_bbox_raw = s_bbox_raw;

                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++) {
                    if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id]) {
                        cls_id = i;
                    }
                }

                float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);
                if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW) {
                    s_cls_raw += CLASSES_NUM;
                    s_bbox_raw += REG * 4;
                    continue;
                }

                float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

                float ltrb[4], sum, dfl;
                for (int i = 0; i < 4; i++) {
                    ltrb[i] = 0.;
                    sum = 0.;
                    for (int j = 0; j < REG; j++) {
                        int index_id = REG * i + j;
                        dfl = std::exp(float(cur_s_bbox_raw[index_id]) * s_bbox_scale[index_id]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                    s_cls_raw += CLASSES_NUM;
                    s_bbox_raw += REG * 4;
                    continue;
                }

                float x1 = (w + 0.5 - ltrb[0]) * 8.0;
                float y1 = (h + 0.5 - ltrb[1]) * 8.0;
                float x2 = (w + 0.5 + ltrb[2]) * 8.0;
                float y2 = (h + 0.5 + ltrb[3]) * 8.0;

                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
            }
        }

        // 中目标特征图处理
        if (output[order[2]].properties.quantiType != NONE || output[order[3]].properties.quantiType != SCALE) {
            std::cout << "Quantization type check failed for medium object feature map!" << std::endl;
            return -1;
        }

        hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[3]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        auto *m_cls_raw = reinterpret_cast<float *>(output[order[2]].sysMem[0].virAddr);
        auto *m_bbox_raw = reinterpret_cast<int32_t *>(output[order[3]].sysMem[0].virAddr);
        auto *m_bbox_scale = reinterpret_cast<float *>(output[order[3]].properties.scale.scaleData);

        for (int h = 0; h < H_16; h++) {
            for (int w = 0; w < W_16; w++) {
                float *cur_m_cls_raw = m_cls_raw;
                int32_t *cur_m_bbox_raw = m_bbox_raw;

                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++) {
                    if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id]) {
                        cls_id = i;
                    }
                }

                if (cur_m_cls_raw[cls_id] < -log(1 / SCORE_THRESHOLD - 1)) {
                    m_cls_raw += CLASSES_NUM;
                    m_bbox_raw += REG * 4;
                    continue;
                }

                float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

                float ltrb[4], sum, dfl;
                for (int i = 0; i < 4; i++) {
                    ltrb[i] = 0.;
                    sum = 0.;
                    for (int j = 0; j < REG; j++) {
                        int index_id = REG * i + j;
                        dfl = std::exp(float(cur_m_bbox_raw[index_id]) * m_bbox_scale[index_id]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                    m_cls_raw += CLASSES_NUM;
                    m_bbox_raw += REG * 4;
                    continue;
                }

                float x1 = (w + 0.5 - ltrb[0]) * 16.0;
                float y1 = (h + 0.5 - ltrb[1]) * 16.0;
                float x2 = (w + 0.5 + ltrb[2]) * 16.0;
                float y2 = (h + 0.5 + ltrb[3]) * 16.0;

                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
            }
        }

        // 大目标特征图处理
        if (output[order[4]].properties.quantiType != NONE || output[order[5]].properties.quantiType != SCALE) {
            std::cout << "Quantization type check failed for large object feature map!" << std::endl;
            return -1;
        }

        hbSysFlushMem(&(output[order[4]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[5]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        auto *l_cls_raw = reinterpret_cast<float *>(output[order[4]].sysMem[0].virAddr);
        auto *l_bbox_raw = reinterpret_cast<int32_t *>(output[order[5]].sysMem[0].virAddr);
        auto *l_bbox_scale = reinterpret_cast<float *>(output[order[5]].properties.scale.scaleData);

        for (int h = 0; h < H_32; h++) {
            for (int w = 0; w < W_32; w++) {
                float *cur_l_cls_raw = l_cls_raw;
                int32_t *cur_l_bbox_raw = l_bbox_raw;

                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++) {
                    if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id]) {
                        cls_id = i;
                    }
                }

                if (cur_l_cls_raw[cls_id] < -log(1 / SCORE_THRESHOLD - 1)) {
                    l_cls_raw += CLASSES_NUM;
                    l_bbox_raw += REG * 4;
                    continue;
                }

                float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

                float ltrb[4], sum, dfl;
                for (int i = 0; i < 4; i++) {
                    ltrb[i] = 0.;
                    sum = 0.;
                    for (int j = 0; j < REG; j++) {
                        int index_id = REG * i + j;
                        dfl = std::exp(float(cur_l_bbox_raw[index_id]) * l_bbox_scale[index_id]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                    l_cls_raw += CLASSES_NUM;
                    l_bbox_raw += REG * 4;
                    continue;
                }

                float x1 = (w + 0.5 - ltrb[0]) * 32.0;
                float y1 = (h + 0.5 - ltrb[1]) * 32.0;
                float x2 = (w + 0.5 + ltrb[2]) * 32.0;
                float y2 = (h + 0.5 + ltrb[3]) * 32.0;

                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
            }
        }

        // NMS处理
        std::vector<std::vector<int>> indices(CLASSES_NUM);
        for (int i = 0; i < CLASSES_NUM; i++) {
            cv::dnn::NMSBoxes(bboxes[i], scores[i], SCORE_THRESHOLD, NMS_THRESHOLD, indices[i], 1.f, NMS_TOP_K);
        }

        // ------------------- 渲染结果 -------------------
        cv::Mat output_frame = frame.clone();
        for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
            for (auto& idx : indices[cls_id]) {
                float x1 = (bboxes[cls_id][idx].x - x_shift) / x_scale;
                float y1 = (bboxes[cls_id][idx].y - y_shift) / y_scale;
                float x2 = x1 + bboxes[cls_id][idx].width / x_scale;
                float y2 = y1 + bboxes[cls_id][idx].height / y_scale;
                float score = scores[cls_id][idx];

                cv::rectangle(output_frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), LINE_SIZE);
                std::string text = object_names[cls_id % CLASSES_NUM] + ": " + std::to_string(static_cast<int>(score * 100)) + "%";
                cv::putText(output_frame, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 255), FONT_THICKNESS, cv::LINE_AA);
            }
        }

        // 写入处理后的帧
        video_writer.write(output_frame);
        cv::imshow("Real-time Detection", output_frame);

        // 释放本帧资源
        hbSysFreeMem(&(input.sysMem[0]));
        for (int i = 0; i < 6; i++)
            hbSysFlushMem(&(output[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        // 打印帧处理时间
        auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - frame_begin);
        std::cout << "Frame processed in: " << frame_time.count() << "ms" << std::endl;
    }

    // ------------------- 资源释放 -------------------
    cap.release();
    video_writer.release();
    for (int i = 0; i < 6; i++)
        hbSysFreeMem(&(output[i].sysMem[0]));
    hbDNNRelease(packed_dnn_handle);

    return 0;
}