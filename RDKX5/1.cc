#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

const std::string INPUT_PATH  = "/home/sunrise/LIME/2.jpg";        // 改成你的输入文件
const std::string OUTPUT_PATH = "/home/sunrise/LIME/2_LIME.jpg";  // 改成你的输出文件
// LIME enhancement function: applies the simplified LIME algorithm to a BGR image.
cv::Mat limeEnhance(const cv::Mat& img, float gamma=0.5f) {
    CV_Assert(!img.empty() && img.type() == CV_8UC3);  // Ensure input is a valid BGR image

    // 1. Convert image to float [0,1] range
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F, 1.0/255.0);

    // Split into B, G, R channels (OpenCV uses BGR order by default)
    std::vector<cv::Mat> channels(3);
    cv::split(imgFloat, channels);
    cv::Mat b = channels[0];
    cv::Mat g = channels[1];
    cv::Mat r = channels[2];

    // 2. Compute initial illumination map (max of R, G, B at each pixel)
    cv::Mat max_rg, illumination;
    cv::max(r, g, max_rg);
    cv::max(max_rg, b, illumination);

    // Create a mask for highlight regions (illumination > 0.8)
    cv::Mat highlightMask;
    cv::compare(illumination, 0.8, highlightMask, cv::CMP_GT);  // 8-bit mask: 255 where illum > 0.8

    // 3. Gamma correction on illumination
    cv::Mat illuminationRefined;
    cv::pow(illumination, gamma, illuminationRefined);  // illumination^gamma for all pixels
    if (cv::countNonZero(highlightMask) > 0) {
        // Further suppress highlights with a stronger gamma
        cv::Mat illumHighlights;
        cv::pow(illumination, gamma * 1.5f, illumHighlights);
        illumHighlights.copyTo(illuminationRefined, highlightMask);  // replace illumRefined where mask is true
    }

    // 4. Smooth the illumination map with a Gaussian blur (3x3 kernel)
    int kernelSize = 3;
    if (kernelSize % 2 == 0) kernelSize++;  // ensure kernel is odd
    cv::GaussianBlur(illuminationRefined, illuminationRefined, cv::Size(kernelSize, kernelSize), 0);

    // Ensure no value is too small (to avoid division blow-up). Clamp illumination to minimum 0.01.
    cv::Mat lowMask;
    cv::compare(illuminationRefined, 0.01, lowMask, cv::CMP_LT);
    illuminationRefined.setTo(0.01, lowMask);

    // 5. Enhance each channel by dividing by the refined illumination and applying channel weights
    const float r_weight = 1.0f, g_weight = 1.0f, b_weight = 1.0f;
    cv::Mat r_enhanced, g_enhanced, b_enhanced;
    cv::divide(r, illuminationRefined, r_enhanced);  // r / illumination
    cv::divide(g, illuminationRefined, g_enhanced);
    cv::divide(b, illuminationRefined, b_enhanced);
    r_enhanced *= r_weight;
    g_enhanced *= g_weight;
    b_enhanced *= b_weight;

    // 6. Merge enhanced channels back to a BGR image
    std::vector<cv::Mat> enhancedChannels = { b_enhanced, g_enhanced, r_enhanced };
    cv::Mat enhancedFloat;
    cv::merge(enhancedChannels, enhancedFloat);

    // 7. Convert back to 8-bit [0,255] and clip values (OpenCV convertTo with saturate cast handles clipping)
    cv::Mat enhanced8U;
    enhancedFloat.convertTo(enhanced8U, CV_8UC3, 255.0);
    return enhanced8U;
}

int main() {
    cv::Mat img = cv::imread(INPUT_PATH, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: 无法读取输入图像: " << INPUT_PATH << std::endl;
        return -1;
    }


    // Apply LIME enhancement
    cv::Mat enhancedImg = limeEnhance(img);

    // Save the enhanced image and output the processing time
    if (!cv::imwrite(OUTPUT_PATH, enhancedImg)) {
        std::cerr << "Error: Failed to write output image \"" << OUTPUT_PATH << "\"\n";
        return 1;
    }
    std::cout << "LIME enhancement completed in " << elapsedMs << " ms." << std::endl;
    return 0;
}
