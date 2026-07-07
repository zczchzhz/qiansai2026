#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本脚本从你给出的完整代码中，抽取了专门用于实现“demo5-limeblock_deeprft”增强逻辑的函数，
且不使用临时目录来读写文件。只需在同目录下放置一张名为 "demo5.jpg" 的图片，运行后会输出
"demo5-limeblock_deeprft.jpg"。
"""

# ==================== 相关 import ====================
import numpy as np
import os
import sys
import traceback
import cv2


# ==================== 1. 水下去模糊函数实现 ====================


def underwater_deblur(img, method='deeprft', strength=0.8):
    """
    水下图像去模糊函数，支持'untv'方法。

    参数:
        img: 输入BGR图像
        method: 去模糊方法('deeprft')
        strength: 去模糊强度，范围[0,1]
    """

    adjusted_strength = strength * 0.6

    print(f"水下去模糊(优化版): 方法={method}, 原始强度={strength}, 调整后强度={adjusted_strength:.2f}")

    if method == 'deeprft':
        # 简化版 DeepRFT：做频域高通 + USM锐化
        img_float = img.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]
        channels = []

        for c in range(img_float.shape[2]):
            channel = img_float[:, :, c]
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)

            # 创建高斯高通
            crow, ccol = h // 2, w // 2
            mask = np.ones((h, w), np.float32)
            r = min(30, min(h, w) // 10)
            x, y = np.ogrid[:h, :w]
            dist = (x - crow) ** 2 + (y - ccol) ** 2
            inside = dist <= r * r
            mask[inside] = 1 - adjusted_strength

            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)

            channels.append(img_back)

        deblurred = np.dstack(channels)
        deblurred_uint8 = np.clip(deblurred * 255, 0, 255).astype(np.uint8)

        # USM锐化
        lab = cv2.cvtColor(deblurred_uint8, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        gaussian = cv2.GaussianBlur(l, (0, 0), 1.5)
        unsharp_mask = cv2.addWeighted(l, 1.5, gaussian, -0.5, 0)

        lab_sharpened = cv2.merge([unsharp_mask, a, b])
        deblurred_sharpened = cv2.cvtColor(lab_sharpened, cv2.COLOR_Lab2BGR)

        return deblurred_sharpened

    else:
        print(f"未知的去模糊方法: {method}，返回原始图像")
        return img


# ==================== 2. lime_light函数实现 ====================

def limenew_enhance(img,  gamma=0.6):
    """
    简化版LIME算法实现，避免复杂的稀疏矩阵优化
    """
    #print("开始执行LIMENEW增强...")

    # 转换为浮点型
    img_float = img.astype(np.float32) / 255.0
    r = img_float[:, :, 2]
    g = img_float[:, :, 1]
    b = img_float[:, :, 0]

    # 初始照度图 (每个像素取RGB的最大值)
    illumination = np.maximum(np.maximum(r, g), b)

    # 创建高光掩码，用于保护高光区域
    highlight_mask = illumination > 0.8

    # 简化的照度图优化: 直接伽马校正+高光区域额外抑制
    illumination_refined = np.power(illumination, gamma)
    illumination_refined[highlight_mask] = np.power(illumination[highlight_mask], gamma * 1.5)

    # 均值滤波平滑
    kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    illumination_refined = cv2.GaussianBlur(illumination_refined, (kernel_size, kernel_size), 0)

    illumination_refined = np.maximum(illumination_refined, 0.01)

    # RGB加权
    r_weight, g_weight, b_weight = 1.0, 1.0, 1.0
    r_enhanced = (r / illumination_refined) * r_weight
    g_enhanced = (g / illumination_refined) * g_weight
    b_enhanced = (b / illumination_refined) * b_weight

    # 组合
    enhanced = np.zeros_like(img_float)
    enhanced[:, :, 2] =  r_enhanced
    enhanced[:, :, 1] =  g_enhanced
    enhanced[:, :, 0] =  b_enhanced

    out_8u = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    #print("LIMENEW增强完成")
    return out_8u


def limeblock_enhance(img,
                      gamma=0.4,
                      block_count=3,
                      overlap_ratio=0.2,
                      deblur=True,
                      deblur_method='deeprft',
                      deblur_strength=0.5):
    """
    LIMEBLOCK算法：对图像不同亮度区域进行自适应划分，分块应用LIMENEW，并进行平滑拼接

    参数:
        img: 输入BGR图像
        gamma: 伽马校正参数
        block_count: 亮度分区数量
        overlap_ratio: 块之间的重叠比例，用于平滑过渡
        deblur: 是否应用水下去模糊
        deblur_method: 去模糊方法('untv'或'deeprft')
        deblur_strength: 去模糊强度

    返回:
        增强后的图像
    """
    print("开始执行LIMEBLOCK算法增强...")

    try:
        if img is None:
            print("错误：LIMEBLOCK输入图像为空")
            return np.zeros((100, 100, 3), dtype=np.uint8)

        print(f"LIMEBLOCK输入图像形状: {img.shape}, 类型: {img.dtype}")

        processed_img = img.copy()

        # 分块逻辑
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist /= hist.sum()
        cdf = np.cumsum(hist)
        thresholds = []
        for i in range(1, block_count):
            tgt = i / block_count
            idx_thresh = np.argmin(np.abs(cdf - tgt))
            thresholds.append(idx_thresh)
        if not thresholds:
            thresholds = [85, 170]

        masks = []
        prev = 0
        for th in thresholds:
            m = np.zeros_like(gray, dtype=np.uint8)
            m[(gray > prev) & (gray <= th)] = 255
            masks.append(m)
            prev = th
        m = np.zeros_like(gray, dtype=np.uint8)
        m[gray > prev] = 255
        masks.append(m)

        expanded_masks = []
        for m in masks:
            kernel_size = int(max(3, min(processed_img.shape[:2]) * overlap_ratio / block_count))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            expanded_masks.append(cv2.dilate(m, kernel))

        result = np.zeros_like(processed_img, dtype=np.float32)
        weight_sum = np.zeros_like(processed_img, dtype=np.float32)

        # 创建保存目录
        os.makedirs('masks_and_enh', exist_ok=True)

        for idx, m in enumerate(expanded_masks, start=1):
            # 1) 保存当前掩码 (单通道)
            cv2.imwrite(f'masks_and_enh/mask_{idx:02d}.png', m)

            # 2) 提取该块区域并增强
            region = cv2.bitwise_and(processed_img, processed_img, mask=m)
            enh = limenew_enhance(region, gamma=gamma)

            # 3) 保存该块增强后的结果 (三通道 BGR)
            cv2.imwrite(f'masks_and_enh/enhanced_{idx:02d}.png', enh)

            # 4) 加权融合回原图
            enh_f = enh.astype(np.float32) / 255.0
            wt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            wt = wt / (wt.max() + 1e-8)
            wt3 = cv2.merge([wt, wt, wt])

            result += enh_f * wt3
            weight_sum += wt3

        # 避免除零
        weight_sum[weight_sum < 1e-6] = 1.0
        final_f = result / weight_sum
        final_u8 = np.clip(final_f * 255, 0, 255).astype(np.uint8)

        # 最终去模糊
        if deblur:
            print(f"执行水下去模糊处理，方法: {deblur_method}, 强度: {deblur_strength}...")
            final_u8 = underwater_deblur(final_u8, method='deeprft', strength=deblur_strength)


        print("LIMEBLOCK算法增强完成")
        return final_u8

    except Exception as e:
        print(f"LIMEBLOCK算法执行出错: {e}")
        traceback.print_exc()
        print("错误恢复：回退到LIMENEW算法")
        return limenew_enhance(img, gamma)



# ==================== 3. 主函数：只演示limeblock_deeprft ====================

if __name__ == "__main__":
    print("程序开始执行...")

    # “limeblock_deeprft”的处理流程：
    # 1) 从当前目录读取 "xxx.jpg"
    # 2) 调用 limeblock_enhance( ... deblur=True, deblur_method='deeprft' ...)
    # 3) 保存结果为 "xxx.jpg"

    input_file = r"data/new2/10_dark.jpg"
    output_file = "data/new2/10_light.jpg"

    if not os.path.exists(input_file):
        print(f"错误: 未找到 {input_file}，请将图片放在脚本同目录下。")
        sys.exit(1)

    img2light = cv2.imread(input_file, cv2.IMREAD_COLOR)
    if img2light is None:
        print(f"无法读取图像文件: {input_file}")
        sys.exit(1)

    print(f"读取图像成功，尺寸: {img2light.shape}")

    # 调用 LIMEBLOCK 算法 + DeepRFT 去模糊
    limeblock_deeprft = limeblock_enhance(
        img2light.copy(),
        gamma=0.55, #分块去模糊参数指定
        block_count=3,  #分块参数指定
        overlap_ratio=0.2,  #融合参数指定
        deblur=True,  # 启用水下去模糊
        deblur_method='deeprft',  # 使用DeepRFT去模糊
        deblur_strength=0.3  # 去模糊强度
    )

    # 保存结果
    cv2.imwrite(output_file, limeblock_deeprft)
    print(f"处理完成！已输出结果：{output_file}")

