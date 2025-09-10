import cv2
import numpy as np
import os

def remove_single_hotspot(image, radius=5, thresh=40, max_area_ratio=1, max_brightness=250):
    """
    仅移除图像中最亮、最大的单个亮点
    :param image: 输入图像 (BGR)
    :param radius: 局部邻域半径，用于平滑背景
    :param thresh: 灰度差阈值
    :param max_area_ratio: 亮点最大面积占整图比例
    :param max_brightness: 亮点最高允许的灰度值
    :return: 修复后的图像, 亮点mask
    """
    h, w = image.shape[:2]
    total_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 2 * radius + 1)

    diff = cv2.subtract(gray, blurred)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_label = None
    best_score = -1

    for i in range(1, num_labels):  # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area_ratio * total_area:
            continue  # 太大，可能不是亮点

        # 取该区域的平均亮度
        mean_brightness = cv2.mean(gray, mask=(labels == i).astype(np.uint8) * 255)[0]

        # 排除过亮的（比如整块白）
        if mean_brightness > max_brightness:
            continue

        # 选择面积和亮度综合最突出的区域
        score = area * mean_brightness
        if score > best_score:
            best_score = score
            best_label = i

    # 没找到合格的亮点
    if best_label is None:
        return image.copy(), np.zeros_like(gray, dtype=np.uint8)

    # 生成最终 mask
    hotspot_mask = (labels == best_label).astype(np.uint8) * 255

    # 膨胀一点，避免修复不干净
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hotspot_mask = cv2.dilate(hotspot_mask, kernel, iterations=2)

    # 修复
    result = cv2.inpaint(image, hotspot_mask, 3, cv2.INPAINT_TELEA)

    return result, hotspot_mask


def batch_process(input_dir, output_dir, **kwargs):
    """
    批量处理文件夹中的图片，仅移除单个最大亮点
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            filepath = os.path.join(input_dir, filename)
            image = cv2.imread(filepath)

            if image is None:
                print(f"跳过 {filename} (读取失败)")
                continue

            result, mask = remove_single_hotspot(image, **kwargs)

            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, result)
            print(f"处理完成: {filename} -> {out_path}")


if __name__ == "__main__":
    input_dir = "output"
    output_dir = "output2"

    # 你可以调这些参数
    batch_process(input_dir, output_dir,
                  radius=5,        # 背景平滑半径
                  thresh=40,       # 亮度差阈值
                  max_area_ratio=0.01,  # 最大亮点面积比例
                  max_brightness=250)   # 亮度上限
