import os
import cv2
import re
import numpy as np

# 路径设置
images_dir = "images"             # 原始图像文件夹
masks_dir = "masks"   # mask png 和 xml 所在文件夹
output_dir = "backgrounds"        # 输出背景图保存文件夹
os.makedirs(output_dir, exist_ok=True)

# 用正则匹配去掉 "_pixelsX"
def base_name(filename):
    return re.sub(r"_pixels\d+", "", filename)

# 收集所有 mask 文件，按 base_name 分组
mask_groups = {}
for file in os.listdir(masks_dir):
    if file.endswith(".png"):
        base = base_name(file)  # e.g. "Misc_1.png"
        mask_groups.setdefault(base, []).append(file)

# 遍历每张图像
for img_name, mask_files in mask_groups.items():
    img_path = os.path.join(images_dir, img_name)
    if not os.path.exists(img_path):
        print(f"跳过 {img_name}，找不到对应的图像")
        continue

    # 读取原图
    img = cv2.imread(img_path)
    if img is None:
        print(f"跳过 {img_name}，图像读取失败")
        continue

    # 初始化空mask
    mask_total = np.zeros(img.shape[:2], dtype=np.uint8)

    # 合并多个mask
    for mask_file in mask_files:
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"跳过 {mask_file}，mask读取失败")
            continue
        mask_total = cv2.bitwise_or(mask_total, (mask > 0).astype("uint8") * 255)

    # inpaint 修复
    result = cv2.inpaint(img, mask_total, 3, cv2.INPAINT_TELEA)

    # 保存结果
    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, result)
    print(f"已处理: {img_name} -> {out_path}")
