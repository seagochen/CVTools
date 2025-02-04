import os
import cv2
import numpy as np
import argparse
import albumentations as A

def crop_images(image_folder, output_folder, ltx, lty):
    """
    对图像进行裁剪，裁剪区域从 (ltx, lty) 开始，边长为图像剩余宽度与高度的最小值。
    """
    if not os.path.exists(image_folder):
        print(f"输入路径 {image_folder} 不存在。")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取 {image_file}，跳过。")
            continue

        height, width = image.shape[:2]
        crop_width = width - ltx
        crop_height = height - lty
        min_side_length = min(crop_width, crop_height)

        cropped_image = image[lty:lty+min_side_length, ltx:ltx+min_side_length]
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, cropped_image)
        print(f"裁剪并保存 {image_file} 到 {output_folder}")

def pad_images(image_folder, output_folder, pad_color=(0, 0, 0)):
    """
    对图像进行填充，使其成为正方形。填充颜色默认为黑色。
    """
    if not os.path.exists(image_folder):
        print(f"输入路径 {image_folder} 不存在。")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取 {image_file}，跳过。")
            continue

        height, width = image.shape[:2]
        max_side = max(width, height)

        # 新建正方形图像并填充背景颜色
        padded_image = np.full((max_side, max_side, 3), pad_color, dtype=np.uint8)
        x_offset = (max_side - width) // 2
        y_offset = (max_side - height) // 2
        padded_image[y_offset:y_offset+height, x_offset:x_offset+width] = image

        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, padded_image)
        print(f"填充并保存 {image_file} 到 {output_folder}")

def image_padding(image, size=640):
    """
    对图像进行填充，保证最终尺寸为 size x size。如果图像尺寸超过目标尺寸，则先缩放。
    """
    h, w = image.shape[:2]
    if h >= size or w >= size:
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))

    h, w = image.shape[:2]
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left

    # 保证非负
    top = max(0, top)
    bottom = max(0, bottom)
    left = max(0, left)
    right = max(0, right)

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def adjust_gamma(image, gamma=1.0):
    """
    对图像应用 Gamma 校正，gamma > 1 会使图像变亮。
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    adjusted_image = cv2.LUT(image, table)
    return adjusted_image

def motion_blur(image, size=15, angle=0):
    """
    对图像应用运动模糊效果。
    
    参数:
      - size: 模糊核大小，决定模糊强度。
      - angle: 模糊角度，单位为度。
    """
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = np.diag(np.ones(size))
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel = kernel / size

    blurred = cv2.filter2D(image, -1, kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    return blurred

def augment_image(image, augmenter):
    """
    利用 Albumentations 定义的数据增强器对图像进行增强。
    """
    return augmenter(image=image)['image']

def augment_and_save(image_folder, augmented_folder, num_augments=3):
    """
    对图像进行数据增强，生成额外的增强样本。默认会生成3个增强版本，
    同时随机触发一些自定义增强（缩放、过曝、运动模糊）。
    """
    if not os.path.exists(augmented_folder):
        os.makedirs(augmented_folder)

    # 定义常规的数据增强操作
    augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=20, shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RGBShift(p=0.5),
        A.CLAHE(p=0.5),
        A.RandomGamma(p=0.5),
        A.GaussNoise(p=0.5),
        A.Blur(p=0.3),
        A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.5),
    ])

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取 {image_file}，跳过。")
            continue

        # 先将图像调整为 640x640 大小（缩放）
        image = cv2.resize(image, (640, 640))
        base_name, ext = os.path.splitext(image_file)
        # 保存原始图像
        cv2.imwrite(os.path.join(augmented_folder, f"{base_name}_original.jpg"), image)

        # 随机触发自定义增强操作
        if np.random.rand() < 0.4:
            # 随机缩放并填充
            random_scale = np.random.uniform(0.5, 0.8)
            shrink_image = cv2.resize(image, (int(image.shape[1] * random_scale), int(image.shape[0] * random_scale)))
            shrink_image = image_padding(shrink_image, 640)
            cv2.imwrite(os.path.join(augmented_folder, f"{base_name}_shrink.jpg"), shrink_image)

            # 过曝处理
            gamma_adjusted_image = adjust_gamma(image, gamma=2.0)
            cv2.imwrite(os.path.join(augmented_folder, f"{base_name}_gamma.jpg"), gamma_adjusted_image)

            # 运动模糊
            motion_blurred_image = motion_blur(image, size=15, angle=45)
            motion_blurred_image = image_padding(motion_blurred_image, 640)
            cv2.imwrite(os.path.join(augmented_folder, f"{base_name}_motion.jpg"), motion_blurred_image)

        # 生成常规增强图像
        for i in range(num_augments):
            aug_img = augment_image(image, augmenter)
            aug_img = image_padding(aug_img, 640)
            cv2.imwrite(os.path.join(augmented_folder, f"{base_name}_aug_{i}.jpg"), aug_img)
            print(f"增强并保存 {base_name}_aug_{i}.jpg 到 {augmented_folder}")

def main():
    parser = argparse.ArgumentParser(
        description="图像预处理与数据增强脚本，支持裁剪、填充和数据增强。")
    subparsers = parser.add_subparsers(dest="command", help="选择操作：crop, pad, augment")

    # 子命令：裁剪
    parser_crop = subparsers.add_parser("crop", help="裁剪图像")
    parser_crop.add_argument("--image_folder", type=str, required=True, help="原始图像文件夹路径")
    parser_crop.add_argument("--output_folder", type=str, default=None, help="裁剪后图像保存路径（默认在原文件夹下创建 cropped 目录）")
    parser_crop.add_argument("--ltx", type=int, required=True, help="裁剪区域左上角 x 坐标")
    parser_crop.add_argument("--lty", type=int, required=True, help="裁剪区域左上角 y 坐标")

    # 子命令：填充
    parser_pad = subparsers.add_parser("pad", help="填充图像为正方形")
    parser_pad.add_argument("--image_folder", type=str, required=True, help="原始图像文件夹路径")
    parser_pad.add_argument("--output_folder", type=str, default=None, help="填充后图像保存路径（默认在原文件夹下创建 padded 目录）")
    parser_pad.add_argument("--pad_color", type=int, nargs=3, default=[0, 0, 0],
                            help="填充颜色 B G R (默认: 0 0 0)")

    # 子命令：数据增强
    parser_aug = subparsers.add_parser("augment", help="对图像进行数据增强")
    parser_aug.add_argument("--image_folder", type=str, required=True, help="原始图像文件夹路径")
    parser_aug.add_argument("--augmented_folder", type=str, default=None,
                            help="增强后图像保存路径（默认在原文件夹下创建 augmented 目录）")
    parser_aug.add_argument("--num_augments", type=int, default=3,
                            help="每张图像生成的常规增强数量（默认: 3）")

    args = parser.parse_args()

    if args.command == "crop":
        output_folder = args.output_folder if args.output_folder else os.path.join(args.image_folder, "cropped")
        crop_images(args.image_folder, output_folder, args.ltx, args.lty)
    elif args.command == "pad":
        output_folder = args.output_folder if args.output_folder else os.path.join(args.image_folder, "padded")
        pad_images(args.image_folder, output_folder, tuple(args.pad_color))
    elif args.command == "augment":
        augmented_folder = args.augmented_folder if args.augmented_folder else os.path.join(args.image_folder, "augmented")
        augment_and_save(args.image_folder, augmented_folder, args.num_augments)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
