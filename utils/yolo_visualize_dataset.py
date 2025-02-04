#!/usr/bin/env python3
import os
import shutil
import hashlib
import argparse
import pandas as pd
from PIL import Image
import cv2
import yaml
from tqdm import tqdm
from collections import defaultdict

#######################################
# 数据验证：检查数据集中是否存在一一对应的图片与标签文件
#######################################
def validate_dataset(dataset_dir):
    """检查 dataset_dir 下的文件是否成对存在，至少要求：
       - 每个 *.txt 文件有对应的图片（png/jpg/jpeg）
       - 每个图片文件有对应的 txt 文件
    """
    files = os.listdir(dataset_dir)
    txt_files = {os.path.splitext(f)[0] for f in files if f.endswith('.txt')}
    img_files = {os.path.splitext(f)[0] for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    
    missing_labels = img_files - txt_files
    missing_imgs = txt_files - img_files
    
    if missing_labels:
        print("以下图片缺少对应的标签文件：")
        for base in missing_labels:
            print(f"  {base}")
    if missing_imgs:
        print("以下标签缺少对应的图片文件：")
        for base in missing_imgs:
            print(f"  {base}")
    
    if missing_labels or missing_imgs:
        raise ValueError("数据集格式不符合要求，必须确保图片和标签一一对应。")
    else:
        print("数据集格式验证通过！")

#######################################
# Step 1. 数据扫描、生成 DataFrame 并统一重命名
#######################################
def scan_folder_and_generate_df(folder_path):
    """
    扫描指定文件夹，查找 txt 和图像文件，并生成一个 DataFrame。
    DataFrame 包含：原始路径、原始 txt 文件名、原始图像文件名、标签文件中行数（items）。
    要求所有文件均位于同一目录下。
    """
    results = []
    files = os.listdir(folder_path)
    # 要求所有文件均在同一目录下
    txt_files = {f for f in files if f.endswith('.txt')}
    img_files = {f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    # 只处理同时存在的基名
    base_names = sorted(txt_files & {os.path.splitext(f)[0] for f in img_files})
    
    for base in base_names:
        txt_name = f"{base}.txt"
        # 按优先级选择图片格式：png > jpg > jpeg
        image_name = ""
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = f"{base}{ext}"
            if candidate in img_files:
                image_name = candidate
                break
        # 统计 txt 文件行数（标签数量）
        items_count = 0
        txt_path = os.path.join(folder_path, txt_name)
        try:
            with open(txt_path, 'r') as f:
                items_count = len(f.readlines())
        except Exception as e:
            items_count = 0
        results.append([folder_path, txt_name, image_name, items_count])
    
    df = pd.DataFrame(results, columns=['base_path', 'label', 'image', 'items'])
    return df

def rename_files_in_dataframe(df):
    """
    根据 txt 文件名（作为基名）生成统一的哈希名，
    构造新的 txt 和图像文件名（重命名后文件均以该哈希命名）。
    """
    new_names = []
    for _, row in df.iterrows():
        base_str = row['label'] if row['label'] else ""
        hash_name = hashlib.md5(base_str.encode()).hexdigest()
        new_txt = f"{hash_name}.txt" if base_str else ""
        image_ext = os.path.splitext(row['image'])[1]
        new_img = f"{hash_name}{image_ext}"
        new_names.append((new_txt, new_img))
    
    df['label_new'] = [x[0] for x in new_names]
    df['image_new'] = [x[1] for x in new_names]
    return df

def filter_dataframe(df):
    """
    过滤出同时存在图片和标签且标签文件中有内容的记录
    """
    return df[(df['label'] != "") & (df['image'] != "") & (df['items'] > 0)]

#######################################
# Step 2. 拷贝文件并统一调整图像尺寸
#######################################
def resize_image(input_image_path, output_image_path, size=(640,640)):
    try:
        with Image.open(input_image_path) as img:
            img = img.resize(size)
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            img.save(output_image_path)
    except Exception as e:
        print(f"Error resizing image {input_image_path}: {e}")

def copy_file(input_file_path, output_file_path):
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        shutil.copy2(input_file_path, output_file_path)
    except Exception as e:
        print(f"Error copying file {input_file_path}: {e}")

def process_and_save_files(df, output_dir, resize_size=(640,640)):
    """
    根据 DataFrame 中的记录，从原始目录中拷贝 txt 和图像文件到 output_dir，
    同时将图像统一调整为 resize_size 尺寸，文件名按重命名后的名字保存。
    """
    for _, row in tqdm(df.iterrows(), total=len(df), desc="拷贝并调整文件"):
        base_path = row['base_path']
        txt_src = os.path.join(base_path, row['label'])
        img_src = os.path.join(base_path, row['image'])
        txt_dst = os.path.join(output_dir, row['label_new'])
        img_dst = os.path.join(output_dir, row['image_new'])
        if os.path.exists(txt_src):
            copy_file(txt_src, txt_dst)
        if os.path.exists(img_src):
            resize_image(img_src, img_dst, size=resize_size)

#######################################
# Step 3. 坐标转换
#######################################
def cxcywh_to_xyxy(cx, cy, w, h, img_size):
    """
    将 normalized (cx,cy,w,h) 转换为 normalized (x1,y1,x2,y2)；
    img_size 为统一后的图像尺寸（例如 640）。
    """
    abs_cx = cx * img_size
    abs_cy = cy * img_size
    abs_w = w * img_size
    abs_h = h * img_size
    x1 = abs_cx - abs_w / 2
    y1 = abs_cy - abs_h / 2
    x2 = abs_cx + abs_w / 2
    y2 = abs_cy + abs_h / 2
    return x1 / img_size, y1 / img_size, x2 / img_size, y2 / img_size

def xyxy_to_cxcywh(x1, y1, x2, y2, img_size):
    """
    将 normalized (x1,y1,x2,y2) 转换为 normalized (cx,cy,w,h)。
    """
    abs_x1 = x1 * img_size
    abs_y1 = y1 * img_size
    abs_x2 = x2 * img_size
    abs_y2 = y2 * img_size
    abs_w = abs_x2 - abs_x1
    abs_h = abs_y2 - abs_y1
    abs_cx = abs_x1 + abs_w / 2
    abs_cy = abs_y1 + abs_h / 2
    return abs_cx / img_size, abs_cy / img_size, abs_w / img_size, abs_h / img_size

def convert_label_file(label_path, input_format, target_format, img_size=640):
    """
    读取 label 文件（格式为：class coord1 coord2 coord3 coord4，均为 normalized 数值），
    根据 input_format 与 target_format 进行转换，并覆盖写回 label 文件。
    """
    new_lines = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = parts[0]
        if input_format == target_format:
            new_line = f"{cls} {' '.join(f'{float(x):.6f}' for x in parts[1:])}\n"
        elif input_format == "xywh" and target_format == "xyxy":
            cx, cy, w, h = map(float, parts[1:5])
            x1, y1, x2, y2 = cxcywh_to_xyxy(cx, cy, w, h, img_size)
            new_line = f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n"
        elif input_format == "xyxy" and target_format == "xywh":
            x1, y1, x2, y2 = map(float, parts[1:5])
            cx, cy, w, h = xyxy_to_cxcywh(x1, y1, x2, y2, img_size)
            new_line = f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        else:
            new_line = line
        new_lines.append(new_line)
    with open(label_path, 'w') as f:
        f.writelines(new_lines)

def convert_all_labels(output_dir, input_format, target_format, img_size=640):
    """
    对 output_dir 中所有 txt 文件进行坐标转换，
    根据 input_format 与 target_format 进行转换；如果二者相同则不做转换。
    """
    if input_format == target_format:
        print("输入格式与目标格式相同，无需转换。")
        return
    files = os.listdir(output_dir)
    for f in tqdm(files, desc="转换坐标"):
        if f.endswith('.txt'):
            label_path = os.path.join(output_dir, f)
            convert_label_file(label_path, input_format, target_format, img_size=img_size)

#######################################
# 以下为可视化相关函数
#######################################
# 定义不同类别对应的颜色（假设最多10个类别）
bbox_colors = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (128, 128, 0),  # Olive
    (128, 128, 128),# Gray
    (0, 128, 255)   # Orange
]

def find_files_with_same_basename(directory):
    """遍历目录，按文件基名分组，返回那些具有多个扩展名的文件组。"""
    file_dict = defaultdict(set)
    for _, _, files in os.walk(directory):
        for file in files:
            basename, ext = os.path.splitext(file)
            file_dict[basename].add(ext.lstrip("."))
    result = [{"basename": basename, "extensions": list(extensions)} 
              for basename, extensions in file_dict.items() if len(extensions) > 1]
    return result

def draw_box_on_image(image, ground_truth: str, index: int, coordinate_format: str):
    """
    在图像上绘制一个边框。
    ground_truth 为字符串，格式为 "class coord1 coord2 coord3 coord4"，
    coordinate_format 为 "xywh" 或 "xyxy"。
    """
    ground_truth = ground_truth.split(" ")
    width, height = image.shape[1], image.shape[0]
    box_color = bbox_colors[index % len(bbox_colors)]
    if coordinate_format == "xywh":
        cls, cx, cy, w, h = int(ground_truth[0]), float(ground_truth[1]), float(ground_truth[2]), float(ground_truth[3]), float(ground_truth[4])
        lx = int((cx - w / 2) * width)
        ly = int((cy - h / 2) * height)
        rx = int((cx + w / 2) * width)
        ry = int((cy + h / 2) * height)
    elif coordinate_format == "xyxy":
        cls, ltx, lty, rbx, rby = int(ground_truth[0]), float(ground_truth[1]), float(ground_truth[2]), float(ground_truth[3]), float(ground_truth[4])
        lx = int(ltx * width)
        ly = int(lty * height)
        rx = int(rbx * width)
        ry = int(rby * height)
    cv2.putText(image, f"Class {cls}", (lx, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    cv2.rectangle(image, (lx, ly), (rx, ry), box_color, 2)
    return image

def visualize_dataset(src_directory, dst_directory, coordinate_format):
    """
    遍历 src_directory 中所有具有相同基名的文件（例如 image 与对应的 txt），
    在图像上绘制边框后保存到 dst_directory 中，绘制时采用 coordinate_format 指定的方式。
    """
    if not os.path.exists(src_directory):
        print(f"The directory {src_directory} does not exist.")
        return
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)
    file_groups = find_files_with_same_basename(src_directory)
    for file_group in tqdm(file_groups, desc="Processing files"):
        image_file_path = find_image_file(file_group["extensions"], src_directory, file_group["basename"])
        if not image_file_path:
            tqdm.write(f"没有找到对应的图片文件: {file_group['basename']}")
            continue
        image = cv2.imread(image_file_path)
        if image is None:
            tqdm.write(f"无法加载图片: {image_file_path}")
            continue
        txt_file_path = os.path.join(src_directory, f"{file_group['basename']}.txt")
        if os.path.exists(txt_file_path):
            with open(txt_file_path, "r") as file:
                try:
                    for idx, line in enumerate(file):
                        image = draw_box_on_image(image, line.strip(), idx, coordinate_format)
                except Exception as e:
                    tqdm.write(f"数据格式无效: {txt_file_path} - {e}")
                    continue
        else:
            tqdm.write(f"没有找到对应的标注文件: {txt_file_path}")
            continue
        output_image_path = os.path.join(dst_directory, f"{file_group['basename']}_bbox.jpg")
        if image is not None:
            cv2.imwrite(output_image_path, image)

#######################################
# 预处理子命令的主函数
#######################################
def preprocess_main(args):
    try:
        validate_dataset(args.input_dir)
    except ValueError as e:
        print(e)
        return

    df = scan_folder_and_generate_df(args.input_dir)
    df = rename_files_in_dataframe(df)
    df = filter_dataframe(df)
    if df.empty:
        print("没有找到符合要求的图片与标签对，退出。")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    process_and_save_files(df, args.output_dir, resize_size=(640,640))
    convert_all_labels(args.output_dir, args.input_format, args.target_format, img_size=640)
    print("预处理步骤已完成。")

#######################################
# 可视化子命令的主函数
#######################################
def visualize_main(args):
    visualize_dataset(args.src_dir, args.dst_dir, args.coordinate_format)
    print("可视化步骤已完成。")

#######################################
# 主函数：使用子命令预处理和可视化
#######################################
def main():
    parser = argparse.ArgumentParser(description="YOLO数据集预处理与可视化脚本")
    subparsers = parser.add_subparsers(dest="command", help="预处理(preprocess) 或 可视化(visualize)")
    
    # 预处理子命令
    preprocess_parser = subparsers.add_parser("preprocess", help="预处理数据集")
    preprocess_parser.add_argument("--input_dir", required=True,
                                   help="原始数据集目录（所有图片与标签均在同一目录，且一一对应）")
    preprocess_parser.add_argument("--output_dir", required=True,
                                   help="输出数据集目录（处理后所有文件均存放于此）")
    preprocess_parser.add_argument("--target_format", required=True, choices=["xywh", "xyxy"],
                                   help="目标标签坐标格式：若选择 xyxy，则转换为 (x1 y1 x2 y2)；选择 xywh 则保持不变")
    preprocess_parser.add_argument("--input_format", default="xywh", choices=["xywh", "xyxy"],
                                   help="原始标签坐标格式，默认为 xywh；若原始数据为 xyxy，请指定为 xyxy")
    
    # 可视化子命令
    visualize_parser = subparsers.add_parser("visualize", help="可视化数据集")
    visualize_parser.add_argument("--src_dir", required=True,
                                  help="待可视化数据集目录（图片与标签均存放于此）")
    visualize_parser.add_argument("--dst_dir", required=True,
                                  help="可视化结果输出目录")
    visualize_parser.add_argument("--coordinate_format", required=True, choices=["xywh", "xyxy"],
                                  help="标签坐标格式，用于绘制边框")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        preprocess_main(args)
    elif args.command == "visualize":
        visualize_main(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
