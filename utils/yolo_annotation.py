import os
import cv2
import torch
import argparse
from ultralytics import YOLO


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="YOLO11 目标检测工具：支持批量图片检测并输出检测结果到txt文件"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="输入图片文件夹路径，文件夹下所有图片都会被检测"
    )
    parser.add_argument(
        "--class",
        type=int,
        default=None,
        help="需要识别的目标类别（例如：0）。若不指定则检测所有类别"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="检测置信度阈值（默认0.5）"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['xywh', 'xyxy'],
        default='xywh',
        help="输出坐标格式：'xywh'为中心点坐标+宽高，'xyxy'为左上角和右下角坐标，默认xywh"
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default=None,
        help="图像resize尺寸，例如640x640；若不指定则使用原图尺寸"
    )
    return parser.parse_args()


def parse_dimension(dimension_str):
    """
    将形如'640x640'的字符串解析为 (width, height) 元组
    """
    try:
        width, height = dimension_str.lower().split('x')
        return int(width), int(height)
    except Exception as e:
        raise ValueError("无法解析尺寸，请使用WxH格式（例如：640x640）") from e


class YoloDetector:
    """
    YOLO模型封装类，便于后续重用和扩展
    """

    def __init__(self, model_path: str, device: str = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 '{model_path}' 不存在")
        self.model = YOLO(model_path)
        # 自动选择设备：cuda > mps > cpu
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = device
        self.model.to(self.device)

    def detect(self, image, conf=0.5, target_class=None, verbose=False):
        """
        对输入图像进行检测

        :param image: 输入图像（BGR格式）
        :param conf: 置信度阈值
        :param target_class: 指定目标类别（int）或 None 表示检测所有类别
        :param verbose: 是否打印详细信息
        :return: 检测结果（ultralytics的Results对象）
        """
        classes_param = [target_class] if target_class is not None else None
        results = self.model(image, conf=conf, classes=classes_param, verbose=verbose)[0]
        return results


def convert_box_to_str(box, out_format='xywh'):
    """
    将检测框转换为字符串格式

    :param box: 单个检测框（包含属性 xyxy、xywh、cls 等）
    :param out_format: 'xywh' 或 'xyxy'
    :return: 字符串，例如 "cx cy w h" 或 "x1 y1 x2 y2"
    """
    cls = int(box.cls.cpu().numpy().item())
    if out_format == 'xyxy':
        coords = box.xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = coords
        coord_str = f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
    else:
        # 默认 xywh 格式，如果模型没有直接提供则从xyxy计算
        if hasattr(box, 'xywh'):
            coords = box.xywh.cpu().numpy()[0]
            cx, cy, w, h = coords
        else:
            coords = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = coords
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
        coord_str = f"{cx:.2f} {cy:.2f} {w:.2f} {h:.2f}"
    return f"{cls} {coord_str}"


def process_image(image_path, detector: YoloDetector, args, resize_dim=None):
    """
    加载单张图片，对其进行检测并返回检测结果字符串列表

    :param image_path: 图片路径
    :param detector: YoloDetector实例
    :param args: 命令行参数对象
    :param resize_dim: 如果不为None，则将图片resize为指定尺寸，格式为 (w, h)
    :return: (results_str_list, used_image) 其中 results_str_list 为每个检测目标的字符串列表，used_image 为检测使用的图像（可能resize过）
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    original_h, original_w = image.shape[:2]
    if resize_dim:
        image = cv2.resize(image, resize_dim)
    return image, original_w, original_h


def save_results(txt_path, results_list):
    """
    将检测结果列表写入txt文件

    :param txt_path: 输出txt文件路径
    :param results_list: 每行检测结果字符串列表
    """
    with open(txt_path, "w") as f:
        for line in results_list:
            f.write(line + "\n")


def process_directory(input_dir, detector: YoloDetector, args):
    """
    批量处理指定文件夹下的所有图片

    :param input_dir: 图片文件夹路径
    :param detector: YoloDetector实例
    :param args: 命令行参数对象
    """
    # 若指定了尺寸，则解析之
    resize_dim = parse_dimension(args.dimension) if args.dimension else None
    if resize_dim:
        print(f"图像将被resize至: {resize_dim}")

    # 构建输出labels文件夹，保存在输入目录下
    output_dir = os.path.join(input_dir, "labels")
    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ])

    if not image_files:
        print(f"错误：在目录 '{input_dir}' 下未找到图片文件")
        return

    print(f"共找到 {len(image_files)} 张图片，开始处理...")
    for img_path in image_files:
        try:
            image, orig_w, orig_h = process_image(img_path, detector, args, resize_dim)
        except ValueError as e:
            print(e)
            continue

        # 进行检测
        results = detector.detect(image, conf=args.conf, target_class=args.__dict__['class'])
        result_lines = []
        for box in results.boxes:
            line = convert_box_to_str(box, out_format=args.format)
            result_lines.append(line)

        # 将检测结果写入txt文件（文件名与图片同名，保存到output_dir）
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(output_dir, base_name + ".txt")
        save_results(txt_path, result_lines)
        print(f"处理 '{img_path}' 完毕，结果保存在 '{txt_path}'")


def main():
    args = parse_args()

    # 指定模型路径，这里使用最新的 yolo11l 模型
    model_path = "/opt/models/yolo11l.pt"
    try:
        detector = YoloDetector(model_path)
    except Exception as e:
        print(e)
        return

    # 检查输入目录
    if not os.path.isdir(args.dir):
        print(f"错误：输入目录 '{args.dir}' 不存在")
        return

    process_directory(args.dir, detector, args)


if __name__ == "__main__":
    main()
