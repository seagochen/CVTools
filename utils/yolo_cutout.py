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
        description="YOLO11 抠图工具：检测图片中的目标并抠出保存"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="输入图片文件夹路径，文件夹下所有图片都会被处理"
    )
    parser.add_argument(
        "--target_class",
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
        "--w_ratio",
        type=float,
        default=1.0,
        help="宽度放大倍率，默认为1.0（不放大），例如1.2表示宽度放大20%%"
    )
    parser.add_argument(
        "--h_ratio",
        type=float,
        default=1.0,
        help="高度放大倍率，默认为1.0（不放大），例如1.2表示高度放大20%%"
    )
    return parser.parse_args()


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
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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


def clip_object(image, box, w_ratio=1.0, h_ratio=1.0):
    """
    根据检测框从图像中抠出目标区域

    :param image: 原图（numpy数组）
    :param box: 检测框对象，使用 box.xyxy
    :param w_ratio: 水平方向放大倍率
    :param h_ratio: 垂直方向放大倍率
    :return: 裁剪后的图像
    """
    # 获取检测框坐标（xyxy格式）
    coords = box.xyxy.cpu().numpy()[0]
    x1, y1, x2, y2 = coords
    # 计算原始宽度和高度
    width = x2 - x1
    height = y2 - y1
    # 计算检测框中心
    cx = x1 + width / 2
    cy = y1 + height / 2
    # 根据放大倍率计算新的宽度和高度
    new_w = width * w_ratio
    new_h = height * h_ratio
    # 计算新的左上角和右下角
    new_x1 = int(round(cx - new_w / 2))
    new_y1 = int(round(cy - new_h / 2))
    new_x2 = int(round(cx + new_w / 2))
    new_y2 = int(round(cy + new_h / 2))
    # 限制坐标在图像范围内
    img_h, img_w = image.shape[:2]
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)
    # 裁剪图像
    clipped = image[new_y1:new_y2, new_x1:new_x2]
    return clipped


def process_image(image_path):
    """
    读取单张图片

    :param image_path: 图片路径
    :return: 读取的图像（原图尺寸）
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    return image


def save_clipped(clipped, out_dir, cls, counts):
    """
    将抠出的目标保存为图片，命名格式为 {类别}_{序号}.png

    :param clipped: 裁剪后的图像
    :param out_dir: 输出目录
    :param cls: 目标类别（整数）
    :param counts: 字典，记录每个类别已保存的数量
    """
    if cls not in counts:
        counts[cls] = 1
    else:
        counts[cls] += 1
    filename = f"{cls}_{counts[cls]:04d}.png"
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, clipped)
    print(f"保存抠图: {out_path}")


def process_directory(input_dir, detector, args):
    """
    批量处理指定文件夹下的所有图片，对检测到的目标进行抠图

    :param input_dir: 图片文件夹路径
    :param detector: YoloDetector实例
    :param args: 命令行参数对象
    """
    # 构建输出文件夹 clipped，保存在输入目录下
    out_dir = os.path.join(input_dir, "clipped")
    os.makedirs(out_dir, exist_ok=True)

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
    # 用于保存每个类别已抠图数量的字典
    counts = {}

    for img_path in image_files:
        try:
            image = process_image(img_path)
        except ValueError as e:
            print(e)
            continue

        # 对图片进行目标检测
        results = detector.detect(image, conf=args.conf, target_class=args.target_class)
        for box in results.boxes:
            cls = int(box.cls.cpu().numpy().item())
            # 按指定倍率抠图
            clipped = clip_object(image, box, w_ratio=args.w_ratio, h_ratio=args.h_ratio)
            save_clipped(clipped, out_dir, cls, counts)


def main():
    args = parse_args()

    # 指定模型路径，这里使用最新的 yolo11l 模型
    model_path = "/opt/models/yolo11l.pt"
    try:
        detector = YoloDetector(model_path)
    except Exception as e:
        print(e)
        return

    if not os.path.isdir(args.dir):
        print(f"错误：输入目录 '{args.dir}' 不存在")
        return

    process_directory(args.dir, detector, args)


if __name__ == "__main__":
    main()
