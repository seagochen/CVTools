import cv2
import os
import argparse

def convert_video_to_images(video_path, output_folder, sample_rate=1, start_frame=0, end_frame=None):
    """
    将视频转换为图片序列。

    参数：
      video_path: 视频文件路径。
      output_folder: 输出图片保存的目录。
      sample_rate: 每秒采样的帧数，默认为1帧每秒。
      start_frame: 采样起始帧位置（默认0，从视频开头开始）。
      end_frame: 采样结束帧位置（默认为None，表示采样到视频末尾）。
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件 '{video_path}' 不存在。")
        return

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'。")
        return

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取视频的FPS（每秒帧数）
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("错误：无法获取视频的FPS。")
        cap.release()
        return

    # 根据sample_rate计算采样间隔（以帧为单位）
    # 如果sample_rate大于视频FPS，则每帧都采样
    frame_interval = int(round(fps / sample_rate)) if sample_rate < fps else 1
    print(f"视频FPS: {fps:.2f}，每秒采样 {sample_rate} 帧，采样间隔: {frame_interval} 帧。")

    # 获取视频文件名（不带扩展名），用于图片命名
    img_base_name = os.path.basename(video_path).split(".")[0]

    count = 0       # 当前视频帧编号
    saved_count = 0 # 已保存的图片数量

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 跳过采样起始帧之前的帧
        if count < start_frame:
            count += 1
            continue

        # 如果设置了采样结束帧，超过该帧则退出循环
        if end_frame is not None and count > end_frame:
            break

        # 从start_frame开始，每隔frame_interval采样一帧
        if (count - start_frame) % frame_interval == 0:
            image_filename = os.path.join(output_folder, f"{img_base_name}_{count:06d}.png")
            cv2.imwrite(image_filename, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"视频处理完成，共处理 {count} 帧，保存了 {saved_count} 张图片。")


def convert_images_to_video(image_folder, output_video, frame_rate=30):
    """
    将图片序列转换为视频。

    参数：
      image_folder: 存放图片的文件夹。程序将读取该目录下所有 png、jpg、jpeg 文件，并按文件名排序。
      output_video: 输出视频文件路径。
      frame_rate: 输出视频的帧率，默认为30。
    """
    # 获取所有图片文件（支持png、jpg、jpeg），并按文件名排序
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = sorted(
        [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    )

    if not image_files:
        print(f"错误：在文件夹 '{image_folder}' 中没有找到图片文件。")
        return

    # 读取第一张图片以获得视频的尺寸
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"错误：无法读取图片 '{image_files[0]}'。")
        return
    height, width, channels = first_img.shape
    frame_size = (width, height)

    # 定义视频编码方式并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 生成 mp4 格式的视频
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, frame_size)
    if not out.isOpened():
        print(f"错误：无法打开视频写入对象，输出视频路径 '{output_video}'。")
        return

    print(f"开始生成视频，帧率: {frame_rate} fps，分辨率: {frame_size}")
    frame_count = 0
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is None:
            print(f"警告：无法读取图片 '{img_file}'，跳过。")
            continue
        # 若图片尺寸与视频尺寸不一致，可以选择调整尺寸（此处默认保持原尺寸，假设一致）
        if (img.shape[1], img.shape[0]) != frame_size:
            img = cv2.resize(img, frame_size)
        out.write(img)
        frame_count += 1

    out.release()
    print(f"视频生成完成，共写入 {frame_count} 帧，保存在 '{output_video}'。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="视频与图片相互转换工具：支持视频转图片(v2i)和图片转视频(i2v)"
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help="选择操作模式：v2i 或 i2v")

    # 视频转图片子命令
    parser_v2i = subparsers.add_parser('v2i', help="将视频转换为图片")
    parser_v2i.add_argument("-i", "--input", required=True, help="输入视频文件路径")
    parser_v2i.add_argument("-o", "--output", required=True, help="输出图片保存的目录")
    parser_v2i.add_argument("--start", type=int, default=0, help="采样起始帧位置（默认0）")
    parser_v2i.add_argument("--end", type=int, default=None, help="采样结束帧位置（默认采样到视频末尾）")
    parser_v2i.add_argument("--sampling", type=float, default=1, help="每秒采样的帧数（默认1）")

    # 图片转视频子命令
    parser_i2v = subparsers.add_parser('i2v', help="将图片序列转换为视频")
    parser_i2v.add_argument("-i", "--input", required=True, help="输入图片文件夹")
    parser_i2v.add_argument("-o", "--output", required=True, help="输出视频文件路径")
    parser_i2v.add_argument("--fps", type=float, default=30, help="输出视频的帧率（默认30）")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == 'v2i':
        convert_video_to_images(
            video_path=args.input,
            output_folder=args.output,
            sample_rate=args.sampling,
            start_frame=args.start,
            end_frame=args.end
        )
    elif args.command == 'i2v':
        convert_images_to_video(
            image_folder=args.input,
            output_video=args.output,
            frame_rate=args.fps
        )
