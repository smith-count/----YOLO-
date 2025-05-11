import subprocess
import os
import cv2
import numpy as np
from PIL import Image

def convert_video_with_ffmpeg(input_file, output_file, video_codec='libx264'):
    """
    使用 FFmpeg 将输入视频文件转换为指定编码的输出视频文件。

    参数:
    - input_file: 输入视频文件的路径。
    - output_file: 输出视频文件的路径。
    - video_codec: 视频编码器，默认为 'libx264'。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件未找到: {input_file}")

    # 确保输出文件以 .mp4 结尾
    if not output_file.lower().endswith('.mp4'):
        output_file += '.mp4'

    # 构建 FFmpeg 命令
    command = [
        'ffmpeg',
        '-i', input_file,  # 输入文件
        '-vcodec', video_codec,  # 视频编码器
        '-y', output_file  # 输出文件，-y 表示覆盖已有文件
    ]

    try:
        # 运行 FFmpeg 命令
        subprocess.run(command, check=True)
        print(f"视频已成功转换为: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"视频转换失败: {e}")

def process_yolo_results(results, class_list=None, conf_thres=0.1):
    global flag
    global detection_mode
    # 1. 结果过滤
    if detection_mode == "单类识别":
        return
    if class_list is not None:
        names = results.names
        keep_idx = [
            i for i, box in enumerate(results.boxes)
            if (names[int(box.cls)] in class_list) and (float(box.conf) >= conf_thres)
        ]
        results.boxes = results.boxes[keep_idx]
        if hasattr(results, 'masks') and results.masks is not None:
            results.masks = results.masks[keep_idx]
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            results.keypoints = results.keypoints[keep_idx]

    if len(results.boxes)==0 :
        flag = False
    # 2. 安全图像转换
    plotted_img = results.plot()

    # 处理不同返回类型
    if isinstance(plotted_img, Image.Image):
        # PIL.Image转numpy数组
        img_np = np.array(plotted_img)
        # 确保是3通道(RGB或BGR)
        if img_np.ndim == 2:  # 灰度图
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:  # 已经是numpy数组
        img_np = plotted_img
        if img_np.ndim == 2:  # 灰度图
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif img_np.shape[2] == 3:  # 确保是BGR
            pass  # 假设已经是BGR

    return img_np






# if __name__ == "__main__":
#     # 示例用法
#     input_mp4_path = r"D:\Python\graduate_design\temp_uploaded_video.mp4"  # 输入 MP4 文件路径
#     output_mp4_path = "output_video_that_streamlit_can_play"  # 输出 MP4 文件路径（无需扩展名，默认 .mp4）
#
#     convert_video_with_ffmpeg(input_mp4_path, output_mp4_path)