import subprocess
import os

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

if __name__ == "__main__":
    # 示例用法
    input_mp4_path = r"D:\Python\graduate_design\temp_uploaded_video.mp4"  # 输入 MP4 文件路径
    output_mp4_path = "output_video_that_streamlit_can_play"  # 输出 MP4 文件路径（无需扩展名，默认 .mp4）

    convert_video_with_ffmpeg(input_mp4_path, output_mp4_path)