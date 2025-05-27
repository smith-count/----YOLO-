import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os
import threading
import time


class YOLOAnalyzerApp:
    def __init__(self, root):
        self.root = root #将主窗口对象保存为类的实例变量-->方便其他变量访问
        self.root.title("YOLO 动物检测")
        self.root.geometry("1200x800")

        # 初始化变量
        self.video_path = ""
        self.image_path = ""
        self.cap = None #变量用于存储视频捕获对象（通常是通过OpenCV的 cv2.VideoCapture 创建的）
        self.video_writer = None #初始化一个视频写入对象
        self.is_processing = False
        self.model = None
        self.output_dir = "../output"
        self.current_frame = None

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 加载YOLO模型
        self.load_model()

        # 创建UI
        self.create_ui()

    def load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(r"/Model/yolo11n.pt")
            messagebox.showinfo("模型加载", "YOLO模型加载成功!")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")


    def create_ui(self):
        """创建用户界面"""
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 控制面板
        control_frame = tk.LabelFrame(main_frame, text="控制面板", padx=5, pady=5)
        control_frame.pack(fill=tk.X, pady=5)

        # 按钮区域
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)

        # 图像按钮
        self.btn_image = tk.Button(btn_frame, text="选择图像", command=self.load_image)
        self.btn_image.pack(side=tk.LEFT, padx=5, pady=5)

        # 视频按钮
        self.btn_video = tk.Button(btn_frame, text="选择视频", command=self.load_video)
        self.btn_video.pack(side=tk.LEFT, padx=5, pady=5)

        # 处理按钮
        self.btn_process = tk.Button(btn_frame, text="开始处理", command=self.start_processing)
        self.btn_process.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_process.config(state=tk.DISABLED)

        # 导出按钮
        self.btn_export = tk.Button(btn_frame, text="导出视频", command=self.export_video)
        self.btn_export.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_export.config(state=tk.DISABLED)

        # 进度条
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)

        # 状态标签
        self.status_label = tk.Label(control_frame, text="就绪", anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=5, pady=5)

        # 显示区域
        display_frame = tk.LabelFrame(main_frame, text="预览", padx=5, pady=5)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 原始图像显示
        self.original_label = tk.Label(display_frame, text="原始图像/视频", bg='white')
        self.original_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 处理后图像显示
        self.processed_label = tk.Label(display_frame, text="处理后结果", bg='white')
        self.processed_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_image(self):
        """加载图像文件"""
        self.reset_state()
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if file_path:
            self.image_path = file_path
            self.status_label.config(text=f"已加载图像: {os.path.basename(file_path)}")

            # 显示原始图像
            self.show_image(file_path, self.original_label)

            # 启用处理按钮
            self.btn_process.config(state=tk.NORMAL) # 动作响应

    def load_video(self):
        """加载视频文件"""
        self.reset_state()
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if file_path:
            self.video_path = file_path
            self.status_label.config(text=f"已加载视频: {os.path.basename(file_path)}")

            # 显示视频第一帧
            self.cap = cv2.VideoCapture(file_path)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.show_cv2_image(frame, self.original_label)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # 启用处理按钮
            self.btn_process.config(state=tk.NORMAL)

    def start_processing(self):
        """开始处理图像或视频"""
        if self.image_path:
            self.process_image()
        elif self.video_path and self.cap:
            self.process_video()

    def process_image(self):
        """处理单张图像"""
        try:
            # 读取图像
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError("无法读取图像文件")

            # 使用YOLO模型处理
            results = self.model(img)

            # 绘制结果
            processed_img = results[0].plot()

            # 显示处理后的图像
            self.show_cv2_image(processed_img, self.processed_label)

            self.status_label.config(text="图像处理完成!")
            self.btn_export.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("错误", f"图像处理失败: {str(e)}")

    def process_video(self):
        """处理视频"""
        if not self.is_processing:
            self.is_processing = True
            self.btn_process.config(text="停止处理")

            # 获取视频信息
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 准备输出视频
            output_path = os.path.join(self.output_dir, f"processed_{os.path.basename(self.video_path)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 在后台线程中处理视频
            self.processing_thread = threading.Thread(
                target=self.process_video_frames,
                args=(total_frames,),
                daemon=True
            )
            self.processing_thread.start()

            # 开始更新UI
            self.update_video_processing()
        else:
            self.is_processing = False
            self.btn_process.config(text="开始处理")

    def process_video_frames(self, total_frames):
        """处理视频帧"""
        frame_count = 0

        while self.is_processing and frame_count < total_frames:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 处理帧
            results = self.model(frame)
            processed_frame = results[0].plot()

            # 保存处理后的帧
            self.video_writer.write(processed_frame)

            # 更新当前帧用于显示
            self.current_frame = processed_frame

            # 更新进度
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            self.progress["value"] = progress

            # 控制处理速度，避免UI卡顿
            time.sleep(0.01)

        # 处理完成或中断
        self.is_processing = False
        if frame_count >= total_frames:
            self.status_label.config(text="视频处理完成!")
            self.btn_export.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="视频处理已停止")

        # 释放资源
        if self.video_writer:
            self.video_writer.release()
        self.btn_process.config(text="开始处理")

    def update_video_processing(self):
        """更新视频处理时的UI"""
        if self.is_processing and self.current_frame is not None:
            self.show_cv2_image(self.current_frame, self.processed_label)
            self.root.after(50, self.update_video_processing)

    def export_video(self):
        """导出处理后的视频"""
        if self.image_path:
            # 导出处理后的图像
            output_path = os.path.join(self.output_dir, f"processed_{os.path.basename(self.image_path)}")
            img = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img)
            messagebox.showinfo("导出成功", f"图像已保存到: {output_path}")
        elif self.video_path:
            # 视频已经在处理时保存
            messagebox.showinfo("导出成功", f"视频已保存到输出目录")

    def show_image(self, image_path, label):
        """显示图像文件"""
        img = Image.open(image_path)
        img.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo

    def show_cv2_image(self, cv2_img, label):
        """显示OpenCV图像"""
        # 转换颜色空间
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        # 转换为PIL图像
        img = Image.fromarray(cv2_img)
        img.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(img)

        # 更新标签
        label.config(image=photo)
        label.image = photo

    def reset_state(self):
        """重置状态"""
        self.is_processing = False
        self.video_path = ""
        self.image_path = ""
        self.current_frame = None
        self.progress["value"] = 0

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        # 重置按钮状态
        self.btn_process.config(text="开始处理", state=tk.DISABLED)
        self.btn_export.config(state=tk.DISABLED)

        # 清空显示
        self.original_label.config(image='', text="原始图像/视频")
        self.processed_label.config(image='', text="处理后结果")

        self.status_label.config(text="就绪")

    def on_closing(self):
        """关闭窗口时的清理工作"""
        self.is_processing = False
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOAnalyzerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()