from datetime import datetime
import zipfile
import shutil
import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import os
import threading
import time
import cv2
import streamlit as st
from util import convert_video_with_ffmpeg

# ======================
# 自定义CSS样式
# ======================
st.markdown("""
<style>
    /* 主容器样式 */
    .main-container {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    /* 动态边框动画 */
    @keyframes border-glow {
        0% { box-shadow: 0 0 10px rgba(76,175,80,0.3); }
        50% { box-shadow: 0 0 15px rgba(33,150,243,0.5); }
        100% { box-shadow: 0 0 10px rgba(76,175,80,0.3); }
    }

    .section-card {
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        animation: border-glow 3s ease-in-out infinite;
    }

    /* 按钮美化 */
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50, #2196F3) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }

    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50, #34495e) !important;
        box-shadow: 2px 0 15px rgba(0,0,0,0.1);
    }

    .sidebar-title {
        color: #fff !important;
        font-size: 1.5rem !important;
        padding: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 初始化全局变量
MODEL_PATHS = {
    "YOLOv11":"D:\Python\graduate_design\Model\yolo11n.pt",
    "YOLO_detect_animals":r"D:\Python\graduate_design\Model\runs\train\exp\weights\best.pt", #
}

current_model = YOLO(r"D:\Python\graduate_design\Model\yolo11n.pt")# 默认
conf_threshold = 0
iou_threshold = 0


# 加载模型
def load_model(model_name):
    global current_model
    model_path = MODEL_PATHS.get(model_name)
    if not model_path:
        st.error("模型路径不存在！")
        return None
    try:
        model = YOLO(model_path)
        st.success(f"成功加载模型：{model_name}")
        return model
    except Exception as e:
        st.error(f"加载模型失败：{e}")
        return None

# # 实时摄像头检测
# def realtime_detection():
#     global current_model
#     st.header("📹 实时摄像头检测")
#
#     # 初始化摄像头
#     camera_placeholder = st.empty()
#     stop_button = st.button("停止检测")
#
#     # 获取摄像头设备（默认0）
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("无法访问摄像头")
#         return
#
#     try:
#         while cap.isOpened() and not stop_button:
#             # 读取视频帧
#             ret, frame = cap.read()
#             if not ret:
#                 st.warning("视频流中断")
#                 break
#
#             # YOLO推理
#             results = current_model.track(
#                 source=frame,
#                 conf=conf_threshold,
#                 iou=iou_threshold,
#                 persist=True,  # 维持跟踪状态
#                 verbose=False
#             )
#
#             # 实时标注
#             annotated_frame = results[0].plot()  # 使用内置绘图方法
#
#             # 显示实时画面
#             camera_placeholder.image(annotated_frame,
#                                      channels="BGR",
#                                      caption="实时检测画面")
#
#             # 控制帧率（根据硬件调整）
#             time.sleep(0.01)  # 约100FPS
#
#     except Exception as e:
#         st.error(f"检测异常: {str(e)}")
#     finally:
#         cap.release()
#         st.info("摄像头已释放")
#
#
# def draw_detections(frame, results, class_names) -> np.ndarray:
#     """在帧上绘制检测框和标签"""
#     annotated = frame.copy()
#     color = (0, 255, 0)  # 绿色边框
#
#     for result in results:
#         for box in result.boxes:
#             # 解析检测结果
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             conf = box.conf.item()
#             cls_id = int(box.cls.item())
#
#             # 绘制元素
#             cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
#             label = f"{class_names[cls_id]} {conf:.2f}"
#             cv2.putText(annotated, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
#
#     return annotated

# 图片检测
def image_detection():
    global current_model
    st.header("图片检测")
    #st.markdown('<div class="dynamic-border">', unsafe_allow_html=True)

    # 上传图片文件
    uploaded_file = st.file_uploader("上传图片文件", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # 创建两列布局
        col1, col2 = st.columns(2)

        # 打开原始图像
        original_image = Image.open(uploaded_file)

        # 左侧显示原始图片
        with col1:
            st.subheader("原始图片")
            st.image(original_image, use_container_width=True)

        # 右侧显示处理进度和结果
        with col2:
            st.subheader("检测结果")

            # 添加处理中的旋转动画
            with st.spinner("YOLO模型正在处理中..."):
                # 使用YOLO模型进行推理
                results = current_model(
                    source=original_image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    save=False
                )

                # 将推理结果绘制到图像上
                annotated_image = results[0].plot()  # 返回带注释的图像（numpy数组）
                annotated_image = Image.fromarray(annotated_image[..., ::-1])  # 转换为PIL图像

                # 显示处理后的图像
                st.image(annotated_image, use_container_width=True)

                # 显示检测统计信息
                num_objects = len(results[0].boxes)
                st.success(f"检测到 {num_objects} 个目标")

                # 将处理后的图像转换为字节流供下载
                buffered = BytesIO()
                annotated_image.save(buffered, format="JPEG")
                img_bytes = buffered.getvalue()

                # 添加右下角下载按钮（使用CSS固定位置）
                st.markdown(
                    """
                    <style>
                    .download-btn {
                        position: fixed;
                        bottom: 20px;
                        right: 20px;
                        z-index: 999;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # 导出按钮
                st.download_button(
                    label="📥 导出处理后的图片",
                    data=img_bytes,
                    file_name="processed_image.jpg",
                    mime="image/jpeg",
                    key="download_image",
                    help="点击下载YOLO处理后的图片",
                    use_container_width=True,
                    on_click=lambda: st.toast("图片导出成功！"),
                    type="primary"
                )

    st.markdown('</div>', unsafe_allow_html=True)

def video_detection():
    global current_model
    st.header("🎥 视频检测系统")

    # 使用容器划分主要功能区域
    upload_container = st.container(border=True)
    with upload_container:
        # 上传区域
        st.subheader("1. 视频上传")
        uploaded_file = st.file_uploader("选择待检测视频文件",
                                         type=["mp4", "avi", "mov"],
                                         label_visibility="collapsed",
                                         help="支持 MP4/AVI/MOV 格式，最大文件大小 500MB")

    if uploaded_file:
        # 保存上传的视频到临时文件
        temp_video_path = "temp_uploaded_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 视频预览和处理区域
        process_container = st.container(border=True)
        with process_container:
            st.subheader("2. 视频处理")

            # 创建对比布局
            col1, col2 = st.columns([0.45, 0.55], gap="large")

            with col1:
                # 原始视频预览
                st.markdown("**原始视频预览**")
                st.video(temp_video_path)

            with col2:
                # 处理状态区域
                status_col1, status_col2 = st.columns([0.7, 0.3])
                with status_col1:
                    st.markdown("**处理状态**")
                    progress_placeholder = st.empty()

        # 处理结果区域
        processed_video_placeholder = st.empty()
        export_placeholder = st.empty()

        # 创建输出目录
        output_dir = os.path.join("temp_results", "processed_video")
        os.makedirs(output_dir, exist_ok=True)
        # 确定输出视频路径
        output_video_name = "output_" + os.path.basename(temp_video_path)
        processed_temp_video_path = os.path.join(output_dir, "temp_processed_video.avi")  # YOLO 输出临时视频
        final_processed_video_path = os.path.join(output_dir, output_video_name)

        try:
            # 获取视频信息
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise ValueError("无法打开上传的视频文件")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # 处理进度和状态管理
            processed_frames = 0
            lock = threading.Lock()

            def process_video():
                nonlocal processed_frames
                try:
                    # 模拟 YOLO 推理和保存视频
                    results = current_model.track(
                        source=temp_video_path,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        stream=True,
                        imgsz=(width, height),
                        save=False,
                        project=None,
                        name=None,
                        verbose=False
                    )

                    # 手动保存 YOLO 处理后的临时视频
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 AVI 格式保存临时视频
                    out = cv2.VideoWriter(processed_temp_video_path, fourcc, fps, (width, height))

                    for frame_result in results:
                        processed_frame = frame_result.plot()  # 示例：绘制检测结果到帧
                        out.write(processed_frame)

                        # 更新处理进度（线程安全）
                        with lock:
                            processed_frames += 1

                    out.release()
                except Exception as e:
                    st.error(f"视频处理内部错误: {str(e)}")

            # 启动处理线程
            process_thread = threading.Thread(target=process_video)
            process_thread.start()

            # 实时更新进度
            while process_thread.is_alive():
                with lock:
                    progress = min(100, int((processed_frames / total_frames) * 100) if total_frames > 0 else 0)
                progress_placeholder.progress(
                    progress,
                    text=f"▏ 处理进度: {progress}% | 已处理 {processed_frames}/{total_frames} 帧"
                )
                time.sleep(0.1)

            process_thread.join()

            # 视频格式转换
            if os.path.exists(processed_temp_video_path):
                convert_video_with_ffmpeg(processed_temp_video_path, final_processed_video_path)

            # 显示处理结果
            with process_container:
                with col2:
                    if os.path.exists(final_processed_video_path):
                        processed_video_placeholder.markdown("**检测结果预览**")
                        processed_video_placeholder.video(final_processed_video_path)

                        # 导出按钮样式优化
                        with export_placeholder:
                            with open(final_processed_video_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ 导出检测结果视频",
                                    data=f,
                                    file_name=output_video_name,
                                    mime="video/mp4",
                                    use_container_width=True,
                                    type="primary"
                                )

        except Exception as e:
            st.error(f"❌ 处理过程中发生错误: {str(e)}", icon="🚨")

        finally:
            # 清理临时文件
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(processed_temp_video_path):
                os.remove(processed_temp_video_path)

def model_usage():
    st.title("🦁 智能动物识别系统")
    st.markdown("---")

    # 侧边栏配置
    with st.sidebar:
        st.markdown('<h1 class="sidebar-title">⚙️ 控制面板</h1>', unsafe_allow_html=True)

        # 模型选择
        model_name = st.selectbox(
            "**选择检测模型**",
            list(MODEL_PATHS.keys()),
            index=0,
            help="选择预训练的YOLO模型"
        )
        current_model = load_model(model_name)

        # 参数设置
        st.markdown("---")
        st.markdown("**⚖️ 检测参数**")
        conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25, step=0.01)
        iou_threshold = st.slider("IoU 阈值", 0.0, 1.0, 0.5, step=0.01)

        # 设备设置
        st.markdown("---")
        st.markdown("**📷 输入源设置**")
        camera_devices = ["默认摄像头", "外部设备1", "外部设备2"]
        selected_camera = st.selectbox("视频输入源", camera_devices)

        # 功能导航
        st.markdown("---")
        st.markdown("**🎯 检测模式**")
        task = st.radio("", ["图片检测", "视频检测", "实时检测"], index=0)

    # 主内容区域
    with st.container():
        #st.markdown('<div class="main-container">', unsafe_allow_html=True)

        if task == "图片检测":
            with st.container():
                # st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.header("📸 图像检测")

                # 文件上传区域
                uploaded_file = st.file_uploader(
                    "上传待检测图片",
                    type=["jpg", "jpeg", "png"],
                    help="支持JPG/JPEG/PNG格式，最大尺寸5MB"
                )

                if uploaded_file:
                    cols = st.columns([1, 1], gap="large")
                    with cols[0]:
                        st.markdown("#### 原始图像")
                        st.image(uploaded_file, use_container_width=True)

                    with cols[1]:
                        with st.spinner("🔍 正在检测中..."):
                            # 模型推理
                            results = current_model.predict(
                                source=Image.open(uploaded_file),
                                conf=conf_threshold,
                                iou=iou_threshold
                            )
                            annotated_image = results[0].plot()[:, :, ::-1]

                            st.markdown("#### 检测结果")
                            st.image(annotated_image, use_container_width=True)

                            # 结果统计
                            num_objects = len(results[0].boxes)
                            st.success(f"✅ 检测到 {num_objects} 个目标")

                            # 导出功能
                            buf = BytesIO()
                            Image.fromarray(annotated_image).save(buf, format="PNG")
                            st.download_button(
                                label="💾 保存检测结果",
                                data=buf.getvalue(),
                                file_name="detection_result.png",
                                mime="image/png",
                                use_container_width=True
                            )
                st.markdown('</div>', unsafe_allow_html=True)

        elif task == "视频检测":
            # with st.container():
            #     st.markdown('<div class="section-card">', unsafe_allow_html=True)
            #     st.header("🎥 视频检测")
            #
            #     # 文件上传区域
            #     uploaded_file = st.file_uploader(
            #         "上传待检测视频",
            #         type=["mp4", "avi"],
            #         help="支持MP4/AVI格式，最大尺寸500MB"
            #     )
            #
            #     if uploaded_file:
            #         # 视频预览与处理
            #         video_path = f"temp_{uploaded_file.name}"
            #         with open(video_path, "wb") as f:
            #             f.write(uploaded_file.getbuffer())
            #
            #         cols = st.columns([1, 1], gap="large")
            #         with cols[0]:
            #             st.markdown("#### 原始视频")
            #             st.video(uploaded_file)
            #
            #         with cols[1]:
            #             st.markdown("#### 处理进度")
            #             progress_bar = st.progress(0)
            #             status_text = st.empty()
            #
            #             # 模拟处理过程
            #             for percent in range(100):
            #                 time.sleep(0.02)
            #                 progress_bar.progress(percent + 1)
            #                 status_text.text(f"▏ 处理进度: {percent + 1}%")
            #
            #             st.success("✅ 处理完成！")
            #             st.download_button(
            #                 label="📥 下载结果视频",
            #                 data=open(video_path, "rb"),
            #                 file_name="processed_video.mp4",
            #                 use_container_width=True
            #             )
            #     st.markdown('</div>', unsafe_allow_html=True)
            video_detection()

        elif task == "实时检测":
            with st.container():
                #st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.header("🌐 实时检测")

                # 摄像头控制
                if st.button("🚀 启动实时检测", use_container_width=True):
                    cap = cv2.VideoCapture(0)
                    frame_placeholder = st.empty()
                    stop_button = st.button("🛑 停止检测", use_container_width=True)

                    while cap.isOpened() and not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("无法获取视频流")
                            break

                        # 实时推理
                        results = current_model.track(
                            frame,
                            conf=conf_threshold,
                            iou=iou_threshold
                        )
                        annotated_frame = results[0].plot()
                        frame_placeholder.image(annotated_frame, channels="BGR")

                    cap.release()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # # 显示示例图片
    # image = Image.open('D:\Python\graduate_design\img.png')  # 确保 'img.png' 文件存在
    # st.image(image, caption='demo', use_container_width=True)

def model_train():
    # 页面设置
    # st.set_page_config(page_title="YOLO模型训练平台", layout="wide")

    # 标题
    st.title("基于YOLO的动物识别模型训练")
    st.markdown("---")

    # 初始化会话状态
    if 'training' not in st.session_state:
        st.session_state.training = False
    if 'process' not in st.session_state:
        st.session_state.process = None
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'dataset_stats' not in st.session_state:
        st.session_state.dataset_stats = {"train": 0, "val": 0}


    # 侧边栏 - 导航
    st.sidebar.title("参数设置")
    page = st.sidebar.radio("选择页面", ["数据集配置", "模型训练", "训练监控"])

    # 数据集配置页面
    if page == "数据集配置":
        with st.container():
            st.markdown('<div class="dynamic-border">', unsafe_allow_html=True)
            st.header("📁 数据集配置")

            # 数据集上传部分
            with st.expander("上传数据集", expanded=True):
                uploaded_file = st.file_uploader("上传ZIP格式的数据集", type=["zip"])

                if uploaded_file is not None:
                    # 创建临时目录
                    temp_dir = "temp_dataset"
                    os.makedirs(temp_dir, exist_ok=True)

                    # 保存上传的文件
                    zip_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # 解压文件
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    # 检查YOLO格式
                    st.info("检查数据集结构...")
                    required_folders = ["images", "labels"]
                    required_splits = ["train", "val"]

                    valid_structure = True
                    for folder in required_folders:
                        folder_path = os.path.join(temp_dir, folder)
                        if not os.path.exists(folder_path):
                            valid_structure = False
                            st.error(f"❌ 缺少 {folder} 文件夹")
                            break
                        for split in required_splits:
                            split_path = os.path.join(folder_path, split)
                            if not os.path.exists(split_path):
                                valid_structure = False
                                st.error(f"❌ 在 {folder} 中缺少 {split} 子文件夹")
                                break

                    if valid_structure:
                        st.success("✅ 数据集结构验证通过 (YOLO格式)")

                        # 计算数据集统计信息
                        train_images = len(os.listdir(os.path.join(temp_dir, "images", "train")))
                        val_images = len(os.listdir(os.path.join(temp_dir, "images", "val")))
                        st.session_state.dataset_stats = {"train": train_images, "val": val_images}

                        # 显示动态统计信息
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("训练图像数量", train_images)
                        with col2:
                            st.metric("验证图像数量", val_images)

                        # 保存数据集选项
                        dataset_name = st.text_input("输入数据集名称", "my_dataset")
                        if st.button("💾 保存数据集"):
                            if dataset_name:
                                dataset_dir = os.path.join("datasets", dataset_name)
                                os.makedirs(dataset_dir, exist_ok=True)

                                # 移动文件
                                for item in os.listdir(temp_dir):
                                    s = os.path.join(temp_dir, item)
                                    d = os.path.join(dataset_dir, item)
                                    if os.path.isdir(s):
                                        shutil.copytree(s, d, dirs_exist_ok=True)
                                    else:
                                        shutil.copy2(s, d)

                                st.success(f"🎉 数据集 '{dataset_name}' 已保存!")
                                st.session_state.selected_dataset = dataset_name

                                # 清理临时文件
                                shutil.rmtree(temp_dir)
            st.markdown('</div>', unsafe_allow_html=True)

    # # 模型训练页面
    elif page == "模型训练":
        with st.container():
            st.markdown('<div class="dynamic-border">', unsafe_allow_html=True)
            st.header("⚙️ 模型训练配置")

            # 选择数据集部分
            dataset_dir = "datasets"
            if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
                datasets = os.listdir(dataset_dir)
                selected_dataset = st.selectbox(
                    "选择数据集",
                    datasets,
                    index=datasets.index(
                        st.session_state.selected_dataset) if st.session_state.selected_dataset in datasets else 0
                )

                # 显示动态数据集信息
                with st.expander("数据集信息", expanded=True):
                    dataset_path = os.path.join(dataset_dir, selected_dataset)
                    if os.path.exists(dataset_path):
                        train_images = len(os.listdir(os.path.join(dataset_path, "images", "train")))
                        val_images = len(os.listdir(os.path.join(dataset_path, "images", "val")))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("训练图像数量", train_images)
                        with col2:
                            st.metric("验证图像数量", val_images)
                    else:
                        st.warning("数据集路径不存在")

                # 动态模型配置部分
                with st.expander("模型配置", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        model_size = st.selectbox("模型尺寸",
                                                  ["nano (n)", "small (s)", "medium (m)", "large (l)", "xlarge (x)"],
                                                  index=1)
                        epochs = st.slider("训练轮次 (epochs)", min_value=1, max_value=500, value=100)
                        batch_size = st.selectbox("批次大小 (batch size)", [4, 8, 16, 32, 64], index=1)

                    with col2:
                        img_size = st.selectbox("图像尺寸 (image size)", [320, 416, 512, 640], index=3)
                        learning_rate = st.slider("学习率 (learning rate)", min_value=0.0001, max_value=0.1, value=0.01,
                                                  step=0.001, format="%.3f")
                        patience = st.number_input("早停耐心值 (patience)", min_value=1, value=50)

                # 动态高级选项部分
                with st.expander("高级选项"):
                    col3, col4 = st.columns(2)

                    with col3:
                        optimizer = st.selectbox("优化器", ["SGD", "Adam", "AdamW"], index=0)
                        weight_decay = st.number_input("权重衰减 (weight decay)", min_value=0.0, value=0.0005,
                                                       step=0.0001, format="%.4f")

                    with col4:
                        augment = st.checkbox("数据增强", value=True)
                        save_period = st.number_input("保存间隔 (save period)", min_value=1, value=10)

                # 动态训练控制部分
                with st.expander("训练控制", expanded=True):
                    if st.button("🚀 开始训练") and not st.session_state.training:
                        st.session_state.training = True

                        # 创建输出目录
                        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = os.path.join("runs", f"train_{current_time}")
                        os.makedirs(output_dir, exist_ok=True)

                        # 显示动态训练信息
                        st.info("训练配置信息:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**数据集:** {selected_dataset}")
                            st.write(f"**模型尺寸:** {model_size}")
                            st.write(f"**训练轮次:** {epochs}")
                        with col2:
                            st.write(f"**批次大小:** {batch_size}")
                            st.write(f"**图像尺寸:** {img_size}")
                            st.write(f"**学习率:** {learning_rate}")

                        # 模拟训练过程
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i in range(1, 101):
                            progress_bar.progress(i)
                            status_text.text(f"训练进度: {i}%")
                            # 这里应该是实际的训练过程

                            # time.sleep(0.1)  # 模拟训练延迟

                        st.session_state.training = False
                        st.success("🎉 训练完成!")

                    elif st.session_state.training:
                        st.warning("训练正在进行中...")
                        if st.button("🛑 停止训练"):
                            st.session_state.training = False
                            st.experimental_rerun()
            else:
                st.warning("没有可用的数据集，请先上传数据集")
            st.markdown('</div>', unsafe_allow_html=True)
    #
    # # 训练监控页面
    elif page == "训练监控":
        st.write("demo")
    #     with st.container():
    #         st.markdown('<div class="dynamic-border">', unsafe_allow_html=True)
    #         st.header("📊 训练监控")
    #
    #         # 检查是否有训练运行
    #         runs_dir = "runs"
    #         if os.path.exists(runs_dir) and len(os.listdir(runs_dir)) > 0:
    #             runs = sorted(os.listdir(runs_dir), reverse=True)
    #             selected_run = st.selectbox("选择训练运行", runs)
    #
    #             run_path = os.path.join(runs_dir, selected_run)
    #
    #             # 动态显示训练结果
    #             with st.expander("训练结果", expanded=True):
    #                 # 模拟结果图表
    #                 col1, col2 = st.columns(2)
    #                 with col1:
    #                     st.line_chart({"损失": [0.8, 0.6, 0.4, 0.3, 0.25, 0.2]}, height=300)
    #                     st.caption("训练损失曲线")
    #                 with col2:
    #                     st.line_chart({"准确率": [0.2, 0.4, 0.6, 0.7, 0.75, 0.8]}, height=300)
    #                     st.caption("验证准确率曲线")
    #
    #             # 动态显示训练日志
    #             with st.expander("训练日志"):
    #                 # 模拟日志内容
    #                 log_content = """
    # [2023-01-01 10:00:00] 训练开始: yolov8s on my_dataset
    # [2023-01-01 10:05:00] Epoch 1/100 - loss: 0.8 - accuracy: 0.2
    # [2023-01-01 10:10:00] Epoch 10/100 - loss: 0.6 - accuracy: 0.4
    # [2023-01-01 10:15:00] Epoch 20/100 - loss: 0.4 - accuracy: 0.6
    # [2023-01-01 10:20:00] Epoch 30/100 - loss: 0.3 - accuracy: 0.7
    # [2023-01-01 10:25:00] Epoch 40/100 - loss: 0.25 - accuracy: 0.75
    # [2023-01-01 10:30:00] Epoch 50/100 - loss: 0.2 - accuracy: 0.8
    # [2023-01-01 10:35:00] 训练完成 - 最佳模型保存在 runs/train_20230101_100000/weights/best.pt
    #                 """
    #                 st.text_area("日志内容", log_content, height=300)
    #         else:
    #             st.warning("没有可用的训练运行")
    #         st.markdown('</div>', unsafe_allow_html=True)

# 主界面
def main():
## 边框功能
    # Streamlit 应用
    st.sidebar.markdown('<h1 class="sidebar-title">🧭 导航菜单</h1>', unsafe_allow_html=True)
    page = st.sidebar.radio("",
                        ["模型选择（已提供）", "模型自定义（训练）"],
                        index=0,
                        format_func=lambda x: "🔍 快速检测" if x == "模型选择（已提供）" else "🛠️ 模型训练"
                        )

    if page == "模型选择（已提供）":
        model_usage()
    else:
        model_train()


# 运行 Streamlit 应用
if __name__ == "__main__":
    main()
