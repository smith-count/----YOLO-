import csv
from datetime import datetime
import zipfile
import shutil
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import os
import threading
import time
import cv2
import streamlit as st
from util import convert_video_with_ffmpeg
import plotly.express as px

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
detection_data = []
detections_df = None
speed_df = None
flag = False # 用于检测有无图像
target_class = []


def process_yolo_results(results, class_list=None, conf_thres=0.1):
    global flag
    # """
    # 处理YOLO结果并返回可直接显示的图像
    # :param results: YOLO检测结果(单个Results对象)
    # :param class_list: 要显示的类别列表
    # :param conf_thres: 置信度阈值
    # :return: 可直接显示的numpy数组图像(BGR格式)
    # """
    # 1. 结果过滤
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

def yolo_results_to_dataframe(results):
    """
    将YOLO检测结果转换为结构化DataFrame
    """
    global detection_data
    result = results[0]

    try:
        for i, box in enumerate(result.boxes, 1):
            detection_data.append({
            "序号": i,
            "类别ID": int(box.cls.item()),
            "类别名称": result.names[int(box.cls.item())],
            "置信度": float(box.conf.item()),
            "x1": round(box.xyxy[0][0].item()),
            "y1": round(box.xyxy[0][1].item()),
            "x2": round(box.xyxy[0][2].item()),
            "y2": round(box.xyxy[0][3].item()),
            "宽度": round(box.xyxy[0][2].item() - box.xyxy[0][0].item()),
            "高度": round(box.xyxy[0][3].item() - box.xyxy[0][1].item())
            })
    except Exception as e:
            st.error(f"处理失败: {str(e)}")
            return

    return pd.DataFrame(detection_data), pd.DataFrame([result.speed])
# 图片检测
def image_detection():
    global current_model
    global detection_data
    global detections_df
    global speed_df
    global flag
    global iou_threshold
    global conf_threshold

    st.header("图片检测")

    # 上传图片文件
    uploaded_file = st.file_uploader("上传图片文件",
                                   type=["jpg", "jpeg", "png"],
                                   label_visibility="visible")

    if uploaded_file:
        flag = True
        # 创建两列布局
        col1, col2 = st.columns(2)

        # 打开原始图像
        original_image = Image.open(uploaded_file)
        img_array = np.array(original_image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 左侧显示原始图片
        with col1:
            st.subheader("原始图片")
            st.image(original_image,  use_container_width=True)

        # 右侧显示处理进度和结果
        with col2:
            st.subheader("检测结果")

            with st.spinner("YOLO模型正在处理中..."):
                try:
                    # 使用YOLO模型进行推理
                    results = current_model(
                        source=img_cv,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        save=False
                    )

                    process_yolo_results(results[0],class_list=target_class,
                                                    )
                    annotated_image = results[0].plot()
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    st.image(annotated_image, use_container_width=True)

                    # 转换结果为DataFrame
                    if hasattr(results[0], 'boxes'):
                        detections_df, speed_df = yolo_results_to_dataframe(results)

                    # 添加下载功能
                    buffered = BytesIO()
                    Image.fromarray(annotated_image).save(buffered, format="JPEG")
                    st.download_button(
                        label="📥 导出检测图片",
                        data=buffered.getvalue(),
                        file_name=f"detection_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
                        mime="image/jpeg",
                        type="primary"
                    )

                except Exception as e:
                    st.error(f"处理失败: {str(e)}")
                    return

        # 下方显示详细数据
        st.divider()
        st.subheader("检测数据详情")

        if flag :
            # 格式化表格
            display_df = detections_df.copy()
            display_df['置信度'] = display_df['置信度'].apply(lambda x: f"{x:.2%}")

            # 交互式表格
            st.dataframe(
                display_df[['序号', '类别名称', '置信度', 'x1', 'y1', 'x2', 'y2']],
                column_config={
                    "置信度": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=1,
                        format="%.2f%%"
                    )
                },
                hide_index=True,
                use_container_width=True
            )

            # 统计信息
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("检测目标总数", len(detections_df))
            with col_stat2:
                st.metric("平均置信度", f"{detections_df['置信度'].mean():.2%}")

            # 类别分布
            st.subheader("类别分布")
            st.bar_chart(detections_df['类别名称'].value_counts())

            # 处理速度和数据下载
            with st.expander("高级选项"):
                tab1, tab2 = st.tabs(["性能指标", "数据导出"])
                with tab1:
                    st.dataframe(
                        speed_df.rename(columns={
                            'preprocess': '预处理(ms)',
                            'inference': '推理(ms)',
                            'postprocess': '后处理(ms)'
                        }),
                        use_container_width=True
                    )
                with tab2:
                    st.download_button(
                        label="📊 导出检测数据(CSV)",
                        data=detections_df.to_csv(index=False).encode('utf-8'),
                        file_name="detection_data.csv",
                        mime="text/csv"
                    )

        # 控制台输出（调试用）
        #     print("==== 检测结果 ====")
        #     print(detections_df[['序号', '类别名称', '置信度', 'x1', 'y1', 'x2', 'y2']].to_string(index=False))
        #     print("\n==== 处理速度 (ms) ====")
        #     print(speed_df.to_string(index=False))



def video_detection():
    global current_model
    global detection_data
    global conf_threshold
    global iou_threshold
    st.header("🎥 视频检测系统")

    # # 在侧边栏添加类别选择功能
    # with st.sidebar.expander("🔍 检测类别设置", expanded=True):
    #     # 获取模型支持的类别列表
    #     class_options = list(current_model.names.values())
    #     selected_classes = st.multiselect(
    #         "选择要检测的类别",
    #         options=class_options,
    #         default=class_options[:3],  # 默认选择前3个类别
    #         help="选择需要检测的目标类别"
    #     )

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
        visualization_placeholder = st.empty()

        # 创建输出目录
        output_dir = os.path.join("temp_results", "processed_video")
        os.makedirs(output_dir, exist_ok=True)
        output_video_name = "output_" + os.path.basename(temp_video_path)
        processed_temp_video_path = os.path.join(output_dir, "temp_processed_video.avi")
        final_processed_video_path = os.path.join(output_dir, output_video_name)
        detection_csv_path = os.path.join(output_dir, "detection_results.csv")

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

            # 初始化检测结果收集
            frame_timestamps = []
            lock = threading.Lock()

            def process_video():
                nonlocal processed_frames
                try:
                    # 初始化CSV文件
                    with open(detection_csv_path, mode='w', newline='') as csv_file:
                        fieldnames = ['frame_num', 'timestamp', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2']
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writeheader()

                    # 使用YOLO进行视频检测
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

                    # 手动保存处理后的视频
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(processed_temp_video_path, fourcc, fps, (width, height))

                    for frame_idx, frame_result in enumerate(results):
                        # 获取当前帧时间戳（秒）
                        current_time = frame_idx / fps

                        # 筛选指定类别的检测结果
                        if target_class:  # 如果用户选择了特定类别
                            keep_idx = [
                                i for i, box in enumerate(frame_result.boxes)
                                if current_model.names[int(box.cls)] in target_class
                            ]
                            frame_result.boxes = frame_result.boxes[keep_idx]
                            if hasattr(frame_result, 'masks') and frame_result.masks is not None:
                                frame_result.masks = frame_result.masks[keep_idx]
                            if hasattr(frame_result, 'keypoints') and frame_result.keypoints is not None:
                                frame_result.keypoints = frame_result.keypoints[keep_idx]

                        # 绘制检测结果到帧
                        processed_frame = frame_result.plot()
                        out.write(processed_frame)

                        # 收集检测数据（只记录选中的类别）
                        for detection in frame_result.boxes:
                            class_id = int(detection.cls)
                            class_name = current_model.names[class_id]
                            conf = float(detection.conf)
                            bbox = detection.xyxy[0].tolist()

                            # 线程安全地更新数据
                            with lock:
                                detection_data.append({
                                    'frame_num': frame_idx,
                                    'timestamp': current_time,
                                    'class': class_name,
                                    'confidence': conf,
                                    'x1': bbox[0],
                                    'y1': bbox[1],
                                    'x2': bbox[2],
                                    'y2': bbox[3]
                                })
                                processed_frames = frame_idx + 1
                                frame_timestamps.append(current_time)

                        # 每处理10帧或结束时保存一次CSV
                        if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                            with lock:
                                with open(detection_csv_path, mode='a', newline='') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                                    writer.writerows(detection_data[-10:])

                    out.release()
                except Exception as e:
                    st.error(f"视频处理内部错误: {str(e)}")

            # 启动处理线程
            processed_frames = 0
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

                        # 导出按钮
                        with export_placeholder:
                            col_export1, col_export2 = st.columns(2)
                            with col_export1:
                                with open(final_processed_video_path, "rb") as f:
                                    st.download_button(
                                        label="⬇️ 导出结果视频",
                                        data=f,
                                        file_name=output_video_name,
                                        mime="video/mp4",
                                        use_container_width=True,
                                        type="primary"
                                    )
                            with col_export2:
                                with open(detection_csv_path, "rb") as f:
                                    st.download_button(
                                        label="📊 导出检测数据(CSV)",
                                        data=f,
                                        file_name="detection_results.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )

            # 数据可视化（只显示选中的类别）
            if os.path.exists(detection_csv_path):
                with visualization_placeholder.container(border=True):
                    st.subheader("3. 检测数据分析")

                    # 加载检测数据
                    df = pd.read_csv(detection_csv_path)

                    # 筛选选中的类别
                    if target_class:
                        df = df[df['class'].isin(target_class)]

                    if not df.empty:
                        # 创建可视化选项卡
                        tab1, tab2, tab3 = st.tabs(["类别分布", "时间趋势", "空间分布"])

                        with tab1:
                            st.markdown("**检测类别统计**")
                            species_counts = df['class'].value_counts().reset_index()
                            species_counts.columns = ['Class', 'Count']
                            fig1 = px.bar(species_counts,
                                          x='Class',
                                          y='Count',
                                          color='Class',
                                          text='Count')
                            st.plotly_chart(fig1, use_container_width=True)

                        with tab2:
                            st.markdown("**检测结果时间分布**")
                            df['time_interval'] = (df['timestamp'] // 5) * 5  # 5秒间隔分组
                            time_dist = df.groupby(['time_interval', 'class']).size().reset_index(name='count')
                            fig2 = px.line(time_dist,
                                           x='time_interval',
                                           y='count',
                                           color='class',
                                           markers=True)
                            fig2.update_xaxes(title="时间 (秒)")
                            fig2.update_yaxes(title="检测数量")
                            st.plotly_chart(fig2, use_container_width=True)

                        with tab3:
                            st.markdown("**检测目标空间分布**")
                            fig3 = px.scatter(df,
                                              x='x1',
                                              y='y1',
                                              color='class',
                                              size='confidence',
                                              hover_data=['frame_num', 'confidence'])
                            fig3.update_xaxes(range=[0, width])
                            fig3.update_yaxes(range=[height, 0])  # 反转Y轴匹配图像坐标
                            st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("未检测到选定类别的目标")

        except Exception as e:
            st.error(f"❌ 处理过程中发生错误: {str(e)}", icon="🚨")

        finally:
            # 清理临时文件
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(processed_temp_video_path):
                os.remove(processed_temp_video_path)


def real_time_detection():
    """实时目标检测函数"""
    st.title("实时目标检测")
    global current_model
    global conf_threshold
    global iou_threshold


    start_button = st.button("开始检测")
    stop_button = st.button("停止检测")

    if start_button:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("无法打开视频源")
            return

        st_frame = st.empty()  # 用于动态更新画面的占位符
        stats_placeholder = st.empty()  # 统计信息占位符

        while cap.isOpened() and not stop_button:
            success, frame = cap.read()
            if not success:
                st.warning("视频流结束")
                break

            # 执行检测
            start_time = time.time()
            results = current_model(frame,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            verbose=False)

            # 计算FPS
            fps = 1 / (time.time() - start_time + 1e-9)

            # 绘制结果
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # 显示统计信息
            num_objects = len(results[0].boxes)
            stats_text = f"""
            **检测统计**  
            • 目标数量: {num_objects}  
            • 置信度阈值: {conf_threshold:.2f}  
            • 实时FPS: {fps:.1f}  
            """

            # 更新界面
            st_frame.image(annotated_frame, caption="实时检测画面", use_container_width=True)
            stats_placeholder.markdown(stats_text)

            # 控制帧率 (默认30FPS)
            time.sleep(1 / 30)

        cap.release()
        cv2.destroyAllWindows()

def model_usage():
    global current_model
    global iou_threshold
    global conf_threshold

    st.title("🦁 基于YOLO的动物识别系统")
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

        if task == "图片检测":
            image_detection()

        elif task == "视频检测":
            video_detection()

        elif task == "实时检测":
            real_time_detection()

import yaml

# 配置常量
TEMP_DIR = "temp_uploads"
DATA_YAML = os.path.join(TEMP_DIR, "data.yaml")
MODEL_YAML = os.path.join(TEMP_DIR, "model.yaml")


def setup_temp_dir():
    """设置临时目录"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


def cleanup_temp_dir():
    """清理临时目录"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


def save_uploaded_file(uploaded_file, save_path):
    """保存单个上传的文件"""
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def create_data_yaml(train_dir, val_dir, class_names):
    """创建data.yaml配置文件"""
    data = {
        'train': train_dir,
        'val': val_dir,
        'nc': len(class_names),
        'names': class_names
    }

    with open(DATA_YAML, 'w') as f:
        yaml.dump(data, f)

    return DATA_YAML



def run_yolo_training(params):
    """执行YOLO训练命令"""
    cmd = [
        'yolo',
        'task=detect',
        'mode=train',
        f'model={params["model_config"]}',
        f'data={params["data_config"]}',
        f'imgsz={params["imgsz"]}',
        f'epochs={params["epochs"]}',
        f'batch={params["batch"]}',
        f'workers={params["workers"]}',
        f'device={params["device"]}',
        f'optimizer={params["optimizer"]}',
        f'project={params["project"]}',
        f'name={params["name"]}',
    ]

    # 添加布尔参数
    if params["cache"]:
        cmd.append('cache=True')
    else:
        cmd.append('cache=False')

    if params["single_cls"]:
        cmd.append('single_cls=True')
    else:
        cmd.append('single_cls=False')

    if params["amp"]:
        cmd.append('amp=True')
    else:
        cmd.append('amp=False')

    if params["close_mosaic"] > 0:
        cmd.append(f'close_mosaic={params["close_mosaic"]}')

    # 执行命令
    st.info("开始训练...")
    st.code(" ".join(cmd))

    # 这里实际上应该使用subprocess或其他方式运行训练
    # 例如: subprocess.run(cmd, check=True)
    # 为了演示，我们只是显示命令

    # 模拟训练过程
    with st.spinner("训练进行中..."):
        import time
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)

    st.success("训练完成!")


def model_train():
    # 页面设置
    """主函数"""
    st.title("YOLO 模型训练配置")

    # 初始化临时目录
    setup_temp_dir()

    # 创建表单
    with st.form("train_form"):
        # 1. 上传模型配置文件
        st.subheader("1. 上传模型配置文件")
        model_file = st.file_uploader(
            "上传YOLO模型配置文件 (.yaml)",
            type=['yaml'],
            help="上传YOLO模型结构配置文件"
        )

        # 2. 上传训练数据
        st.subheader("2. 上传训练数据")
        col1, col2 = st.columns(2)

        with col1:
            train_images = st.file_uploader(
                "上传训练图片",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="上传训练集图片文件"
            )

            train_labels = st.file_uploader(
                "上传训练标注",
                type=['txt'],
                accept_multiple_files=True,
                help="上传训练集标注文件 (YOLO格式)"
            )

        with col2:
            val_images = st.file_uploader(
                "上传验证图片",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="上传验证集图片文件"
            )

            val_labels = st.file_uploader(
                "上传验证标注",
                type=['txt'],
                accept_multiple_files=True,
                help="上传验证集标注文件 (YOLO格式)"
            )

        # 3. 配置数据集信息
        st.subheader("3. 配置数据集信息")
        class_names = st.text_area(
            "类别名称 (每行一个)",
            value="class1\nclass2\nclass3",
            help="输入所有类别名称，每行一个"
        )

        # 4. 训练参数配置
        st.subheader("4. 训练参数配置")
        col1, col2 = st.columns(2)

        with col1:
            epochs = st.number_input(
                "训练轮数 (epochs)",
                min_value=1,
                max_value=1000,
                value=100,
                step=1
            )

            batch = st.number_input(
                "批量大小 (batch)",
                min_value=1,
                max_value=64,
                value=8,
                step=1
            )

            imgsz = st.number_input(
                "图像大小 (imgsz)",
                min_value=32,
                max_value=1280,
                value=640,
                step=32
            )

        with col2:
            workers = st.number_input(
                "工作线程数 (workers)",
                min_value=0,
                max_value=16,
                value=0,
                step=1
            )

            device = st.text_input(
                "设备 (device)",
                value="0",
                help="使用的GPU设备ID，如'0'或'0,1,2'，CPU使用'cpu'"
            )

            optimizer = st.selectbox(
                "优化器 (optimizer)",
                options=["SGD", "Adam", "AdamW", "RMSprop"],
                index=0
            )

        # 5. 高级选项
        st.subheader("5. 高级选项")
        cache = st.checkbox(
            "使用缓存 (cache)",
            value=False,
            help="是否使用缓存加速训练"
        )

        single_cls = st.checkbox(
            "单类别检测 (single_cls)",
            value=False,
            help="是否为单类别检测任务"
        )

        amp = st.checkbox(
            "自动混合精度 (amp)",
            value=True,
            help="是否使用自动混合精度训练"
        )

        close_mosaic = st.number_input(
            "关闭马赛克增强的轮数 (close_mosaic)",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            help="最后N轮关闭马赛克增强"
        )

        # 6. 输出配置
        st.subheader("6. 输出配置")
        project = st.text_input(
            "项目目录 (project)",
            value="runs/train",
            help="训练结果保存的根目录"
        )

        name = st.text_input(
            "实验名称 (name)",
            value="exp",
            help="训练实验的名称"
        )

        # 提交按钮
        submitted = st.form_submit_button("开始训练")

        if submitted:
            # 验证必要文件是否上传
            if not model_file:
                st.error("请上传模型配置文件!")
                return

            if not train_images or not val_images:
                st.error("请上传训练和验证图片!")
                return

            if not train_labels or not val_labels:
                st.error("请上传训练和验证标注!")
                return

            # 保存模型配置文件
            model_config_path = save_uploaded_file(model_file, MODEL_YAML)

            # 创建目录结构
            train_img_dir = os.path.join(TEMP_DIR, "train", "images")
            train_label_dir = os.path.join(TEMP_DIR, "train", "labels")
            val_img_dir = os.path.join(TEMP_DIR, "val", "images")
            val_label_dir = os.path.join(TEMP_DIR, "val", "labels")

            os.makedirs(train_img_dir, exist_ok=True)
            os.makedirs(train_label_dir, exist_ok=True)
            os.makedirs(val_img_dir, exist_ok=True)
            os.makedirs(val_label_dir, exist_ok=True)

            # 保存训练数据
            for img in train_images:
                save_uploaded_file(img, os.path.join(train_img_dir, img.name))

            for label in train_labels:
                save_uploaded_file(label, os.path.join(train_label_dir, label.name))

            # 保存验证数据
            for img in val_images:
                save_uploaded_file(img, os.path.join(val_img_dir, img.name))

            for label in val_labels:
                save_uploaded_file(label, os.path.join(val_label_dir, label.name))

            # 创建data.yaml
            class_list = [name.strip() for name in class_names.split('\n') if name.strip()]
            data_config_path = create_data_yaml(
                os.path.join(TEMP_DIR, "train"),
                os.path.join(TEMP_DIR, "val"),
                class_list
            )

            # 收集所有参数
            params = {
                "model_config": model_config_path,
                "data_config": data_config_path,
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "workers": workers,
                "device": device,
                "optimizer": optimizer,
                "close_mosaic": close_mosaic,
                "cache": cache,
                "single_cls": single_cls,
                "amp": amp,
                "project": project,
                "name": name,
            }

            # 显示配置信息
            st.subheader("训练配置摘要")
            st.json(params)

            # 运行训练
            run_yolo_training(params)

    # 清理临时目录
    if st.button("清理临时文件"):
        cleanup_temp_dir()
        st.success("临时文件已清理!")



def main():
    global target_class
    # 导航菜单
    st.sidebar.markdown('<h1 class="sidebar-title">🧭 导航菜单</h1>', unsafe_allow_html=True)
    page = st.sidebar.radio("",
                          ["模型选择（已提供）", "模型自定义（训练）"],
                          index=0,
                          format_func=lambda x: "🔍 快速检测" if x == "模型选择（已提供）" else "🛠️ 模型训练"
                          )



    if page == "模型选择（已提供）":
        # 全局设置区域（显示在导航菜单下方）
        with st.sidebar.expander("⚙️ 检测设置", expanded=True):
            # 单类/多类识别选择
            detection_mode = st.radio(
                "检测模式",
                ["多类识别", "单类识别"],
                index=0,
                help="选择是否只检测特定类别的目标"
            )

            # 类选择器
            if detection_mode == "单类识别":
                # 这里替换为你的实际类别列表
                class_options = ["cat", "dog", "bird", "teddy bear"]
                target_class = st.selectbox(
                    "选择要识别的目标类别",
                    options=class_options,
                    index=0
                )
            else:
                target_class = ["cat", "dog", "bird", "teddy bear"]
        model_usage()
    else:
        model_train()


# 运行 Streamlit 应用
if __name__ == "__main__":
    main()
