import csv
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
from io import BytesIO
import threading
import time
import streamlit as st
from util import *
import plotly.express as px
import warnings

# 这必须是第一个 Streamlit 调用！
st.set_page_config(
    page_title="基于YOLO模型的动物识别系统"
)

# 自定义CSS样式
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
    "YOLOv8_Provided":"D:\Python\graduate_design\Model\yolov8n.pt",
    "YOLOv11_Provided":"D:\Python\graduate_design\Model\yolo11n.pt",
}

current_model = YOLO(r"D:\Python\graduate_design\Model\yolo11n.pt")# 默认
conf_threshold = 0
iou_threshold = 0
detection_data = []
detections_df = None
speed_df = None
flag = False # 用于检测有无图像
target_class = []
detection_mode = None


# 加载模型
# 将模型应用到全局变量current_mode
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

# 将YOLO检测结果转换为结构化DataFrame用于数据清洗
def yolo_results_to_dataframe(results):
    global detection_data
    # 获取Yolo模型解析的数据
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

# 处理Yolo结果 # 根据检测内容（Target_list）清洗Yolo检测结果
def process_yolo_results(results, class_list=None):
    global flag

    if class_list is not None:
        names = results.names
        keep_idx = [
            i for i, box in enumerate(results.boxes)
            if (names[int(box.cls)] in class_list)
        ]
        results.boxes = results.boxes[keep_idx]
        if hasattr(results, 'masks') and results.masks is not None:
            results.masks = results.masks[keep_idx]
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            results.keypoints = results.keypoints[keep_idx]

    if len(results.boxes)==0 :
        flag = False
    # 2. 图像转换
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

# 图片检测
def image_detection():
    global current_model
    global detection_data
    global detections_df
    global speed_df
    global flag
    global iou_threshold
    global conf_threshold
    global target_class

    # 界面设计
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

                    process_yolo_results(results[0],class_list=target_class,)
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

        # 数据可视化 # 添加分支保证在没有数据上传时不会产生空错误
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

# 视频检测
def video_detection():
    global current_model
    global detection_data
    global conf_threshold
    global iou_threshold
    global target_class
    st.header("🎥 视频检测系统")

    # 使用容器划分主要功能区域
    upload_container = st.container(border=True)
    with upload_container:
        # 上传区域
        st.subheader("1. 视频上传")
        uploaded_file = st.file_uploader("选择待检测视频文件",
                                         type=["mp4", "avi", "mov"],
                                         label_visibility="hidden",
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
                        if detection_mode=="单类识别":  # 如果用户选择了特定类别
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
            progress_placeholder.progress(100,text=f"▏ 处理进度: 100% | 已处理 {total_frames}/{total_frames} 帧 | 处理完成")

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
                    # if target_class:
                    #     df = df[df['class'].isin(target_class)]
                    # 确保 target_class 是列表形式（即使只有一个类别）
                    if target_class:
                        if isinstance(target_class, str):  # 如果是字符串，转为列表
                            target_class = [target_class]
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
            # 结束-清理临时文件
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(processed_temp_video_path):
                os.remove(processed_temp_video_path)

# 实时检测
def real_time_detection():
    st.title("📷 实时目标检测")
    global current_model
    global conf_threshold
    global iou_threshold
    global detection_mode
    global target_class

    # 使用列布局创建控制面板
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("控制面板")
        start_button = st.button("▶️ 开始检测", key="start", help="启动摄像头并开始实时检测")
        stop_button = st.button("⏹️ 停止检测", key="stop", help="停止检测并关闭摄像头")

        # 显示当前参数设置
        st.markdown("### 当前参数")
        st.markdown(f"- 模型: `{current_model.__class__.__name__}`")
        st.markdown(f"- 置信度阈值: `{conf_threshold:.2f}`")
        st.markdown(f"- IOU阈值: `{iou_threshold:.2f}`")

        # 性能指标占位符
        performance_placeholder = st.empty()

    with col2:
        # 视频流显示区域
        video_placeholder = st.empty()

        # 检测结果统计区域 - 初始为空，开始检测后才会显示内容
        stats_placeholder = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ 无法打开视频源，请检查摄像头连接")
            return

        # 初始化性能指标
        fps_list = []
        detection_times = []

        while cap.isOpened() and not stop_button:
            success, frame = cap.read()
            if not success:
                st.warning("⚠️ 视频流中断")
                break

            # 执行检测
            start_time = time.time()
            results = current_model(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            detection_time = time.time() - start_time

            # 计算FPS
            fps = 1 / (detection_time + 1e-9)
            fps_list.append(fps)
            detection_times.append(detection_time)
            avg_fps = sum(fps_list[-10:]) / min(10, len(fps_list))
            avg_detection_time = sum(detection_times[-10:]) / min(10, len(detection_times))

            # 绘制结果
            process_yolo_results(results[0], class_list=target_class)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # 更新视频流显示
            video_placeholder.image(
                annotated_frame,
                caption="实时检测画面",
                use_container_width=True,
                channels="RGB"
            )

            # 更新统计信息
            num_objects = len(results[0].boxes)
            classes_detected = results[0].boxes.cls.unique().tolist() if num_objects > 0 else []

            with stats_placeholder.container():
                st.markdown("### 实时统计")
                col_stat1, col_stat2, col_stat3 = st.columns(3)

                with col_stat1:
                    st.metric("目标数量", num_objects)
                    st.metric("检测类别数", len(classes_detected))

                with col_stat2:
                    st.metric("实时FPS", f"{fps:.1f}")
                    st.metric("平均FPS", f"{avg_fps:.1f}")

                with col_stat3:
                    st.metric("检测时间", f"{detection_time * 1000:.1f}ms")
                    st.metric("平均检测时间", f"{avg_detection_time * 1000:.1f}ms")

                # 显示检测到的类别
                if classes_detected:
                    st.markdown("**检测到的类别:**")
                    st.write(", ".join([str(int(cls)) for cls in classes_detected]))

            # 更新性能指标
            performance_placeholder.markdown("### 性能指标")
            performance_placeholder.markdown(f"- 最近10帧平均FPS: `{avg_fps:.1f}`")
            performance_placeholder.markdown(f"- 最近10帧平均检测时间: `{avg_detection_time * 1000:.1f} ms`")

            # 控制帧率 (默认30FPS)
            time.sleep(max(0, 1 / 30 - detection_time))

        cap.release()
        cv2.destroyAllWindows()
        video_placeholder.empty()
        stats_placeholder.empty()
        performance_placeholder.empty()
        st.success("✅ 检测已停止")

def model_usage():
    global current_model
    global iou_threshold
    global conf_threshold

    st.title("🦁 基于YOLO模型的动物识别系统")
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

        # 功能导航
        st.markdown("---")
        st.markdown("**🎯 检测模式**")
        task = st.radio(" ", ["图片检测", "视频检测", "实时检测"], index=0,label_visibility="hidden")

    # 主内容区域
    with st.container():

        if task == "图片检测":
            image_detection()

        elif task == "视频检测":
            video_detection()

        elif task == "实时检测":
            real_time_detection()

# YOLO模型训练系统的Streamlit应用
def model_train():

    # 忽略警告
    warnings.filterwarnings('ignore')

    # 标题
    st.title("🤖 YOLO模型训练系统")

    # 初始化session_state
    if 'training' not in st.session_state:
        st.session_state.training = False
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'log' not in st.session_state:
        st.session_state.log = ""
    if 'epoch_progress' not in st.session_state:
        st.session_state.epoch_progress = 0

    # 侧边栏配置
    with st.sidebar:
        st.header("训练参数配置")

        # 模型选择
        model_path = st.text_input(
            "模型路径(.pt)",
            r"D:\Python\graduate_design\Model\yolo11n.pt"
        )

        # 数据配置
        data_path = st.text_input(
            "数据配置文件路径(yaml)",
            r"D:\Python\graduate_design\Model\dataset\data.yaml"
        )

        # 基本参数
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("训练轮次(epochs)", 1, 1000, 100)
            batch_size = st.number_input("批量大小(batch)", 1, 64, 8)
        with col2:
            img_size = st.number_input("图像大小(imgsz)", 320, 1280, 640, step=32)

        # 高级参数
        with st.expander("高级参数"):
            single_cls = st.checkbox("单类别检测(single_cls)", False)
            close_mosaic = st.number_input("关闭马赛克增强的轮次", 0, 100, 10)
            optimizer = st.selectbox("优化器", ["SGD", "Adam", "AdamW", "RMSprop"])
            amp = st.checkbox("混合精度训练(amp)", True)
            cache = st.checkbox("缓存数据集(cache)", False)

        # 设备选择
        device_options = ["CPU"] + ["CUDA"]
        device = st.selectbox("训练设备(device)", device_options, index=1)

        # 输出设置
        project = st.text_input("项目目录(project)", "runs/train")
        name = st.text_input("实验名称(name)", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 训练状态显示
    with st.container():
        # 第一行：全局训练进度
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("**全局进度**")
        with col2:
            progress_bar = st.progress(st.session_state.progress)

        # 第二行：当前epoch进度
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("**当前Epoch**")
        with col2:
            epoch_progress_bar = st.progress(st.session_state.epoch_progress)

        # 第三行：状态信息
        status_placeholder = st.empty()

        # 第四行：日志显示（保持原样）
        log_placeholder = st.empty()

    # 训练控制区域
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("开始训练", disabled=st.session_state.training):
            st.session_state.training = True
            st.session_state.progress = 0
            st.session_state.epoch_progress = 0
            st.session_state.log = ""

    with col2:
        if st.button("停止训练", disabled=not st.session_state.training):
            # 停止训练并重置所有状态
            st.session_state.training = False
            st.session_state.progress = 0
            st.session_state.epoch_progress = 0
            st.session_state.log = ""

            # 清除训练产生的临时文件
            import shutil
            output_dir = os.path.join(project, name) if project and name else "runs/train/exp"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                st.success(f"已清除训练输出目录: {output_dir}")

            # 重置UI显示
            status_placeholder.info("训练已停止并重置")
            progress_bar.progress(0)
            epoch_progress_bar.progress(0)
            log_placeholder.code("训练已终止")



    # 训练函数
    def train_model():
        try:
            # 验证路径
            if not os.path.exists(model_path):
                st.error(f"模型文件不存在: {model_path}")
                return
            if not os.path.exists(data_path):
                st.error(f"数据配置文件不存在: {data_path}")
                return

            # 初始化模型
            model = YOLO(model_path)

            # 训练回调 - 适配最新版API
            def on_train_epoch_end(trainer):
                epoch = trainer.epoch
                epochs = trainer.epochs
                progress = min((epoch + 1) / epochs, 1.0)  # 确保不超过1.0
                st.session_state.progress = progress
                progress_bar.progress(progress)
                status_placeholder.info(
                    f"训练中: 第 {epoch + 1}/{epochs} 轮次 ({(progress * 100):.1f}%)")

                # 更新日志
                log_entry = f"Epoch {epoch + 1}/{epochs} - "
                log_entry += f"mAP50: {trainer.metrics.get('metrics/mAP50(B)', 0):.3f} "
                log_entry += f"Precision: {trainer.metrics.get('metrics/precision(B)', 0):.3f} "
                log_entry += f"Recall: {trainer.metrics.get('metrics/recall(B)', 0):.3f}\n"

                st.session_state.log += log_entry
                log_placeholder.code(st.session_state.log)

            # 初始化进度跟踪变量
            current_batch = 0
            batches_per_epoch = 1  # 初始化为1避免除零错误

            def on_train_epoch_start(trainer):
                nonlocal current_batch, batches_per_epoch
                current_batch = 0  # 重置batch计数器
                batches_per_epoch = max(len(trainer.train_loader), 1)  # 确保至少为1

            def on_train_batch_end(trainer):
                nonlocal current_batch
                current_batch += 1
                epoch_progress = min(current_batch / batches_per_epoch, 1.0)  # 确保不超过1.0
                st.session_state.epoch_progress = epoch_progress
                epoch_progress_bar.progress(epoch_progress)

            # 注册回调
            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            # 开始训练
            results = model.train(
                data=data_path,
                cache=False,
                imgsz=img_size,
                epochs=epochs,
                single_cls=single_cls,
                batch=batch_size,
                close_mosaic=close_mosaic,
                workers=0,
                device=device.split(":")[-1] if ":" in device else device,
                optimizer=optimizer,
                amp=amp,
                project=project,
                name=name
            )

            # 训练完成
            st.session_state.training = False
            status_placeholder.success("🎉 训练完成!")

            # 显示最终结果
            if results:
                st.subheader("训练结果摘要")
                cols = st.columns(3)
                cols[0].metric("mAP50", f"{results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
                cols[1].metric("精确度", f"{results.results_dict.get('metrics/precision(B)', 0):.3f}")
                cols[2].metric("召回率", f"{results.results_dict.get('metrics/recall(B)', 0):.3f}")

        except Exception as e:
            st.session_state.training = False
            status_placeholder.error(f"训练错误: {str(e)}")


    # 训练状态监控
    if st.session_state.training:
        train_model()

    # 帮助信息
    with st.expander("使用帮助"):
        st.markdown("""
            ### 使用说明

            1. **模型路径**：指定预训练模型(.pt)的完整路径
            2. **数据配置**：指定数据集配置文件(.yaml)的完整路径
            3. **训练参数**：
               - 训练轮次：通常100-300轮
               - 批量大小：根据GPU内存调整(8,16,32等)
               - 图像大小：建议640x640
            4. **设备选择**：默认使用GPU 0，无GPU则选CPU

            ### 常见问题
            - 如果训练卡住，请检查GPU内存是否不足
            - 路径错误时请使用绝对路径
            - 训练日志会自动显示在下方
            """)

def main():
    global target_class
    global detection_mode

    # 导航菜单
    st.sidebar.markdown('<h1 class="sidebar-title">🧭 导航菜单</h1>', unsafe_allow_html=True)
    page = st.sidebar.radio(" ",
                          ["模型选择（已提供）", "模型自定义（训练）"],
                          index=0,
                          format_func=lambda x: "🔍 快速检测" if x == "模型选择（已提供）" else "🛠️ 模型训练",
                            label_visibility="hidden"
                            )

    if page == "模型选择（已提供）":
        # 全局设置区域（显示在导航菜单下方）
        with st.sidebar.expander("⚙️ 检测设置", expanded=True):
            # 单类/多类识别选择
            detection_mode = st.radio(
                "检测模式",
                ["多类识别", "单类识别"],
                index=1,
                help="选择是否只检测特定类别的目标"
            )

            # 类选择器
            if detection_mode == "单类识别":
                # 这里替换为你的实际类别列表
                class_options = ["cat", "dog", "bird","sheep","cow", "teddy bear"]
                target_class = st.selectbox(
                    "选择要识别的目标类别",
                    options=class_options,
                    index=0
                )
            else:
                target_class = ["cat", "dog", "bird","sheep","cow",
                                        "teddy bear","zebra","giraffe","elephant","person"]
        model_usage()
    else:
        model_train()


# 运行 Streamlit 应用
if __name__ == "__main__":
    main()
