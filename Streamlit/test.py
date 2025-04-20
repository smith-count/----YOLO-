def video_detection():
    global current_model
    st.header("视频检测")
    st.markdown('<div class="dynamic-border">', unsafe_allow_html=True)

    # 上传视频文件
    uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # 保存上传的视频到临时文件
        temp_video_path = "temp_uploaded_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 创建两列布局：原视频和处理后视频
        col1, col2 = st.columns(2)

        # 在原视频列显示原始视频
        with col1:
            st.subheader("原始视频")
            st.video(temp_video_path)

        # 在处理后视频列显示处理进度和结果
        with col2:
            st.subheader("YOLO处理后的视频")

            # 创建输出目录
            output_dir = os.path.join("temp_results", "processed_video")
            os.makedirs(output_dir, exist_ok=True)

            # 确定输出视频路径
            output_video_name = "output_" + os.path.basename(temp_video_path)
            processed_video_path = os.path.join(output_dir, output_video_name)

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

                # 视频处理进度和状态
                processed_frames = 0
                lock = threading.Lock()  # 用于线程安全地更新 `processed_frames`

                # 视频处理函数
                def process_video():
                    nonlocal processed_frames
                    try:
                        # 模拟 YOLO 推理和保存视频
                        # 假设 current_model.track() 是一个生成器，返回每一帧的结果
                        # 实际实现需要根据你的 YOLO 推理代码调整
                        results = current_model.track(
                            source=temp_video_path,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            stream=True,
                            imgsz=(width, height),
                            save=False,  # 禁用模型自带的保存功能，我们手动保存
                            project=None,  # 禁用默认保存路径
                            name=None,  # 禁用默认文件名
                            verbose=False
                        )

                        # 手动保存处理后的视频（伪代码，需根据实际推理结果调整）
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

                        for frame_result in results:
                            # 假设 frame_result 是处理后的帧图像 (numpy array)
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

                # 进度更新循环
                progress_placeholder = st.empty()  # 用于动态更新进度条
                progress = 0
                while process_thread.is_alive():
                    with lock:  # 线程安全地读取 `processed_frames`
                        progress = min(100, int((processed_frames / total_frames) * 100) if total_frames > 0 else 0)
                    progress_placeholder.progress(progress, text=f"处理进度: {progress}%")
                    time.sleep(0.1)

                process_thread.join()

                # 检查输出文件
                if os.path.exists(processed_video_path):
                    # 导出按钮
                    with open(processed_video_path, "rb") as f:
                        st.download_button(
                            label="📥 导出处理后的视频",
                            data=f,
                            file_name=output_video_name,
                            mime="video/mp4"
                        )
                else:
                    st.error("未找到输出视频，可能处理失败")

            except Exception as e:
                st.error(f"视频处理出错: {str(e)}")

            finally:
                # 清理临时文件
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

        output_mp4_path = "output_video_that_streamlit_can_play"  # 输出 MP4 文件路径（无需扩展名，默认 .mp4）

        convert_video_with_ffmpeg(processed_video_path, output_mp4_path)
        st.video(r"D:\Python\graduate_design\output_video_that_streamlit_can_play.mp4")