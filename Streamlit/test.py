import streamlit as st
import warnings
from ultralytics import YOLO
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 页面设置
st.set_page_config(
    page_title="YOLO模型训练平台",
    page_icon=":rocket:",
    layout="wide"
)

# 标题
st.title("🚀 YOLO模型训练平台")

# 侧边栏设置
with st.sidebar:
    st.header("训练参数配置")

    # 模型配置文件路径
    model_cfg = st.text_input(
        "模型配置文件路径(yaml)",
        r"/Model\dataset\yolo11.yaml",
        help="YOLO模型配置文件的完整路径"
    )

    # 数据配置文件路径
    data_cfg = st.text_input(
        "数据配置文件路径(yaml)",
        r"D:\Python\graduate_design\Model\dataset\data.yaml",
        help="包含数据集信息的YAML文件路径"
    )

    # 基本参数
    epochs = st.number_input("训练轮次(epochs)", 1, 1000, 100)
    batch_size = st.number_input("批量大小(batch)", 1, 64, 8)
    img_size = st.number_input("图像大小(imgsz)", 320, 1280, 640, step=32)

    # 高级参数
    with st.expander("高级参数"):
        single_cls = st.checkbox("单类别检测(single_cls)", False)
        close_mosaic = st.number_input("关闭马赛克增强的轮次(close_mosaic)", 0, 100, 10)
        workers = st.number_input("数据加载线程数(workers)", 0, 8, 0)
        optimizer = st.selectbox("优化器(optimizer)", ["SGD", "Adam", "AdamW", "RMSprop"])
        amp = st.checkbox("混合精度训练(amp)", True)
        cache = st.checkbox("缓存数据集(cache)", False)

    # 设备选择
    device_options = ["CPU"] + [f"CUDA:{i}" for i in range(4)]
    device = st.selectbox("训练设备(device)", device_options, index=1)

    # 输出设置
    project = st.text_input("项目目录(project)", "runs/train")
    name = st.text_input("实验名称(name)", "exp")

# 训练按钮
if st.button("开始训练", type="primary"):
    # 验证路径
    if not os.path.exists(model_cfg):
        st.error(f"模型配置文件不存在: {model_cfg}")
    elif not os.path.exists(data_cfg):
        st.error(f"数据配置文件不存在: {data_cfg}")
    else:
        try:
            # 显示训练信息
            st.info("训练配置信息:")
            config_table = {
                "参数": ["模型配置", "数据配置", "训练轮次", "批量大小", "图像大小",
                         "单类别", "关闭马赛克", "线程数", "优化器", "混合精度", "缓存", "设备"],
                "值": [model_cfg, data_cfg, epochs, batch_size, img_size,
                       single_cls, close_mosaic, workers, optimizer, amp, cache, device]
            }
            st.table(config_table)

            # 初始化模型
            with st.spinner("正在初始化模型..."):
                model = YOLO(model_cfg)

            # 开始训练
            st.subheader("训练日志")
            with st.empty():
                progress_bar = st.progress(0)
                status_text = st.text("训练准备中...")


                # 训练回调函数
                def on_train_epoch_end(epoch, epochs, **kwargs):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"训练中: 第 {epoch + 1}/{epochs} 轮次 ({(progress * 100):.1f}%)")


                # 执行训练
                results = model.train(
                    data=data_cfg,
                    cache=cache,
                    imgsz=img_size,
                    epochs=epochs,
                    single_cls=single_cls,
                    batch=batch_size,
                    close_mosaic=close_mosaic,
                    workers=workers,
                    device=device.split(":")[-1] if ":" in device else device,
                    optimizer=optimizer,
                    amp=amp,
                    project=project,
                    name=name,
                    verbose=False,  # 使用我们自己的进度显示
                    callback={
                        "on_train_epoch_end": on_train_epoch_end
                    }
                )

                progress_bar.progress(1.0)
                status_text.text("训练完成!")

            # 显示训练结果
            st.success("🎉 训练完成!")

            # 显示关键指标
            if results:
                st.subheader("训练结果摘要")
                metrics = results.results_dict
                cols = st.columns(3)
                cols[0].metric("mAP50", f"{metrics.get('metrics/mAP50(B)', 0):.3f}")
                cols[1].metric("精确度", f"{metrics.get('metrics/precision(B)', 0):.3f}")
                cols[2].metric("召回率", f"{metrics.get('metrics/recall(B)', 0):.3f}")

        except Exception as e:
            st.error(f"训练过程中发生错误: {str(e)}")

# 使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### YOLO模型训练平台使用指南

    1. **配置文件路径**:
       - 确保提供正确的模型配置文件和数据集配置文件路径
       - 路径可以是绝对路径或相对路径

    2. **基本参数**:
       - 训练轮次(epochs): 训练的总轮次
       - 批量大小(batch): 每次迭代处理的图像数量
       - 图像大小(imgsz): 输入模型的图像尺寸

    3. **高级参数**:
       - 单类别检测: 如果您的任务只有一个类别，请勾选
       - 关闭马赛克增强: 指定在最后多少轮次关闭马赛克数据增强
       - 线程数: 数据加载的线程数(0表示在主线程中加载)

    4. **训练设备**:
       - 选择CPU或可用的GPU设备

    5. **开始训练**:
       - 配置好所有参数后，点击"开始训练"按钮
       - 训练进度和日志将显示在主界面
    """)

# 注意事项
st.info("""
💡 **注意事项**: 
- 训练过程可能需要较长时间，请耐心等待
- 确保有足够的GPU内存(如果使用GPU)
- 训练过程中不要关闭浏览器标签页
""")