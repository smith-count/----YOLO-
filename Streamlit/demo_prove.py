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

# 初始化全局变量
MODEL_PATHS = {
    "YOLOv11":"D:\Python\graduate_design\Model\yolo11n.pt",
    "YOLO_detect_animals":r"D:\Python\graduate_design\Model\runs\train\exp\weights\best.pt", #
}

current_model = YOLO(r"D:\Python\graduate_design\Model\yolo11n.pt")# 默认
conf_threshold = 0
iou_threshold = 0

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


# ======================
# 模型使用界面优化
# ======================
def model_usage():



# ======================
# 模型训练界面优化（保持类似结构）
# ======================
def model_train():

# （保持原有训练逻辑，添加类似样式优化）

# ======================
# 主函数
# ======================
def main():
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


if __name__ == "__main__":
    main()