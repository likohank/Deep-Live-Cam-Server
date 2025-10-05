from typing import Any, List
import cv2
import threading
import gfpgan
import os
from queue import Queue
import numpy as np

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
import platform
import torch
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)

# GFPGAN 对象池
FACE_ENHANCER_POOL = None
POOL_LOCK = threading.Lock()
POOL_SIZE = 15  # 根据显存调整，建议4-8个

NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

def create_dummy_frame():
    """创建一个用于预热的虚拟帧"""
    # 创建一个128x128的虚拟人脸图像（灰色背景，中间有个"脸"的形状）
    dummy_frame = np.ones((128, 128, 3), dtype=np.uint8) * 128
    # 在中间画一个简单的"脸"
    cv2.circle(dummy_frame, (64, 40), 20, (255, 255, 255), -1)  # 头
    cv2.circle(dummy_frame, (50, 35), 5, (0, 0, 0), -1)  # 左眼
    cv2.circle(dummy_frame, (78, 35), 5, (0, 0, 0), -1)  # 右眼
    cv2.ellipse(dummy_frame, (64, 50), (15, 10), 0, 0, 180, (0, 0, 0), 2)  # 嘴
    return dummy_frame

def pre_warm_enhancer(enhancer):
    """预热模型，强制加载到显存"""
    try:
        print("预热 GFPGAN 模型...")
        dummy_frame = create_dummy_frame()
        # 进行一次虚拟推理，触发模型完全加载
        _, _, _ = enhancer.enhance(dummy_frame, paste_back=True)
        print("模型预热完成")
    except Exception as e:
        print(f"模型预热失败: {e}")

def init_face_enhancer_pool():
    """初始化 GFPGAN 对象池并进行预热"""
    global FACE_ENHANCER_POOL
    
    with POOL_LOCK:
        if FACE_ENHANCER_POOL is None:
            FACE_ENHANCER_POOL = Queue(maxsize=POOL_SIZE)
            
            model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
            
            # 获取设备信息
            selected_device = None
            if torch.cuda.is_available():
                selected_device = torch.device("cuda")
            elif torch.backends.mps.is_available() and platform.system() == "Darwin":
                selected_device = torch.device("mps")
            else:
                selected_device = torch.device("cpu")
            
            print(f"初始化 GFPGAN 对象池，大小: {POOL_SIZE}，设备: {selected_device}")
            
            for i in range(POOL_SIZE):
                enhancer = gfpgan.GFPGANer(
                    model_path=model_path, 
                    upscale=1, 
                    device=selected_device
                )
                
                # 预热模型（只在第一个实例时预热，避免重复占用太多显存）
                #if i == 0:
                pre_warm_enhancer(enhancer)
                
                FACE_ENHANCER_POOL.put(enhancer)
            
            print("GFPGAN 对象池初始化完成")

def get_face_enhancer_from_pool():
    """从池中获取 GFPGAN 实例"""
    if FACE_ENHANCER_POOL is None:
        init_face_enhancer_pool()
    
    return FACE_ENHANCER_POOL.get()

def return_face_enhancer_to_pool(enhancer):
    """将 GFPGAN 实例归还到池中"""
    if FACE_ENHANCER_POOL is not None:
        FACE_ENHANCER_POOL.put(enhancer)

def pre_check() -> bool:
    download_directory_path = models_dir
    conditional_download(
        download_directory_path,
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        ],
    )
    return True

def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True

def enhance_face(temp_frame: Frame) -> Frame:
    """使用对象池进行人脸增强"""
    enhancer = get_face_enhancer_from_pool()
    try:
        _, _, temp_frame = enhancer.enhance(temp_frame, paste_back=True)
        return temp_frame
    except Exception as e:
        print(f"人脸增强失败: {e}")
        # 发生错误时返回原始帧
        return temp_frame
    finally:
        return_face_enhancer_to_pool(enhancer)

def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame

def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames)

def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame
