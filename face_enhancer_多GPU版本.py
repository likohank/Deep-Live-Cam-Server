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

# 多GPU对象池
FACE_ENHANCER_POOLS = {}
POOL_LOCKS = {}
GPU_COUNT = torch.cuda.device_count()
POOL_SIZE_PER_GPU = 2  # 每个GPU上的实例数

NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

def create_dummy_frame():
    """创建一个用于预热的虚拟帧"""
    dummy_frame = np.ones((128, 128, 3), dtype=np.uint8) * 128
    cv2.circle(dummy_frame, (64, 40), 20, (255, 255, 255), -1)
    cv2.circle(dummy_frame, (50, 35), 5, (0, 0, 0), -1)
    cv2.circle(dummy_frame, (78, 35), 5, (0, 0, 0), -1)
    cv2.ellipse(dummy_frame, (64, 50), (15, 10), 0, 0, 180, (0, 0, 0), 2)
    return dummy_frame

def pre_warm_enhancer(enhancer):
    """预热模型，强制加载到显存"""
    try:
        dummy_frame = create_dummy_frame()
        _, _, _ = enhancer.enhance(dummy_frame, paste_back=True)
    except Exception as e:
        print(f"模型预热失败: {e}")

def init_face_enhancer_pools():
    """为每个GPU初始化对象池"""
    global FACE_ENHANCER_POOLS, POOL_LOCKS
    
    model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
    
    print(f"检测到 {GPU_COUNT} 个GPU，为每个GPU创建 {POOL_SIZE_PER_GPU} 个实例")
    
    for gpu_id in range(GPU_COUNT):
        pool_key = f"gpu_{gpu_id}"
        POOL_LOCKS[pool_key] = threading.Lock()
        FACE_ENHANCER_POOLS[pool_key] = Queue(maxsize=POOL_SIZE_PER_GPU)
        
        device = torch.device(f"cuda:{gpu_id}")
        
        print(f"在 GPU {gpu_id} 上初始化 GFPGAN 对象池...")
        
        for i in range(POOL_SIZE_PER_GPU):
            enhancer = gfpgan.GFPGANer(
                model_path=model_path, 
                upscale=1, 
                device=device
            )
            
            # 预热第一个实例
            if i == 0:
                pre_warm_enhancer(enhancer)
            
            FACE_ENHANCER_POOLS[pool_key].put(enhancer)
        
        print(f"GPU {gpu_id} 对象池初始化完成")

def get_face_enhancer_from_pool(gpu_id=None):
    """从指定GPU的池中获取实例，如果未指定则轮询"""
    if not FACE_ENHANCER_POOLS:
        init_face_enhancer_pools()
    
    # 如果没有指定GPU，则使用轮询策略
    if gpu_id is None:
        gpu_id = getattr(thread_local, 'current_gpu', 0)
        gpu_id = (gpu_id + 1) % GPU_COUNT
        thread_local.current_gpu = gpu_id
    
    pool_key = f"gpu_{gpu_id}"
    
    with POOL_LOCKS[pool_key]:
        return FACE_ENHANCER_POOLS[pool_key].get()

def return_face_enhancer_to_pool(enhancer, gpu_id):
    """将实例归还到指定GPU的池中"""
    pool_key = f"gpu_{gpu_id}"
    
    if pool_key in FACE_ENHANCER_POOLS:
        with POOL_LOCKS[pool_key]:
            FACE_ENHANCER_POOLS[pool_key].put(enhancer)

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

# 线程局部存储，用于跟踪每个线程使用的GPU
thread_local = threading.local()

def enhance_face(temp_frame: Frame) -> Frame:
    """使用多GPU对象池进行人脸增强"""
    # 为当前线程选择GPU
    if not hasattr(thread_local, 'current_gpu'):
        thread_local.current_gpu = 0
    
    gpu_id = thread_local.current_gpu
    thread_local.current_gpu = (gpu_id + 1) % GPU_COUNT
    
    enhancer = get_face_enhancer_from_pool(gpu_id)
    try:
        _, _, temp_frame = enhancer.enhance(temp_frame, paste_back=True)
        return temp_frame
    except Exception as e:
        print(f"GPU {gpu_id} 人脸增强失败: {e}")
        return temp_frame
    finally:
        return_face_enhancer_to_pool(enhancer, gpu_id)

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