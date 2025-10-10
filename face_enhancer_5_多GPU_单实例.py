from typing import Any, List
import cv2
import threading
import gfpgan
import os
from queue import Queue
import numpy as np

import time
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

# 双 GPU 对象池
FACE_ENHANCER_POOLS = None
POOL_LOCK = threading.Lock()
POOL_SIZE_PER_GPU = 1  # 每个 GPU 4个实例，总共8个

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
        print("预热 GFPGAN 模型...")
        dummy_frame = create_dummy_frame()
        _, _, _ = enhancer.enhance(dummy_frame, paste_back=True)
        print("模型预热完成")
    except Exception as e:
        print(f"模型预热失败: {e}")

def init_face_enhancer_pool():
    """初始化双 GPU GFPGAN 对象池"""
    global FACE_ENHANCER_POOLS

    with POOL_LOCK:
        if FACE_ENHANCER_POOLS is None:
            FACE_ENHANCER_POOLS = []
            
            model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
            
            # 检测可用 GPU
            available_gpus = []
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"检测到 {gpu_count} 个 GPU")
                
                #for i in range(1,gpu_count):  #把显卡0留给其他模型使用
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f}GB")
                    available_gpus.append(i)
            
            if not available_gpus:
                print("未检测到 GPU，使用 CPU")
                available_gpus = [torch.device("cpu")]
            else:
                # 使用前两个 GPU（如果有的话）
                #available_gpus = available_gpus[:2]

                #使用所有GPU
                #available_gpus = available_gpus[1:]
                available_gpus = available_gpus[:]
                print(f"使用 GPU: {available_gpus}")

            # 为每个 GPU 创建对象池
            for gpu_id in available_gpus:
                if isinstance(gpu_id, int):
                    device = torch.device(f"cuda:{gpu_id}")
                else:
                    device = gpu_id
                    
                pool = Queue(maxsize=POOL_SIZE_PER_GPU)
                print(f"在 {device} 上初始化 GFPGAN 对象池，大小: {POOL_SIZE_PER_GPU}")

                for i in range(POOL_SIZE_PER_GPU):
                    enhancer = gfpgan.GFPGANer(
                        model_path=model_path,
                        upscale=1,
                        device=device
                    )

                    # 预热第一个实例
                    if i == 0:
                        pre_warm_enhancer(enhancer)

                    # 为增强器标记所属 GPU
                    enhancer.gpu_id = gpu_id if isinstance(gpu_id, int) else -1
                    pool.put(enhancer)

                FACE_ENHANCER_POOLS.append(pool)

            print(f"双 GPU GFPGAN 对象池初始化完成，总共 {len(available_gpus) * POOL_SIZE_PER_GPU} 个实例")

# 轮询计数器，用于负载均衡
_gpu_selector = 0
_selector_lock = threading.Lock()

def get_face_enhancer_from_pool():
    """从池中获取 GFPGAN 实例（负载均衡）"""
    global _gpu_selector
    
    if FACE_ENHANCER_POOLS is None:
        init_face_enhancer_pool()

    with _selector_lock:
        current_pool_index = _gpu_selector % len(FACE_ENHANCER_POOLS)
        _gpu_selector += 1
        
    pool = FACE_ENHANCER_POOLS[current_pool_index]
    
    # 如果当前池为空，尝试其他池
    if pool.empty():
        for i in range(1,len(FACE_ENHANCER_POOLS)):
            alt_index = (current_pool_index + i) % len(FACE_ENHANCER_POOLS)
            alt_pool = FACE_ENHANCER_POOLS[alt_index]
            if not alt_pool.empty():
                pool = alt_pool
                current_pool_index = alt_index  #kangkang
                break
    
    return pool.get(), current_pool_index

def return_face_enhancer_to_pool(enhancer, pool_index):
    """将 GFPGAN 实例归还到对应的池中"""
    if FACE_ENHANCER_POOLS is not None and pool_index < len(FACE_ENHANCER_POOLS):
        FACE_ENHANCER_POOLS[pool_index].put(enhancer)

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
    """使用双 GPU 对象池进行人脸增强"""
    fetch_start = time.time()
    enhancer, pool_index = get_face_enhancer_from_pool()

    enhance_start = time.time()
    fetch_cost_time = enhance_start - fetch_start

    try:
        _, _, temp_frame = enhancer.enhance(temp_frame, paste_back=True)
        enhance_cost_time = time.time() - enhance_start
        print(f"face_enhancer fetch耗时{fetch_cost_time:.3f}s  enhance耗时{enhance_cost_time:.3f}s")

        return temp_frame
    except Exception as e:
        print(f"人脸增强失败 (GPU {enhancer.gpu_id}): {e}")
        return temp_frame
    finally:
        return_face_enhancer_to_pool(enhancer, pool_index)

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
