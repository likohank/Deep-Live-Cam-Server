from typing import Any, List
import cv2
import multiprocessing as mp
import gfpgan
import os
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

# 设置多进程启动方法
mp.set_start_method('spawn', force=True)

NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

# 全局多进程池
ENHANCER_PROCESS_POOL = None
TASK_QUEUE = None
RESULT_QUEUE = None
PROCESSES = []
POOL_INITIALIZED = False

PROCESSES_PER_GPU = 5

def create_dummy_frame():
    """创建一个用于预热的虚拟帧"""
    dummy_frame = np.ones((128, 128, 3), dtype=np.uint8) * 128
    cv2.circle(dummy_frame, (64, 40), 20, (255, 255, 255), -1)
    cv2.circle(dummy_frame, (50, 35), 5, (0, 0, 0), -1)
    cv2.circle(dummy_frame, (78, 35), 5, (0, 0, 0), -1)
    cv2.ellipse(dummy_frame, (64, 50), (15, 10), 0, 0, 180, (0, 0, 0), 2)
    return dummy_frame

def create_enhancer(device_id=0):
    """创建 GFPGAN 增强器实例"""
    model_path = os.path.join(models_dir, "GFPGANv1.4.pth")

    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{device_id}")
        print(f"增强进程使用 GPU {device_id}")
    else:
        device = torch.device("cpu")
        print("增强进程使用 CPU")

    enhancer = gfpgan.GFPGANer(
        model_path=model_path,
        upscale=1,
        device=device
    )
    
    return enhancer


def init_face_enhancer_pool():
    """初始化多进程增强池 - 同步版本，会阻塞直到所有进程就绪"""
    global ENHANCER_PROCESS_POOL, TASK_QUEUE, RESULT_QUEUE, PROCESSES, POOL_INITIALIZED, PROCESSES_PER_GPU

    processes_per_gpu = PROCESSES_PER_GPU
    
    if POOL_INITIALIZED:
        return True
    
    print("正在初始化多进程人脸增强池...")
    start_time = time.time()
    
    # 检测可用 GPU
    available_gpus = []
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个 GPU，用于人脸增强")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f}GB")
            available_gpus.append(i)
    
    if not available_gpus:
        print("未检测到 GPU，使用 CPU 进行增强")
        available_gpus = [-1]  # 使用 CPU
    
    # 创建进程间通信队列
    TASK_QUEUE = mp.Queue(maxsize=50)
    RESULT_QUEUE = mp.Queue(maxsize=50)
    
    # 创建进程
    PROCESSES = []
    process_id = 0
    
    # 准备就绪事件列表
    ready_events = []
    
    print("--"*40)
    print(available_gpus)
    print("--"*40)
    for gpu_id in available_gpus[1:]:
    #for gpu_id in available_gpus:
        for i in range(processes_per_gpu):
            # 为每个进程创建就绪事件
            ready_event = mp.Event()
            ready_events.append(ready_event)
            
            process = mp.Process(
                target=enhancer_worker_with_ready_signal,
                args=(process_id, gpu_id, TASK_QUEUE, RESULT_QUEUE, ready_event),
                daemon=True
            )
            process.start()
            PROCESSES.append(process)
            process_id += 1
    
    # 等待所有进程准备就绪
    print("等待所有增强进程初始化完成...")
    for i, event in enumerate(ready_events):
        event.wait(timeout=120.0)  # 60秒超时
        if event.is_set():
            print(f"增强进程 {i} 已就绪")
        else:
            print(f"警告: 增强进程 {i} 初始化超时")
    
    ENHANCER_PROCESS_POOL = {
        'processes': PROCESSES,
        'task_queue': TASK_QUEUE,
        'result_queue': RESULT_QUEUE,
        'next_task_id': 0
    }
    
    POOL_INITIALIZED = True
    init_time = time.time() - start_time
    print(f"多进程增强池初始化完成，总共 {len(PROCESSES)} 个进程，耗时 {init_time:.2f} 秒")
    return True

def enhancer_worker_with_ready_signal(process_id, device_id, task_queue, result_queue, ready_event):
    """带就绪信号的增强工作进程"""
    print(f"增强进程 {process_id} 启动，使用设备: GPU{device_id}")
    
    # 创建独立的 enhancer 实例
    enhancer = create_enhancer(device_id)
    
    # 预热模型
    try:
        print(f"进程 {process_id} 预热模型...")
        dummy_frame = create_dummy_frame()
        _, _, _ = enhancer.enhance(dummy_frame, paste_back=True)
        print(f"进程 {process_id} 模型预热完成")
    except Exception as e:
        print(f"进程 {process_id} 模型预热失败: {e}")
    
    # 发送就绪信号
    ready_event.set()
    
    while True:
        try:
            # 获取任务
            time1 = time.time()
            task_id, frame_data = task_queue.get()
            if task_id is None:  # 终止信号
                break
            start_time = time.time()
            get_time = start_time - time1
            
            # 执行增强
            _, _, enhanced_frame = enhancer.enhance(frame_data, paste_back=True)
            
            end_time = time.time()
            cost_time = end_time - start_time
            
            # 发送结果
            result_queue.put((task_id, enhanced_frame, cost_time))

            put_time = time.time() - end_time
            print(f"enhancer_worker_with_ready_signal get耗时 {get_time:.3f}s, 增强耗时 {cost_time:.3f}s, put耗时 {put_time:.3f}s-----------------------------")
            
        except Exception as e:
            print(f"增强进程 {process_id} 处理失败: {e}")
            # 发送原始帧作为失败结果
            result_queue.put((task_id, frame_data, 0))
    
    print(f"增强进程 {process_id} 退出")

def shutdown_face_enhancer_pool():
    """关闭多进程增强池"""
    global ENHANCER_PROCESS_POOL, POOL_INITIALIZED
    
    if not POOL_INITIALIZED:
        return
    
    print("正在关闭多进程增强池...")
    
    # 发送终止信号
    for _ in PROCESSES:
        TASK_QUEUE.put((None, None))
    
    # 等待进程结束
    for process in PROCESSES:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
    
    # 清空队列
    while not TASK_QUEUE.empty():
        TASK_QUEUE.get()
    while not RESULT_QUEUE.empty():
        RESULT_QUEUE.get()
    
    ENHANCER_PROCESS_POOL = None
    POOL_INITIALIZED = False
    print("多进程增强池已关闭")

# 任务ID计数器
_task_id_counter = 0
_task_id_lock = mp.Lock()

def get_next_task_id():
    """获取下一个任务ID"""
    global _task_id_counter
    with _task_id_lock:
        _task_id_counter += 1
        return _task_id_counter

def enhance_face(temp_frame: Frame) -> Frame:
    """使用多进程池进行人脸增强"""
    global ENHANCER_PROCESS_POOL
    
    if not POOL_INITIALIZED:
        print("错误: 多进程增强池未初始化!")
        return temp_frame
    
    # 获取任务ID
    task_id = get_next_task_id()
    
    # 提交任务
    fetch_start = time.time()
    try:
        TASK_QUEUE.put((task_id, temp_frame), timeout=1.0)
    except:
        print("任务队列已满，跳过增强")
        return temp_frame
    
    fetch_cost_time = time.time() - fetch_start
    
    # 等待结果
    enhance_start = time.time()
    timeout_count = 0
    while timeout_count < 3:  # 最多尝试3次
        try:
            result_id, result_frame, enhance_time = RESULT_QUEUE.get(timeout=5.0)
            if result_id == task_id:
                total_enhance_time = time.time() - enhance_start
                if total_enhance_time > 1.0:  # 如果总等待时间超过1秒，打印警告
                    print(f"face_enhancer 提交耗时 {fetch_cost_time:.3f}s, 增强耗时 {enhance_time:.3f}s, 总等待 {total_enhance_time:.3f}s")
                return result_frame
        except:
            timeout_count += 1
            print(f"等待增强结果超时 ({timeout_count}/3)，任务ID: {task_id}")
    
    print(f"增强任务 {task_id} 最终超时，返回原帧")
    return temp_frame

# 原有的其他函数保持不变
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
