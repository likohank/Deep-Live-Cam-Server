import cv2
import numpy as np
import time
import threading
import gfpgan
import torch
import os



def load_frame_from_file(file_path="frame.dat"):
    """加载 Frame 类型的二进制数据"""
    try:
        with open(file_path, 'rb') as f:
            frame_data = f.read()
        
        # 直接将二进制数据转换为 numpy 数组
        # 由于 Frame 本质上是 numpy.ndarray，我们可以直接使用 frombuffer
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        
        # 尝试解码为图像（如果是编码格式）
        decoded_frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        if decoded_frame is not None:
            print(f"成功加载并解码帧: {decoded_frame.shape}")
            return decoded_frame
        
        # 如果不是编码格式，尝试直接重塑为图像形状
        # 这里需要知道原始帧的形状，你可能需要根据实际情况调整
        # 如果不知道确切形状，可以尝试一些常见的尺寸
        common_shapes = [
            (480, 640, 3),    # 480p
            (720, 1280, 3),   # 720p  
            (1080, 1920, 3),  # 1080p
            (2160, 3840, 3),  # 4K
        ]
        
        for shape in common_shapes:
            height, width, channels = shape
            expected_size = height * width * channels
            if len(frame) == expected_size:
                reshaped_frame = frame.reshape((height, width, channels))
                print(f"成功重塑帧: {reshaped_frame.shape}")
                return reshaped_frame
        
        print(f"无法确定帧的形状，数据大小: {len(frame)} 字节")
        return None
            
    except Exception as e:
        print(f"加载帧文件失败: {e}")
        return None

def create_enhancer(device_id=0):
    """创建 GFPGAN 增强器实例"""
    model_path = "models/GFPGANv1.4.pth"  # 根据你的实际路径调整
    
    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{device_id}")
        print(f"使用 GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    
    enhancer = gfpgan.GFPGANer(
        model_path=model_path,
        upscale=1,
        device=device
    )
    
    return enhancer

def benchmark_worker(thread_id, iterations, temp_frame, device_id=0):
    """压测工作线程"""
    print(f"线程 {thread_id} 启动，使用设备: {'GPU' + str(device_id) if device_id >= 0 else 'CPU'}")
    
    # 创建独立的 enhancer 实例
    enhancer = create_enhancer(device_id)
    
    thread_times = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            # 执行增强操作
            _, _, enhanced_frame = enhancer.enhance(temp_frame, paste_back=True)
            end_time = time.time()
            cost_time = end_time - start_time
            
            thread_times.append(cost_time)
            print(f"线程 {thread_id}-迭代 {i+1} - 处理耗时: {cost_time:.3f}s")
            
        except Exception as e:
            end_time = time.time()
            cost_time = end_time - start_time
            print(f"线程 {thread_id}-迭代 {i+1} - 处理失败: {e}, 耗时: {cost_time:.3f}s")
    
    # 统计结果
    if thread_times:
        avg_time = np.mean(thread_times)
        max_time = np.max(thread_times)
        min_time = np.min(thread_times)
        print(f"线程 {thread_id} 完成: 平均 {avg_time:.3f}s, 最小 {min_time:.3f}s, 最大 {max_time:.3f}s")
    
    return thread_times

def run_simple_benchmark(thread_count=4, iterations_per_thread=5, frame_file="frame.dat", device_id=0):
    """运行简单压测"""
    print(f"开始压测配置:")
    print(f"- 线程数: {thread_count}")
    print(f"- 每线程迭代次数: {iterations_per_thread}")
    print(f"- 设备ID: {device_id}")
    print(f"- 帧文件: {frame_file}")
    
    # 加载测试帧
    temp_frame = load_frame_from_file(frame_file)
    if temp_frame is None:
        print("无法加载测试帧，退出压测")
        return
    
    # 创建线程和结果容器
    threads = []
    results = {}
    
    # 启动线程
    for i in range(thread_count):
        thread = threading.Thread(
            target=lambda idx=i: results.update({idx: benchmark_worker(idx, iterations_per_thread, temp_frame, device_id)})
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 输出统计结果
    print("\n=== 压测结果统计 ===")
    all_times = []
    for thread_id, times in results.items():
        if times:
            avg_time = np.mean(times)
            max_time = np.max(times)
            min_time = np.min(times)
            print(f"线程 {thread_id}: 平均 {avg_time:.3f}s, 最小 {min_time:.3f}s, 最大 {max_time:.3f}s")
            all_times.extend(times)
    
    if all_times:
        overall_avg = np.mean(all_times)
        overall_min = np.min(all_times)
        overall_max = np.max(all_times)
        print(f"\n总体统计:")
        print(f"- 平均耗时: {overall_avg:.3f}s")
        print(f"- 最小耗时: {overall_min:.3f}s") 
        print(f"- 最大耗时: {overall_max:.3f}s")
        print(f"- 总处理次数: {len(all_times)}")
        print(f"- 总耗时: {np.sum(all_times):.3f}s")
        print(f"- 吞吐量: {len(all_times)/np.sum(all_times):.2f} 帧/秒")

def run_multi_gpu_benchmark(threads_per_gpu=2, iterations_per_thread=5, frame_file="frame.dat"):
    """多GPU压测"""
    if not torch.cuda.is_available():
        print("未检测到GPU，使用单CPU模式")
        run_simple_benchmark(threads_per_gpu, iterations_per_thread, frame_file, -1)
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU，每个GPU启动 {threads_per_gpu} 个线程")
    
    all_results = {}
    
    for gpu_id in range(gpu_count):
        print(f"\n=== 在 GPU {gpu_id} 上启动压测 ===")
        results = {}
        threads = []
        
        temp_frame = load_frame_from_file(frame_file)
        if temp_frame is None:
            print("无法加载测试帧，跳过该GPU")
            continue
        
        for i in range(threads_per_gpu):
            thread_id = f"GPU{gpu_id}-Thread{i}"
            thread = threading.Thread(
                target=lambda idx=thread_id: results.update({idx: benchmark_worker(idx, iterations_per_thread, temp_frame, gpu_id)})
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        all_results.update(results)
    
    # 输出总体统计
    print("\n=== 多GPU压测总体统计 ===")
    all_times = []
    for thread_id, times in all_results.items():
        if times:
            avg_time = np.mean(times)
            all_times.extend(times)
            print(f"{thread_id}: 平均 {avg_time:.3f}s")
    
    if all_times:
        overall_avg = np.mean(all_times)
        print(f"\n所有GPU总体平均耗时: {overall_avg:.3f}s")
        print(f"总处理次数: {len(all_times)}")
        print(f"总吞吐量: {len(all_times)/np.sum(all_times):.2f} 帧/秒")

if __name__ == "__main__":
    # 单设备压测示例
    print("=== 单设备压测 ===")
    run_simple_benchmark(
        thread_count=1,           # 线程数量
        iterations_per_thread=100,  # 每个线程的迭代次数
        frame_file="frame.dat",   # 帧文件路径
        device_id=0               # 设备ID (GPU编号，-1表示CPU)
    )
    
    # 多GPU压测示例（取消注释使用）
    # print("\n=== 多GPU压测 ===")
    # run_multi_gpu_benchmark(
    #     threads_per_gpu=2,        # 每个GPU的线程数
    #     iterations_per_thread=5,  # 每个线程的迭代次数
    #     frame_file="frame.dat"    # 帧文件路径
    # )
