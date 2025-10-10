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
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        
        # 尝试解码为图像（如果是编码格式）
        decoded_frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        if decoded_frame is not None:
            print(f"成功加载并解码帧: {decoded_frame.shape}")
            return decoded_frame
        
        # 如果不是编码格式，尝试直接重塑为图像形状
        # 这里需要知道原始帧的形状，你可能需要根据实际情况调整
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

def benchmark_worker(thread_id, iterations, frame_data, device_id=0):
    """压测工作线程"""
    print(f"线程 {thread_id} 启动，使用设备: GPU{device_id}")
    
    # 创建独立的 enhancer 实例
    enhancer = create_enhancer(device_id)
    
    thread_times = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            # 使用 frame_data 进行增强操作
            _, _, enhanced_frame = enhancer.enhance(frame_data, paste_back=True)
            end_time = time.time()
            cost_time = end_time - start_time
            
            thread_times.append(cost_time)
            print(f"线程 {thread_id}-迭代 {i+1} - GPU{device_id} - 处理耗时: {cost_time:.3f}s")
            
        except Exception as e:
            end_time = time.time()
            cost_time = end_time - start_time
            print(f"线程 {thread_id}-迭代 {i+1} - GPU{device_id} - 处理失败: {e}, 耗时: {cost_time:.3f}s")
    
    # 统计结果
    if thread_times:
        avg_time = np.mean(thread_times)
        max_time = np.max(thread_times)
        min_time = np.min(thread_times)
        print(f"线程 {thread_id} (GPU{device_id}) 完成: 平均 {avg_time:.3f}s, 最小 {min_time:.3f}s, 最大 {max_time:.3f}s")
    
    return thread_times, device_id

def run_dual_gpu_benchmark(total_threads=10, iterations_per_thread=5, frame_file="frame.dat"):
    """运行双GPU压测"""
    print(f"开始双GPU压测配置:")
    print(f"- 总线程数: {total_threads}")
    print(f"- 每线程迭代次数: {iterations_per_thread}")
    print(f"- 帧文件: {frame_file}")
    
    # 检查GPU数量
    if not torch.cuda.is_available():
        print("未检测到GPU，退出压测")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU")
    
    if gpu_count < 2:
        print("GPU数量不足2个，无法进行双GPU压测")
        return
    
    # 加载测试帧
    frame_data = load_frame_from_file(frame_file)
    if frame_data is None:
        print("无法加载测试帧，退出压测")
        return
    
    # 计算每个GPU分配的线程数
    threads_per_gpu = total_threads // 2
    print(f"每个GPU分配 {threads_per_gpu} 个线程")
    
    # 创建线程和结果容器
    threads = []
    results = {}
    
    # 启动线程 - 前一半使用GPU0，后一半使用GPU1
    for i in range(total_threads):
        # 分配GPU：前一半用GPU0，后一半用GPU1
        device_id = 0 if i < threads_per_gpu else 1
        
        thread = threading.Thread(
            target=lambda idx=i, dev_id=device_id: results.update({idx: benchmark_worker(idx, iterations_per_thread, frame_data, dev_id)})
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 输出统计结果
    print("\n=== 双GPU压测结果统计 ===")
    
    # 按GPU分组统计
    gpu0_times = []
    gpu1_times = []
    all_times = []
    
    for thread_id, (times, device_id) in results.items():
        if times:
            avg_time = np.mean(times)
            max_time = np.max(times)
            min_time = np.min(times)
            print(f"线程 {thread_id} (GPU{device_id}): 平均 {avg_time:.3f}s, 最小 {min_time:.3f}s, 最大 {max_time:.3f}s")
            
            if device_id == 0:
                gpu0_times.extend(times)
            else:
                gpu1_times.extend(times)
            all_times.extend(times)
    
    # GPU0统计
    if gpu0_times:
        gpu0_avg = np.mean(gpu0_times)
        gpu0_min = np.min(gpu0_times)
        gpu0_max = np.max(gpu0_times)
        print(f"\nGPU0 统计:")
        print(f"- 平均耗时: {gpu0_avg:.3f}s")
        print(f"- 最小耗时: {gpu0_min:.3f}s") 
        print(f"- 最大耗时: {gpu0_max:.3f}s")
        print(f"- 总处理次数: {len(gpu0_times)}")
        print(f"- 总耗时: {np.sum(gpu0_times):.3f}s")
        print(f"- 吞吐量: {len(gpu0_times)/np.sum(gpu0_times):.2f} 帧/秒")
    
    # GPU1统计
    if gpu1_times:
        gpu1_avg = np.mean(gpu1_times)
        gpu1_min = np.min(gpu1_times)
        gpu1_max = np.max(gpu1_times)
        print(f"\nGPU1 统计:")
        print(f"- 平均耗时: {gpu1_avg:.3f}s")
        print(f"- 最小耗时: {gpu1_min:.3f}s") 
        print(f"- 最大耗时: {gpu1_max:.3f}s")
        print(f"- 总处理次数: {len(gpu1_times)}")
        print(f"- 总耗时: {np.sum(gpu1_times):.3f}s")
        print(f"- 吞吐量: {len(gpu1_times)/np.sum(gpu1_times):.2f} 帧/秒")
    
    # 总体统计
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
        print(f"- 总吞吐量: {len(all_times)/np.sum(all_times):.2f} 帧/秒")

# 如果你知道帧的确切形状，可以使用这个更精确的加载函数
def load_frame_with_shape(file_path="frame.dat", shape=(480, 640, 3)):
    """使用已知形状加载 Frame 数据"""
    try:
        with open(file_path, 'rb') as f:
            frame_data = f.read()
        
        # 直接将二进制数据转换为 numpy 数组并重塑
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(shape)
        print(f"成功加载帧: {frame.shape}")
        return frame
    except Exception as e:
        print(f"加载帧文件失败: {e}")
        return None

if __name__ == "__main__":
    # 双GPU压测示例
    run_dual_gpu_benchmark(
        total_threads=12,         # 总线程数量
        iterations_per_thread=50,  # 每个线程的迭代次数
        frame_file="frame.dat"    # 帧文件路径
    )
