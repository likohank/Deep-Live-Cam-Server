import cv2
import numpy as np
import time
import multiprocessing as mp
import gfpgan
import torch
import os

# 设置多进程启动方法为 'spawn'
mp.set_start_method('spawn', force=True)

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
    
    # 避免在子进程中调用 torch.cuda.get_device_name
    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{device_id}")
        print(f"进程使用 GPU {device_id}")
    else:
        device = torch.device("cpu")
        print("进程使用 CPU")
    
    enhancer = gfpgan.GFPGANer(
        model_path=model_path,
        upscale=1,
        device=device
    )
    
    return enhancer

def benchmark_worker(process_id, iterations, frame_data, device_id, result_queue, ready_event, start_event):
    """压测工作进程"""
    print(f"进程 {process_id} 启动，使用设备: GPU{device_id}")
    
    # 创建独立的 enhancer 实例
    enhancer = create_enhancer(device_id)
    
    # 预热模型（可选）
    try:
        print(f"进程 {process_id} 预热模型...")
        _, _, _ = enhancer.enhance(frame_data, paste_back=True)
        print(f"进程 {process_id} 模型预热完成")
    except Exception as e:
        print(f"进程 {process_id} 模型预热失败: {e}")
    
    # 通知主进程已准备就绪
    print(f"进程 {process_id} 准备就绪，等待开始信号")
    ready_event.set()
    
    # 等待所有进程准备就绪后的开始信号
    start_event.wait()
    
    # 记录进程开始时间
    process_start_time = time.time()
    
    process_times = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            # 使用 frame_data 进行增强操作
            _, _, enhanced_frame = enhancer.enhance(frame_data, paste_back=True)
            end_time = time.time()
            cost_time = end_time - start_time
            
            process_times.append(cost_time)
            print(f"进程 {process_id}-迭代 {i+1} - GPU{device_id} - 处理耗时: {cost_time:.3f}s")
            
        except Exception as e:
            end_time = time.time()
            cost_time = end_time - start_time
            print(f"进程 {process_id}-迭代 {i+1} - GPU{device_id} - 处理失败: {e}, 耗时: {cost_time:.3f}s")
    
    # 记录进程结束时间
    process_end_time = time.time()
    process_total_time = process_end_time - process_start_time
    
    # 统计结果并发送到队列
    if process_times:
        avg_time = np.mean(process_times)
        max_time = np.max(process_times)
        min_time = np.min(process_times)
        print(f"进程 {process_id} (GPU{device_id}) 完成: 平均 {avg_time:.3f}s, 最小 {min_time:.3f}s, 最大 {max_time:.3f}s")
    
    # 将结果发送到主进程
    result_queue.put({
        'process_id': process_id,
        'device_id': device_id,
        'times': process_times,
        'avg_time': avg_time if process_times else 0,
        'min_time': min_time if process_times else 0,
        'max_time': max_time if process_times else 0,
        'process_total_time': process_total_time,
        'process_start_time': process_start_time,
        'process_end_time': process_end_time
    })

def run_multiprocess_benchmark(total_processes=10, iterations_per_process=5, frame_file="frame.dat"):
    """运行多进程压测"""
    print(f"开始多进程压测配置:")
    print(f"- 总进程数: {total_processes}")
    print(f"- 每进程迭代次数: {iterations_per_process}")
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
    
    # 计算每个GPU分配的进程数
    processes_per_gpu = total_processes // 2
    print(f"每个GPU分配 {processes_per_gpu} 个进程")
    
    # 创建进程间通信对象
    result_queue = mp.Queue()
    ready_events = [mp.Event() for _ in range(total_processes)]
    start_event = mp.Event()
    
    # 创建进程
    processes = []
    for i in range(total_processes):
        # 分配GPU：前一半用GPU0，后一半用GPU1
        device_id = 0 if i < processes_per_gpu else 1
        
        process = mp.Process(
            target=benchmark_worker,
            args=(i, iterations_per_process, frame_data, device_id, result_queue, ready_events[i], start_event)
        )
        processes.append(process)
    
    # 启动所有进程
    for process in processes:
        process.start()
    
    # 等待所有进程准备就绪
    print("等待所有进程加载模型并准备就绪...")
    for event in ready_events:
        event.wait()
    
    print("所有进程已准备就绪，开始测试...")
    
    # 记录整体开始时间
    overall_start_time = time.time()
    
    # 发送开始信号给所有进程
    start_event.set()
    
    # 等待所有进程完成
    for process in processes:
        process.join()
    
    # 记录整体结束时间
    overall_end_time = time.time()
    overall_total_time = overall_end_time - overall_start_time
    
    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # 输出统计结果
    print("\n=== 多进程压测结果统计 ===")
    
    # 按GPU分组统计
    gpu0_times = []
    gpu1_times = []
    all_times = []
    
    # 进程时间统计
    process_times = []
    
    for result in results:
        process_id = result['process_id']
        device_id = result['device_id']
        times = result['times']
        process_total_time = result['process_total_time']
        
        process_times.append(process_total_time)
        
        if times:
            avg_time = result['avg_time']
            max_time = result['max_time']
            min_time = result['min_time']
            
            # 计算每个进程的平均每帧处理时间（包括进程启动开销）
            frames_per_second = len(times) / process_total_time if process_total_time > 0 else 0
            
            print(f"进程 {process_id} (GPU{device_id}):")
            print(f"  - 进程总耗时: {process_total_time:.3f}s")
            print(f"  - 处理帧数: {len(times)}")
            print(f"  - 平均每帧时间: {process_total_time/len(times):.3f}s")
            print(f"  - 进程吞吐量: {frames_per_second:.2f} 帧/秒")
            print(f"  - 增强操作平均耗时: {avg_time:.3f}s, 最小 {min_time:.3f}s, 最大 {max_time:.3f}s")
            
            if device_id == 0:
                gpu0_times.extend(times)
            else:
                gpu1_times.extend(times)
            all_times.extend(times)
    
    # 整体时间统计
    total_frames = len(all_times)
    overall_fps = total_frames / overall_total_time if overall_total_time > 0 else 0
    avg_time_per_frame_overall = overall_total_time / total_frames if total_frames > 0 else 0
    
    print(f"\n=== 整体时间统计 ===")
    print(f"整体开始时间: {time.strftime('%H:%M:%S', time.localtime(overall_start_time))}")
    print(f"整体结束时间: {time.strftime('%H:%M:%S', time.localtime(overall_end_time))}")
    print(f"整体总耗时: {overall_total_time:.3f}s")
    print(f"总处理帧数: {total_frames}")
    print(f"整体平均每帧时间: {avg_time_per_frame_overall:.3f}s")
    print(f"整体吞吐量: {overall_fps:.2f} 帧/秒")
    
    # 进程时间分析
    if process_times:
        fastest_process = np.min(process_times)
        slowest_process = np.max(process_times)
        avg_process_time = np.mean(process_times)
        
        print(f"\n=== 进程时间分析 ===")
        print(f"最快进程耗时: {fastest_process:.3f}s")
        print(f"最慢进程耗时: {slowest_process:.3f}s")
        print(f"平均进程耗时: {avg_process_time:.3f}s")
        print(f"进程间时间差异: {slowest_process - fastest_process:.3f}s")
    
    # GPU0统计
    if gpu0_times:
        gpu0_avg = np.mean(gpu0_times)
        gpu0_min = np.min(gpu0_times)
        gpu0_max = np.max(gpu0_times)
        gpu0_total_frames = len(gpu0_times)
        gpu0_total_time = np.sum(gpu0_times)
        gpu0_fps = gpu0_total_frames / gpu0_total_time if gpu0_total_time > 0 else 0
        
        print(f"\nGPU0 统计:")
        print(f"- 平均增强耗时: {gpu0_avg:.3f}s")
        print(f"- 最小增强耗时: {gpu0_min:.3f}s") 
        print(f"- 最大增强耗时: {gpu0_max:.3f}s")
        print(f"- 总处理帧数: {gpu0_total_frames}")
        print(f"- 总增强耗时: {gpu0_total_time:.3f}s")
        print(f"- 纯增强吞吐量: {gpu0_fps:.2f} 帧/秒")
    
    # GPU1统计
    if gpu1_times:
        gpu1_avg = np.mean(gpu1_times)
        gpu1_min = np.min(gpu1_times)
        gpu1_max = np.max(gpu1_times)
        gpu1_total_frames = len(gpu1_times)
        gpu1_total_time = np.sum(gpu1_times)
        gpu1_fps = gpu1_total_frames / gpu1_total_time if gpu1_total_time > 0 else 0
        
        print(f"\nGPU1 统计:")
        print(f"- 平均增强耗时: {gpu1_avg:.3f}s")
        print(f"- 最小增强耗时: {gpu1_min:.3f}s") 
        print(f"- 最大增强耗时: {gpu1_max:.3f}s")
        print(f"- 总处理帧数: {gpu1_total_frames}")
        print(f"- 总增强耗时: {gpu1_total_time:.3f}s")
        print(f"- 纯增强吞吐量: {gpu1_fps:.2f} 帧/秒")
    
    # 总体统计
    if all_times:
        overall_avg = np.mean(all_times)
        overall_min = np.min(all_times)
        overall_max = np.max(all_times)
        total_enhance_time = np.sum(all_times)
        total_enhance_fps = total_frames / total_enhance_time if total_enhance_time > 0 else 0
        
        print(f"\n总体统计:")
        print(f"- 平均增强耗时: {overall_avg:.3f}s")
        print(f"- 最小增强耗时: {overall_min:.3f}s") 
        print(f"- 最大增强耗时: {overall_max:.3f}s")
        print(f"- 总处理帧数: {total_frames}")
        print(f"- 总增强耗时: {total_enhance_time:.3f}s")
        print(f"- 纯增强吞吐量: {total_enhance_fps:.2f} 帧/秒")
        print(f"- 整体吞吐量: {overall_fps:.2f} 帧/秒")
        
        # 计算效率
        efficiency = total_enhance_time / (overall_total_time * total_processes) if overall_total_time > 0 else 0
        print(f"- 并行效率: {efficiency*100:.2f}%")

if __name__ == "__main__":
    # 多进程压测示例
    run_multiprocess_benchmark(
        total_processes=6,         # 总进程数量
        iterations_per_process=50,  # 每个进程的迭代次数
        frame_file="frame.dat"      # 帧文件路径
    )
