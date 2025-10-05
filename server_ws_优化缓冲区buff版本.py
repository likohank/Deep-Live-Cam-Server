import os
import modules
import modules.globals
import cv2
import onnxruntime
from modules.face_analyser import get_one_face
from PIL import Image
import asyncio
import websockets
import numpy as np
import base64
import time
from queue import Queue
import threading
import torch
import collections

from modules.processors.frame import face_swapper
from modules.processors.frame import face_enhancer

def configurar_providers():
    providers = onnxruntime.get_available_providers()
    print(f"[ort] available providers: {providers}")
    if "CUDAExecutionProvider" in providers:
        print("[ort] using CUDAExecutionProvider")
        return [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "cudnn_conv_use_max_workspace": "1",
                "do_copy_in_default_stream": "1"
            }),
            "CPUExecutionProvider",
        ]
    print("[ort] fallback CPUExecutionProvider")
    return ["CPUExecutionProvider"]

# 配置执行提供者
modules.globals.execution_providers = configurar_providers()
print(f"Providers configurados: {modules.globals.execution_providers}")

# 执行一个预处理，加载模型
source_path = "photos/dilma.jpg"
target_path = "photos/elon.jpg"
output_path = "photos/hi.jpg"

face = cv2.imread(source_path)
source_face = get_one_face(face)
temp_frame = cv2.imread(target_path)
face_swapper.process_frame(source_face, temp_frame)

def check_gpu_memory():
    """检查GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU显存: 已分配 {allocated:.2f}GB / 保留 {reserved:.2f}GB"
    return "GPU不可用"

class FrameBuffer:
    """帧缓冲区，用于平滑视频流"""
    def __init__(self, max_size=15):  # 大约0.5秒的缓冲（假设30fps）
        self.buffer = collections.OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
        
    def put(self, sequence, frame_b64):
        """放入帧"""
        with self.lock:
            # 如果缓冲区已满，移除最旧的帧
            if len(self.buffer) >= self.max_size:
                oldest_seq = next(iter(self.buffer))
                del self.buffer[oldest_seq]
            
            self.buffer[sequence] = frame_b64
    
    def get_latest(self):
        """获取最新的帧"""
        with self.lock:
            if not self.buffer:
                return None, None
            
            # 返回最新的帧
            latest_seq = max(self.buffer.keys())
            return latest_seq, self.buffer[latest_seq]
    
    def get_oldest(self):
        """获取最旧的帧"""
        with self.lock:
            if not self.buffer:
                return None, None
            
            # 返回最旧的帧
            oldest_seq = next(iter(self.buffer))
            return oldest_seq, self.buffer[oldest_seq]
    
    def size(self):
        """返回缓冲区大小"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()

class FaceSwapServer:
    def __init__(self, source_image_path="photos/cr7.jpg", max_workers=8):
        self.configurar_qualidade()
        self.carregar_source_face(source_image_path)
        self.clientes_ativos = set()

        # 预初始化 GFPGAN 对象池
        print("正在初始化 GFPGAN 对象池...")
        face_enhancer.init_face_enhancer_pool()
        print("GFPGAN 对象池初始化完成")

        # 帧序列号和最新帧管理
        self.frame_sequence = 0
        self.sequence_lock = threading.Lock()
        
        # 帧缓冲区
        self.frame_buffer = FrameBuffer(max_size=50)  # 约0.5秒缓冲
        
        # 队列配置
        self.raw_frames = Queue()  # 原始帧队列，元素为 (sequence, frame)
        self.processamento_ativo = True
        
        # 统计信息
        self.processed_count = 0
        self.dropped_count = 0  # 丢弃的帧数
        self.buffer_dropped_count = 0  # 缓冲区丢弃的帧数
        self.start_time = time.time()
        self.stats_lock = threading.Lock()
        
        # 发送控制
        self.last_sent_sequence = -1
        self.send_interval = 0.033  # 目标发送间隔（约30fps）
        self.last_sent_time = time.time()

        # 启动工作线程
        self.worker_threads = []
        for i in range(max_workers):
            t = threading.Thread(target=self.processar_frames, name=f"Worker-{i}")
            t.daemon = True
            t.start()
            self.worker_threads.append(t)

        # 启动统计线程
        self.stats_thread = threading.Thread(target=self.mostrar_estatisticas)
        self.stats_thread.daemon = True
        self.stats_thread.start()

        print(f"服务器初始化完成，工作线程数: {max_workers}")

    def configurar_qualidade(self):
        modules.globals.color_adjustment = True
        modules.globals.mouth_mask = True
        modules.globals.mask_feather_ratio = 8
        modules.globals.mask_down_size = 0.1
        modules.globals.mask_size = 0.5
        modules.globals.source_image_scaling_factor = 2

    def carregar_source_face(self, source_image_path):
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            raise Exception(f"Erro ao carregar imagem fonte: {source_image_path}")

        self.source_face = get_one_face(source_img)
        if self.source_face is None:
            raise Exception("Nenhum rosto encontrado na imagem fonte")
        print("源人脸加载成功")

    def decodificar_imagem(self, img_b64):
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def codificar_imagem(self, frame):
        # 调整JPEG质量以减少带宽使用
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')

    def get_next_sequence(self):
        """获取下一个帧序列号"""
        with self.sequence_lock:
            seq = self.frame_sequence
            self.frame_sequence += 1
            return seq

    def processar_frames(self):
        thread_name = threading.current_thread().name
        print(f"线程 {thread_name} 启动")

        while self.processamento_ativo:
            try:
                # 获取待处理帧（带序列号）
                sequence, frame = self.raw_frames.get(timeout=1.0)
                
                # 检查缓冲区是否已满，如果满则丢弃最旧的帧
                if self.frame_buffer.size() >= 15:  # 缓冲区已满
                    oldest_seq, _ = self.frame_buffer.get_oldest()
                    if oldest_seq is not None and sequence > oldest_seq + 5:
                        # 如果当前帧比最旧帧新很多，丢弃最旧帧为新帧腾出空间
                        self.frame_buffer.clear()
                        with self.stats_lock:
                            self.buffer_dropped_count += 1
                
                t1 = time.time()
                
                # 人脸交换
                new_face = face_swapper.process_frame(self.source_face, frame)
                
                # 人脸增强（使用对象池）
                new_face = face_enhancer.enhance_face(new_face)
                
                processing_time = time.time() - t1
                
                # 更新统计信息
                with self.stats_lock:
                    self.processed_count += 1
                
                # 将处理后的帧编码并放入缓冲区
                frame_b64 = self.codificar_imagem(new_face)
                self.frame_buffer.put(sequence, frame_b64)
                
                # 打印处理时间（限制频率）
                if self.processed_count % 10 == 0:
                    print(f"线程 {thread_name} - 处理时间: {processing_time:.3f}s, 序列号: {sequence}")
                    
            except Exception as e:
                # 忽略超时异常
                if "empty" not in str(e):
                    print(f"线程 {thread_name} 处理错误: {e}")
                continue

    def mostrar_estatisticas(self):
        """显示处理统计信息"""
        last_count = 0
        last_dropped = 0
        last_buffer_dropped = 0
        while self.processamento_ativo:
            try:
                time.sleep(5)
                with self.stats_lock:
                    current_count = self.processed_count
                    current_dropped = self.dropped_count
                    current_buffer_dropped = self.buffer_dropped_count
                    elapsed_time = time.time() - self.start_time
                    
                processed_since_last = current_count - last_count
                dropped_since_last = current_dropped - last_dropped
                buffer_dropped_since_last = current_buffer_dropped - last_buffer_dropped
                
                fps = processed_since_last / 5
                avg_fps = current_count / elapsed_time if elapsed_time > 0 else 0
                
                gpu_info = check_gpu_memory()
                
                buffer_size = self.frame_buffer.size()
                
                total_dropped = dropped_since_last + buffer_dropped_since_last
                drop_rate = total_dropped / max(processed_since_last + total_dropped, 1) * 100
                
                print(f"[统计] 总处理帧数: {current_count}, 丢弃帧数: {current_dropped}, 缓冲区丢弃: {current_buffer_dropped}")
                print(f"[性能] 瞬时FPS: {fps:.2f}, 平均FPS: {avg_fps:.2f}, 总丢弃率: {drop_rate:.1f}%")
                print(f"[队列] 原始帧: {self.raw_frames.qsize()}, 缓冲区大小: {buffer_size}")
                print(f"[显存] {gpu_info}")
                print("-" * 50)
                
                last_count = current_count
                last_dropped = current_dropped
                last_buffer_dropped = current_buffer_dropped
            except Exception as e:
                print(f"统计线程错误: {e}")

    async def enviar_frames_processados(self):
        """发送处理完成的帧给客户端"""
        print("发送循环启动")
        sent_count = 0
        
        while True:
            try:
                current_time = time.time()
                
                # 检查是否达到发送间隔
                if current_time - self.last_sent_time >= self.send_interval:
                    # 从缓冲区获取最新的帧
                    sequence, frame_b64 = self.frame_buffer.get_latest()
                    
                    if frame_b64 is not None and sequence != self.last_sent_sequence:
                        # 发送帧
                        clientes_desconectados = set()
                        for websocket in self.clientes_ativos:
                            try:
                                await websocket.send('FRAME:' + frame_b64)
                            except Exception as e:
                                print(f"发送帧到客户端错误: {e}")
                                clientes_desconectados.add(websocket)

                        for cliente in clientes_desconectados:
                            self.clientes_ativos.discard(cliente)
                        
                        self.last_sent_sequence = sequence
                        self.last_sent_time = current_time
                        sent_count += 1
                        
                        # 每发送100帧打印一次
                        if sent_count % 100 == 0:
                            print(f"已发送 {sent_count} 帧到客户端, 当前序列号: {sequence}")
                
                # 短暂休眠，避免过度占用CPU
                await asyncio.sleep(0.001)

            except Exception as e:
                print(f"发送循环错误: {e}")
                await asyncio.sleep(0.01)

    async def processar_cliente(self, websocket, path):
        self.clientes_ativos.add(websocket)
        client_count = len(self.clientes_ativos)
        print(f"新客户端连接。当前客户端数: {client_count}")

        try:
            async for mensagem in websocket:
                if mensagem.startswith('FRAME:'):
                    img_b64 = mensagem[len('FRAME:'):]
                    frame = self.decodificar_imagem(img_b64)
                    sequence = self.get_next_sequence()
                    
                    # 如果原始帧队列过大，丢弃最旧的帧
                    if self.raw_frames.qsize() > 8:  # 适当增加队列大小
                        try:
                            self.raw_frames.get_nowait()  # 丢弃一帧旧数据
                            with self.stats_lock:
                                self.dropped_count += 1
                        except:
                            pass
                    
                    self.raw_frames.put((sequence, frame))
                else:
                    print(f"未知消息: {mensagem[:50]}...")
                    await websocket.send('ERRO: Mensagem desconhecida')

        except Exception as e:
            print(f"客户端连接错误: {e}")

        finally:
            self.clientes_ativos.discard(websocket)
            client_count = len(self.clientes_ativos)
            print(f"客户端断开。剩余客户端: {client_count}")

    def __del__(self):
        self.processamento_ativo = False
        for t in self.worker_threads:
            t.join(timeout=0.5)

async def main():
    # 根据GPU内存调整工作线程数
    max_workers = 24
    
    server = FaceSwapServer(max_workers=max_workers)
    print(f"启动 WebSocket 服务器在端口 8765...")

    async with websockets.serve(
        server.processar_cliente,
        '0.0.0.0', 8765,
        ping_interval=30,
        ping_timeout=10,
        max_size=10 * 1024 * 1024  # 10MB 最大消息大小
    ):
        print("服务器运行中。等待连接...")
        await asyncio.gather(
            server.enviar_frames_processados(),
            asyncio.Future()  # 保持服务器运行
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("服务器被用户中断")
    except Exception as e:
        print(f"服务器错误: {e}")
