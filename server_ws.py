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
import heapq  # 用于优先级队列

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
    """检查所有GPU显存使用情况"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        memory_info = []
        
        for i in range(gpu_count):
            try:
                # 使用device上下文来安全地检查每个GPU
                with torch.cuda.device(i):
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    memory_info.append(f"GPU{i}: {allocated:.2f}GB/{reserved:.2f}GB")
            except Exception as e:
                memory_info.append(f"GPU{i}: 错误")
        
        return " | ".join(memory_info)
    return "GPU不可用"

class FrameWithSequence:
    """带序列号的帧"""
    def __init__(self, sequence, frame):
        self.sequence = sequence
        self.frame = frame
    
    def __lt__(self, other):
        # 用于优先级队列比较
        return self.sequence < other.sequence

class FaceSwapServer:
    def __init__(self, source_image_path="photos/cr7.jpg", max_workers=10):
        self.configurar_qualidade()
        self.carregar_source_face(source_image_path)
        self.clientes_ativos = set()

        # 预初始化 GFPGAN 对象池
        print("正在初始化 GFPGAN 对象池...")
        face_enhancer.init_face_enhancer_pool()
        print("GFPGAN 对象池初始化完成")

        # 帧序列号
        self.frame_sequence = 0
        self.sequence_lock = threading.Lock()
        
        # 队列配置 - 使用优先级队列确保顺序
        self.raw_frames = Queue()  # 原始帧队列，元素为 (sequence, frame)
        self.processed_frames = []  # 处理完成的帧，使用堆确保顺序
        self.processed_frames_lock = threading.Lock()
        self.next_sequence_to_send = 0  # 下一个要发送的序列号
        
        self.frames_para_enviar = Queue()
        self.processamento_ativo = True
        
        # 统计信息
        self.processed_count = 0
        self.start_time = time.time()
        self.stats_lock = threading.Lock()

        # 启动工作线程
        self.worker_threads = []
        for i in range(max_workers):
            t = threading.Thread(target=self.processar_frames, name=f"Worker-{i}")
            t.daemon = True
            t.start()
            self.worker_threads.append(t)

        # 启动发送线程
        self.thread_envio = threading.Thread(target=self.preparar_frames_para_envio)
        self.thread_envio.daemon = True
        self.thread_envio.start()

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

        #modules.globals.color_adjustment = True
        #modules.globals.mouth_mask = True        # 保持嘴部自然度
        #modules.globals.mask_feather_ratio = 6   # 适中的羽化
        #modules.globals.mask_down_size = 0.15    # 适中的下采样
        #modules.globals.mask_size = 0.4          # 适中的遮罩范围
        #modules.globals.source_image_scaling_factor = 1.5  # 适中的源图像缩放

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
                t_total_start_0 = time.time()
                # 获取待处理帧（带序列号）
                sequence, frame = self.raw_frames.get(timeout=1.0)
                

                # 总开始时间
                t_total_start = time.time()
                
                # 1. 人脸检测时间
                t_detect_start = time.time()
                #target_face = get_one_face(frame)
                detect_time = time.time() - t_detect_start
                
                #if target_face:
                if True:
                    # 2. 人脸交换时间
                    t_swap_start = time.time()
                    new_face = face_swapper.process_frame(self.source_face, frame)
                    swap_time = time.time() - t_swap_start
                    
                    # 3. 人脸增强时间
                    t_enhance_start = time.time()
                    new_face = face_enhancer.enhance_face(new_face)
                    enhance_time = time.time() - t_enhance_start
                    
                    total_time = time.time() - t_total_start
                    
                    # 更新统计信息
                    with self.stats_lock:
                        self.processed_count += 1
                    
                    lock_push_start = time.time()
                    # 将处理后的帧按序列号放入优先级队列
                    with self.processed_frames_lock:
                        suc_lock_push_time = time.time()
                        suc_lock_push_cost = suc_lock_push_time - lock_push_start 

                        heapq.heappush(self.processed_frames, FrameWithSequence(sequence, new_face))
                        push_cost_time = time.time() - suc_lock_push_time

                    total_time_0 = time.time() - t_total_start_0
                    # 每帧都打印详细时间（前50帧），之后每10帧打印一次
                    if self.processed_count <= 50 or self.processed_count % 10 == 0:
                        print(f"线程 {thread_name} -总耗时{total_time_0:.3f}s;push获取锁耗时{suc_lock_push_cost:.3f}s;push耗时:{push_cost_time:.3f}s 核心三步总时间: {total_time:.3f}s | 检测: {detect_time:.3f}s | 交换: {swap_time:.3f}s | 增强: {enhance_time:.3f}s | 序列号: {sequence}")
                else:
                    # 如果没有检测到人脸，也记录时间
                    total_time = time.time() - t_total_start
                    swap_time = 0
                    enhance_time = 0
                    
                    with self.stats_lock:
                        self.processed_count += 1
                    
                    with self.processed_frames_lock:
                        heapq.heappush(self.processed_frames, FrameWithSequence(sequence, frame))
                    
                    if self.processed_count <= 50 or self.processed_count % 1 == 0:
                        print(f"线程 {thread_name} - 总时间: {total_time:.3f}s | 检测: {detect_time:.3f}s | 无人脸跳过 | 序列号: {sequence}")
                        
            except Exception as e:
                # 忽略超时异常
                if len( str(e) )>4:
                    print(f"线程 {thread_name} 处理错误: {e}")
                continue

    def preparar_frames_para_envio(self):
        """准备按顺序发送的帧"""
        print("帧准备线程启动")
        empty_cycles = 0  # 连续空循环计数
        
        while self.processamento_ativo:
            try:
                frame_obj = None
                
                # 短暂获取锁检查堆顶元素
                with self.processed_frames_lock:
                    ##if self.processed_frames and self.processed_frames[0].sequence == self.next_sequence_to_send:
                    #if self.processed_frames and (abs(self.processed_frames[0].sequence - self.next_sequence_to_send) < 10 or len(self.processed_frames)>10):
                    #    frame_obj = heapq.heappop(self.processed_frames)
                    #    frame_b64 = self.codificar_imagem(frame_obj.frame)
                    #    #print("++++++++++++++++++++=%s++++++++++++++"%(self.next_sequence_to_send))
                    #    #print("++++++++@@@@@@@@@++++=%s ++++++++++++++"%(frame_obj.sequence))
                    #    if frame_obj.sequence>self.next_sequence_to_send:
                    #        self.frames_para_enviar.put((frame_obj.sequence, frame_b64))
                    #    self.next_sequence_to_send = frame_obj.sequence
                    #if self.processed_frames and (abs(self.processed_frames[0].sequence - self.next_sequence_to_send) < 3 or len(self.processed_frames)>10):
                    if self.processed_frames and self.processed_frames[0].sequence == self.next_sequence_to_send:
                        frame_obj = heapq.heappop(self.processed_frames)
                        empty_cycles = 0  # 重置空循环计数
                    else:
                        empty_cycles += 1
                
                if frame_obj is not None:
                    # 在锁外进行编码和入队操作
                    frame_b64 = self.codificar_imagem(frame_obj.frame)
                    self.frames_para_enviar.put((frame_obj.sequence, frame_b64))
                    self.next_sequence_to_send += 1
                    
                    # 每处理10帧打印一次进度
                    if self.next_sequence_to_send % 10 == 0:
                        print(f"已准备发送帧序列: {self.next_sequence_to_send}")
                else:
                    # 根据空循环次数动态调整等待时间
                    if empty_cycles > 100:  # 连续100次空循环，等待时间增加
                        time.sleep(0.01)
                    elif empty_cycles > 10:  # 连续10次空循环，中等等待
                        time.sleep(0.005)
                    else:  # 短暂等待
                        time.sleep(0.001)
                        
            except Exception as e:
                print(f"帧准备错误: {e}")
                time.sleep(0.01)


    def mostrar_estatisticas(self):
        """显示处理统计信息"""
        last_count = 0
        while self.processamento_ativo:
            try:
                time.sleep(5)
                with self.stats_lock:
                    current_count = self.processed_count
                    elapsed_time = time.time() - self.start_time
                    
                processed_since_last = current_count - last_count
                fps = processed_since_last / 5
                avg_fps = current_count / elapsed_time if elapsed_time > 0 else 0
                
                gpu_info = check_gpu_memory()
                
                with self.processed_frames_lock:
                    pending_frames = len(self.processed_frames)
                
                print(f"[统计] 总处理帧数: {current_count}, 瞬时FPS: {fps:.2f}, 平均FPS: {avg_fps:.2f}")
                print(f"[队列] 原始帧: {self.raw_frames.qsize()}, 待发送: {self.frames_para_enviar.qsize()}, 乱序帧: {pending_frames}")
                print(f"[序列] 下一个发送: {self.next_sequence_to_send}")
                print(f"[显存] {gpu_info}")
                print("-" * 50)
                
                last_count = current_count
            except Exception as e:
                print(f"统计线程错误: {e}")

    async def enviar_frames_processados(self):
        print("发送循环启动")
        last_sent_time = time.time()
        sent_count = 0
        
        while True:
            try:
                current_time = time.time()
                
                # 限制发送频率（最大30FPS）
                #if current_time - last_sent_time < 0.033:  # 约30FPS
                # 限制发送频率（最大20FPS）
                if current_time - last_sent_time < 0.05:  # 约20FPS
                    await asyncio.sleep(0.001)
                    continue
                
                if not self.frames_para_enviar.empty():
                    sequence, frame_b64 = self.frames_para_enviar.get_nowait()
                    last_sent_time = current_time
                    sent_count += 1

                    clientes_desconectados = set()
                    for websocket in self.clientes_ativos:
                        try:
                            await websocket.send('FRAME:' + frame_b64)
                        except Exception as e:
                            print(f"发送帧到客户端错误: {e}")
                            clientes_desconectados.add(websocket)

                    for cliente in clientes_desconectados:
                        self.clientes_ativos.discard(cliente)
                        
                    # 每发送100帧打印一次
                    if sent_count % 100 == 0:
                        print(f"已发送 {sent_count} 帧到客户端, 当前序列号: {sequence}")
                else:
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
                    
                    # 如果原始帧队列过大，丢弃旧帧
                    if self.raw_frames.qsize() > 10:
                        try:
                            None
                            #self.raw_frames.get_nowait()  # 丢弃一帧旧数据
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
        if hasattr(self, 'thread_envio'):
            self.thread_envio.join(timeout=1.0)
        for t in self.worker_threads:
            t.join(timeout=0.5)

async def main():
    # 根据GPU内存调整工作线程数
    max_workers = 1
    
    server = FaceSwapServer(max_workers=max_workers)
    print(f"启动 WebSocket 服务器在端口 8765...")

    async with websockets.serve(
        server.processar_cliente,
        '0.0.0.0', 8765,
        ping_interval=30,
        ping_timeout=60,
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
