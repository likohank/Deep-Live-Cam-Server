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
from time import time
from queue import Queue
import threading

from modules.processors.frame import face_swapper
# from modules.processors.frame import face_enhancer


def configurar_providers():
    providers = onnxruntime.get_available_providers()
    print(f"Providers disponíveis: {providers}")
    if 'TensorrtExecutionProvider' in providers:
        return ['TensorrtExecutionProvider']

    if 'CUDAExecutionProvider' in providers:
        print("Usando CUDA para processamento")
        return ['CUDAExecutionProvider']
    
    print("Usando CPU para processamento")
    return ['CPUExecutionProvider']

# modules.globals.many_faces = True

# verificar_gpu()
modules.globals.execution_providers = configurar_providers()
print(f"Providers configurados: {modules.globals.execution_providers}")


# Executar um processamento inicial, para carregar o modelo
source_path = "photos/dilma.jpg"
target_path = "photos/elon.jpg"
output_path = "photos/hi.jpg"

face = cv2.imread(source_path)  
source_face = get_one_face(face)
temp_frame = cv2.imread(target_path)
# frame = temp_frame

face_swapper.process_frame(source_face, temp_frame)


# ========================
# Teste de processamento

class FaceSwapServer:
    def __init__(self, source_image_path="photos/cr7.jpg", max_workers=10):
        self.configurar_qualidade()
        self.carregar_source_face(source_image_path)
        self.clientes_ativos = set()
        
        self.processed_frames = Queue()
        self.raw_frames = Queue()
        self.frames_para_enviar = Queue()
        self.processamento_ativo = True

        for i in range(max_workers):
            t = threading.Thread(target=self.processar_frames)
            t.start()

        self.thread_envio = threading.Thread(target=self.preparar_frames_para_envio)
        self.thread_envio.start()

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

    def decodificar_imagem(self, img_b64):
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def codificar_imagem(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def processar_frames(self):
        print(f"Thread {threading.current_thread().name} iniciada")
              
        while self.processamento_ativo:
            frame = self.raw_frames.get()
            t1 = time()
            # print(f"Thread {threading.current_thread().name} - Inicio")
            new_face = face_swapper.process_frame(self.source_face, frame)
            # new_face = face_enhancer.enhance_face(new_face)
            print(f"Thread {threading.current_thread().name} - Tempo de processamento: {time() - t1:.2f}s")
            self.processed_frames.put(new_face)


    def preparar_frames_para_envio(self):
        print("Thread de preparação de frames iniciada")
        while self.processamento_ativo:
            try:
                frame = self.processed_frames.get()
                frame_b64 = self.codificar_imagem(frame)
                self.frames_para_enviar.put(frame_b64)
            except Exception as e:
                print(f"Erro na thread de preparação: {e}")
                continue

    async def enviar_frames_processados(self):
        print("Loop de envio iniciado")
        while True:
            try:
                if not self.frames_para_enviar.empty():
                    frame_b64 = self.frames_para_enviar.get_nowait()
                    
                    # print(f"Clientes ativos: {self.clientes_ativos}")
                    clientes_desconectados = set()
                    
                    for websocket in self.clientes_ativos:
                        try:
                            # print(f"Enviando frame para cliente {websocket}")
                            await websocket.send('FRAME:' + frame_b64)
                        except Exception as e:
                            print(f"Erro ao enviar frame para cliente: {e}")
                            clientes_desconectados.add(websocket)
                    
                    for cliente in clientes_desconectados:
                        self.clientes_ativos.discard(cliente)
                else:
                    await asyncio.sleep(0.01)  # Pequena pausa para não sobrecarregar a CPU
                    
            except Exception as e:
                print(f"Erro no loop de envio: {e}")
                continue

    async def processar_cliente(self, websocket, *args):
        self.clientes_ativos.add(websocket)
        print(f"Novo cliente conectado. Total de clientes: {len(self.clientes_ativos)}")
        
        try:
            async for mensagem in websocket:
                if mensagem.startswith('FRAME:'):
                    img_b64 = mensagem[len('FRAME:'):]
                    frame = self.decodificar_imagem(img_b64)
                    self.raw_frames.put(frame)
                else:
                    print(f"Mensagem desconhecida recebida: {mensagem[:50]}...")
                    await websocket.send('ERRO: Mensagem desconhecida')

        except Exception as e:
            print(f"Erro na conexão: {e}")
        
        finally:
            self.clientes_ativos.discard(websocket)
            print(f"Cliente desconectado. Clientes restantes: {len(self.clientes_ativos)}")

    def __del__(self):
        self.processamento_ativo = False
        if hasattr(self, 'thread_envio'):
            self.thread_envio.join(timeout=1.0)


async def main():
    server = FaceSwapServer(max_workers=10)
    print("Iniciando servidor WebSocket na porta 8765...")
    
    async with websockets.serve(server.processar_cliente, 
                              '0.0.0.0', 8765, 
                              ping_interval=30, 
                              ping_timeout=10):
        print("Servidor rodando. Aguardando conexões...")
        await asyncio.gather(
            server.enviar_frames_processados(),
            asyncio.Future()  # Mantém o servidor rodando
        )

if __name__ == "__main__":
    asyncio.run(main())

