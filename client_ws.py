import cv2
import base64
import numpy as np
import time
import threading
from queue import Queue, Empty, Full
import websocket

class WebcamClient:
    def __init__(self, server_url='192.168.1.5:8765'):
        self.server_url = server_url
        self.cap = None
        self.raw_queue = Queue(maxsize=2)  # Buffer mínimo para evitar acúmulo
        self.processed_queue = Queue(maxsize=2)  # Buffer mínimo para evitar latência
        self.running = True
        self.n = 0
        self.ultima_captura = 0
        self.fps_desejado = 10
        self.intervalo = 1.0 / self.fps_desejado

    def codificar_imagem(self, frame):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Qualidade JPEG reduzida para maior velocidade
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')

    def decodificar_imagem(self, img_b64):
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def on_message(self, ws, message):
        if message.startswith('FRAME:'):
            frame_b64 = message[len('FRAME:'):]
            frame_processado = self.decodificar_imagem(frame_b64)
            try:
                self.processed_queue.put_nowait(frame_processado)  # Non-blocking
            except Full:
                self.processed_queue.get_nowait()  # Descarta frame antigo
                self.processed_queue.put_nowait(frame_processado)

    def on_error(self, ws, error):
        print(f"Erro de conexão: {error}")
        self.running = False

    def on_close(self, ws, close_status_code, close_msg):
        print("Conexão fechada")
        self.running = False

    def on_open(self, ws):
        print("Conectado ao servidor")
        threading.Thread(target=self.enviar_frames, args=(ws,), daemon=True).start()

    def enviar_frames(self, ws):
        while self.running:
            try:
                frame = self.raw_queue.get_nowait()
                if frame is not None:
                    img_b64 = self.codificar_imagem(frame)
                    self.n += 1
                    ws.send('FRAME:' + img_b64)
            except Empty:
                time.sleep(0.001)  # Pequena pausa quando não há frames

    def capturar_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                agora = time.time()
                if agora - self.ultima_captura >= self.intervalo:
                    try:
                        self.raw_queue.put_nowait(frame)
                        self.ultima_captura = agora
                    except Full:
                        continue  # Pula frame se a fila estiver cheia
            else:
                print("Erro ao capturar frame")
                break

    def executar(self):
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(f"ws://{self.server_url}",
                                  on_open=self.on_open,
                                  on_message=self.on_message,
                                  on_error=self.on_error,
                                  on_close=self.on_close)

        ws_thread = threading.Thread(target=ws.run_forever, 
                                   kwargs={'ping_interval': 30, 
                                         'ping_timeout': 10},
                                   daemon=True)
        ws_thread.start()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduz resolução
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Reduz resolução
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        captura_thread = threading.Thread(target=self.capturar_frames, daemon=True)
        captura_thread.start()

        cv2.namedWindow('Deep Live Cam')
        
        try:
            while self.running:
                try:
                    frame_processado = self.processed_queue.get_nowait()
                    cv2.imshow('Deep Live Cam', frame_processado)
                except Empty:
                    pass

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Usuário pressionou 'q' para sair")
                    break
                
        except KeyboardInterrupt:
            print("\nInterrupção detectada")
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            ws.close()
            
def main():
    client = WebcamClient()
    client.executar()

if __name__ == "__main__":
    main()
