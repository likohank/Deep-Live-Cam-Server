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
        self.raw_queue = Queue(maxsize=20)
        self.processed_queue = Queue(maxsize=20)
        self.running = True
        self.n = 0

    def codificar_imagem(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
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
                self.processed_queue.put(frame_processado, timeout=0.1)
            except Full:
                pass

    def on_error(self, ws, error):
        print(f"Erro de conexão: {error}")
        self.running = False

    def on_close(self, ws, close_status_code, close_msg):
        print("Conexão fechada")
        self.running = False

    def on_open(self, ws):
        print("Conectado ao servidor")
        def enviar_frames():
            while self.running:
                # try:
                frame = self.raw_queue.get()
                if frame is not None:
                    img_b64 = self.codificar_imagem(frame)
                    # print(f"{self.n}Frame enviado")
                    self.n += 1
                    ws.send('FRAME:' + img_b64)
                # except Empty:
                    # time.sleep(0.01) 
                # except Exception as e:
                    # print(f"Erro ao enviar frame: {e}")
                    # time.sleep(0.1)
        
        threading.Thread(target=enviar_frames, daemon=True).start()

    def executar(self):
        # try:
        websocket.enableTrace(False)  # Desabilita logs de debug
        ws = websocket.WebSocketApp(f"ws://{self.server_url}",
                                    on_open=self.on_open,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)

        ws_thread = threading.Thread(target=ws.run_forever, 
                                    kwargs={'ping_interval': 30, 
                                            'ping_timeout': 10})
        ws_thread.daemon = True
        ws_thread.start()

        self.cap = cv2.VideoCapture(1)
        ultima_captura = time.time()
        fps_desejado = 10
        intervalo = 1.0 / fps_desejado

        while self.running:
            ret, frame = self.cap.read()
            agora = time.time()
            
            if (agora - ultima_captura) >= intervalo and not self.raw_queue.full():
                self.raw_queue.put(frame)
                ultima_captura = agora

            try:
                frame_processado = self.processed_queue.get(timeout=0.01)
                cv2.imshow('Deep Live Cam', frame_processado)
            except Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Usuário pressionou 'q' para sair")
                break

        self.running = False
        ws.close()
        ws_thread.join(timeout=1.0)

        # except Exception as e:
        #     print(f"Erro: {e}")

        # finally:
        #     self.running = False
        #     if self.cap:
        #         self.cap.release()
        #     cv2.destroyAllWindows()
        #     print("Cliente finalizado")

def main():
    client = WebcamClient()
    client.executar()

if __name__ == "__main__":
    main()