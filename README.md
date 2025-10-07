# Deep-Live-Cam Server [by Asimov Academy]

Este é um fork otimizado do [Deep-Live-Cam original](https://github.com/hacksider/Deep-Live-Cam), focado em processamento distribuído via WebSocket para melhor performance em deepfakes em tempo real.

## Diferencial desta Versão

Esta versão foi modificada para separar o processamento de imagens da interface do usuário, permitindo:

- Processamento remoto em servidores com GPUs potentes
- Menor latência na interface do usuário
- Possibilidade de múltiplos clientes conectados ao mesmo servidor
- Ideal para streaming e aplicações em tempo real

### Arquitetura

```
Cliente (Webcam) <-> WebSocket <-> Servidor (GPU) 
```

- **Cliente**: Captura frames da webcam e exibe resultados
- **WebSocket**: Comunicação em tempo real de baixa latência
- **Servidor**: Processa as imagens usando GPU dedicada

## Instalação

### Requisitos do Sistema

- Python 3.10
- CUDA Toolkit 11.8.0 (para GPUs NVIDIA)
- ffmpeg

先用pip 安装 torch==2.6.0 torchvision==0.21.0,再安装 basicsr==1.4.2
### Configuração do Ambiente

1. Clone o repositório:
```bash
git clone https://github.com/asimov-academy/Deep-Live-Cam-Server
cd Deep-Live-Cam-Server
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/MacOS
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Baixe os modelos necessários:
- [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
- [inswapper_128_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)

Coloque os arquivos na pasta "models".

### Configuração GPU (NVIDIA)

1. Instale o CUDA Toolkit 11.8.0
2. Configure o onnxruntime-gpu:
```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3
```

## Uso

### Servidor

1. Inicie o servidor WebSocket:
```bash
python server_ws.py
```

O servidor iniciará na porta 8765 por padrão.

### Cliente

1. Configure o endereço do servidor no arquivo de configuração
2. Execute o cliente:
```bash
python client.py
```

## Configuração em Nuvem

Para melhor performance, recomendamos hospedar o servidor em uma máquina com GPU dedicada. Algumas opções:

- Google Cloud com GPUs NVIDIA T4
- AWS EC2 com instâncias g4dn
- Servidores dedicados com GPUs



-----------------------------------------------------------------------------------------
使用腾讯云GPU服务器 + ubuntu 22.04 + 不预装 驱动

1. https://blog.csdn.net/xundh/article/details/127974227  直接开始第二步 安装cuda，因为cuda 安装程序 带 驱动。
2. 下载 cuda_11.8.0_520.61.05_linux.run
3. wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
4. sudo sh cuda_11.8.0_520.61.05_linux.run
5. 勾选上驱动，然后安装
   
   export PATH=$PATH:/usr/local/cuda-11.8/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
   
7. 测试 nvcc --version  ;  nvidia-smi


8. 下载cudnn 8.6 https://developer.nvidia.com/rdp/cudnn-archive, 下载到本地
9. sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.6.0.163/cudnn-local-FAED14DD-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8                 # 这里输入到=按tab补全即可，安装运行库
sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8          # 通过tab自动补全，安装developer 库
sudo apt-get install libcudnn8-samples=8.6.0.163-1+cuda11.8  # 通过tab自动补全，安装示例和文档


8.测试 cudnn
cp -r /usr/src/cudnn_samples_v8/ $HOME 
cd ~/cudnn_samples_v8/mnistCUDNN/
make clean && make
sudo apt-get install libfreeimage3 libfreeimage-dev
./mnistCUDNN

9.下载 minconda https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_25.7.0-2-Linux-x86_64.sh
10. ./conda init
11. conda create -n py311 python==3.11
12. conda activate py311

13. 安装ffmpeg sudo apt install ffmpeg

14. 准备安装 requirements.txt
15. pip install cython
    pip install -i https://mirrors.aliyun.com/pypi/simple tb-nightly
    pip install --upgrade pip setuptools wheel
    pip install torch==2.6.0
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple basicsr==1.4.2

16. 安装 requirements.txt，内容如下
    cog==0.14.12
    customtkinter==5.2.2
    cv2_enumerate_cameras==1.1.18.3
    facexlib==0.3.0
    insightface==0.7.3
    numpy<2
    onnxruntime_gpu==1.16.3
    opencv_python_headless==4.11.0.86
    opennsfw2==0.14.0
    Pillow==11.2.1
    pygrabber==0.2
    PyYAML==6.0.2
    realesrgan==0.3.0
    scikit_learn==1.2.2
    tensorflow==2.20.0
    torch==2.7.0
    torchvision
    tqdm==4.66.4
    websocket_client==1.8.0
    websockets==12.0
    
    #torch_tensorrt==2.7.0


18. pip install -r requirements.txt
19. 启动程序前 第一次 先设置个 代理， 因为要下载些东西
    export http_proxy="socks5://8YZsweP:Rtw2111j@18.55.140.95:10120"
    export https_proxy="socks5://8YZsweP:Rtw2111j@18.55.140.95:10120"
    export http_proxy=;export https_proxy=
20. python server_ws.py


21. 客户端 本地台式机安装
       pip install websocket_client==1.8.0
       pip install websocket_client==1.8.0
       pip install opencv_python==4.8.0.74
    python client_ws.py



22. 如果遇到 TensorrtExecutionProvider 错误
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
    tar -xzf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
    sudo mv TensorRT-8.6.1.6 /usr/local/tensorrt
    echo 'export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export PATH=/usr/local/tensorrt/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
    ls /usr/local/tensorrt/lib/libnvinfer.so.8

23. 出现ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'的原因大概是原先的“名字”改了，但是安装的basicsr包中的名字没有改，所以会报错。
只要在miniconda3/lib/python3.12/site-packages/basicsr/data/degradations.py文件中第8行将
原from torchvision.transforms.functional_tensor import rgb_to_grayscale
改成from torchvision.transforms._functional_tensor import rgb_to_grayscale
或者改成from torchvision.transforms.functional import rgb_to_grayscale
均能够解决问题


-------------------------------------------------
 server_ws.py 的一些优化

1.开启人脸增强的开关

from modules.processors.frame import face_enhancer
new_face = face_enhancer.enhance_face(new_face)

2.增大frame 交换
FaceSwapServer(max_workers=30)


3.关闭 TensorrtExecutionProvider  优化加速
#if 'TensorrtExecutionProvider' in providers:
#    return ['TensorrtExecutionProvider']

4.返回给客户端 png，增强画质(好像不起作用)
#_, buffer = cv2.imencode('.jpg', frame)
_, buffer = cv2.imencode('.png', frame)
