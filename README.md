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

## Contribuição

Sinta-se à vontade para contribuir com o projeto através de pull requests ou reportando issues.

## Licença

Este projeto segue a mesma licença do projeto original, com restrições para uso não-comercial dos modelos do InsightFace.
