FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

USER root

# Explicitly set GPU-related env vars to avoid CPU fallback
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV FORCE_CUDA=1  
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Forces CUDA support (for packages like torchvision)

WORKDIR /workspace
COPY . .
RUN pip install -r requirements.gpu.txt  && rm -rf /root/.cache/pip

# CMD ["python", "src/train.py"]
CMD ["tail", "-f", "/dev/null"]
