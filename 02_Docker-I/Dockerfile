FROM python:3.9-slim
USER root

# using workspace as it is mentioned in testing script
WORKDIR /workspace
COPY train.py /workspace/train.py

# Since python3.9 has numpy=2+ version we are using 1.26.4 here for torch=1.9.0 compactability
RUN pip3 --no-cache-dir install numpy==1.26.4
RUN pip3 --no-cache-dir install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
CMD ["python3", "train.py"]
