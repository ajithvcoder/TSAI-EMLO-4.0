FROM public.ecr.aws/docker/library/python:3.12-slim

# Copy the Lambda adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Set environment variables
ENV PORT=8000

# Set working directory
WORKDIR /var/task

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app.py ./
COPY icons.py ./
COPY model.onnx ./

# Set command
CMD exec uvicorn --host 0.0.0.0 --port $PORT app:app 