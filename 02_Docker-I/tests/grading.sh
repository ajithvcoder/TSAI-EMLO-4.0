#!/bin/bash

set -e

DOCKER_IMAGE_NAME="student_image"
DOCKER_CONTAINER_NAME="grading_container"
DOCKERFILE="Dockerfile"
TRAINING_SCRIPT="train.py"

# 1. Check the Dockerfile exists
if [ ! -f $DOCKERFILE ]; then
    echo "🚫 $DOCKER_IMAGE_NAME does not exist"
    exit 1
fi

echo "🚚 Building the Docker image..."
docker build -t $DOCKER_IMAGE_NAME .

# 2. Check the size of the Docker image
image_size=$(docker inspect $DOCKER_IMAGE_NAME --format='{{.Size}}')
image_size_gb=$((image_size/1000000000))
image_size_mb=$(((image_size%1000000000)/1000000))
if [ $image_size -gt 1100000000 ]; then
    echo "💥 Docker image is too large. Size: $image_size_gb GB $image_size_mb MB"
    exit 1
else
    echo "✅ Docker image size is acceptable. Size: $image_size_gb GB $image_size_mb MB"
fi

# 3. Check the training script runs successfully
echo "🏫 Running the training script..."
start_time=$(date +%s)
docker run --name $DOCKER_CONTAINER_NAME --rm -v$(pwd):/workspace $DOCKER_IMAGE_NAME python /workspace/$TRAINING_SCRIPT
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "🕒 Total runtime of the training script: $runtime seconds"

# 4. Check that the checkpoint is saved to host system
if [ ! -f model_checkpoint.pth ]; then
    echo "🚫 Model checkpoint was not correctly saved to the host system"
    exit 1
else
    checkpoint_size=$(du -sh model_checkpoint.pth | cut -f1)
    echo "✅ Model checkpoint saved correctly to the host system. Size: $checkpoint_size"
fi

# 5. Check the training script can resume from a checkpoint
echo "⏱ Checking the training script can resume from a checkpoint..."
docker run --name $DOCKER_CONTAINER_NAME --rm -v $(pwd):/workspace $DOCKER_IMAGE_NAME python /workspace/$TRAINING_SCRIPT --resume

echo "🎉 All checks passed!"
