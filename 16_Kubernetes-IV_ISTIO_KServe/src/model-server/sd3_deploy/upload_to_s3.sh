#!/bin/bash

# Configuration
S3_BUCKET="mybucket-emlo-mumbai"
MODEL_DIR="sd-small-model"
CONFIG_DIR="./config"
MAR_FILE="../model-store/sd_small.mar"

# TODO: change to s3 again
# Upload model files to S3
echo "Uploading model files to S3..."
aws s3 cp --recursive ${MODEL_DIR} s3://${S3_BUCKET}/sd_small/sd-small-model/

# Upload config file to S3
echo "Uploading config file to S3..."
aws s3 cp ${CONFIG_DIR}/config.properties s3://${S3_BUCKET}/sd_small/config/

# Upload MAR file if it exists
echo "Uploading MAR file to S3..."
aws s3 cp ${MAR_FILE} s3://${S3_BUCKET}/sd_small/model-store/

echo "Upload complete!" 