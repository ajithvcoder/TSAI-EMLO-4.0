#!/bin/bash
#=============================================
# Project        : Download and Unzip Dataset
# Version        : 0.1.0
# Description    : download raw datasets from kaggle
# CREATED        : 02-16-2025

# Schedule execution of many runs
# Run from root folder with: bash shscripts/download_zips.sh
#=============================================
echo "Script to Download Dataset From Kaggle"


echo "************* Sports Dataset *************"
curl -L -o ./data/raw/sports-classiunfication.zip https://www.kaggle.com/api/v1/datasets/download/gpiosenka/sports-classification
echo "******************************************"


echo "*********** FruitsVeg Dataset ************"
curl -L -o ./data/raw/fruit-and-vegetable-image-recognition.zip https://www.kaggle.com/api/v1/datasets/download/kritikseth/fruit-and-vegetable-image-recognition
echo "******************************************"

echo "************ Making Dirs *****************"
mkdir -p ./data/processed/sports/
mkdir -p ./data/processed/vegfruits/
echo "******************************************"


echo "*********** Unzipping 📁 *****************"
unzip ./data/raw/sports-classiunfication.zip -d ./data/processed/sports/
unzip ./data/raw/fruit-and-vegetable-image-recognition.zip -d ./data/processed/vegfruits/
echo "******************************************"