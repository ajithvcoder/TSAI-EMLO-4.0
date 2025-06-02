# Setup Torchserve
```bash
# install jdk & config JAVA_HOME
sudo apt install --no-install-recommends -y openjdk-11-jre-headless

# JAVA Installation
sudo apt install default-jdk
update-alternatives --config java
JAVA_HOME="/lib/jvm/java-11-openjdk-amd64" # keep it in .bashrc
```

# SPORTS

```bash 
# make mar file
torch-model-archiver --model-name msports --serialized-file checkpoints/onnxs/sports.onnx --handler src/backend/torchserve_app/sports_handler.py --export-path checkpoints/model_stores/sports/ -f --version 0.0.1 --extra-files checkpoints/model_stores/sports/index_to_name.json 

# start server
torchserve --start --model-store checkpoints/model_stores/sports/ --ts-config checkpoints/model_stores/sports/config.properties --enable-model-api --disable-token-auth

# inference
curl http://localhost:8080/ping
curl -X OPTIONS http://localhost:8080 -o src/backend/torchserve_app/sports_swagger.json
curl http://localhost:8080/predictions/msports -F 'data=@data/processed/sports/train/speed skating/001.jpg'

# management
curl http://localhost:8081/models
curl http://localhost:8081/models/msports
curl -v -X PUT "http://localhost:8081/models/msports?min_workers=1&batch_size=10"

# metrics
curl http://localhost:8082/metrics

# stop server
torchserve --stop
```

# VegFruits

```bash

# make mar file
torch-model-archiver --model-name mvegfruits --serialized-file checkpoints/onnxs/vegfruits.onnx --handler src/backend/torchserve_app/vegfruits_handler.py --export-path checkpoints/model_stores/vegfruits/ -f --version 0.0.1 --extra-files checkpoints/model_stores/vegfruits/index_to_name.json 

# start server
torchserve --start --model-store checkpoints/model_stores/vegfruits/ --ts-config checkpoints/model_stores/vegfruits/config.properties --enable-model-api --disable-token-auth

# inference
curl http://localhost:8080/ping
curl -X OPTIONS http://localhost:8080 -o src/backend/torchserve_app/vegfruits_swagger.json
curl http://localhost:8080/predictions/mvegfruits -F 'data=@data/processed/vegfruits/validation/lettuce/Image_8.jpg'

# management
curl http://localhost:8081/models
curl http://localhost:8081/models/mvegfruits
curl -v -X PUT "http://localhost:8081/models/mvegfruits?min_workers=1&batch_size=10"

# metrics
curl http://localhost:8082/metrics

# stop server
torchserve --stop
```

### Torchserve - 

torch-model-archiver   --model-name sports-classifier     --version 1.0      --export-path ./checkpoints/model_stores/sports/mar_file  --handler  ./src/backend/torchserve_app/sports_handler.py      --serialized-file ./checkpoints/pths/sports_cpu.pt     --extra-files checkpoints/model_stores/sports/index_to_name.json




torch-model-archiver --model-name sports-classifier --serialized-file checkpoints/onnxs/sports.onnx --handler src/backend/torchserve_app/sports_handler.py --export-path checkpoints/model_stores/sports/ -f --version 0.0.1 --extra-files checkpoints/model_stores/sports/index_to_name.json


torch-model-archiver --model-name vegfruits-classifier --serialized-file checkpoints/onnxs/vegfruits.onnx --handler src/backend/torchserve_app/vegfruits_handler.py --export-path checkpoints/model_stores/vegfruits/ -f --version 0.0.1 --extra-files checkpoints/model_stores/vegfruits/index_to_name.json

