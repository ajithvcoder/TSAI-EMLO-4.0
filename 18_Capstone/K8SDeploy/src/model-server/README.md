Use below command for preparing mar file 

### Torchserve - Preparation file

torch-model-archiver   --model-name sports-classifier     --version 1.0      --export-path ./checkpoints/model_stores/sports/mar_file  --handler  ./src/backend/torchserve_app/sports_handler.py      --serialized-file ./checkpoints/pths/sports_cpu.pt     --extra-files checkpoints/model_stores/sports/index_to_name.json

And then upload to s3 location

