import os
import subprocess


print("Creating MAR file")
CONFIG_TEMPLATE = """
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8085
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
load_models={0}
max_response_size=655350000
model_store=/mnt/models/model-store
default_response_timeout=600
enable_metrics_api=true
metrics_format=prometheus
number_of_netty_threads=4
job_queue_size=10
model_snapshot={{"name":"startup.cfg","modelCount":1,"models":{{"{0}":{{"1.0":{{"defaultVersion":true,"marName":"{0}.mar","minWorkers":1,"maxWorkers":1,"batchSize":1,"maxBatchDelay":100,"responseTimeout":600}}}}}}}}
"""
MODELS_PATH = [
    "imagenet-m1",
    "imagenet-m2",
    "imagenet-m3",
    # "dog-classifier",
    # "food-classifier",
    # "imagenet-vit",
    # "indian-food-classifier",
]

def setup_directories(base_path, model_name):
    paths = {
        "root": os.path.join(base_path, model_name),
        "config": os.path.join(base_path, model_name, "config"),
        "store": os.path.join(base_path, model_name, "model-store"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    print(f"Created directory structure for: {model_name}")
    return paths

def generate_config(model_path, model_name):
    config_path = os.path.join(model_path, "config", "config.properties")
    with open(config_path, "w") as cfg:
        cfg.write(CONFIG_TEMPLATE.format(model_name))
    print(f"Generated configuration for: {model_name}")

def package_model(model_name, storage_path, model_path):
    command = [
        "torch-model-archiver",
        "--model-name", model_name,
        "--handler", "classifier_handler.py",
        "--extra-files", f"models/{model_path}/",
        "--version", "1.0",
        "--export-path", storage_path,
    ]
    try:
        print(f"Packaging {model_name}.mar at {storage_path} from {model_path}")
        subprocess.run(command, check=True)
        print(f"Successfully packaged: {model_name}.mar")
    except subprocess.CalledProcessError as err:
        log.error(f"Error packaging {model_name}: {err}", exc_info=True)

if __name__ == "__main__":
    base_directory = "./packaged-models"
    for model_path in MODELS_PATH:
        paths = setup_directories(base_directory, model_path)
        model_name = "imagenet-classifier"
        generate_config(paths["root"], model_name)
        package_model(model_name, paths["store"], model_path)
        print(f"### Completed processing for {model_path} ###")
