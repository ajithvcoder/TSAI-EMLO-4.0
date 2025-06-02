# ensure both vegfruits.onnx and sports.onnx is written to s3
# Run from root folder

import os
import shutil
import boto3
from pathlib import Path
import subprocess
from dotenv import load_dotenv
import argparse

# Set environment variables for AWS
load_dotenv()

def upload_folder_to_s3(local_folder, bucket_name, s3_prefix):
    """
    Uploads a folder to S3, maintaining the folder structure.
    
    Parameters:
    - local_folder: Path to the local folder
    - bucket_name: Target S3 bucket name
    - s3_prefix: Path in the bucket where files will be uploaded (e.g., "models/sports")
    """
    s3 = boto3.client('s3')

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # S3 expects forward slashes

            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(local_path, bucket_name, s3_key)

    print("Upload complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Process some integers.')
    parser.add_argument('-p', '--path', default="dev") 

    args = parser.parse_args()
    print(args.path)
    all_projects = ["vegfruits", "sports"]

    for project in all_projects:

        # Input values (update as needed)
        onnx_path = Path("checkpoints/onnxs") / f"{project}.onnx"
        config_path =  Path("checkpoints/model_stores") / project / "config.properties"
        index_to_name_path = Path("checkpoints/model_stores") / project / "index_to_name.json"
        pth_path = Path("checkpoints/pths") / f"{project}_cpu.pt"
        accuracy_path = f"output_{project}.txt"
        # You need to define a model handler or use a default one if appropriate
        # Example uses a custom handler named 'custom_handler.py' in the current dir
        handler_path = f"src/backend/torchserve_app/{project}_handler.py"   # Update if needed

        s3_bucket = "mybucket-emlo-mumbai"
        s3_key_prefix = f"kserve-ig/{project}-classifier-{args.path}"  # e.g., "models/sports"
        # s3://mybucket-emlo-mumbai/kserve-ig/sports-classifier-dev/

        # Step 1: Create destination directory
        deploy_dir = Path("checkpoints/deployment") / project
        deploy_dir.mkdir(parents=True, exist_ok=True)

        config_dir  = Path("checkpoints/deployment") / project / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        model_store_dir =  Path("checkpoints/deployment") / project / "model-store"
        model_store_dir.mkdir(parents=True, exist_ok=True)

        model_store_dir =  Path("checkpoints/deployment") / project / "pths"
        model_store_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: Copy files to deployment folder
        shutil.copy(onnx_path, deploy_dir / f"{project}.onnx")
        shutil.copy(config_path, deploy_dir / "config" / "config.properties")
        shutil.copy(index_to_name_path, deploy_dir / "index_to_name.json")
        shutil.copy(pth_path, deploy_dir / "pths" / f"{project}_cpu.pt")
        shutil.copy(accuracy_path, deploy_dir / f"output_{project}.txt")

        # Step 3: Generate .mar file
        mar_output = deploy_dir / f"{project}.mar"

        """
        torch-model-archiver   --model-name sports-classifier     --version 1.0     
        --export-path ./checkpoints/model_stores/sports/mar_file 
        --handler  ./src/backend/torchserve_app/sports_handler.py     
        --serialized-file ./checkpoints/pths/sports_cpu.pt   
        --extra-files {deploy_dir}/index_to_name.json

        torch-model-archiver --model-name {project}-classifier 
        --serialized-file {deploy_dir}/{project}.onnx 
        --handler src/backend/torchserve_app/{project}_handler.py 
        --export-path {deploy_dir}/model-store/ -f --version 0.0.1 
        --extra-files {deploy_dir}/index_to_name.json

        """

        subprocess.run([
            "torch-model-archiver",
            "--model-name", f"{project}-classifier",
            "--version", "1.0",
            "--serialized-file", str(deploy_dir / f"{project}.onnx"),
            "--handler", handler_path,
            "--extra-files", str(deploy_dir / "index_to_name.json"),
            "--export-path", str(deploy_dir /"model-store"),
            "--force"
        ], check=True)

        upload_folder_to_s3(deploy_dir, s3_bucket, s3_key_prefix)

        # s3.upload_file(mar_file_path, s3_bucket, s3_key)
        print(f"Uploaded {deploy_dir} to s3://{s3_bucket}/{s3_key_prefix}")
