import boto3
from pathlib import Path
import os

def download_folder_from_s3(bucket_name, s3_prefix, local_folder):
    """
    Downloads a folder from S3, maintaining the folder structure.

    Parameters:
    - bucket_name: S3 bucket name
    - s3_prefix: S3 prefix (acts like a folder), e.g., "models/vegfruits/"
    - local_folder: Path to local destination folder
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    for page in pages:
        for obj in page.get('Contents', []):
            s3_key = obj['Key']

            # Skip directory markers
            if s3_key.endswith('/'):
                continue

            # Maintain relative path
            relative_path = os.path.relpath(s3_key, s3_prefix)
            local_path = os.path.join(local_folder, relative_path)

            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            print(f"Downloading s3://{bucket_name}/{s3_key} to {local_path}")
            s3.download_file(bucket_name, s3_key, local_path)


    print("Download complete.")



if __name__ == "__main__":

    all_projects = ["vegfruits", "sports"]

    for project in all_projects:

        model_store_dir = Path(os.path.join(os.getcwd() , "s3_files" , project))
        model_store_dir.mkdir(parents=True, exist_ok=True)
        file_count = sum(
            1 for entry in os.listdir(model_store_dir)
            if os.path.isfile(os.path.join(model_store_dir, entry))
        )
        if file_count < 1:
            s3_bucket = "mybucket-emlo-mumbai"
            s3_key_prefix = f"kserve-ig/{project}-classifier-prod" 
            download_folder_from_s3(s3_bucket, s3_key_prefix, model_store_dir)
        else:
            print("Files are present continue")