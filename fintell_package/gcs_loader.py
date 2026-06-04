from pathlib import Path
from google.cloud import storage

def upload_folder_to_bucket(
    project_id: str,
    bucket_name: str,
    local_folder: str
) -> None:
    """
    Recursively uploads a folder and all its subfolders
    to a GCS bucket while preserving
    the directory structure.

    Args:
        project_id (str): GCP project ID.
        bucket_name (str): Name of the GCS bucket.
        local_folder (str): Local folder to upload.
    """

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    base_folder = Path(local_folder)

    for file in base_folder.rglob("*"):

        if file.is_file():

            cloud_path = str(
                file.relative_to(base_folder)
            ).replace("\\", "/")

            blob = bucket.blob(cloud_path)

            blob.upload_from_filename(str(file))

            print(f"✅ Upload : {cloud_path}")

    print("🚀 Upload terminé")


def upload_file_to_bucket(
    project_id: str,
    bucket_name: str,
    local_file: str,
    destination_path: str
) -> None:
    """
    Upload a file into a GCP bucket.

    Args:
        project_id (str): GCP project ID.
        bucket_name (str): Name of the GCS bucket.
        local_folder (str): Local folder to upload.
        destination_path : GCS folder destination.
    """

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    blob = bucket.blob(destination_path)

    blob.upload_from_filename(local_file)

    print(f"✅ Upload : {local_file}")
