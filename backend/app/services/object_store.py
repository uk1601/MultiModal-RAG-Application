from google.cloud import storage
from google.oauth2 import service_account
import tempfile
from google.api_core.exceptions import Forbidden
import os
from app.config.settings import settings

def download_file_from_gcs(file_name):
    try:
        credentials = service_account.Credentials.from_service_account_file(settings.GCP_JSON)
        client = storage.Client(credentials=credentials)

        # Define the bucket name and full object path
        bucket_name = "assignment2-damg7245-t1"
        object_name = f"gaia_extracted_pdfs/{file_name}"

        # Get the bucket and blob
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Define the local file path
        local_file_path = os.path.join(temp_dir, file_name)

        # Download the file
        blob.download_to_filename(local_file_path)
        print(f"🗳️🗳️🗳️🗳️Successful downloading the required file:{local_file_path} 🗳️🗳️🗳️🗳️")
        return local_file_path

    except Forbidden as e:
        print(f"Permission denied: {e}")
        print("Please check the service account permissions and ensure it has access to the bucket and object.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
if __name__ == "__main__":
    print(download_file_from_gcs("021a5339-744f-42b7-bd9b-9368b3efda7a.pdf"))