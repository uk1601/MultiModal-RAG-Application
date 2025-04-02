import requests
import boto3
import os
from dotenv import load_dotenv
from airflow.models import Variable

load_dotenv()

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=Variable.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=Variable.get("AWS_SECRET_ACCESS_KEY"),
    region_name=Variable.get("AWS_REGION")
)

def upload_to_s3(file_content, bucket, s3_path):
    try:
        if not isinstance(file_content, (bytes, str)):
            print(f"Unexpected file_content type: {type(file_content)}")
        if not isinstance(bucket, str) or not isinstance(s3_path, str):
            print(f"Unexpected bucket or s3_path type: bucket={type(bucket)}, s3_path={type(s3_path)}")
        s3_client.put_object(Bucket=bucket, Key=s3_path, Body=file_content)
        s3_url = f"https://{bucket}.s3.amazonaws.com/{s3_path}"
        print(f"Uploaded to S3: {s3_url}")
        return s3_url
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        return ""

def download_pdf(title, pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        if response.content:  # Ensure response has content
            s3_path = f"assignment3/pdfs/{sanitize_filename(title)}.pdf"
            return upload_to_s3(response.content, os.getenv("AWS_BUCKET_NAME"), s3_path)
        else:
            print(f"No content in PDF for {title}.")
            return ""
    except Exception as e:
        print(f"Failed to download PDF for {title}. Error: {e}")
        return ""

def save_image(title, image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        if response.content:
            s3_path = f"assignment3/images/{sanitize_filename(title)}.jpg"
            return upload_to_s3(response.content, os.getenv("AWS_BUCKET_NAME"), s3_path)
        else:
            print(f"No content in image for {title}.")
            return ""
    except Exception as e:
        print(f"Failed to save image for {title}. Error: {e}")
        return ""

def sanitize_filename(name):
    return "".join([c if c.isalnum() or c in " ._-()" else "_" for c in name])
