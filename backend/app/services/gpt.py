import base64
import json
import os
import tempfile

import requests
import logging

from google.oauth2 import service_account
from openai import OpenAI

from app.config.settings import settings
from app.services.tools import tools
from fastapi import HTTPException

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EvaluationModel:
    def __init__(self, objective: str, file_attachments: list, model: str,
                 query: str = None, additional_context: str = None):
        self.objective = objective
        self.file_attachments = file_attachments
        self.model = model
        self.response = None
        self.query = query
        self.additional_context = additional_context


from google.cloud import storage


def download_file_from_gcs(gcs_url: str) -> str:
    """
    Downloads a file from Google Cloud Storage given its URL.

    Args:
        gcs_url (str): The GCS file link in the format gs://bucket_name/object_name.

    Returns:
        str: Path to the downloaded file.
    """
    try:
        logger.info(f"Attempting to download file from GCS: {gcs_url}")

        # Extract the bucket name and object name from the GCS URL
        if not gcs_url.startswith("gs://"):
            raise HTTPException(status_code=400, detail="Invalid GCS URL format")

        gcs_url_parts = gcs_url[5:].split("/", 1)
        if len(gcs_url_parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid GCS URL format")

        bucket_name = gcs_url_parts[0]
        object_name = gcs_url_parts[1]

        # Initialize a client and download the file
        credentials = service_account.Credentials.from_service_account_file(settings.GCP_JSON)
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        # Create a temporary directory to store the file
        temp_dir = tempfile.mkdtemp()

        # Define the local file path
        local_file_path = os.path.join(temp_dir, object_name.split("/")[-1])

        # Download the file to the temporary location
        blob.download_to_filename(local_file_path)

        logger.info(f"File downloaded successfully: {local_file_path}")
        return local_file_path

    except Exception as e:
        logger.exception("Error occurred while downloading file from GCS.")
        raise HTTPException(status_code=500, detail=f"Error downloading file from GCS: {str(e)}")

def get_image_data_url(image_file: str, image_format: str) -> str:
    """
    Helper function to convert an image file to a data URL string.

    Args:
        image_file (str): The path to the image file.
        image_format (str): The format of the image file.

    Returns:
        str: The data URL of the image.
    """
    try:
        logger.info(f"Converting image to data URL: {image_file}")
        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{image_format};base64,{image_data}"
    except FileNotFoundError:
        logger.error(f"File not found: {image_file}")
        raise HTTPException(status_code=404, detail=f"Could not read '{image_file}'.")

def handle_file_reading(file_path: str):
    logger.info(f"Handling file reading for: {file_path}")
    if file_path and os.path.exists(file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        tool = None
        logger.info(f"File extension identified: {ext}")

        # Identify the appropriate tool based on file extension
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            tool = "ReadImage"
        elif ext in ['.xlsx', '.xls']:
            tool = "ReadExcel"
        elif ext == '.csv':
            tool = "ReadCSV"
        elif ext == '.zip':
            tool = "ReadZIP"
        elif ext == '.pdf':
            tool = "ReadPDF"
        elif ext == '.json':
            tool = "ReadJSON"
        elif ext == '.py':
            tool = "ReadPython"
        elif ext == '.docx':
            tool = "ReadDOCX"
        elif ext == '.pptx':
            tool = "ReadPPTX"
        elif ext in ['.mp3', '.wav']:
            tool = "ReadAudio"
        elif ext == '.pdb':
            tool = "ReadPDB"
        elif ext == '.jsonld':
            tool = "ReadJSONLD"
        elif ext == '.txt':
            tool = "ReadTXT"

        if tool and tool != "ReadImage" and tool in tools:
            logger.info(f"Using tool: {tool} for file: {file_path}")
            context = tools[tool](file_path)
            if tool == "ReadAudio":
                try:
                    audio_data = json.loads(context)
                    transcription = audio_data.get('transcription', '')
                    context += f"\nTranscription: {transcription}"
                except json.JSONDecodeError:
                    logger.error("Failed to decode audio metadata.")
                    raise HTTPException(status_code=500, detail="Failed to decode audio metadata.")
            return context, tool
        elif tool == "ReadImage":
            context = get_image_data_url(file_path, ext)
            return context, tool
        else:
            logger.info(f"No specific tool found for file: {file_path}, treating as plain text")
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), tool
    else:
        logger.error(f"File could not be found: {file_path}")
        raise HTTPException(status_code=404, detail=f"File '{file_path}' could not be found.")


def generate_summarization_prompt():
    prompt = """
    ```markdown
    Summarize the following information concisely. 
    - **Output Format**: Provide the response in well-structured markdown, using clear and concise language.
    - **Instructions**: Explain what the data is about and infer key insights based on it.
    - **Important**: If the data provided does not contain enough information to explain the topic fully, clearly state 
    that the information is insufficient and provide no additional speculation.
    ```
    """
    return prompt

def generate_query_prompt(evaluation_query):
    prompt = f"""
    ```markdown
    Answer the following query based on the given context. 
    - **Query**: {evaluation_query}
    - **Output Format**: Structure the answer in markdown format, ensuring it is clear and well-organized.
    - **Important**: If the provided context does not contain the required information to fully answer the query, 
    clearly state that the data is insufficient and avoid making assumptions or speculating.
    ```
    """
    return prompt
def evaluate(evaluation: EvaluationModel):
    logger.info(f"Starting evaluation with objective: {evaluation.objective}")
    context_list = []
    tools_used = []
    for file_url in evaluation.file_attachments:
        logger.info(f"Processing file attachment: {file_url}")
        file_path = download_file_from_gcs(file_url)
        context, tool = handle_file_reading(file_path)
        context_list.append(context)
        tools_used.append(tool)

    if evaluation.objective == "Summarize":
        objective_prompt = generate_summarization_prompt()
    elif evaluation.objective == "Query":
        objective_prompt = generate_query_prompt(evaluation.query)
    else:
        logger.error("Invalid objective provided.")
        raise HTTPException(status_code=400, detail="Invalid objective. Must be either 'Summarize' or 'Query'.")

    full_context = "\n".join(context_list)
    prompt = f"""
    {objective_prompt}
    Context: {full_context}
    Additional Context (This is the full text that is extracted from the source: {evaluation.additional_context}
    """

    message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
    logger.info(f"Generated prompt message for OpenAI: {message}")

    try:
        response = client.chat.completions.create(
            model=evaluation.model,
            messages=[message],
            temperature=0
        )

        if response.choices:
            evaluation.response = response.choices[0].message.content
            logger.info("Received response from OpenAI model.")
        else:
            logger.error("No response received from the model.")
            raise HTTPException(status_code=500, detail="No response received from the model.")

    except Exception as e:
        logger.exception("Error occurred while interacting with OpenAI API.")
        raise HTTPException(status_code=500, detail=f"Error from OpenAI API: {str(e)}")

    return evaluation

# Example usage of the evaluate function
if __name__ == "__main__":
    evaluation = EvaluationModel(
        objective="Summarize",
        file_attachments=["gs://assignment2-damg7245-t1/extracted_by_opensource_pdf/b3654e47-4307-442c-a09c-945b33b913c6/tables/table_page_1_1.csv"],
        model="gpt-4o-mini"
    )
    updated_evaluation = evaluate(evaluation)
    logger.info(f"Final response: {updated_evaluation.response}")
    print(updated_evaluation.response)