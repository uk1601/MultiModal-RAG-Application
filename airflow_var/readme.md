# Part : Data Ingestion and Database Population with Airflow

## Project Overview

This project automates data ingestion and populates a Snowflake database by:
1. Scraping publication data from a specified website.
2. Storing media assets (PDFs and images) in an AWS S3 bucket.
3. Ingesting metadata into a Snowflake database.

### Workflow Summary

The data ingestion process is implemented using an Airflow DAG, `data_ingestion_combined`, which performs the following steps:
1. **Data Scraping**: Collects publication details, including title, summary, images, and PDFs.
2. **S3 Upload**: Uploads scraped PDFs and images to AWS S3.
3. **Snowflake Database Population**: Inserts metadata into a Snowflake table.

## Prerequisites

### 1. Environment and Tools

- **GCP VM Instance** with Docker and Docker Compose installed.
- **Airflow with Celery Executor** (configured in `docker-compose.yaml`).
- **AWS S3 Bucket** for storing images and PDFs.
- **Snowflake Database** with a table to store publication metadata.

### 2. Required Variables and Secrets

The setup uses environment variables and Airflow variables to manage configurations for Snowflake, AWS S3, and Airflow, keeping sensitive data secure.

#### Airflow Variables (configured in `variables.json`)
- **AWS_ACCESS_KEY_ID**
- **AWS_SECRET_ACCESS_KEY**
- **AWS_BUCKET_NAME**: Name of the AWS S3 bucket.
- **AWS_REGION**: AWS region of the S3 bucket.
- **SNOWFLAKE_ACCOUNT**
- **SNOWFLAKE_USER**
- **SNOWFLAKE_PASSWORD**
- **SNOWFLAKE_WAREHOUSE**
- **SNOWFLAKE_DATABASE**
- **SNOWFLAKE_SCHEMA**
- **SNOWFLAKE_TABLE**
- **AIRFLOW_UID**: UID for Airflow, set to `50000` in this setup.



### Directory Structure

```
.
├── dags/
│   ├── scraper_dag.py
│   ├── publications_data.csv  # Sample data file (optional)
│   └── scripts/
│       ├── aws_s3.py
│       ├── scraper.py
│       └── snowflake_utils.py
├── docker-compose.yaml
├── requirements.txt
└── variables.json
```


## Setup and Execution

### Step 1: Clone the Repository and Upload Project Files
1. SSH into your GCP VM instance.
2. Clone or transfer this project’s files into your desired directory on the VM.

    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

### Step 2: Configure Environment Variables
- **Airflow Variables**: Upload `variables.json` to Airflow to ensure it has access to necessary credentials and configurations.

    ```bash
    airflow variables import variables.json
    ```

### Step 3: Build and Deploy Docker Containers with Docker Compose

- **Build the docker image using dockerfile**: This will build the docker image with apache airflow image and with the all the necessary pakcages installed.

    ```bash
    docker-compose build --no-cache
    ```

- **Run Airflow Init container**: This will set up the airflow init component in the docker-compose, it'll initialise the DB setup and create necessary users for airflow.

    ```bash
    docker-compose run airflow-init
    ```

- **Build and Run Docker Compose**: This will set up all Airflow components, PostgreSQL, Redis, and Selenium for headless scraping.

    ```bash
    docker-compose up --build -d
    ```

- **Verify Deployment**: Confirm all services are running.

    ```bash
    docker-compose ps
    ```
    
- **Access Airflow Web UI**: Open `<your-gcp-vm-ip>:8080` in your browser to access the Airflow UI.

## Step 4: Configure and Trigger the Airflow DAG
1. **Enable the DAG**:
    - In the Airflow Web UI, locate `data_ingestion_combined` in the list of DAGs.
    - Switch the DAG to **ON** to activate it.

2. **Run the DAG**:
    - Trigger the DAG manually from the UI to verify it runs successfully.



## DAG Structure and Execution

The DAG `data_ingestion_combined` is scheduled to run daily and includes a single combined task that executes the following functions sequentially:

### Task Breakdown

1. **Scrape Publications**:
    - Executes the `scrape_publications()` function in `scraper.py`, using Selenium to gather publication data, including title, summary, images, and PDFs.
    - Data is stored temporarily in a DataFrame.

2. **Upload Files to S3**:
    - Calls `aws_s3.py` functions `save_image()` and `download_pdf()` to upload images and PDFs to S3.
    - Stores the S3 URLs in the DataFrame for database insertion.

3. **Ingest Data into Snowflake**:
    - Uses `setup_snowflake_database()` and `upload_dataframe_to_snowflake()` in `snowflake_utils.py` to ensure the Snowflake table is set up and to upload publication metadata.

## Important Notes

- **Error Handling and Retries**:
    - The DAG is configured to retry on failure with a `retry_delay` of 5 minutes.
    - `combined_task_callable` includes a `try-finally` block to ensure the Selenium driver is closed even if an error occurs.

- **Execution Timeout**:
    - The `execution_timeout` is set to 30 minutes, but this can be adjusted based on observed execution times in production.

- **Logging and Monitoring**:
    - Logs are accessible in the `logs/` directory, allowing you to track task success or troubleshoot any issues.
