from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from scripts.scraper import scrape_publications, close_driver
from scripts.snowflake_utils import connect_to_snowflake, setup_snowflake_database, upload_dataframe_to_snowflake
import pandas as pd

# Define default arguments for the DAG
default_args = {
    "start_date": datetime(2024, 10, 29),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),  # Allow up to 15 minutes for the task
    "catchup": False,
}

# Define the DAG
with DAG("data_ingestion_combined", default_args=default_args, schedule_interval="@daily") as dag:

    # Combined task function
    def combined_task_callable():
        # Step 1: Scrape publications and store in a DataFrame
        #publications_df = pd.read_csv("/opt/airflow/dags/publications_data.csv")
        publications_df = scrape_publications()  # Uncomment if scrape_publications is needed

        # Step 2: Set up Snowflake database and table if not already set up
        conn = connect_to_snowflake()
        try:
            setup_snowflake_database(conn)

            # Step 3: Upload scraped DataFrame to Snowflake
            upload_dataframe_to_snowflake(publications_df, conn)
        finally:
            conn.close()

        # Step 4: Close the Selenium driver
        close_driver()

    # Create a single task to perform all steps
    combined_task = PythonOperator(
        task_id="combined_task",
        python_callable=combined_task_callable,
        execution_timeout=timedelta(minutes=30),
    )
