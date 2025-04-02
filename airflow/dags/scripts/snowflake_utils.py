import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector.errors import ProgrammingError

load_dotenv(override=True)

# Snowflake configuration from environment variables
DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
TABLE_NAME = os.getenv("SNOWFLAKE_TABLE", "PUBLICATIONS")


def connect_to_snowflake():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE")
    )

def setup_snowflake_database(conn):
    if not DATABASE or not SCHEMA:
        raise ValueError("Database or schema not specified in the environment variables.")

    try:
        cursor = conn.cursor()

        # Check if the database exists
        cursor.execute(f"SHOW DATABASES LIKE '{DATABASE}'")
        database_exists = cursor.fetchone() is not None

        if not database_exists:
            cursor.execute(f"CREATE DATABASE {DATABASE}")
            print(f"New database '{DATABASE}' created successfully.")
        else:
            print(f"Database '{DATABASE}' already exists.")
        cursor.execute(f"USE DATABASE {DATABASE};")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};")
        cursor.execute(f"USE SCHEMA {SCHEMA};")
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            ID INT PRIMARY KEY,
            Title STRING,
            Summary STRING,
            Image_URL STRING,
            PDF_URL STRING
        );
        """)
        print(f"Database '{DATABASE}', schema'{SCHEMA}', and table '{TABLE_NAME}' setup complete.")
    except ProgrammingError as e:
        print(f"Error during setup: {e}")
    finally:
        cursor.close()

def validate_dataframe(df):
    required_columns = {"ID","Title", "Summary", "Image Path", "PDF Path"}
    if not required_columns.issubset(df.columns):
        raise ValueError("DataFrame columns do not match the required table structure.")

def upload_dataframe_to_snowflake(df, conn):
    if df.empty:
        print("DataFrame is empty. Skipping upload.")
        return

    validate_dataframe(df)
    
    cursor = conn.cursor()
    try:
        rows_inserted=0
        for _, row in df.iterrows():
            check_sql = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE ID = %s"
            cursor.execute(check_sql, (row["ID"],))
            count = cursor.fetchone()[0]
            
            if count == 0:
                sql = f"""
                INSERT INTO {TABLE_NAME} (ID,Title, Summary, Image_URL, PDF_URL)
                VALUES (%s, %s, %s, %s, %s);
                """
                cursor.execute(sql, (row["ID"], row["Title"], row["Summary"], row["Image Path"], row["PDF Path"]))
                rows_inserted +=1
        
        conn.commit()
        print(f"Data upload successful. {rows_inserted} new rows inserted into Snowflake.")
        print(f"Snowflake Database Details:")
        print(f"  - Database: {DATABASE}")
        print(f"  - Schema: {SCHEMA}")
        print(f"  - Table: {TABLE_NAME}")
    except ProgrammingError as e:
        print(f"Error uploading data: {e}")
    finally:
        cursor.close()
