import os
import snowflake.connector
from snowflake.connector import ProgrammingError
from dotenv import load_dotenv
load_dotenv(override=True)
class SnowflakeConnection:
    _instance = None

    def __new__(cls):
        """Ensure only one instance of the connection (Singleton pattern)"""
        if cls._instance is None:
            cls._instance = super(SnowflakeConnection, cls).__new__(cls)
            cls._instance._conn = None
        return cls._instance

    def connect(self):
        """Create a connection to Snowflake if it doesn't already exist"""
        if self._conn is None:
            try:
                print(f"User:{os.getenv('SNOWFLAKE_USER')}")
                self._conn = snowflake.connector.connect(
                    user=os.getenv('SNOWFLAKE_USER'),
                    password=os.getenv('SNOWFLAKE_PASSWORD'),
                    account=os.getenv('SNOWFLAKE_ACCOUNT'),
                    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
                    database=os.getenv('SNOWFLAKE_DATABASE'),
                    schema=os.getenv('SNOWFLAKE_SCHEMA')
                )

                print("Connection successful")
            except ProgrammingError as e:
                print(f"Error connecting to Snowflake: {e}")
        return self._conn

    def close_connection(self):
        """Close the Snowflake connection if it exists"""
        if self._conn:
            try:
                self._conn.close()
                print("Connection closed")
                self._conn = None  # Set connection to None after closing
            except ProgrammingError as e:
                print(f"Error closing Snowflake connection: {e}")


# Helper function to execute any query
def execute_query(query):
    try:
        conn = SnowflakeConnection().connect()
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        print("Query executed successfully")
        return result
    except ProgrammingError as e:
        print(f"Error executing query: {e}")
        return None

# Helper function to get all publications
def get_all_publications():
    query = "SELECT * FROM publications order by id limit 20;"
    return execute_query(query)

# Main function
def main():
    # Get all publications
    publications = get_all_publications()
    if publications:
        print("Publications:", publications)

    # Close the connection at the end
    SnowflakeConnection().close_connection()

if __name__ == "__main__":
    main()
