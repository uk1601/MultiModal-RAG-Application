import logging

from fastapi.logger import logger
from snowflake.connector import ProgrammingError, SnowflakeConnection


from app.services.snowflake import SnowflakeConnection


class PublicationService:
    def __init__(self):
        self.conn = SnowflakeConnection().connect()

    def create_publication(self, title, summary, image_url, pdf_url):
        """Insert a new publication into the database"""
        query = """
        INSERT INTO CFAPUBLICATIONS1.CFAPUBLICATIONS1.PUBLICATIONS (TITLE, SUMMARY, IMAGE_URL, PDF_URL)
        VALUES (%s, %s, %s, %s);
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (title, summary, image_url, pdf_url))
            self.conn.commit()
            print("Publication created successfully")
        except ProgrammingError as e:
            print(f"Error creating publication: {e}")
        finally:
            cursor.close()

    def get_publication_by_id(self, publication_id):
        """Retrieve a single publication by its ID"""
        query = f"SELECT * FROM CFAPUBLICATIONS.CFAPUBLICATIONS.PUBLICATIONS WHERE ID = {publication_id};"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            publication = cursor.fetchone()
            if publication:
                publication = dict(zip([column[0] for column in cursor.description], publication))
                logging.info(f"Fetched publication: {publication}")
            return publication
        except ProgrammingError as e:
            print(f"Error fetching publication: {e}")
        finally:
            cursor.close()

    def get_all_publications(self, page=1, per_page=10):
        """Retrieve all publications with pagination."""
        offset = (page - 1) * per_page
        count_query = "SELECT COUNT(*) FROM CFAPUBLICATIONS.CFAPUBLICATIONS.PUBLICATIONS;"
        data_query = f"SELECT * FROM CFAPUBLICATIONS.CFAPUBLICATIONS.PUBLICATIONS ORDER BY ID LIMIT {per_page} OFFSET {offset};"

        try:
            cursor = self.conn.cursor()

            # Get total count of publications
            cursor.execute(count_query)
            total_count = cursor.fetchone()[0]

            # Calculate total pages
            total_pages = (total_count + per_page - 1) // per_page

            # Fetch the publications for the current page
            cursor.execute(data_query)
            publications = cursor.fetchall()
            logging.info(f"Fetched {len(publications)} publications on page {page}")

            # Convert the list of tuples to a list of dictionaries
            publications = [dict(zip([column[0] for column in cursor.description], row)) for row in publications]
            logging.info(f"Publications: {publications}")
            # Prepare paginated response
            paginated_response = {
                "total_count": total_count,
                "total_pages": total_pages,
                "current_page": page,
                "per_page": per_page,
                "next_page": page + 1 if page < total_pages else None,
                "previous_page": page - 1 if page > 1 else None,
                "publications": publications,
            }
            return paginated_response

        except ProgrammingError as e:
            print(f"Error fetching publications: {e}")
            return {
                "total_count": 0,
                "total_pages": 0,
                "current_page": page,
                "per_page": per_page,
                "publications": [],
                "error": str(e)
            }
        finally:
            cursor.close()

    def update_publication(self, publication_id, title, summary, image_url, pdf_url):
        """Update an existing publication by its ID"""
        query = """
        UPDATE CFAPUBLICATIONS.CFAPUBLICATIONS.PUBLICATIONS 
        SET TITLE = %s, SUMMARY = %s, IMAGE_URL = %s, PDF_URL = %s 
        WHERE ID = %s;
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (title, summary, image_url, pdf_url, publication_id))
            self.conn.commit()
            print("Publication updated successfully")
        except ProgrammingError as e:
            print(f"Error updating publication: {e}")
        finally:
            cursor.close()

    def delete_publication(self, publication_id):
        """Delete a publication by its ID"""
        query = f"DELETE FROM CFAPUBLICATIONS.CFAPUBLICATIONS.PUBLICATIONS WHERE ID = {publication_id};"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
            print("Publication deleted successfully")
        except ProgrammingError as e:
            print(f"Error deleting publication: {e}")
        finally:
            cursor.close()

    def close_connection(self):
        """Close the Snowflake connection"""
        SnowflakeConnection().close_connection()


def test():
    # Initialize the service
    publication_service = PublicationService()

    try:
        # 1. Create a new publication
        print("Creating a new publication...")
        publication_service.create_publication(
            title="Test Title",
            summary="This is a test summary",
            image_url="https://example.com/image.png",
            pdf_url="https://example.com/document.pdf"
        )

        # 2. Get all publications (with pagination, page 1, 10 per page)
        print("\nFetching all publications...")
        publications = publication_service.get_all_publications(page=1, per_page=10)
        print(f"Publications on page 1: {publications}")

        # 3. Fetch a single publication by ID
        if publications:
            first_publication_id = publications[0][0]  # Assuming the first column is the ID
            print(f"\nFetching publication with ID {first_publication_id}...")
            publication = publication_service.get_publication_by_id(first_publication_id)
            print(f"Publication: {publication}")

        # 4. Update the first publication
        if publications:
            print(f"\nUpdating publication with ID {first_publication_id}...")
            publication_service.update_publication(
                publication_id=first_publication_id,
                title="Updated Title",
                summary="This is an updated summary",
                image_url="https://example.com/updated_image.png",
                pdf_url="https://example.com/updated_document.pdf"
            )
            print(f"Publication {first_publication_id} updated successfully")

            # Fetch the updated publication
            updated_publication = publication_service.get_publication_by_id(first_publication_id)
            print(f"Updated Publication: {updated_publication}")

        # 5. Delete the first publication
        if publications:
            print(f"\nDeleting publication with ID {first_publication_id}...")
            publication_service.delete_publication(first_publication_id)
            print(f"Publication {first_publication_id} deleted successfully")

            # Try to fetch the deleted publication (it should not exist)
            deleted_publication = publication_service.get_publication_by_id(first_publication_id)
            print(f"Deleted Publication (should be None): {deleted_publication}")

    finally:
        # Close the Snowflake connection
        publication_service.close_connection()


if __name__ == "__main__":
    test()
