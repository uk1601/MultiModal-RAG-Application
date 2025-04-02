from fastapi import APIRouter, Depends, HTTPException, Body, Query
from fastapi.security import OAuth2PasswordBearer
from typing import List, Optional

from pydantic import BaseModel

from app.models.publication import Publication
from app.services.PublicationService import PublicationService  # Assuming the `PublicationService` is in this module
from app.services.auth_service import verify_token  # Assuming token validation is handled in auth_service
import logging

# Initialize the router for publication routes
router = APIRouter()

# Define OAuth2 scheme for JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Configure loggingx
logging.basicConfig(level=logging.INFO)

class PaginatedResponse(BaseModel):
    total_count: int
    total_pages: int
    current_page: int
    per_page: int
    next_page: Optional[int]
    previous_page: Optional[int]
    publications: List[dict]
# Dependency to get the service
def get_publication_service():
    return PublicationService()


# Helper to verify the token and get the user email
def get_current_user(token: str = Depends(oauth2_scheme)):
    user_email = verify_token(token)
    if not user_email:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_email



# Route to retrieve all publications with pagination
@router.get("/publications", tags=["Publications"], response_model=PaginatedResponse)
async def get_publications(
        page: int = Query(1, description="Page number"),
        per_page: int = Query(10, description="Number of publications per page"),
        token: str = Depends(oauth2_scheme),
        publication_service: PublicationService = Depends(get_publication_service)
):
    """Retrieve all publications with pagination."""
    try:
        # Verify token
        user_email = get_current_user(token)

        # Fetch publications with pagination
        logging.info(f"Fetching publications for page {page} and per_page {per_page}")
        publications = publication_service.get_all_publications(page=page, per_page=per_page)
        return publications

    except Exception as e:
        logging.error(f"Error while fetching publications: {str(e)}")
        raise HTTPException(status_code=500, detail="Error while fetching publications")


# Route to retrieve a single publication by ID
@router.get("/publications/{publication_id}", tags=["Publications"], response_model=dict)
async def get_publication_by_id(
        publication_id: int,
        token: str = Depends(oauth2_scheme),
        publication_service: PublicationService = Depends(get_publication_service)
):
    """Retrieve a publication by ID."""
    try:
        # Verify token
        user_email = get_current_user(token)

        # Fetch publication by ID
        publication = publication_service.get_publication_by_id(publication_id)
        if not publication:
            raise HTTPException(status_code=404, detail="Publication not found")
        return publication

    except Exception as e:
        logging.error(f"Error while fetching publication: {str(e)}")
        raise HTTPException(status_code=500, detail="Error while fetching publication")


