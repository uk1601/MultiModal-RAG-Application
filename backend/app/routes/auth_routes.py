from fastapi import APIRouter, Depends, Form, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List
import re  # For email validation
from app.controllers.auth_controller import register_user, login_user, get_all_users
from app.models.user_model import UserRegisterModel, UserLoginModel, UserListResponse
from app.services.database_service import get_db
from app.services.auth_service import get_current_user  # Assuming token validation is handled in auth_service
from app.models.user_model import User  # Assuming you have a SQLAlchemy User model

# Initialize the router for authentication routes
router = APIRouter()

# Define OAuth2 scheme for JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Helper function for validating email format
def validate_email(email: str):
    email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    if not re.match(email_regex, email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format."
        )

# Helper function to check username
def validate_username(username: str):
    if len(username) < 3 or len(username) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be between 3 and 20 characters long."
        )
    if not re.match("^[a-zA-Z0-9_]+$", username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username can only contain alphanumeric characters and underscores."
        )

# Check uniqueness of email or username in the database
def check_unique_username_email(username: str, email: str, db: Session):
    print("Checking for unique username and email")
    print(f"Username: {username}, Email: {email}")
    user_with_username = db.query(User).filter(User.Username == username).first()
    user_with_email = db.query(User).filter(User.Email == email).first()
    if user_with_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists."
        )
    if user_with_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists."
        )

# Route for user registration (not protected)
@router.post("/register", tags=["Authentication"])
def register(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Validate username, email, and password
        validate_username(username)
        validate_email(email)
        if len(password) < 8 or not any(c.isdigit() for c in password) or not any(c.isalpha() for c in password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long and contain both letters and numbers."
            )

        # Check for unique username and email in the database
        check_unique_username_email(username, email, db)

        # Proceed with user registration
        register_data = UserRegisterModel(Username=username, Email=email, Password=password)
        return register_user(register_data, db)
    except HTTPException as e:
        raise e
    except IntegrityError:
        # If the integrity of the database is violated (e.g., unique constraint failure)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or Email already in use."
        )
    except Exception as e:
        # Catching all other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Route for user login
@router.post("/login", tags=["Authentication"])
def login(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Validate email format
        validate_email(email)

        login_data = UserLoginModel(Email=email, Password=password)
        return login_user(login_data, db)
    except HTTPException as e:
        raise e
    except Exception as e:
        # Handling any unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Logout endpoint
@router.post("/logout", tags=["Authentication"])
def logout(current_user: str = Depends(get_current_user)):
    try:
        # If you have token invalidation logic, implement it here.
        return {"message": "Successfully logged out."}
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
