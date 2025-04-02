from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
from app.models.user_model import User, UserRegisterModel, UserLoginModel, UserListResponse
from app.services.auth_service import hash_password, verify_password, create_access_token, create_refresh_token
from app.services.database_service import get_db
from typing import List

# Custom error response function for detailed error messaging
def create_error_response(message: str, details: str = None):
    return {"error": message, "details": details}

# User registration logic with detailed error handling
def register_user(user: UserRegisterModel, db: Session = Depends(get_db)):
    # Check if a user with the given email already exists
    db_user = db.query(User).filter(User.Email == user.Email).first()
    if db_user:
        raise HTTPException(status_code=400, detail=create_error_response("Registration failed", "Email already registered"))
    
    # Check if a user with the given username already exists
    db_user_by_username = db.query(User).filter(User.Username == user.Username).first()
    if db_user_by_username:
        raise HTTPException(status_code=400, detail=create_error_response("Registration failed", "Username already taken"))

    # Hash the password using bcrypt
    hashed_password = hash_password(user.Password)
    
    # Create a new user record in the database
    new_user = User(
        Username=user.Username, 
        Email=user.Email, 
        PasswordHash=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User registered successfully"}

# User login logic with detailed error handling
def login_user(user: UserLoginModel, db: Session = Depends(get_db)):
    # Fetch the user from the database by email
    db_user = db.query(User).filter(User.Email == user.Email).first()
    if not db_user:
        raise HTTPException(status_code=404, detail=create_error_response("Login failed", "Email not found"))
    
    # Verify the provided password with bcrypt
    if not verify_password(user.Password, db_user.PasswordHash):
        raise HTTPException(status_code=401, detail=create_error_response("Login failed", "Incorrect password"))

    # Generate JWT tokens
    access_token = create_access_token(data={"sub": db_user.Email})
    refresh_token = create_refresh_token(data={"sub": db_user.Email})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

# Retrieve a list of all registered users (admin functionality or debugging purposes)
def get_all_users(db: Session = Depends(get_db)) -> List[UserListResponse]:
    users = db.query(User).all()
    return [UserListResponse(UserId=user.UserId, Email=user.Email) for user in users]