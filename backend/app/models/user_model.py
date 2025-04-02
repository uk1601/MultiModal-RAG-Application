from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field, EmailStr, validator, constr
import regex
from datetime import datetime

# SQLAlchemy Base model
Base = declarative_base()

# SQLAlchemy model for Users table in the database
class User(Base):
    __tablename__ = "Users"

    UserId = Column(Integer, primary_key=True, index=True)  # Automatically generated unique UserId
    Username = Column(String(255), nullable=False, unique=True)  # Username cannot be blank and must be unique
    Email = Column(String(255), nullable=False, unique=True)  # Email cannot be blank and must be unique
    PasswordHash = Column(String, nullable=False)  # Hashed password, cannot be blank
    CreatedAt = Column(TIMESTAMP, default=datetime.utcnow)  # Automatically set timestamp when a new record is created

# Pydantic model for user registration input validation
class UserRegisterModel(BaseModel):
    Username: constr(min_length=3, max_length=50) = Field(..., 
      description="Username must be alphanumeric, between 3 to 50 characters, and cannot be blank")

    # Custom validator for Username
    @validator("Username")
    def validate_username(cls, value):  # Changed self to cls
        if not regex.match(r'^\w+$', value):
            raise ValueError("Username must be alphanumeric")
        return value

    Email: EmailStr = Field(..., description="Email must be valid")
    Password: str = Field(..., min_length=8, description="Password cannot be blank and must have at least 8 characters")

    @validator("Password")
    def validate_password(cls, value):  # Changed self to cls
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(char.isdigit() for char in value):
            raise ValueError("Password must contain at least one number")
        if not any(char.isupper() for char in value):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in value):
            raise ValueError("Password must contain at least one lowercase letter")
        return value

# Pydantic model for user login input validation
class UserLoginModel(BaseModel):
    Email: EmailStr = Field(..., description="Email must be valid")  # Validates email format
    Password: str = Field(..., min_length=8, description="Password cannot be blank")  # Validates password with a minimum length of 8 characters

# Pydantic model for response when fetching user details
class UserListResponse(BaseModel):
    UserId: int  # User ID for identification
    Email: EmailStr  # Email of the user

    class Config:
        from_attributes = True  # Allows ORM models to be converted to Pydantic models
