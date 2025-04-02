from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.config.settings import settings
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.database_service import get_db

# OAuth2PasswordBearer dependency
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Initialize password context (bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create access token with expiry using RS256
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "token_type": "access", "sub": str(data["sub"])})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_PRIVATE_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

# Create refresh token with longer expiry (e.g., 7 days)
def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "token_type": "refresh", "sub": str(data["sub"])})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_PRIVATE_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

# Verify JWT token using RS256 algorithm
def verify_token(token: str):
    try:
        payload = jwt.decode(token, settings.JWT_PUBLIC_KEY, algorithms=[settings.JWT_ALGORITHM])
        token_type = payload.get("token_type")
        if token_type not in ["access", "refresh"]:
            raise JWTError("Invalid token type")
        email: str = payload.get("sub")
        if email is None:
            raise JWTError("Missing user information")
        return email
    except JWTError as e:
        print(f"Token verification error: {str(e)}")  # Optional: Add logging or detailed error handling here
        return None

# Hash password using bcrypt
def hash_password(password: str):
    return pwd_context.hash(password)

# Verify hashed password using bcrypt
def verify_password(password: str, hashed_password: str):
    return pwd_context.verify(password, hashed_password)

# Helper function to extract and validate the user from the token
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, settings.JWT_PUBLIC_KEY, algorithms=[settings.JWT_ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise JWTError("Missing user information")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")