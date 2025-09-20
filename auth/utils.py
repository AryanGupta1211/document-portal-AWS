import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from .database import get_db, User, TokenBlacklist
from dotenv import load_dotenv

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-this-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 15

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

class AuthUtils:
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str, credentials_exception: HTTPException) -> str:
        """Verify a JWT token and return the username"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            return username
        except JWTError:
            raise credentials_exception

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, email: str, username: str, password: str) -> User:
    """Create a new user"""
    hashed_password = AuthUtils.get_password_hash(password)
    db_user = User(
        email=email,
        username=username,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user"""
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not AuthUtils.verify_password(password, user.hashed_password):
        return None
    return user

def is_token_blacklisted(db: Session, token: str) -> bool:
    """Check if token is blacklisted"""
    return db.query(TokenBlacklist).filter(TokenBlacklist.token == token).first() is not None

def blacklist_token(db: Session, token: str):
    """Add token to blacklist"""
    blacklisted_token = TokenBlacklist(token=token)
    db.add(blacklisted_token)
    db.commit()

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
#     """Get current authenticated user"""
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
    
#     # Check if token is blacklisted
#     if is_token_blacklisted(db, token):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Token has been invalidated",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     username = AuthUtils.verify_token(token, credentials_exception)
#     user = get_user_by_username(db, username)
#     if user is None:
#         raise credentials_exception
#     return user

def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Get current authenticated user (from cookie)"""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    # Check if token is blacklisted
    if is_token_blacklisted(db, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been invalidated",
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    username = AuthUtils.verify_token(token, credentials_exception)
    user = get_user_by_username(db, username)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
#     """Get current active user"""
#     if not current_user.is_active:
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user