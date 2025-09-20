from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .database import get_db
from .schemas import UserCreate, UserLogin, Token, UserResponse
from .utils import (
    create_user, authenticate_user, get_user_by_username, get_user_by_email,
    AuthUtils, get_current_active_user, blacklist_token, ACCESS_TOKEN_EXPIRE_MINUTES
)
from pathlib import Path

# Initialize router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Templates
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@router.get("/", response_class=HTMLResponse)
async def auth_page(request: Request):
    """Serve the login/register page"""
    return templates.TemplateResponse(request, "auth.html", {"request": request})

@router.post("/register", response_model=dict)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Check if user already exists
        if get_user_by_email(db, user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        if get_user_by_username(db, user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Create user
        user = create_user(
            db=db,
            email=user_data.email,
            username=user_data.username,
            password=user_data.password
        )
        
        return {
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login")
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user and return JWT in cookie"""
    user = authenticate_user(db, user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = AuthUtils.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    response = JSONResponse(content={"message": "Login successful"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,            # prevents JavaScript access
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",           # CSRF protection
        secure=False              # True if using HTTPS
    )
    return response


@router.post("/logout")
async def logout(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if token:
        blacklist_token(db, token)

    response = JSONResponse(content={"message": "Successfully logged out"})
    response.delete_cookie("access_token")
    return response


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_active_user)):
    """Get current user info"""
    return current_user

@router.get("/verify-token")
async def verify_token(current_user: dict = Depends(get_current_active_user)):
    """Verify if token is valid"""
    return {"valid": True, "user": current_user.username}