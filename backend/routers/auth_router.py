"""
Authentication Router
FedMedShield Framework - Clinical User Registration & JWT Authentication
"""

import os
from fastapi import APIRouter, HTTPException, status, Depends
from backend.models.auth_models import UserRegister, UserLogin, Token, UserResponse, UserRole
from backend.utils.jwt_utils import verify_password, get_password_hash, create_access_token
from backend.middleware.auth_middleware import get_current_user
from datetime import timedelta

router = APIRouter(prefix="/auth", tags=["Authentication"])

ENABLE_DEMO_AUTH = os.getenv("ENABLE_DEMO_AUTH", "true").lower() in ("true", "1", "yes")

# In-memory users store for development / demo responsiveness
MOCK_USERS_DB = {}
if ENABLE_DEMO_AUTH:
    MOCK_USERS_DB = {
        "dr_smith": {
            "id": 1,
            "username": "dr_smith",
            "email": "dr.smith@fedmedshield.org",
            "hashed_password": get_password_hash("password123"),
            "full_name": "Dr. Sarah Smith, MD",
            "role": "researcher",
            "institution": "General Hospital - New York",
            "is_active": True
        },
        "admin": {
            "id": 2,
            "username": "admin",
            "email": "admin@fedmedshield.org",
            "hashed_password": get_password_hash("admin123"),
            "full_name": "System Administrator",
            "role": "admin",
            "institution": "FedMedShield Global Operations",
            "is_active": True
        }
    }


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister):
    # Security guardrail: disallow self-assigning admin role
    if user_data.role == UserRole.ADMIN or str(user_data.role).lower() == "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role cannot be self-assigned via public registration."
        )

    if user_data.username in MOCK_USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered."
        )

    new_user = {
        "id": len(MOCK_USERS_DB) + 1,
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": get_password_hash(user_data.password),
        "full_name": user_data.full_name or user_data.username,
        "role": user_data.role,
        "institution": user_data.institution or "General Hospital",
        "is_active": True
    }
    MOCK_USERS_DB[user_data.username] = new_user
    return new_user


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    user = MOCK_USERS_DB.get(credentials.username)
    if not user or not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"], "institution": user["institution"]},
        expires_delta=timedelta(hours=24)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user["role"],
        "username": user["username"]
    }


@router.get("/me", response_model=UserResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    username = current_user.get("sub") or current_user.get("username")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user token payload"
        )
    user = MOCK_USERS_DB.get(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found in active directory"
        )
    return user

