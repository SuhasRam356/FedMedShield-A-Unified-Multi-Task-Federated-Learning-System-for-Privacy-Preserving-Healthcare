"""
Pydantic Schemas for Authentication and Role-Based Access Control
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    HOSPITAL_LEAD = "hospital_lead"
    AUDITOR = "auditor"


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = "Clinical Investigator"
    role: UserRole = UserRole.RESEARCHER
    institution: Optional[str] = "General Hospital"


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    institution: str
    is_active: bool

    class Config:
        from_attributes = True
