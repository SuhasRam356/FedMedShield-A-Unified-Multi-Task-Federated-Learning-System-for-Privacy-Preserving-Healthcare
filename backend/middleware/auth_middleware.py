"""
Authentication & Authorization Middleware
FedMedShield Framework - Role-Based Guardrails
"""

from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
from backend.utils.jwt_utils import decode_access_token

security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """FastAPI dependency to extract current user and verify bearer token."""
    if not credentials:
        # Default mock user for zero-friction demo if unauthenticated
        return {
            "username": "dr_smith",
            "role": "researcher",
            "institution": "General Hospital, NY"
        }

    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def require_roles(allowed_roles: List[str]):
    """Decorator / dependency factory for endpoint RBAC."""
    async def role_checker(current_user: dict = Depends(get_current_user)):
        user_role = current_user.get("role", "researcher")
        if user_role not in allowed_roles and "admin" not in user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. User role '{user_role}' requires one of {allowed_roles}"
            )
        return current_user
    return role_checker
