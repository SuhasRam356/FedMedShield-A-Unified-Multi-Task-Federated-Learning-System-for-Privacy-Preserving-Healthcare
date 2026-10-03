"""
JWT and Cryptographic Password Utilities (Standard Library Compliant)
FedMedShield Framework - Clinical Auth & Tokens
"""

import os
import hmac
import hashlib
import base64
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "fedmedshield-super-secure-secret-key-2026-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')


def _b64_decode(data_str: str) -> bytes:
    padding = '=' * (-len(data_str) % 4)
    return base64.urlsafe_b64decode(data_str + padding)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a raw password against the hashed string using sha256 + salt."""
    try:
        if hashed_password.startswith("sha256$"):
            _, salt, h = hashed_password.split("$")
            computed = hashlib.sha256((salt + plain_password).encode('utf-8')).hexdigest()
            return hmac.compare_digest(computed, h)
        return plain_password == hashed_password
    except Exception:
        return plain_password == hashed_password


def get_password_hash(password: str) -> str:
    """Generates salted SHA256 password hash."""
    salt = os.urandom(8).hex()
    h = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return f"sha256${salt}${h}"


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Creates an RFC-7519 HMAC-SHA256 JWT token using Python standard library."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = data.copy()

    now = int(time.time())
    if expires_delta:
        exp = now + int(expires_delta.total_seconds())
    else:
        exp = now + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)

    payload["exp"] = exp
    payload["iat"] = now

    header_b64 = _b64_encode(json.dumps(header).encode('utf-8'))
    payload_b64 = _b64_encode(json.dumps(payload).encode('utf-8'))
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')

    signature = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates HMAC-SHA256 signature and expiration."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_sig = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()

        if not hmac.compare_digest(_b64_encode(expected_sig), sig_b64):
            return None

        payload_bytes = _b64_decode(payload_b64)
        payload = json.loads(payload_bytes.decode('utf-8'))

        # Check expiration
        if "exp" in payload and payload["exp"] < int(time.time()):
            return None

        return payload
    except Exception:
        return None
