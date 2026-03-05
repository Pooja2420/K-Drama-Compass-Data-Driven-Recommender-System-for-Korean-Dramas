"""JWT authentication utilities for the K-Drama Compass API."""

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from src.api.schemas import TokenData

SECRET_KEY = os.getenv("SECRET_KEY", "kdrama-compass-dev-secret-change-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory demo users (replace with a database in production)
_USERS: dict[str, str] = {
    "demo": pwd_context.hash("kdrama123"),
    "admin": pwd_context.hash("admin123"),
}


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the *hashed* password."""
    return pwd_context.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> bool:
    """Return True if the username / password pair is valid."""
    hashed = _USERS.get(username)
    if not hashed:
        return False
    return verify_password(password, hashed)


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """Create and return a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData:
    """Decode *token* and return TokenData (username=None on failure)."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        return TokenData(username=username)
    except JWTError:
        return TokenData()
