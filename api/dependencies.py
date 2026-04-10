"""
FastAPI dependencies: API key auth, file validation, rate limiting headers.
"""

from fastapi import Header, HTTPException, UploadFile, status
from core.config import get_settings

settings = get_settings()

# Demo: single shared key. In production use a DB of hashed keys.
VALID_API_KEYS = {"demo-key-hackathon", settings.secret_key}


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
    return x_api_key


async def validate_resume_file(file: UploadFile) -> bytes:
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.max_file_size_mb} MB",
        )
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain",
    }
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )
    return content
