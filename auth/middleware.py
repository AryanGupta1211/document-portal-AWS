from fastapi import Request, HTTPException, status
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import json

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to protect routes that require authentication"""
    
    def __init__(self, app, protected_paths: list = None):
        super().__init__(app)
        # Default protected paths (your existing API endpoints)
        self.protected_paths = protected_paths or [
            "/analyze",
            "/compare", 
            "/chat/index",
            "/chat/query"
        ]
        # Paths that don't require authentication
        self.public_paths = [
            "/",
            "/auth",
            "/auth/",
            "/auth/login",
            "/auth/register",
            "/auth/logout",
            "/health",
            "/static",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Get the request path
        path = request.url.path
        
        # Skip middleware for public paths and static files
        if any(path.startswith(public_path) for public_path in self.public_paths):
            return await call_next(request)
        
        # Check if path requires protection
        if any(path.startswith(protected_path) for protected_path in self.protected_paths):
            # Check for Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                # For browser requests (HTML), redirect to login
                if request.headers.get("accept", "").startswith("text/html"):
                    return RedirectResponse(url="/auth/", status_code=302)
                # For API requests, return 401
                else:
                    return Response(
                        content=json.dumps({"detail": "Authentication required"}),
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        media_type="application/json"
                    )
        
        response = await call_next(request)
        return response