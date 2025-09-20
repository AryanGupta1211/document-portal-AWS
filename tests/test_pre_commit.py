import pytest
from fastapi import status
from unittest.mock import patch, MagicMock
import json

class TestPreCommitSuite:
    """Pre-commit test suite - fast unit tests without heavy dependencies"""

    def test_health_check(self, client):
        """Test 1: Health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "document-portal"

    def test_root_redirect_no_token(self, client):
        """Test 2: Root endpoint redirects to auth when no token"""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 302
        assert "/auth/" in response.headers["location"]

    def test_auth_page_loads(self, client):
        """Test 3: Auth page loads successfully"""
        response = client.get("/auth/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_user_registration_valid_data(self, client, sample_user_data):
        """Test 4: User registration with valid data"""
        response = client.post("/auth/register", json=sample_user_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "User registered successfully"
        assert data["user"]["username"] == sample_user_data["username"]
        assert data["user"]["email"] == sample_user_data["email"]

    def test_user_registration_duplicate_email(self, client, sample_user_data):
        """Test 5: User registration fails with duplicate email"""
        # Register first user
        client.post("/auth/register", json=sample_user_data)
        
        # Try to register with same email
        duplicate_data = sample_user_data.copy()
        duplicate_data["username"] = "different_user"
        
        response = client.post("/auth/register", json=duplicate_data)
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]

    def test_user_registration_invalid_password(self, client):
        """Test 6: User registration fails with invalid password"""
        invalid_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "123",  # Too short
            "confirm_password": "123"
        }
        
        response = client.post("/auth/register", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_user_login_valid_credentials(self, client, sample_user_data):
        """Test 7: User login with valid credentials"""
        # Register user first
        client.post("/auth/register", json=sample_user_data)
        
        login_data = {
            "username": sample_user_data["username"],
            "password": sample_user_data["password"]
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        assert response.json()["message"] == "Login successful"
        assert "access_token" in response.cookies

    def test_user_login_invalid_credentials(self, client, sample_user_data):
        """Test 8: User login fails with invalid credentials"""
        # Register user first
        client.post("/auth/register", json=sample_user_data)
        
        invalid_login = {
            "username": sample_user_data["username"],
            "password": "wrongpassword"
        }
        
        response = client.post("/auth/login", json=invalid_login)
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_protected_route_without_auth(self, client):
        """Test 9: Protected route denies access without authentication"""
        response = client.get("/dashboard")
        assert response.status_code == 401

    @patch('api.main.DocHandler')
    @patch('api.main.DocumentAnalyzer')
    @patch('api.main.read_pdf_via_handler')
    def test_analyze_endpoint_without_auth(self, mock_read_pdf, mock_analyzer, mock_handler, client, sample_pdf_file):
        """Test 10: Analyze endpoint denies access without authentication"""
        filename, content, content_type = sample_pdf_file
        
        response = client.post(
            "/analyze",
            files={"file": (filename, content, content_type)}
        )
        assert response.status_code == 401