import pytest
import tempfile
import shutil
import os
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock

# Import your app and database dependencies
from api.main import app
from auth.database import Base, get_db
from auth.utils import AuthUtils

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Override the database dependency
app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create test database tables"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    """Test client fixture"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def db_session():
    """Database session fixture"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def temp_dirs():
    """Create temporary directories for file operations"""
    temp_upload = tempfile.mkdtemp(prefix="test_upload_")
    temp_faiss = tempfile.mkdtemp(prefix="test_faiss_")
    
    # Set environment variables
    os.environ["UPLOAD_BASE"] = temp_upload
    os.environ["FAISS_BASE"] = temp_faiss
    
    yield {"upload": temp_upload, "faiss": temp_faiss}
    
    # Cleanup
    shutil.rmtree(temp_upload, ignore_errors=True)
    shutil.rmtree(temp_faiss, ignore_errors=True)

@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123",
        "confirm_password": "testpass123"
    }

@pytest.fixture
def auth_headers(client, sample_user_data):
    """Create authenticated user and return auth headers"""
    # Register user
    client.post("/auth/register", json=sample_user_data)
    
    # Login user
    login_response = client.post("/auth/login", json={
        "username": sample_user_data["username"],
        "password": sample_user_data["password"]
    })
    
    # Extract token from cookie
    cookies = login_response.cookies
    access_token = cookies.get("access_token")
    
    return {"Cookie": f"access_token={access_token}"}

@pytest.fixture
def sample_pdf_file():
    """Create a mock PDF file for testing"""
    content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF"
    return ("test.pdf", content, "application/pdf")

@pytest.fixture
def mock_doc_handler():
    """Mock DocHandler for testing without actual PDF processing"""
    with patch('api.main.DocHandler') as mock_handler:
        mock_instance = MagicMock()
        mock_instance.save_pdf.return_value = "/tmp/test.pdf"
        mock_handler.return_value = mock_instance
        yield mock_handler

@pytest.fixture
def mock_document_analyzer():
    """Mock DocumentAnalyzer for testing"""
    with patch('api.main.DocumentAnalyzer') as mock_analyzer:
        mock_instance = MagicMock()
        mock_instance.analyze_document.return_value = {
            "summary": "Test document summary",
            "key_points": ["Point 1", "Point 2"],
            "word_count": 100
        }
        mock_analyzer.return_value = mock_instance
        yield mock_analyzer

@pytest.fixture
def mock_read_pdf():
    """Mock PDF reading function"""
    with patch('api.main.read_pdf_via_handler') as mock_read:
        mock_read.return_value = "Sample PDF content for testing purposes."
        yield mock_read

@pytest.fixture
def auth_cookies(client, sample_user_data):
    """Get authentication cookies for making authenticated requests"""
    # Register user if needed (ignore if already exists)
    register_response = client.post("/auth/register", json=sample_user_data)
    
    # Login to get authentication cookie
    login_data = {
        "username": sample_user_data["username"],
        "password": sample_user_data["password"]
    }
    login_response = client.post("/auth/login", json=login_data)
    assert login_response.status_code == 200, f"Login failed: {login_response.text}"
    
    return login_response.cookies


@pytest.fixture
def authenticated_user_token(client, sample_user_data):
    """Get a raw JWT token for testing"""
    # Register user (ignore if exists)
    client.post("/auth/register", json=sample_user_data)
    
    # Login to get token
    login_data = {
        "username": sample_user_data["username"],
        "password": sample_user_data["password"]
    }
    login_response = client.post("/auth/login", json=login_data)
    assert login_response.status_code == 200
    
    # Extract token from cookies
    access_token = None
    for cookie in login_response.cookies:
        if cookie.name == "access_token":
            access_token = cookie.value
            break
    
    assert access_token, "No access_token found in cookies"
    return access_token