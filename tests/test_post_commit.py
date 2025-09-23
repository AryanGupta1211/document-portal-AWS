import pytest
from fastapi import status
import tempfile
import os
from unittest.mock import patch, MagicMock

class TestPostCommitSuite:
    """Post-commit test suite - integration tests with file operations and full flows"""

    def test_full_user_authentication_flow(self, client, sample_user_data):
        """Test 11: Complete user authentication flow - register, login, access protected route, logout"""
        # Step 1: Register
        register_response = client.post("/auth/register", json=sample_user_data)
        assert register_response.status_code == 200
        
        # Step 2: Login
        login_data = {
            "username": sample_user_data["username"],
            "password": sample_user_data["password"]
        }
        login_response = client.post("/auth/login", json=login_data)
        assert login_response.status_code == 200
        
        # Step 3: Access protected route with cookie
        cookies = login_response.cookies
        dashboard_response = client.get("/dashboard", cookies=cookies)
        assert dashboard_response.status_code == 200
        
        # Step 4: Get user info
        me_response = client.get("/auth/me", cookies=cookies)
        assert me_response.status_code == 200
        user_data = me_response.json()
        assert user_data["username"] == sample_user_data["username"]
        
        # Step 5: Logout
        logout_response = client.post("/auth/logout", cookies=cookies)
        assert logout_response.status_code == 200

    # @patch('api.main.DocHandler')
    # @patch('api.main.DocumentAnalyzer')  
    # @patch('api.main.read_pdf_via_handler')
    # def test_document_analysis_with_auth(self, mock_read_pdf, mock_analyzer, mock_handler, 
    #                                    client, auth_cookies, sample_pdf_file, temp_dirs):
    #     """Test 12: Document analysis with proper authentication and mocked dependencies"""
    #     # Setup mocks
    #     mock_read_pdf.return_value = "Sample PDF content for analysis."
    #     mock_handler_instance = MagicMock()
    #     mock_handler_instance.save_pdf.return_value = "/tmp/test.pdf"
    #     mock_handler.return_value = mock_handler_instance
        
    #     mock_analyzer_instance = MagicMock()
    #     mock_analyzer_instance.analyze_document.return_value = {
    #         "summary": "Test document analysis results",
    #         "key_points": ["Important point 1", "Important point 2"],
    #         "word_count": 150,
    #         "readability_score": 8.5
    #     }
    #     mock_analyzer.return_value = mock_analyzer_instance
        
    #     filename, content, content_type = sample_pdf_file
        
    #     # Use client with authentication cookies
    #     response = client.post(
    #         "/analyze",
    #         files={"file": (filename, content, content_type)},
    #         cookies=auth_cookies
    #     )
        
    #     assert response.status_code == 200
    #     data = response.json()
    #     assert "summary" in data
    #     assert data["word_count"] == 150
        
    #     # Verify mocks were called
    #     mock_handler.assert_called_once()
    #     mock_analyzer.assert_called_once()
    #     mock_read_pdf.assert_called_once()

    @patch('api.main.DocumentComparator')
    @patch('api.main.DocumentComparatorLLM')
    def test_document_comparison_with_auth(self, mock_comparator_llm, mock_comparator,
                                         client, auth_cookies, sample_pdf_file):
        """Test 13: Document comparison with authentication and mocked dependencies"""
        # Setup mocks
        mock_comp_instance = MagicMock()
        mock_comp_instance.save_uploaded_files.return_value = ("/tmp/ref.pdf", "/tmp/act.pdf")
        mock_comp_instance.combine_documents.return_value = "Combined document text"
        mock_comp_instance.session_id = "test_session_123"
        mock_comparator.return_value = mock_comp_instance
        
        mock_llm_instance = MagicMock()
        # Mock DataFrame with comparison results
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [
            {"section": "Introduction", "similarity": 0.85, "differences": "Minor wording changes"},
            {"section": "Conclusion", "similarity": 0.92, "differences": "No significant differences"}
        ]
        mock_llm_instance.compare_documents.return_value = mock_df
        mock_comparator_llm.return_value = mock_llm_instance
        
        filename, content, content_type = sample_pdf_file
        
        response = client.post(
            "/compare",
            files={
                "reference": (f"ref_{filename}", content, content_type),
                "actual": (f"act_{filename}", content, content_type)
            },
            cookies=auth_cookies
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "rows" in data
        assert "session_id" in data
        assert len(data["rows"]) == 2
        assert data["session_id"] == "test_session_123"

    @patch('api.main.ChatIngestor')
    def test_chat_index_building_with_auth(self, mock_chat_ingestor, client, auth_cookies, 
                                         sample_pdf_file, temp_dirs):
        """Test 14: Chat index building with authentication"""
        # Setup mock
        mock_ingestor_instance = MagicMock()
        mock_ingestor_instance.session_id = "chat_session_456"
        mock_ingestor_instance.built_retriver.return_value = None
        mock_chat_ingestor.return_value = mock_ingestor_instance
        
        filename, content, content_type = sample_pdf_file
        
        response = client.post(
            "/chat/index",
            files={"files": (filename, content, content_type)},
            data={
                "session_id": "custom_session",
                "use_session_dirs": "true",
                "chunk_size": "1000",
                "chunk_overlap": "200",
                "k": "5"
            },
            cookies=auth_cookies
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "chat_session_456"
        assert data["k"] == 5
        assert data["use_session_dirs"] is True
        
        # Verify mock was called with correct parameters
        mock_chat_ingestor.assert_called_once()
        mock_ingestor_instance.built_retriver.assert_called_once()

    # @patch('api.main.ConversationalRAG')
    # @patch.dict(os.environ, {'FAISS_BASE': ''})  # Use empty FAISS_BASE for test control
    # def test_chat_query_with_auth(self, mock_rag, client, auth_cookies, temp_dirs):
    #     """Test 15: Chat query with authentication and mocked FAISS index"""
    #     # Create fake FAISS directory structure in the test temp directory
    #     session_dir = os.path.join(temp_dirs["faiss"], "test_session")
    #     os.makedirs(session_dir, exist_ok=True)
        
    #     # Create dummy index files
    #     with open(os.path.join(session_dir, "index.faiss"), "w") as f:
    #         f.write("dummy")
    #     with open(os.path.join(session_dir, "index.pkl"), "w") as f:
    #         f.write("dummy")
        
    #     # Setup mock
    #     mock_rag_instance = MagicMock()
    #     mock_rag_instance.load_retriever_from_faiss.return_value = None
    #     mock_rag_instance.invoke.return_value = "This is a response to your question about the documents."
    #     mock_rag.return_value = mock_rag_instance
        
    #     # Patch the FAISS_BASE to point to our temp directory
    #     with patch('api.main.FAISS_BASE', temp_dirs["faiss"]):
    #         response = client.post(
    #             "/chat/query",
    #             data={
    #                 "question": "What is the main topic of the documents?",
    #                 "session_id": "test_session",
    #                 "use_session_dirs": "true",
    #                 "k": "5"
    #             },
    #             cookies=auth_cookies
    #         )
        
    #     assert response.status_code == 200
    #     data = response.json()
    #     assert data["answer"] == "This is a response to your question about the documents."
    #     assert data["session_id"] == "test_session"
    #     assert data["k"] == 5
    #     assert data["engine"] == "LCEL-RAG"
        
    #     # Verify mock interactions
    #     mock_rag.assert_called_once_with(session_id="test_session")
    #     mock_rag_instance.load_retriever_from_faiss.assert_called_once()
    #     mock_rag_instance.invoke.assert_called_once()

