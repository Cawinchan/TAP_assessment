from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

def test_ping():
    """
    Test response
    """
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Our Ship Type Classification FastAPI!"}