import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import app

@pytest.fixture
def client():
    """Fixture that creates a TestClient instance for testing API endpoints."""
    return TestClient(app)

@pytest.fixture
def base_url():
    """Fixture that provides the base URL for API endpoints."""
    return "http://127.0.0.1:7860"

@pytest.fixture
def test_image_path():
    """Fixture that provides a path to test images."""
    return Path(__file__).parent / "test_data" / "test_image.png"

@pytest.fixture
def test_model_path():
    """Fixture that provides a path to test model files."""
    return Path(__file__).parent / "test_data" / "test_model.safetensors" 