import unittest
from pathlib import Path
import sys
import os
import shutil
from unittest.mock import patch, MagicMock
import json

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import check_model_status, check_and_download_models

class TestModelManagement(unittest.TestCase):
    """Test cases for model management functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_models_dir = Path.home() / ".flux" / "models"
        self.original_models_dir = None
        
        # Backup existing models directory if it exists
        if self.test_models_dir.exists():
            self.original_models_dir = self.test_models_dir.with_suffix(".bak")
            shutil.move(self.test_models_dir, self.original_models_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove test models directory
        if self.test_models_dir.exists():
            shutil.rmtree(self.test_models_dir)
        
        # Restore original models directory if it existed
        if self.original_models_dir and self.original_models_dir.exists():
            shutil.move(self.original_models_dir, self.test_models_dir)
    
    def test_model_status_check(self):
        """Test that model status is correctly reported"""
        # Test when models don't exist
        status = check_model_status("flux-schnell")
        self.assertEqual(status, "❌ Not downloaded")
        
        # Create dummy model files
        self.test_models_dir.mkdir(parents=True, exist_ok=True)
        (self.test_models_dir / "flux-schnell-flow.safetensors").touch()
        (self.test_models_dir / "flux-schnell-ae.safetensors").touch()
        
        clip_dir = self.test_models_dir / "flux-schnell" / "text_encoder"
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "config.json").touch()
        (clip_dir / "model.safetensors").touch()
        
        t5_dir = self.test_models_dir / "flux-schnell" / "text_encoder_2"
        t5_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "config.json").touch()
        (clip_dir / "model.safetensors").touch()
        (t5_dir / "config.json").touch()
        (t5_dir / "model.safetensors.index.json").touch()
        
        # Test when all files exist
        status = check_model_status("flux-schnell")
        self.assertEqual(status, "✅ Downloaded")
    
    @patch("flux_app.hf_hub_download")
    def test_model_download(self, mock_hf_download):
        """Test that models can be downloaded"""
        # Mock HuggingFace Hub download to create temporary files
        def mock_download(repo_id, filename):
            temp_file = self.test_models_dir / f"temp_{filename.replace('/', '_')}"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a valid JSON file for the index
            if filename.endswith("model.safetensors.index.json"):
                with open(temp_file, "w") as f:
                    json.dump({"weight_map": {"weight1": "file1.safetensors"}}, f)
            else:
                temp_file.touch()
            
            return str(temp_file)
        
        mock_hf_download.side_effect = mock_download
        
        # Test downloading schnell model
        check_and_download_models("flux-schnell")
        self.assertTrue((self.test_models_dir / "flux-schnell-flow.safetensors").exists())
        self.assertTrue((self.test_models_dir / "flux-schnell-ae.safetensors").exists())
        
        # Test environment variables are set
        self.assertEqual(
            os.environ.get("FLUX_SCHNELL"),
            str(self.test_models_dir / "flux-schnell-flow.safetensors")
        )
        self.assertEqual(
            os.environ.get("AE"),
            str(self.test_models_dir / "flux-schnell-ae.safetensors")
        )

if __name__ == "__main__":
    unittest.main(verbosity=2) 