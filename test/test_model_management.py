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
        # Use HuggingFace cache directory structure
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.model_dir = self.cache_dir / "models--black-forest-labs--FLUX.1-schnell"
        self.snapshot_dir = self.model_dir / "snapshots" / "test"
        
        # Create test directories
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create text encoder directories
        self.clip_dir = self.snapshot_dir / "text_encoder"
        self.t5_dir = self.snapshot_dir / "text_encoder_2"
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.t5_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup existing directory if it exists
        if self.model_dir.exists():
            self.original_dir = self.model_dir.with_suffix(".bak")
            if self.original_dir.exists():
                shutil.rmtree(self.original_dir)
            shutil.move(self.model_dir, self.original_dir)
            
            # Recreate the directories after moving
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            self.clip_dir.mkdir(parents=True, exist_ok=True)
            self.t5_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove test directory
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
        
        # Restore original directory if it existed
        if hasattr(self, 'original_dir') and self.original_dir.exists():
            shutil.move(self.original_dir, self.model_dir)
    
    def test_model_status_check(self):
        """Test that model status is correctly reported"""
        # Test when models don't exist
        status = check_model_status("flux-schnell")
        self.assertEqual(status, "❌ Not downloaded")
        
        # Create model files
        (self.snapshot_dir / "flux1-schnell.safetensors").touch()
        (self.snapshot_dir / "ae.safetensors").touch()
        
        (self.clip_dir / "config.json").touch()
        (self.clip_dir / "model.safetensors").touch()
        
        (self.t5_dir / "config.json").touch()
        (self.t5_dir / "model.safetensors").touch()
        (self.t5_dir / "model.safetensors.index.json").touch()
        
        # Test when all files exist
        status = check_model_status("flux-schnell")
        self.assertEqual(status, "✅ Downloaded")
    
    @patch("flux_app.hf_hub_download")
    def test_model_download(self, mock_hf_download):
        """Test that models can be downloaded"""
        # Mock HuggingFace Hub download
        def mock_download(repo_id, filename):
            target_file = self.snapshot_dir / filename
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a valid JSON file for the index
            if filename.endswith("model.safetensors.index.json"):
                with open(target_file, "w") as f:
                    json.dump({"weight_map": {"weight1": "file1.safetensors"}}, f)
            else:
                target_file.touch()
            
            return str(target_file)
        
        mock_hf_download.side_effect = mock_download
        
        # Test downloading schnell model
        check_and_download_models("flux-schnell")
        
        # Verify files were created
        self.assertTrue((self.snapshot_dir / "flux1-schnell.safetensors").exists())
        self.assertTrue((self.snapshot_dir / "ae.safetensors").exists())
        
        # Verify environment variables
        self.assertEqual(
            os.environ.get("FLUX_SCHNELL"),
            str(self.snapshot_dir / "flux1-schnell.safetensors")
        )
        self.assertEqual(
            os.environ.get("AE"),
            str(self.snapshot_dir / "ae.safetensors")
        )

if __name__ == "__main__":
    unittest.main(verbosity=2) 