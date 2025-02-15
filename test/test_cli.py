import unittest
from pathlib import Path
import sys
import argparse
from unittest.mock import patch, MagicMock

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import main, check_system_compatibility

class TestCLI(unittest.TestCase):
    """Test cases for command-line interface"""
    
    def test_system_compatibility(self):
        """Test system compatibility check"""
        # Test Apple Silicon Mac
        with patch("sys.platform", "darwin"), \
             patch("platform.machine", return_value="arm64"):
            self.assertTrue(check_system_compatibility())
        
        # Test Intel Mac
        with patch("sys.platform", "darwin"), \
             patch("platform.machine", return_value="x86_64"):
            with self.assertRaises(SystemError) as cm:
                check_system_compatibility()
            self.assertEqual(str(cm.exception), "This application requires an Apple Silicon Mac")
        
        # Test non-macOS
        with patch("sys.platform", "linux"), \
             patch("platform.machine", return_value="x86_64"):
            with self.assertRaises(SystemError) as cm:
                check_system_compatibility()
            self.assertEqual(str(cm.exception), "This application only runs on macOS")
    
    def test_cli_arguments(self):
        """Test command line argument parsing"""
        test_args = [
            "--enable-api",
            "--api-port", "8000",
            "--listen-all",
            "--download-models",
            "--force-download"
        ]
        
        with patch("sys.argv", ["flux_app.py"] + test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--enable-api", action="store_true")
            parser.add_argument("--api-port", type=int, default=7860)
            parser.add_argument("--listen-all", action="store_true")
            parser.add_argument("--download-models", action="store_true")
            parser.add_argument("--force-download", action="store_true")
            
            args = parser.parse_args()
            
            self.assertTrue(args.enable_api)
            self.assertEqual(args.api_port, 8000)
            self.assertTrue(args.listen_all)
            self.assertTrue(args.download_models)
            self.assertTrue(args.force_download)
    
    @patch("uvicorn.run")
    @patch("flux_app.check_system_compatibility")
    @patch("flux_app.check_and_download_models")
    def test_main_api_mode(self, mock_download, mock_check, mock_uvicorn):
        """Test main function in API mode"""
        test_args = ["flux_app.py", "--enable-api", "--listen-all"]
        
        with patch("sys.argv", test_args):
            main()
            
            mock_check.assert_called_once()
            mock_uvicorn.assert_called_once()
            self.assertEqual(mock_uvicorn.call_args[1]["host"], "0.0.0.0")
            self.assertEqual(mock_uvicorn.call_args[1]["port"], 7860)
    
    @patch("gradio.Blocks.launch")
    @patch("flux_app.check_system_compatibility")
    @patch("flux_app.check_and_download_models")
    @patch("flux_app.create_ui")
    def test_main_ui_mode(self, mock_create_ui, mock_download, mock_check, mock_launch):
        """Test main function in UI mode"""
        test_args = ["flux_app.py"]
        
        # Mock create_ui to return a MagicMock
        mock_demo = MagicMock()
        mock_create_ui.return_value = mock_demo
        
        with patch("sys.argv", test_args):
            main()
            
            mock_check.assert_called_once()
            mock_create_ui.assert_called_once_with(enable_api=False, api_port=7860)
            mock_demo.launch.assert_called_once_with(
                server_name="127.0.0.1",
                server_port=7860,
                share=False,
                show_error=True,
                inbrowser=True
            )

if __name__ == "__main__":
    unittest.main(verbosity=2) 