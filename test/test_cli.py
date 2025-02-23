import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import check_system_compatibility

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
            "--port", "8000",
            "--listen-all"
        ]
        
        with patch("sys.argv", ["flux_app.py"] + test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, default=7860)
            parser.add_argument("--listen-all", action="store_true")
            
            args = parser.parse_args()
            
            self.assertEqual(args.port, 8000)
            self.assertTrue(args.listen_all)
    
    def test_default_options(self):
        """Test default command line options"""
        with patch("sys.argv", ["flux_app.py"]):
            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, default=7860)
            parser.add_argument("--listen-all", action="store_true")
            
            args = parser.parse_args()
            
            self.assertEqual(args.port, 7860)
            self.assertFalse(args.listen_all)
    
    def test_invalid_port(self):
        """Test invalid port number"""
        with patch("sys.argv", ["flux_app.py", "--port", "invalid"]), \
             patch("sys.stderr", new=MagicMock()):  # Suppress error output
            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int)
            
            with self.assertRaises(SystemExit):
                parser.parse_args()
    
    def test_server_config(self):
        """Test server configuration with different options"""
        test_cases = [
            {
                "args": ["flux_app.py"],
                "expected_host": "127.0.0.1",
                "expected_port": 7860
            },
            {
                "args": ["flux_app.py", "--listen-all"],
                "expected_host": "0.0.0.0",
                "expected_port": 7860
            },
            {
                "args": ["flux_app.py", "--port", "8080"],
                "expected_host": "127.0.0.1",
                "expected_port": 8080
            }
        ]
        
        for case in test_cases:
            with self.subTest(args=case["args"]):
                # Parse arguments
                parser = argparse.ArgumentParser()
                parser.add_argument("--port", type=int, default=7860)
                parser.add_argument("--listen-all", action="store_true")
                
                with patch("sys.argv", case["args"]):
                    args = parser.parse_args()
                
                # Verify configuration
                host = "0.0.0.0" if args.listen_all else "127.0.0.1"
                self.assertEqual(host, case["expected_host"])
                self.assertEqual(args.port, case["expected_port"])


if __name__ == "__main__":
    unittest.main(verbosity=2) 