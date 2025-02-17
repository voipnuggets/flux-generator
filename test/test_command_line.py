import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import main, check_system_compatibility

class TestCommandLine(unittest.TestCase):
    """Test cases for command line options"""
    
    def setUp(self):
        """Set up test environment"""
        self.original_argv = sys.argv
    
    def tearDown(self):
        """Clean up test environment"""
        sys.argv = self.original_argv
    
    def test_default_options(self):
        """Test default command line options"""
        sys.argv = ["flux_app.py"]
        
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            main()
            args = mock_parse.return_value
            
            # Verify default values
            self.assertEqual(args.host, "127.0.0.1")
            self.assertEqual(args.port, 7860)
            self.assertFalse(getattr(args, "listen_all", False))
            self.assertFalse(getattr(args, "listen_local", False))
            self.assertFalse(getattr(args, "download_models", False))
            self.assertFalse(getattr(args, "force_download", False))
    
    def test_host_option(self):
        """Test --host option"""
        test_host = "192.168.1.100"
        sys.argv = ["flux_app.py", "--host", test_host]
        
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            main()
            args = mock_parse.return_value
            self.assertEqual(args.host, test_host)
    
    def test_port_option(self):
        """Test --port option"""
        test_port = 8080
        sys.argv = ["flux_app.py", "--port", str(test_port)]
        
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            main()
            args = mock_parse.return_value
            self.assertEqual(args.port, test_port)
    
    def test_listen_all_option(self):
        """Test --listen-all option"""
        sys.argv = ["flux_app.py", "--listen-all"]
        
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            main()
            args = mock_parse.return_value
            self.assertTrue(args.listen_all)
            self.assertEqual(args.host, "0.0.0.0")
    
    def test_listen_local_option(self):
        """Test --listen-local option"""
        sys.argv = ["flux_app.py", "--listen-local"]
        
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            main()
            args = mock_parse.return_value
            self.assertTrue(args.listen_local)
            self.assertEqual(args.host, "192.168.1.1")
    
    def test_download_models_option(self):
        """Test --download-models option"""
        sys.argv = ["flux_app.py", "--download-models"]
        
        with patch("argparse.ArgumentParser.parse_args") as mock_parse, \
             patch("flux_app.check_and_download_models") as mock_download:
            main()
            args = mock_parse.return_value
            self.assertTrue(args.download_models)
            mock_download.assert_called()
    
    def test_force_download_option(self):
        """Test --force-download option"""
        sys.argv = ["flux_app.py", "--force-download"]
        
        with patch("argparse.ArgumentParser.parse_args") as mock_parse, \
             patch("flux_app.check_and_download_models") as mock_download:
            main()
            args = mock_parse.return_value
            self.assertTrue(args.force_download)
            mock_download.assert_called_with(mock.ANY, True)
    
    def test_mutually_exclusive_options(self):
        """Test that --listen-all and --listen-local are mutually exclusive"""
        sys.argv = ["flux_app.py", "--listen-all", "--listen-local"]
        
        with self.assertRaises(SystemExit), \
             patch("sys.stderr", new=MagicMock()):  # Suppress error output
            main()
    
    def test_help_option(self):
        """Test --help option"""
        sys.argv = ["flux_app.py", "--help"]
        
        with self.assertRaises(SystemExit) as cm, \
             patch("sys.stdout", new=MagicMock()):  # Suppress help output
            main()
        self.assertEqual(cm.exception.code, 0)
    
    def test_invalid_port(self):
        """Test invalid port number"""
        sys.argv = ["flux_app.py", "--port", "invalid"]
        
        with self.assertRaises(SystemExit), \
             patch("sys.stderr", new=MagicMock()):  # Suppress error output
            main()
    
    def test_port_availability_check(self):
        """Test port availability check"""
        test_port = 7860
        sys.argv = ["flux_app.py", "--port", str(test_port)]
        
        with patch("flux_app.check_port_available") as mock_check_port, \
             patch("flux_app.find_available_port") as mock_find_port:
            
            # Test when port is available
            mock_check_port.return_value = True
            main()
            mock_check_port.assert_called_with("127.0.0.1", test_port)
            mock_find_port.assert_not_called()
            
            # Test when port is not available
            mock_check_port.return_value = False
            mock_find_port.return_value = test_port + 1
            main()
            mock_check_port.assert_called_with("127.0.0.1", test_port)
            mock_find_port.assert_called_with("127.0.0.1", test_port)
    
    @patch("uvicorn.Server")
    @patch("fastapi.FastAPI")
    def test_server_startup(self, mock_fastapi, mock_server):
        """Test server startup with different options"""
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
                "args": ["flux_app.py", "--listen-local"],
                "expected_host": "192.168.1.1",
                "expected_port": 7860
            },
            {
                "args": ["flux_app.py", "--port", "8080"],
                "expected_host": "127.0.0.1",
                "expected_port": 8080
            }
        ]
        
        for case in test_cases:
            sys.argv = case["args"]
            with self.subTest(args=case["args"]):
                main()
                mock_server.assert_called()
                config = mock_server.call_args[0][0]
                self.assertEqual(config.host, case["expected_host"])
                self.assertEqual(config.port, case["expected_port"])

if __name__ == "__main__":
    unittest.main(verbosity=2) 