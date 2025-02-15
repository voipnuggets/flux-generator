import unittest
from pathlib import Path
import sys
import gradio as gr
from unittest.mock import patch, MagicMock

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import create_ui, download_model_ui, generate_for_ui

class TestUIHandlers(unittest.TestCase):
    """Test cases for UI event handlers and Gradio interface"""
    
    @patch("gradio.Blocks")
    @patch("gradio.Tab")
    @patch("gradio.Column")
    @patch("gradio.Row")
    @patch("gradio.Textbox")
    @patch("gradio.Button")
    @patch("gradio.Slider")
    @patch("gradio.Image")
    @patch("gradio.Markdown")
    @patch("gradio.Group")
    @patch("gradio.Radio")
    @patch("gradio.Checkbox")
    @patch("gradio.Number")
    def test_create_ui(self, *mocks):
        """Test that UI is created with correct components"""
        # Set up mock returns
        mock_blocks = MagicMock()
        mock_blocks.blocks = {}
        mocks[0].return_value = mock_blocks  # Blocks mock
        
        # Create UI
        demo = create_ui(enable_api=True, api_port=7860)
        
        # Verify component creation
        self.assertIsInstance(demo, MagicMock)
        for mock in mocks:
            self.assertGreater(mock.call_count, 0, f"Mock {mock} was not called")
    
    @patch("flux_app.check_and_download_models")
    def test_download_model_ui(self, mock_download):
        """Test model download UI handler"""
        # Test successful download
        mock_download.return_value = None
        result = download_model_ui("flux-schnell", force=False)
        self.assertTrue("✅" in result)
        self.assertTrue("Successfully downloaded" in result)
        mock_download.assert_called_with("flux-schnell", False)
        
        # Test error case
        mock_download.side_effect = Exception("Download failed")
        result = download_model_ui("flux-schnell", force=False)
        self.assertTrue("❌" in result)
        self.assertTrue("Error downloading" in result)
    
    @patch("flux_app.generate_images")
    def test_generate_for_ui(self, mock_generate):
        """Test image generation UI handler"""
        # Mock successful generation
        mock_generate.return_value = ["data:image/png;base64,test123"]
        
        # Test with specific values
        image = generate_for_ui(
            prompt="test image",
            model_type="schnell",
            num_steps=2,
            guidance_scale=4.0,
            width=512,
            height=512,
            seed=42
        )
        self.assertEqual(image, "data:image/png;base64,test123")
        mock_generate.assert_called_once()
        
        # Test with default values
        mock_generate.reset_mock()
        image = generate_for_ui(
            prompt="test image",
            model_type="schnell",
            num_steps=None,
            guidance_scale=4.0,
            width=512,
            height=512,
            seed=None
        )
        self.assertEqual(image, "data:image/png;base64,test123")
        mock_generate.assert_called_once()

if __name__ == "__main__":
    unittest.main(verbosity=2) 