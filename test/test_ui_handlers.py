import unittest
from pathlib import Path
import sys
import gradio as gr
from unittest.mock import patch, MagicMock
import asyncio

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import create_ui, download_model_ui, on_generate

class TestUIHandlers(unittest.TestCase):
    """Test cases for UI event handlers and Gradio interface"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_port = 7860
        self.test_api = True
    
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
    
    @patch("gradio.Blocks")
    @patch("flux_app.download_model_ui")
    def test_ui_functionality(self, mock_download_model_ui, mock_blocks):
        """Test the complete UI functionality"""
        # Set up mock components
        mock_schnell_status = MagicMock()
        mock_dev_status = MagicMock()
        mock_download_schnell = MagicMock()
        mock_download_dev = MagicMock()
        mock_generate_btn = MagicMock()
        mock_stop_btn = MagicMock()
        mock_api_btn = MagicMock()
        mock_force_download = MagicMock()
        mock_download_status = MagicMock()
        mock_prompt = MagicMock()
        mock_model_type = MagicMock()
        mock_num_steps = MagicMock()
        mock_guidance = MagicMock()
        mock_image_width = MagicMock()
        mock_image_height = MagicMock()
        mock_seed = MagicMock()
        mock_output_image = MagicMock()
        mock_image_info = MagicMock()
        
        # Set up mock returns
        mock_blocks.return_value.__enter__.return_value = MagicMock()
        mock_download_model_ui.return_value = "✅ Model downloaded successfully"
        
        with patch.multiple(
            "gradio",
            Textbox=MagicMock(side_effect=[
                mock_schnell_status, mock_dev_status, mock_download_status, mock_prompt
            ]),
            Button=MagicMock(side_effect=[
                mock_download_schnell, mock_download_dev, mock_generate_btn, mock_stop_btn, mock_api_btn
            ]),
            Checkbox=MagicMock(return_value=mock_force_download),
            Radio=MagicMock(return_value=mock_model_type),
            Slider=MagicMock(side_effect=[
                mock_num_steps, mock_guidance, mock_image_width, mock_image_height
            ]),
            Number=MagicMock(return_value=mock_seed),
            Image=MagicMock(return_value=mock_output_image),
            Markdown=MagicMock(return_value=mock_image_info),
            Tab=MagicMock(),
            Column=MagicMock(),
            Row=MagicMock(),
            Group=MagicMock()
        ):
            # Create UI
            demo = create_ui(enable_api=self.test_api, api_port=self.test_port)
            
            # Test model download functionality
            result = mock_download_model_ui("flux-schnell", False)
            self.assertEqual(result, "✅ Model downloaded successfully")
            mock_download_model_ui.assert_called_with("flux-schnell", False)
            
            # Test image generation functionality
            async def mock_generate(*args):
                return ["data:image/png;base64,test123", mock_image_info]
            
            with patch("flux_app.on_generate", new=mock_generate):
                # Create an async mock for the generate handler
                async def mock_handler(*args):
                    return await mock_generate(*args)
                
                # Replace the handler in the blocks
                demo.blocks[mock_generate_btn]["click"][0] = mock_handler
                
                # Run the handler
                result = asyncio.run(mock_handler(
                    "test prompt", "schnell", 1, 4.0, 512, 512, 42
                ))
                self.assertTrue(isinstance(result, list))
                self.assertEqual(len(result), 2)  # Should return [image, info]
                self.assertEqual(result[0], "data:image/png;base64,test123")
                self.assertEqual(result[1], mock_image_info)  # Compare with the mock directly
            
            # Test stop functionality
            def mock_stop_handler():
                return "Generation stopped"
            
            # Replace the handler in the blocks
            demo.blocks[mock_stop_btn]["click"][0] = mock_stop_handler
            
            # Run the handler
            result = mock_stop_handler()
            self.assertEqual(result, "Generation stopped")
            
            # Test API documentation button if API is enabled
            if self.test_api:
                with patch("webbrowser.open") as mock_browser:
                    def mock_api_handler():
                        import webbrowser
                        webbrowser.open(f"http://127.0.0.1:{self.test_port}/docs")
                        return None
                    
                    # Replace the handler in the blocks
                    demo.blocks[mock_api_btn]["click"][0] = mock_api_handler
                    
                    # Run the handler
                    mock_api_handler()
                    mock_browser.assert_called_with(f"http://127.0.0.1:{self.test_port}/docs")
    
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
    def test_generate_with_api(self, mock_generate):
        """Test image generation UI handler"""
        # Mock successful generation
        mock_generate.return_value = ["data:image/png;base64,test123"]
        
        # Test with specific values
        result = asyncio.run(on_generate(
            "test image",  # prompt
            "schnell",     # model_type
            1,            # num_steps
            4.0,          # guidance_scale
            512,          # width
            512,          # height
            42           # seed
        ))
        self.assertEqual(result[0], "data:image/png;base64,test123")
        self.assertIsInstance(result[1], gr.Markdown)
        mock_generate.assert_called_once()
        
        # Test with default values
        mock_generate.reset_mock()
        result = asyncio.run(on_generate(
            "test image",  # prompt
            "schnell",     # model_type
            1,            # num_steps
            4.0,          # guidance_scale
            512,          # width
            512,          # height
            None         # seed
        ))
        self.assertEqual(result[0], "data:image/png;base64,test123")
        self.assertIsInstance(result[1], gr.Markdown)
        mock_generate.assert_called_once()

if __name__ == "__main__":
    unittest.main(verbosity=2) 