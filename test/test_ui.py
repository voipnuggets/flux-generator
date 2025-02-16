import inspect
import importlib.util
import sys
import unittest
from pathlib import Path

class TestUI(unittest.TestCase):
    """Test cases for UI implementation"""
    
    @classmethod
    def setUpClass(cls):
        """Load the flux_app module once for all tests"""
        module_path = Path(__file__).parent.parent / "flux_app.py"
        spec = importlib.util.spec_from_file_location("flux_app", module_path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.source = inspect.getsource(cls.module)
        cls.create_ui_source = inspect.getsource(cls.module.create_ui)
    
    def test_no_html_imports(self):
        """Test that there are no HTML-related imports"""
        html_related_imports = [
            "from fastapi.templating import Jinja2Templates",
            "from fastapi.staticfiles import StaticFiles",
            "from fastapi.responses import HTMLResponse",
            "Request, Form"
        ]
        
        found_imports = [imp for imp in html_related_imports if imp in self.source]
        self.assertEqual(len(found_imports), 0, 
                        f"Found HTML-related imports that should be removed: {found_imports}")
    
    def test_no_html_code(self):
        """Test that there is no HTML/template-related code"""
        html_related_code = [
            "templates = Jinja2Templates",
            "StaticFiles(directory=",
            "TemplateResponse",
            "@app.get(\"/\", response_class=HTMLResponse)",
            "templates.TemplateResponse"
        ]
        
        found_code = [code for code in html_related_code if code in self.source]
        self.assertEqual(len(found_code), 0,
                        f"Found HTML/template-related code that should be removed: {found_code}")
    
    def test_gradio_components(self):
        """Test that all required Gradio components are present"""
        required_components = [
            "gr.Blocks",
            "gr.Column",
            "gr.Row",
            "gr.Tabs",
            "gr.Tab",
            "gr.Textbox",
            "gr.Button",
            "gr.Slider",
            "gr.Image",
            "gr.Markdown",
            "gr.Group",
            "gr.Radio",
            "gr.Checkbox",
            "gr.Number"
        ]
        
        missing_components = [comp for comp in required_components 
                            if comp not in self.create_ui_source]
        self.assertEqual(len(missing_components), 0,
                        f"Missing Gradio components: {missing_components}")
    
    def test_gradio_theme(self):
        """Test that Gradio theme is configured correctly with Google fonts"""
        self.assertIn("gr.themes.Soft(", self.create_ui_source,
                     "Gradio theme should be configured with Soft theme")
        self.assertIn("font=gr.themes.GoogleFont(\"Inter\")", self.create_ui_source,
                     "Theme should use Inter Google font")
        self.assertIn("font_mono=gr.themes.GoogleFont(\"IBM Plex Mono\")", self.create_ui_source,
                     "Theme should use IBM Plex Mono Google font for monospace")
    
    def test_css_styling(self):
        """Test that CSS styling is present"""
        self.assertIn("css=css", self.create_ui_source,
                     "CSS styling should be configured")
        self.assertIn(".container", self.create_ui_source,
                     "Container CSS class should be defined")

    def test_generate_button_functionality(self):
        """Test that the generate button works correctly"""
        # Create a mock response
        mock_response = self.module.SDAPIResponse(
            images=["data:image/png;base64,test123"],
            parameters={
                "prompt": "test prompt",
                "model": "schnell",
                "steps": 2,
                "cfg_scale": 4.0,
                "width": 512,
                "height": 512,
                "seed": 42
            },
            info="Generated with Flux schnell model"
        )

        # Create a mock txt2img function
        original_txt2img = self.module.txt2img
        txt2img_called = False
        async def mock_txt2img(request):
            nonlocal txt2img_called
            txt2img_called = True
            return mock_response

        try:
            # Replace txt2img with mock
            self.module.txt2img = mock_txt2img

            # Call the UI generate function
            import asyncio
            result = asyncio.run(self.module.on_generate(
                "test prompt",  # prompt
                "schnell",      # model_type
                2,             # num_steps
                4.0,           # guidance_scale
                512,           # width
                512,           # height
                42            # seed
            ))

            # Verify txt2img was called
            self.assertTrue(txt2img_called, "txt2img endpoint was not called")

            # Verify the result
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], "data:image/png;base64,test123")
            self.assertIsInstance(result[1], self.module.gr.Markdown)
            self.assertIn("Generated with Flux schnell model", result[1].value)

        except Exception as e:
            self.fail(f"Generate function failed: {str(e)}")
        finally:
            # Restore original txt2img
            self.module.txt2img = original_txt2img

    async def test_ui_uses_txt2img_endpoint(self):
        """Test that the UI generate button uses the txt2img endpoint"""
        # Create a mock response
        mock_response = self.module.SDAPIResponse(
            images=["data:image/png;base64,test123"],
            parameters={
                "prompt": "test prompt",
                "model": "schnell",
                "steps": 2,
                "cfg_scale": 4.0,
                "width": 512,
                "height": 512,
                "seed": 42
            },
            info="Generated with Flux schnell model"
        )

        # Create a mock txt2img function
        original_txt2img = self.module.txt2img
        txt2img_called = False
        async def mock_txt2img(request):
            nonlocal txt2img_called
            txt2img_called = True
            self.assertEqual(request.prompt, "test prompt")
            self.assertEqual(request.model, "schnell")
            self.assertEqual(request.steps, 2)
            self.assertEqual(request.cfg_scale, 4.0)
            self.assertEqual(request.width, 512)
            self.assertEqual(request.height, 512)
            self.assertEqual(request.seed, 42)
            return mock_response

        try:
            # Replace txt2img with mock
            self.module.txt2img = mock_txt2img

            # Call the UI generate function
            import asyncio
            result = asyncio.run(self.module.on_generate(
                "test prompt",  # prompt
                "schnell",      # model_type
                2,             # num_steps
                4.0,           # guidance_scale
                512,           # width
                512,           # height
                42            # seed
            ))

            # Verify txt2img was called
            self.assertTrue(txt2img_called, "txt2img endpoint was not called")

            # Verify the result
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], "data:image/png;base64,test123")
            self.assertIsInstance(result[1], self.module.gr.Markdown)
            self.assertIn("Generated with Flux schnell model", result[1].value)

        finally:
            # Restore original txt2img
            self.module.txt2img = original_txt2img

if __name__ == "__main__":
    unittest.main(verbosity=2) 