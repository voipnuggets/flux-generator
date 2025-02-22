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
    
    def test_css_styling(self):
        """Test that CSS styling is present"""
        self.assertIn("css=css", self.create_ui_source,
                     "CSS styling should be configured")
        self.assertIn(".container", self.create_ui_source,
                     "Container CSS class should be defined")

if __name__ == "__main__":
    unittest.main(verbosity=2) 