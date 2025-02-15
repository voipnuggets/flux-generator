import inspect
import importlib.util
import sys
from pathlib import Path

def test_ui_implementation():
    """Test that the UI implementation uses Gradio exclusively"""
    
    # Load the flux_app module
    module_path = Path(__file__).parent.parent / "flux_app.py"
    spec = importlib.util.spec_from_file_location("flux_app", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get source code
    source = inspect.getsource(module)
    
    # Check for HTML-related imports
    html_related_imports = [
        "from fastapi.templating import Jinja2Templates",
        "from fastapi.staticfiles import StaticFiles",
        "from fastapi.responses import HTMLResponse",
        "Request, Form"
    ]
    
    found_imports = [imp for imp in html_related_imports if imp in source]
    if found_imports:
        print("❌ Found HTML-related imports that should be removed:")
        for imp in found_imports:
            print(f"  - {imp}")
    
    # Check for HTML/template-related code
    html_related_code = [
        "templates = Jinja2Templates",
        "StaticFiles(directory=",
        "TemplateResponse",
        "@app.get(\"/\", response_class=HTMLResponse)",
        "templates.TemplateResponse"
    ]
    
    found_code = [code for code in html_related_code if code in source]
    if found_code:
        print("❌ Found HTML/template-related code that should be removed:")
        for code in found_code:
            print(f"  - {code}")
    
    # Check for required Gradio components
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
                         if comp not in source]
    if missing_components:
        print("❌ Missing Gradio components:")
        for comp in missing_components:
            print(f"  - {comp}")
    
    # Final verdict
    assert not found_imports, "Found HTML-related imports that should be removed"
    assert not found_code, "Found HTML/template-related code that should be removed"
    assert not missing_components, "Missing required Gradio components"

if __name__ == "__main__":
    test_ui_implementation() 