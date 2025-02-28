"""
PyInstaller build script for Flux Generator

Prerequisites:
> python3 -m venv venv
> source venv/bin/activate (or source venv/bin/activate.fish for fish shell)
> python3 -m pip install -r requirements.txt
> python3 -m pip install pyinstaller
> python3 build.py

Platform specific libraries that MIGHT be needed:
MacOS:
- brew install portaudio
"""

import os
import platform
import sys
import PyInstaller.__main__

def build(signing_key=None):
    app_name = 'Flux\\ Generator'
    compile(signing_key)
    
    macos = platform.system() == 'Darwin'
    if macos and signing_key:
        # Codesign
        os.system(
            f'codesign --deep --force --verbose --sign "{signing_key}" dist/{app_name}.app --options runtime')
        
        zip_name = zip()
        
        if signing_key:
            keychain_profile = signing_key.split('(')[0].strip()
            # Notarize
            os.system(f'xcrun notarytool submit --wait --keychain-profile "{keychain_profile}" --verbose dist/{zip_name}')
            input(f'Check whether notarization was successful using \n\t xcrun notarytool history --keychain-profile {keychain_profile}.\nYou can check debug logs using \n\t xcrun notarytool log --keychain-profile "{keychain_profile}" <run-id>')
            
            # Staple
            os.system(f'xcrun stapler staple dist/{app_name}.app')
            
            # Zip the signed, stapled file
            zip_name = zip()

def compile(signing_key=None):
    # Path to main application script
    app_script = 'flux_app.py'
    
    # Common PyInstaller options
    pyinstaller_options = [
        '--clean',
        '--noconfirm',
        
        # --- Basics --- #
        '--name=Flux Generator',
        '--icon=resources/icon.png',  # You'll need to create this
        '--windowed',  # GUI application
        
        # Where to find necessary packages
        '--paths=./venv/lib/python3.11/site-packages',
        
        # Required imports
        '--hidden-import=mlx',
        '--hidden-import=gradio',
        '--hidden-import=fastapi',
        '--hidden-import=transformers',
        '--hidden-import=huggingface_hub',
        '--hidden-import=soundfile',
        '--hidden-import=scipy',
        '--hidden-import=tqdm',
        
        # Static files and resources
        '--add-data=musicgen:musicgen',
        '--add-data=resources/*:resources',
        
        # Main script
        app_script
    ]
    
    # Platform-specific options
    if platform.system() == 'Darwin':  # MacOS
        if signing_key:
            pyinstaller_options.extend([
                f'--codesign-identity={signing_key}'
            ])
    
    # Run PyInstaller
    PyInstaller.__main__.run(pyinstaller_options)
    print('Done. Check dist/ for executables.')

def zip():
    # Zip the app
    print('Zipping the executables')
    app_name = 'Flux\\ Generator'
    zip_name = 'Flux-Generator'
    
    if platform.system() == 'Darwin':  # MacOS
        if platform.processor() == 'arm':
            zip_name = zip_name + '-MacOS-M-Series' + '.zip'
        else:
            zip_name = zip_name + '-MacOS-Intel' + '.zip'
        # Special zip command for macos to keep the complex directory metadata intact
        zip_cli_command = 'cd dist/; ditto -c -k --sequesterRsrc --keepParent ' + app_name + '.app ' + zip_name
    
    os.system(zip_cli_command)
    return zip_name

if __name__ == '__main__':
    apple_code_signing_key = None
    if len(sys.argv) > 1:
        apple_code_signing_key = sys.argv[1]  # python3 build.py "Developer ID Application: ... (...)"
        print("apple_code_signing_key: ", apple_code_signing_key)
    elif len(sys.argv) == 1 and platform.system() == 'Darwin':
        input("Are you sure you don't want to sign your code? ")
    
    build(apple_code_signing_key) 