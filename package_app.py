import PyInstaller.__main__
import sys
import os
from pathlib import Path
import shutil
import plistlib
import platform

def create_info_plist(app_path):
    """Create the Info.plist file for the app bundle"""
    info = {
        'CFBundleName': 'FLUX Generator',
        'CFBundleDisplayName': 'FLUX Generator',
        'CFBundleIdentifier': 'com.mlx.fluxgenerator',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': '????',
        'CFBundleExecutable': 'FLUX Generator',
        'CFBundleIconFile': 'AppIcon.icns',
        'LSMinimumSystemVersion': '11.0.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
        'LSApplicationCategoryType': 'public.app-category.graphics-design',
        # Specify that this app only runs on Apple Silicon
        'LSArchitecturePriority': ['arm64'],
        'LSRequiresNativeExecution': True,
    }
    
    plist_path = app_path / 'Contents' / 'Info.plist'
    with open(plist_path, 'wb') as f:
        plistlib.dump(info, f)

def check_system_compatibility():
    """Check if we're on an Apple Silicon Mac"""
    if sys.platform != 'darwin':
        raise SystemError("This app can only be built on macOS")
    
    if platform.machine() != 'arm64':
        raise SystemError("This app can only be built on Apple Silicon (M1/M2) Macs")

def create_app_bundle():
    # Verify we're on the right system
    check_system_compatibility()
    
    app_name = "FLUX Generator"
    bundle_name = f"{app_name}.app"
    
    # Define paths
    root_dir = Path(__file__).parent
    dist_dir = root_dir / 'dist'
    bundle_path = dist_dir / bundle_name
    contents_path = bundle_path / 'Contents'
    macos_path = contents_path / 'MacOS'
    resources_path = contents_path / 'Resources'
    frameworks_path = contents_path / 'Frameworks'
    
    # Clean previous build
    if bundle_path.exists():
        shutil.rmtree(bundle_path)
    
    # Create directory structure
    for path in [macos_path, resources_path, frameworks_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Run PyInstaller with Apple Silicon specific options
    PyInstaller.__main__.run([
        'standalone_app.py',
        '--name=%s' % app_name,
        '--windowed',
        '--noconfirm',
        '--clean',
        '--target-arch=arm64',
        '--hidden-import=mlx.core',
        '--hidden-import=mlx.nn',
        '--hidden-import=huggingface_hub',
        '--hidden-import=gradio',
        '--collect-data=mlx',
        '--collect-data=huggingface_hub',
        '--collect-data=gradio',
        f'--distpath={macos_path}',
        f'--workpath={dist_dir}/build',
        f'--specpath={dist_dir}',
    ])
    
    # Copy MLX and other dependencies
    python_lib_path = Path(sys.executable).parent / 'lib'
    for lib_version in python_lib_path.glob('python3.*'):
        mlx_path = lib_version / 'site-packages' / 'mlx'
        if mlx_path.exists():
            shutil.copytree(mlx_path, frameworks_path / 'mlx', dirs_exist_ok=True)
            break
    
    # Copy icon if it exists
    icon_path = root_dir / 'AppIcon.icns'
    if icon_path.exists():
        shutil.copy2(icon_path, resources_path / 'AppIcon.icns')
    
    # Create Info.plist
    create_info_plist(bundle_path)
    
    # Create models directory in app bundle
    models_path = resources_path / 'models'
    models_path.mkdir(exist_ok=True)
    
    # Create a DMG for distribution
    create_dmg(dist_dir, bundle_path)
    
    print("\nBuild complete!")
    print(f"App bundle created at: {bundle_path}")
    print(f"DMG installer created at: {dist_dir / bundle_name.replace('.app', '.dmg')}")
    print("\nNote: This app requires an Apple Silicon Mac (M1/M2) to run.")

def create_dmg(dist_dir, bundle_path):
    """Create a DMG file for distribution"""
    dmg_name = bundle_path.name.replace('.app', '.dmg')
    os.system(f'hdiutil create -volname "FLUX Generator" -srcfolder "{bundle_path}" -ov -format UDZO "{dist_dir / dmg_name}"')

if __name__ == "__main__":
    try:
        create_app_bundle()
    except SystemError as e:
        print(f"Error: {e}")
        print("This application can only be built on Apple Silicon (M1/M2) Macs.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during build: {e}")
        sys.exit(1) 