#!/usr/bin/env python3
"""
Environment check script for the Competitive Pricing Strategy System
This script verifies that the project environment is correctly set up
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

# Required directories
REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/engineered",
    "data/examples",
    "models/category_models",
    "models/improved",
    "visualizations",
    "logs",
    "pricing_strategies",
]

# Required Python packages
REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "xgboost",
    "streamlit",
    "joblib",
    "tqdm",
    "plotly",
]

def check_directories():
    """Check if required directories exist"""
    print("Checking required directories...")
    missing_dirs = []
    
    for directory in REQUIRED_DIRS:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
            
    if missing_dirs:
        print("❌ Missing directories:")
        for directory in missing_dirs:
            print(f"  - {directory}")
        print("\nCreating missing directories...")
        for directory in missing_dirs:
            os.makedirs(directory, exist_ok=True)
        print("✅ Created missing directories")
    else:
        print("✅ All required directories exist")
    
    return len(missing_dirs) == 0

def check_packages():
    """Check if required Python packages are installed"""
    print("\nChecking required Python packages...")
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nYou can install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def check_python_version():
    """Check if Python version is compatible"""
    print("\nChecking Python version...")
    required_version = (3, 10)
    current_version = sys.version_info
    
    if current_version.major < required_version[0] or \
       (current_version.major == required_version[0] and current_version.minor < required_version[1]):
        print(f"❌ Python version {required_version[0]}.{required_version[1]} or higher is required")
        print(f"  Current version: {current_version.major}.{current_version.minor}.{current_version.micro}")
        return False
    else:
        print(f"✅ Python version {current_version.major}.{current_version.minor}.{current_version.micro} is compatible")
        return True

def check_file_permissions():
    """Check if we have proper permissions for the project directories"""
    print("\nChecking file permissions...")
    permission_errors = []
    
    for directory in REQUIRED_DIRS:
        path = Path(directory)
        if path.exists():
            try:
                # Try creating a temporary file
                temp_file = path / ".permission_check"
                temp_file.touch()
                temp_file.unlink()
            except (PermissionError, OSError):
                permission_errors.append(directory)
    
    if permission_errors:
        print("❌ Permission issues with directories:")
        for directory in permission_errors:
            print(f"  - {directory}")
        return False
    else:
        print("✅ File permissions are correct")
        return True

def check_application_startup():
    """Check if the application starts up correctly"""
    print("\nChecking application startup...")
    streamlit_path = Path("streamlit_app.py")
    
    if not streamlit_path.exists():
        print("❌ Streamlit application file not found")
        return False
    
    try:
        # Just try to compile the python file to check for syntax errors
        with open(streamlit_path, 'r') as f:
            compile(f.read(), 'streamlit_app.py', 'exec')
        print("✅ Streamlit application file looks good")
        return True
    except Exception as e:
        print(f"❌ Error in Streamlit application file: {e}")
        return False

def main():
    """Main function"""
    print("=" * 50)
    print(" Competitive Pricing Strategy - Environment Check")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check Python version
    all_checks_passed &= check_python_version()
    
    # Check required directories
    all_checks_passed &= check_directories()
    
    # Check required packages
    all_checks_passed &= check_packages()
    
    # Check file permissions
    all_checks_passed &= check_file_permissions()
    
    # Check application startup
    all_checks_passed &= check_application_startup()
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✅ All checks passed! Your environment is ready.")
        print("\nYou can run the application with:")
        print("  streamlit run streamlit_app.py")
    else:
        print("❌ Some checks failed. Please fix the issues above before running the application.")
    print("=" * 50)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 