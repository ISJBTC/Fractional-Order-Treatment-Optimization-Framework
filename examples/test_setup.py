#!/usr/bin/env python3
"""
Setup Test Example
==================
Basic test to verify the project setup works correctly.
"""

import sys
import os
from pathlib import Path

def test_setup():
    """Test that the project setup is correct."""
    print("TESTING PROJECT SETUP")
    print("=" * 40)
    
    # Test directory structure
    print("\nChecking directory structure...")
    
    expected_dirs = [
        "cancer_model",
        "cancer_model/core", 
        "cancer_model/protocols",
        "cancer_model/simulation",
        "cancer_model/visualization",
        "cancer_model/utils",
        "examples",
        "tests",
        "results"
    ]
    
    all_dirs_exist = True
    for dir_path in expected_dirs:
        if Path(dir_path).exists():
            print(f"   OK: {dir_path}")
        else:
            print(f"   MISSING: {dir_path}")
            all_dirs_exist = False
    
    # Test Python packages
    print("\nChecking Python packages...")
    
    required_packages = ["numpy", "matplotlib", "pandas", "seaborn", "scipy", "sklearn"]
    packages_ok = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   OK: {package}")
        except ImportError:
            print(f"   MISSING: {package} (run: pip install -r requirements.txt)")
            packages_ok = False
    
    # Test files
    print("\nChecking configuration files...")
    
    config_files = ["requirements.txt", "setup.py", ".vscode/settings.json", ".gitignore"]
    files_ok = True
    
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"   OK: {file_path}")
        else:
            print(f"   MISSING: {file_path}")
            files_ok = False
    
    # Summary
    print("\nSETUP STATUS")
    print("-" * 20)
    print(f"Directory structure: {'OK' if all_dirs_exist else 'ISSUES'}")
    print(f"Python packages: {'OK' if packages_ok else 'ISSUES'}")
    print(f"Configuration files: {'OK' if files_ok else 'ISSUES'}")
    
    if all_dirs_exist and packages_ok and files_ok:
        print("\nPROJECT SETUP SUCCESSFUL!")
        print("\nNext steps:")
        print("   1. Copy module files into cancer_model/ subdirectories")
        print("   2. Run: pip install -e .")
        print("   3. Start coding your cancer model!")
    else:
        print("\nSetup has issues - please check the errors above")
    
    return all_dirs_exist and packages_ok and files_ok

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
