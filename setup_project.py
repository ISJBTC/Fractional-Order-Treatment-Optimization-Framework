#!/usr/bin/env python3
"""
Cancer Model Project Setup Script - Windows Compatible
======================================================
Creates the complete project structure without Unicode issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_main_init():
    """Create main package __init__.py with imports."""
    print("Creating main package __init__.py...")
    
    init_content = '''"""
Cancer Model: Modular Cancer Treatment Modeling System
=====================================================

A comprehensive framework for modeling cancer dynamics, treatment protocols, 
and patient-specific responses using fractional calculus and advanced 
mathematical modeling techniques.

Quick Start:
-----------
>>> from cancer_model import CancerModelRunner
>>> runner = CancerModelRunner()
>>> results = runner.run_basic_analysis()
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Cancer Model Team"
__email__ = "contact@cancermodel.org"

print(f"Cancer Model v{__version__} - Setup Complete!")
print("Next: Copy module files and run 'pip install -e .'")
'''
    
    with open("cancer_model/__init__.py", "w", encoding='utf-8') as f:
        f.write(init_content)
    print("   Created: cancer_model/__init__.py")

def create_test_example():
    """Create a simple test example."""
    print("Creating test example...")
    
    test_content = '''#!/usr/bin/env python3
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
    print("\\nChecking directory structure...")
    
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
    print("\\nChecking Python packages...")
    
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
    print("\\nChecking configuration files...")
    
    config_files = ["requirements.txt", "setup.py", ".vscode/settings.json", ".gitignore"]
    files_ok = True
    
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"   OK: {file_path}")
        else:
            print(f"   MISSING: {file_path}")
            files_ok = False
    
    # Summary
    print("\\nSETUP STATUS")
    print("-" * 20)
    print(f"Directory structure: {'OK' if all_dirs_exist else 'ISSUES'}")
    print(f"Python packages: {'OK' if packages_ok else 'ISSUES'}")
    print(f"Configuration files: {'OK' if files_ok else 'ISSUES'}")
    
    if all_dirs_exist and packages_ok and files_ok:
        print("\\nPROJECT SETUP SUCCESSFUL!")
        print("\\nNext steps:")
        print("   1. Copy module files into cancer_model/ subdirectories")
        print("   2. Run: pip install -e .")
        print("   3. Start coding your cancer model!")
    else:
        print("\\nSetup has issues - please check the errors above")
    
    return all_dirs_exist and packages_ok and files_ok

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
'''
    
    with open("examples/test_setup.py", "w", encoding='utf-8') as f:
        f.write(test_content)
    print("   Created: examples/test_setup.py")

def create_readme():
    """Create README.md file."""
    print("Creating README.md...")
    
    readme_content = """# Cancer Model: Modular Treatment Modeling System

A comprehensive, modular system for modeling cancer treatment dynamics using fractional calculus and advanced mathematical techniques.

## Quick Start

This project has been set up with the following structure:

```
cancer_model_project/
├── cancer_model/           # Main package
│   ├── core/              # Core mathematical models  
│   ├── protocols/         # Treatment protocols
│   ├── simulation/        # Simulation runners
│   ├── visualization/     # Plotting tools
│   └── utils/            # Utilities and main runners
├── examples/             # Ready-to-run examples
├── tests/               # Unit tests
├── results/             # Generated outputs
└── docs/               # Documentation
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install the package in development mode:**
```bash
pip install -e .
```

3. **Test the setup:**
```bash
python examples/test_setup.py
```

## Next Steps

1. Copy the module files into their respective directories
2. Complete the installation with `pip install -e .`
3. Run the examples to verify everything works
4. Start developing your cancer model analysis!

## Support

If you encounter any issues during setup, check:
- Python version (3.8+ required)
- Virtual environment activation
- All dependencies installed correctly

Created by Cancer Model Setup Script v1.0
"""
    
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    print("   Created: README.md")

def install_packages():
    """Install required Python packages."""
    print("Installing Python packages...")
    print("This may take a few minutes...")
    
    try:
        # Install main packages
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "numpy", "scipy", "matplotlib", "seaborn", "pandas", 
                       "scikit-learn", "tqdm", "plotly"], check=True)
        
        # Install development packages
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "pytest", "black", "flake8", "mypy", 
                       "jupyter", "ipykernel"], check=True)
        
        print("Packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Package installation encountered issues: {e}")
        print("You can install them manually with: pip install -r requirements.txt")
        return False

def main():
    """Complete the setup that was partially done."""
    print("COMPLETING CANCER MODEL PROJECT SETUP")
    print("=" * 50)
    
    try:
        # Complete the remaining setup steps
        create_main_init()
        create_test_example()
        create_readme()
        
        # Install packages
        packages_installed = install_packages()
        
        # Final summary
        print("\nSETUP COMPLETE!")
        print("=" * 50)
        
        print("\nCreated:")
        print("   • Complete directory structure")
        print("   • Configuration files")
        print("   • VS Code settings")
        print("   • Test example")
        
        if packages_installed:
            print("   • Python packages installed")
        
        print("\nNext Steps:")
        print("   1. Copy module files into cancer_model/ subdirectories")
        print("   2. Run: pip install -e .")
        print("   3. Test: python examples/test_setup.py")
        print("   4. Open in VS Code and start coding!")
        
        print("\nReady to start your cancer modeling project!")
        
    except Exception as e:
        print(f"\nSetup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()