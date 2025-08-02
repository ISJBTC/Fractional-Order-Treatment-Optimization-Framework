# Cancer Model Project Setup Script for Windows PowerShell
# Run this script in your activated virtual environment

Write-Host "🔬 Setting up Cancer Model Project..." -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green

# Check if virtual environment is activated
if ($env:VIRTUAL_ENV -eq $null) {
    Write-Host "⚠️  Warning: Virtual environment not detected" -ForegroundColor Yellow
    Write-Host "Please activate your virtual environment first:" -ForegroundColor Yellow
    Write-Host "cancer_model_env\Scripts\activate" -ForegroundColor Yellow
    Read-Host "Press Enter to continue anyway or Ctrl+C to exit"
}

# Create directory structure
Write-Host "`n📁 Creating directory structure..." -ForegroundColor Cyan

$directories = @(
    "cancer_model",
    "cancer_model\core",
    "cancer_model\protocols", 
    "cancer_model\simulation",
    "cancer_model\visualization",
    "cancer_model\utils",
    "examples",
    "tests", 
    "notebooks",
    "results",
    "results\figures",
    "results\data", 
    "results\reports",
    "docs",
    ".vscode"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   Created: $dir" -ForegroundColor Gray
    }
}

# Create __init__.py files
Write-Host "`n📄 Creating __init__.py files..." -ForegroundColor Cyan

$initFiles = @(
    "cancer_model\__init__.py",
    "cancer_model\core\__init__.py",
    "cancer_model\protocols\__init__.py",
    "cancer_model\simulation\__init__.py",
    "cancer_model\visualization\__init__.py", 
    "cancer_model\utils\__init__.py",
    "tests\__init__.py"
)

foreach ($file in $initFiles) {
    if (!(Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "   Created: $file" -ForegroundColor Gray
    }
}

# Create requirements.txt
Write-Host "`n📦 Creating requirements.txt..." -ForegroundColor Cyan
@"
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
plotly>=5.0.0

# Development dependencies
pytest>=6.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.910
jupyter>=1.0.0
ipykernel>=6.0.0
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8

# Create setup.py
Write-Host "📦 Creating setup.py..." -ForegroundColor Cyan
@"
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cancer-model",
    version="1.0.0",
    author="Cancer Model Team",
    author_email="contact@cancermodel.org",
    description="Modular Cancer Treatment Modeling System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cancer-model=cancer_model.utils.main_runner:main",
        ],
    },
)
"@ | Out-File -FilePath "setup.py" -Encoding UTF8

# Create .vscode/settings.json
Write-Host "🔧 Creating VS Code settings..." -ForegroundColor Cyan
@"
{
    "python.defaultInterpreterPath": "./cancer_model_env/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": false,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.testing.autoTestDiscoverOnSaveEnabled": true,
    "files.associations": {
        "*.py": "python"
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/cancer_model_env": true,
        "**/.mypy_cache": true
    },
    "python.analysis.extraPaths": [
        "./cancer_model"
    ]
}
"@ | Out-File -FilePath ".vscode\settings.json" -Encoding UTF8

# Create .vscode/launch.json
@"
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Basic Analysis",
            "type": "python",
            "request": "launch",
            "program": "`${workspaceFolder}/examples/basic_analysis.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "cwd": "`${workspaceFolder}",
            "env": {
                "PYTHONPATH": "`${workspaceFolder}"
            }
        },
        {
            "name": "Python: Optimization Example", 
            "type": "python",
            "request": "launch",
            "program": "`${workspaceFolder}/examples/optimization_example.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "cwd": "`${workspaceFolder}",
            "env": {
                "PYTHONPATH": "`${workspaceFolder}"
            }
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "`${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "cwd": "`${workspaceFolder}",
            "env": {
                "PYTHONPATH": "`${workspaceFolder}"
            }
        }
    ]
}
"@ | Out-File -FilePath ".vscode\launch.json" -Encoding UTF8

# Create .gitignore
Write-Host "📝 Creating .gitignore..." -ForegroundColor Cyan
@"
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*`$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
cancer_model_env/
venv/
env/

# VS Code
.vscode/settings.json.bak

# Results and outputs
results/
*.png
*.jpg
*.pdf
*.csv
*.xlsx

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/

# mypy
.mypy_cache/

# Coverage
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8

# Now create the main module files
Write-Host "`n🧬 Creating core module files..." -ForegroundColor Cyan

# Create cancer_model/__init__.py
@"
"""
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

# Import main classes for easy access
from .utils.main_runner import CancerModelRunner
from .simulation.simulation_runner import SimulationRunner
from .core.model_parameters import ModelParameters, PatientProfiles, FineTuningPresets
from .protocols.treatment_protocols import TreatmentProtocols, ProtocolOptimizer
from .visualization.visualization_module import VisualizationEngine

# Package metadata
__version__ = "1.0.0"
__author__ = "Cancer Model Team"
__email__ = "contact@cancermodel.org"

# Define what gets imported with "from cancer_model import *"
__all__ = [
    'CancerModelRunner',
    'SimulationRunner',
    'ModelParameters',
    'PatientProfiles', 
    'FineTuningPresets',
    'TreatmentProtocols',
    'ProtocolOptimizer',
    'VisualizationEngine',
    '__version__',
    '__author__',
    '__email__'
]

print(f"Cancer Model v{__version__} loaded successfully!")
"@ | Out-File -FilePath "cancer_model\__init__.py" -Encoding UTF8

# Create a simple example file to test the setup
Write-Host "📝 Creating test example..." -ForegroundColor Cyan
@"
#!/usr/bin/env python3
"""
Simple Test Example
==================
Basic test to verify the cancer model setup works.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test basic imports
        import numpy
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        print("✅ Basic scientific packages imported successfully")
        
        # Test cancer model imports (will fail until modules are created)
        # Uncomment these after creating the full modules:
        # from cancer_model import CancerModelRunner
        # print("✅ Cancer model imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_directories():
    """Test that directory structure was created correctly."""
    print("🗂️  Testing directory structure...")
    
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
    
    all_exist = True
    for dir_path in expected_dirs:
        if not (project_root / dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            all_exist = False
        else:
            print(f"✅ Found: {dir_path}")
    
    return all_exist

def main():
    print("🔬 CANCER MODEL SETUP TEST")
    print("=" * 40)
    
    # Test directory structure
    dirs_ok = test_directories()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n📋 SETUP STATUS")
    print("-" * 20)
    print(f"Directory structure: {'✅ OK' if dirs_ok else '❌ ISSUES'}")
    print(f"Package imports: {'✅ OK' if imports_ok else '❌ ISSUES'}")
    
    if dirs_ok and imports_ok:
        print("\n🎉 Setup test passed!")
        print("Next steps:")
        print("1. Copy the module files into their respective directories")
        print("2. Run: pip install -e .")
        print("3. Test with full examples")
    else:
        print("\n⚠️  Setup test revealed issues")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
"@ | Out-File -FilePath "examples\test_setup.py" -Encoding UTF8

# Create README.md
Write-Host "📖 Creating README.md..." -ForegroundColor Cyan
@"
# Cancer Model: Modular Treatment Modeling System

A comprehensive, modular system for modeling cancer treatment dynamics using fractional calculus and advanced mathematical techniques.

## 🚀 Quick Start

This project has been set up with the following structure:

``````
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
``````

## 📦 Installation

1. **Install dependencies:**
``````bash
pip install -r requirements.txt
``````

2. **Install the package in development mode:**
``````bash
pip install -e .
``````

3. **Test the setup:**
``````bash
python examples\test_setup.py
``````

## 🔧 Next Steps

1. Copy the module files into their respective directories
2. Complete the installation with `pip install -e .`
3. Run the examples to verify everything works
4. Start developing your cancer model analysis!

## 📞 Support

If you encounter any issues during setup, check:
- Python version (3.8+ required)
- Virtual environment activation
- All dependencies installed correctly

Created by Cancer Model Setup Script v1.0
"@ | Out-File -FilePath "README.md" -Encoding UTF8

# Install packages
Write-Host "`n📦 Installing Python packages..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

try {
    pip install numpy scipy matplotlib seaborn pandas scikit-learn tqdm plotly
    pip install pytest black flake8 mypy jupyter ipykernel
    Write-Host "✅ Packages installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Package installation encountered issues" -ForegroundColor Yellow
    Write-Host "You may need to install them manually:" -ForegroundColor Yellow
    Write-Host "pip install -r requirements.txt" -ForegroundColor Yellow
}

# Final summary
Write-Host "`n🎉 SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green

Write-Host "`n📁 Created directory structure with:" -ForegroundColor Cyan
Write-Host "   • Main package directories" -ForegroundColor Gray
Write-Host "   • Configuration files" -ForegroundColor Gray  
Write-Host "   • VS Code settings" -ForegroundColor Gray
Write-Host "   • Example and test structure" -ForegroundColor Gray

Write-Host "`n🔧 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Copy the module files into cancer_model/ subdirectories" -ForegroundColor Yellow
Write-Host "   2. Run: pip install -e ." -ForegroundColor Yellow
Write-Host "   3. Test setup: python examples\test_setup.py" -ForegroundColor Yellow
Write-Host "   4. Open in VS Code and start coding!" -ForegroundColor Yellow

Write-Host "`n🚀 Ready to start your cancer modeling project!" -ForegroundColor Green

# Keep window open
Write-Host "`nPress any key to continue..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")