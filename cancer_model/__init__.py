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
from .core.cancer_model_core import CancerModel
from .core.pharmacokinetics import PharmacokineticModel, CircadianRhythm, DrugScheduling
from .core.fractional_math import EnhancedFractionalDerivative

# Package metadata
__version__ = "1.0.0"
__author__ = "Cancer Model Team"
__email__ = "contact@cancermodel.org"
__description__ = "Modular Cancer Treatment Modeling System"

# Define what gets imported with "from cancer_model import *"
__all__ = [
    # Main runner
    'CancerModelRunner',
    
    # Core components
    'SimulationRunner',
    'ModelParameters',
    'PatientProfiles', 
    'FineTuningPresets',
    'CancerModel',
    'EnhancedFractionalDerivative',
    
    # Treatment protocols
    'TreatmentProtocols',
    'ProtocolOptimizer',
    'DrugScheduling',
    
    # Pharmacokinetics
    'PharmacokineticModel',
    'CircadianRhythm',
    
    # Visualization
    'VisualizationEngine',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__description__'
]

print(f"Cancer Model v{__version__} loaded successfully!")