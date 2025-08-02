"""
Module 2: Model Parameters and Configuration
===========================================
Contains all model parameters, patient profiles, and configuration settings.
Easily adjustable for fine-tuning model behavior.
"""

import numpy as np
from cancer_model.core.fractional_math import initialize_fractional_calculators


class ModelParameters:
    """Central parameter management class for easy fine-tuning"""
    
    def __init__(self, patient_profile=None):
        self.base_params = self._get_base_parameters()
        self.patient_profile = patient_profile
        self.params = self._apply_patient_profile()
        self.fractional_calculators = initialize_fractional_calculators(self.params['alpha'])
    
    def _get_base_parameters(self):
        """Base parameters - modify these for fine-tuning"""
        return {
            # === FRACTIONAL ORDER ===
            'alpha': 0.93,        # Fractional order (0.8-0.99 range)
            
            # === TUMOR DYNAMICS ===
            'K': 1000,            # Carrying capacity
            'lambda1': 0.003,     # Growth rate of sensitive cells
            'lambda2': 0.002,     # Growth rate of partially resistant cells
            'lambda_R1': 0.006,   # Type 1 resistant cell growth
            'lambda_R2': 0.005,   # Type 2 resistant cell growth
            
            # === IMMUNE SYSTEM ===
            'beta1': 0.005,       # Cytotoxic immune killing rate
            'beta2': 0.001,       # Regulatory immune suppression
            'phi1': 0.1,          # Baseline cytotoxic immune production
            'phi2': 0.001,        # Tumor-induced immune recruitment
            'phi3': 0.0003,       # Regulatory immune recruitment
            'delta_I': 0.04,      # Immune cell death rate
            
            # === RESISTANCE MECHANISMS ===
            'immune_resist_factor1': 0.10,  # Type 1 immune resistance
            'immune_resist_factor2': 0.05,  # Type 2 immune resistance
            'omega_R1': 0.004,    # Type 1 resistance development rate
            'omega_R2': 0.003,    # Type 2 resistance development rate
            'resistance_floor': 0.01,       # Minimum resistance level
            
            # === METASTASIS ===
            'gamma': 0.0001,      # Metastasis formation rate
            'delta_P': 0.01,      # Metastasis death rate
            
            # === ANGIOGENESIS ===
            'alpha_A': 0.01,      # Angiogenesis stimulation rate
            'delta_A': 0.1,       # Angiogenesis factor degradation
            
            # === CELL STATE TRANSITIONS ===
            'kappa_Q': 0.001,     # Rate of entering quiescence
            'lambda_Q': 0.0005,   # Rate of leaving quiescence
            'kappa_S': 0.0005,    # Senescence induction rate
            'delta_S': 0.005,     # Senescent cell death rate
            
            # === THERAPY EFFECTIVENESS ===
            'etaE': 0.01,         # Hormone therapy effectiveness
            'etaH': 0.01,         # HER2 therapy effectiveness
            'etaC': 0.01,         # Chemotherapy effectiveness
            
            # === TREATMENT RESISTANCE EFFECTS ===
            'immuno_resist_boost': 0.5,
            'continuous_resist_dev': 2.0,
            'adaptive_resist_dev': 1.2,
            
            # === PHARMACOKINETICS ===
            'absorption_rate': 0.5,       # Drug absorption rate
            'elimination_rate': 0.1,      # Drug elimination rate
            'distribution_vol': 70.0,     # Distribution volume (L)
            'bioavailability': 0.85,      # Bioavailability fraction
            'max_drug_effect': 1.0,       # Maximum drug effect
            'EC50': 0.3,                  # Concentration for half-maximal effect
            'hill_coef': 1.5,             # Hill coefficient for drug effect
            
            # === CIRCADIAN RHYTHM ===
            'circadian_amplitude': 0.2,   # Amplitude of circadian oscillation
            'circadian_phase': 0.0,       # Phase shift of circadian rhythm
            'circadian_period': 24.0,     # Period of circadian rhythm (hours)
            
            # === GENETIC/EPIGENETIC ===
            'mutation_rate': 0.0001,      # Base mutation rate (TUNE THIS)
            'epigenetic_silencing': 0.002, # Epigenetic silencing rate
            'genetic_instability': 1.0,   # Genetic instability factor
            
            # === MICROENVIRONMENT ===
            'hypoxia_threshold': 0.3,     # Tumor size where hypoxia starts
            'acidosis_factor': 0.01,      # Acidosis impact factor
            'metabolic_switch_rate': 0.02, # Rate of switching to glycolysis
            'microenv_stress_factor': 1.0, # Microenvironmental stress factor
            
            # === TREATMENT SCHEDULING ===
            'treatment_cycle_period': 21, # Days in treatment cycle
            'treatment_active_days': 7,   # Active treatment days per cycle
            'rest_period_days': 14,       # Rest days (no treatment) per cycle
            'treatment_intensity': 1.0,   # Treatment intensity multiplier
            
            # === PATIENT FACTORS ===
            'age_factor': 1.0,            # Age impact on treatment response
            'performance_status': 1.0,    # Patient performance status
            'bmi_factor': 1.0,            # BMI impact factor
            'prior_treatment_factor': 1.0, # Prior treatment impact
            'liver_function': 1.0,        # Liver function impact on drug metabolism
            'kidney_function': 1.0,       # Kidney function impact on drug clearance
            'immune_status': 1.0,         # Baseline immune status
            
            # === DEFAULT CONTROLS ===
            'uE': 0.0,            # Hormone therapy control
            'uH': 0.0,            # HER2 therapy control
            'uC': 0.0,            # Chemotherapy control
            'uI': 0.0,            # Immunotherapy control
        }
    
    def _apply_patient_profile(self):
        """Apply patient-specific parameter modifications"""
        params = self.base_params.copy()
        
        if self.patient_profile:
            for key, value in self.patient_profile.items():
                if key in params:
                    params[key] = value
        
        return params
    
    def get_parameter(self, key):
        """Get a specific parameter value"""
        return self.params.get(key, None)
    
    def set_parameter(self, key, value):
        """Set a specific parameter value for fine-tuning"""
        if key in self.params:
            self.params[key] = value
            print(f"Parameter '{key}' updated to {value}")
        else:
            print(f"Warning: Parameter '{key}' not found in model parameters")
    
    def get_all_parameters(self):
        """Get all parameters as dictionary"""
        return self.params.copy()
    
    def update_parameters(self, param_dict):
        """Update multiple parameters at once"""
        for key, value in param_dict.items():
            self.set_parameter(key, value)


class PatientProfiles:
    """Patient profile definitions for personalized medicine modeling"""
    
    @staticmethod
    def get_profile(profile_type='average'):
        """Get patient-specific parameter modifications"""
        profiles = {
            'average': {
                'age_factor': 1.0,
                'performance_status': 1.0,
                'bmi_factor': 1.0,
                'prior_treatment_factor': 1.0,
                'liver_function': 1.0,
                'kidney_function': 1.0,
                'immune_status': 1.0
            },
            'young': {
                'age_factor': 1.2,
                'performance_status': 1.2,
                'immune_status': 1.3,
                'liver_function': 1.1,
                'kidney_function': 1.1,
                'mutation_rate': 0.00008,
                'lambda1': 0.0035,  # Slightly higher growth rates
                'lambda2': 0.0025
            },
            'elderly': {
                'age_factor': 0.8,
                'performance_status': 0.8,
                'immune_status': 0.7,
                'liver_function': 0.9,
                'kidney_function': 0.85,
                'mutation_rate': 0.00015,
                'delta_I': 0.05,     # Faster immune decline
                'absorption_rate': 0.4  # Slower drug absorption
            },
            'compromised': {
                'performance_status': 0.7,
                'immune_status': 0.6,
                'liver_function': 0.7,
                'kidney_function': 0.7,
                'mutation_rate': 0.00015,
                'genetic_instability': 1.2,
                'beta1': 0.003,      # Reduced immune killing
                'phi1': 0.08         # Lower baseline immune production
            },
            'high_metabolism': {
                'absorption_rate': 0.6,
                'elimination_rate': 0.15,
                'liver_function': 1.2,
                'metabolic_switch_rate': 0.03
            },
            'low_metabolism': {
                'absorption_rate': 0.4,
                'elimination_rate': 0.08,
                'liver_function': 0.85,
                'metabolic_switch_rate': 0.015
            },
            'prior_treatment': {
                'prior_treatment_factor': 1.3,
                'resistance_floor': 0.02,
                'omega_R1': 0.006,   # Higher resistance development
                'omega_R2': 0.004,
                'immune_status': 0.9  # Slightly compromised immune system
            }
        }
        
        if profile_type in profiles:
            return profiles[profile_type]
        else:
            print(f"Warning: Profile '{profile_type}' not found. Using 'average' profile.")
            return profiles['average']


class InitialConditions:
    """Define initial conditions for all model compartments"""
    
    @staticmethod
    def get_standard_conditions():
        """Standard initial conditions"""
        return np.array([
            190,    # N1: Sensitive cells
            10,     # N2: Partially resistant cells
            40,     # I1: Cytotoxic immune cells
            10,     # I2: Regulatory immune cells
            0.1,    # P: Metastatic potential
            1,      # A: Angiogenesis factor
            0.1,    # Q: Quiescent cells
            1.0,    # R1: Type 1 resistant cells
            1.0,    # R2: Type 2 resistant cells
            0.1,    # S: Senescent cells
            0.0,    # D: Drug concentration
            0.0,    # Dm: Metabolized drug
            1.0,    # G: Genetic stability
            1.0,    # M: Metabolism status
            0.0     # H: Hypoxia level
        ])
    
    @staticmethod
    def get_conditions_for_profile(profile_type='average'):
        """Get initial conditions adjusted for patient profile"""
        base_conditions = InitialConditions.get_standard_conditions()
        
        # Adjust based on patient profile
        if profile_type == 'young':
            base_conditions[0] *= 1.1  # Slightly higher initial tumor burden
            base_conditions[2] *= 1.2  # Higher initial immune response
        elif profile_type == 'elderly':
            base_conditions[2] *= 0.8  # Lower initial immune response
            base_conditions[3] *= 1.1  # Slightly higher regulatory immune cells
        elif profile_type == 'compromised':
            base_conditions[2] *= 0.7  # Much lower immune response
            base_conditions[12] *= 0.95  # Slightly lower genetic stability
        
        return base_conditions


# === FINE-TUNING PRESETS ===
class FineTuningPresets:
    """Predefined parameter sets for different model behaviors"""
    
    @staticmethod
    def high_resistance_scenario():
        """Parameters for scenarios with high resistance development"""
        return {
            'omega_R1': 0.008,        # Double resistance development
            'omega_R2': 0.006,
            'resistance_floor': 0.05,  # Higher baseline resistance
            'mutation_rate': 0.0003,   # Triple mutation rate
            'genetic_instability': 1.5
        }
    
    @staticmethod
    def strong_immune_response():
        """Parameters for enhanced immune response"""
        return {
            'beta1': 0.008,           # Higher immune killing
            'phi1': 0.15,             # Higher baseline immune production
            'phi2': 0.002,            # Enhanced tumor-induced recruitment
            'delta_I': 0.03           # Slower immune decline
        }
    
    @staticmethod
    def aggressive_tumor():
        """Parameters for aggressive tumor growth"""
        return {
            'lambda1': 0.005,         # Higher growth rates
            'lambda2': 0.004,
            'lambda_R1': 0.008,
            'lambda_R2': 0.007,
            'gamma': 0.0003,          # Higher metastasis rate
            'K': 1200                 # Higher carrying capacity
        }
    
    @staticmethod
    def drug_sensitive():
        """Parameters for high drug sensitivity"""
        return {
            'etaE': 0.02,             # Double therapy effectiveness
            'etaH': 0.02,
            'etaC': 0.02,
            'EC50': 0.2,              # Lower concentration needed
            'max_drug_effect': 1.2    # Higher maximum effect
        }
    
    @staticmethod
    def realistic_resistance():
        """More realistic resistance development parameters"""
        return {
            'omega_R1': 0.01,         # Higher resistance development
            'omega_R2': 0.008,
            'resistance_floor': 0.1,   # Much higher baseline
            'mutation_rate': 0.0005,   # 5x higher mutation rate
            'immune_resist_factor1': 0.3,  # Higher immune resistance
            'immune_resist_factor2': 0.2
        }
