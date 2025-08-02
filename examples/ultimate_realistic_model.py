#!/usr/bin/env python3
"""
Ultimate Realistic Cancer Model - Final Version
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cancer_model import CancerModelRunner, PatientProfiles
from cancer_model.core.model_parameters import ModelParameters

class RealisticCancerModelRunner(CancerModelRunner):
    """Enhanced cancer model runner with realistic resistance parameters"""
    
    def __init__(self, resistance_level='medium', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resistance_level = resistance_level
        
        # Define realistic parameter sets
        self.resistance_params = {
            'low': {
                'omega_R1': 0.2,
                'omega_R2': 0.15,
                'etaE': 0.05,
                'etaH': 0.05,
                'etaC': 0.05
            },
            'medium': {
                'omega_R1': 1.0,
                'omega_R2': 0.8, 
                'etaE': 0.1,
                'etaH': 0.1,
                'etaC': 0.1
            },
            'high': {
                'omega_R1': 2.0,
                'omega_R2': 1.6,
                'etaE': 0.2,
                'etaH': 0.2,
                'etaC': 0.2
            },
            'extreme': {
                'omega_R1': 5.0,
                'omega_R2': 4.0,
                'etaE': 0.5,
                'etaH': 0.5,
                'etaC': 0.5
            }
        }
        
        print(f'Initialized realistic cancer model with {resistance_level} resistance level')
    
    def _force_realistic_parameters(self, patient_profile_name):
        """Force realistic resistance parameters"""
        
        patient_profile = PatientProfiles.get_profile(patient_profile_name)
        model_params = ModelParameters(patient_profile)
        
        # FORCE realistic parameters
        resistance_config = self.resistance_params[self.resistance_level]
        for param, value in resistance_config.items():
            model_params.params[param] = value
        
        return model_params
    
    def run_single_simulation(self, patient_profile_name, protocol_name, 
                            simulation_days=500, use_circadian=True):
        """Run simulation with forced realistic parameters"""
        
        print(f'Running realistic simulation: {patient_profile_name} + {protocol_name}')
        
        from cancer_model.core.cancer_model_core import CancerModel
        from cancer_model.core.pharmacokinetics import PharmacokineticModel, CircadianRhythm
        from cancer_model.protocols.treatment_protocols import TreatmentProtocols
        from cancer_model.core.fractional_math import safe_solve_ivp
        from cancer_model.core.model_parameters import InitialConditions
        
        # Force realistic parameters
        model_params = self._force_realistic_parameters(patient_profile_name)
        params = model_params.get_all_parameters()
        
        omega_r1 = params['omega_R1']
        omega_r2 = params['omega_R2'] 
        eta_e = params['etaE']
        print(f'  Using: omega_R1={omega_r1}, omega_R2={omega_r2}, etaE={eta_e}')
        
        # Create model components
        pk_model = PharmacokineticModel(params)
        circadian_model = CircadianRhythm(params)
        cancer_model = CancerModel(params, pk_model, circadian_model)
        
        # Get treatment protocol
        treatment_protocols = TreatmentProtocols()
        protocol = treatment_protocols.get_protocol(protocol_name, PatientProfiles.get_profile(patient_profile_name))
        
        # Setup simulation
        t_span = [0, simulation_days]
        t_eval = np.linspace(0, simulation_days, simulation_days + 1)
        initial_conditions = InitialConditions.get_conditions_for_profile(patient_profile_name)
        temp_func = protocol.get('temperature', lambda t: 37.0)
        
        def model_function(t, y):
            current_temp = temp_func(t)
            return cancer_model.enhanced_temperature_cancer_model(
                t, y, protocol['drugs'], current_temp, use_circadian
            )
        
        # Run simulation
        result = safe_solve_ivp(model_function, t_span, initial_conditions, 'RK45', t_eval)
        
        if result.success:
            metrics = self._calculate_realistic_metrics(result, protocol)
            time_series = self._extract_time_series(result)
            
            resistance = metrics['final_resistance_fraction']
            efficacy = metrics['treatment_efficacy_score']
            print(f'  Results: Resistance={resistance:.1f}%, Efficacy={efficacy:.2f}')
            
            return {
                'success': True,
                'metrics': metrics,
                'time_series': time_series,
                'protocol': protocol,
                'patient_profile': PatientProfiles.get_profile(patient_profile_name),
                'model_params': params,
                'resistance_level': self.resistance_level
            }
        else:
            return {
                'success': False,
                'error_message': result.message
            }
    
    def _calculate_realistic_metrics(self, result, protocol):
        """Calculate metrics with corrected resistance calculation"""
        
        # Extract state variables
        N1 = result.y[0]
        N2 = result.y[1]
        Q = result.y[6]
        R1 = result.y[7]
        R2 = result.y[8]
        S = result.y[9]
        D = result.y[10]
        
        # Calculate tumor dynamics
        total_tumor = N1 + N2 + Q + R1 + R2 + S
        total_resistant = R1 + R2
        resistance_fraction = (total_resistant / total_tumor * 100)
        
        # Basic metrics
        initial_burden = total_tumor[0]
        final_burden = total_tumor[-1]
        percent_reduction = 100 * (1 - final_burden / initial_burden)
        
        # Efficacy score (accounts for resistance)
        final_resistance_pct = resistance_fraction[-1]
        treatment_efficacy = percent_reduction / (1 + final_resistance_pct/50)  # Adjusted formula
        
        return {
            'initial_burden': initial_burden,
            'final_burden': final_burden, 
            'percent_reduction': percent_reduction,
            'final_resistance_fraction': final_resistance_pct,
            'treatment_efficacy_score': treatment_efficacy,
            'max_drug_concentration': np.max(D),
            'resistance_growth': total_resistant[-1] - total_resistant[0],
            'immune_activation': 1.0,  # Simplified for now
            'genetic_instability': 0.1,  # Simplified for now
        }
    
    def _extract_time_series(self, result):
        """Extract time series data"""
        
        N1, N2, I1, I2, P, A, Q, R1, R2, S, D, Dm, G, M, H = result.y
        total_tumor = N1 + N2 + Q + R1 + R2 + S
        resistance_fraction = (R1 + R2) / total_tumor * 100
        
        return {
            'time': result.t,
            'total_tumor': total_tumor,
            'sensitive_cells': N1,
            'resistant_type1': R1,
            'resistant_type2': R2,
            'resistance_fraction': resistance_fraction,
            'drug_concentration': D,
            'immune_cells': I1
        }

def test_realistic_model():
    print('TESTING ULTIMATE REALISTIC CANCER MODEL')
    print('=' * 60)
    
    # Test different resistance levels
    resistance_levels = ['low', 'medium', 'high']
    
    for level in resistance_levels:
        print(f'\nTesting {level.upper()} resistance level:')
        
        runner = RealisticCancerModelRunner(resistance_level=level, output_dir=f'results/realistic_{level}')
        result = runner.run_single_simulation('average', 'standard', 100)
        
        if result['success']:
            resistance = result['metrics']['final_resistance_fraction']
            efficacy = result['metrics']['treatment_efficacy_score']
            reduction = result['metrics']['percent_reduction']
            
            print(f'  Final resistance: {resistance:.1f}%')
            print(f'  Tumor reduction: {reduction:.1f}%')
            print(f'  Efficacy score: {efficacy:.2f}')
            
            if 5 <= resistance <= 50:
                print(f'   Realistic resistance achieved!')
            else:
                print(f'   May need adjustment')
        else:
            print(f'   Simulation failed')
    
    print(f'\n Ultimate realistic cancer model testing complete!')

if __name__ == '__main__':
    test_realistic_model()
