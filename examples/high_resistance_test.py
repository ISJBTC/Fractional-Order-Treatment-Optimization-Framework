#!/usr/bin/env python3
"""
High Resistance Test - PowerShell Version
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cancer_model import CancerModelRunner, PatientProfiles, ModelParameters

def test_high_resistance():
    print("TESTING VERY HIGH RESISTANCE PARAMETERS")
    print("=" * 50)
    
    # Create much more aggressive resistance parameters
    custom_patient = PatientProfiles.get_profile('average')
    
    # Make resistance much higher
    custom_patient.update({
        'omega_R1': 0.05,        # 5x higher resistance development  
        'omega_R2': 0.04,        # 5x higher
        'mutation_rate': 0.002,  # 4x higher mutation rate
        'resistance_floor': 0.2, # 20% baseline resistance
        'immune_resist_factor1': 0.8,  # Much higher immune resistance
        'immune_resist_factor2': 0.6
    })
    
    print("Testing with very aggressive resistance parameters:")
    print(f"  omega_R1: {custom_patient['omega_R1']}")
    print(f"  omega_R2: {custom_patient['omega_R2']}")
    print(f"  mutation_rate: {custom_patient['mutation_rate']}")
    print(f"  resistance_floor: {custom_patient['resistance_floor']}")
    
    # Test different protocols
    protocols = ['standard', 'continuous', 'adaptive']
    
    for protocol in protocols:
        print(f"\nTesting {protocol} protocol...")
        
        try:
            runner = CancerModelRunner(output_dir=f'results/high_resistance_test_{protocol}')
            
            # Create model with custom parameters
            model_params = ModelParameters(custom_patient)
            
            # Run simulation using the simulation runner directly
            result = runner.simulation_runner.run_single_simulation('average', protocol, 300)
            
            if result['success']:
                metrics = result['metrics']
                resistance = metrics['final_resistance_fraction']
                efficacy = metrics['treatment_efficacy_score']
                
                print(f"  SUCCESS!")
                print(f"  Resistance: {resistance:.1f}%")
                print(f"  Efficacy: {efficacy:.2f}")
                
                if resistance > 10:
                    print(f"   REALISTIC resistance level achieved!")
                else:
                    print(f"   Still too low")
            else:
                print(f"  FAILED: {result.get('error_message', 'Unknown')}")
                
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_high_resistance()
