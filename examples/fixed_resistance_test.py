#!/usr/bin/env python3
"""
Fixed Resistance Model Test - Corrected
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cancer_model import CancerModelRunner, PatientProfiles, ModelParameters

def test_fixed_resistance():
    print('TESTING FIXED RESISTANCE DEVELOPMENT')
    print('=' * 50)
    
    # Create parameters that will actually develop resistance
    patient = PatientProfiles.get_profile('average')
    
    # MUCH more aggressive resistance parameters
    patient.update({
        'omega_R1': 0.5,         # 125x higher than original 0.004!
        'omega_R2': 0.4,         # 133x higher than original 0.003!
        'mutation_rate': 0.01,   # 100x higher
        'etaE': 0.1,             # 10x higher treatment effect
        'etaH': 0.1,             # 10x higher treatment effect  
        'etaC': 0.1,             # 10x higher treatment effect
        'genetic_instability': 3.0  # 3x more unstable
    })
    
    print('EXTREMELY AGGRESSIVE PARAMETERS:')
    for key in ['omega_R1', 'omega_R2', 'etaE', 'etaH', 'etaC']:
        print(f'  {key}: {patient[key]}')
    
    # Test calculation manually first
    print(f'\nMANUAL CALCULATION CHECK:')
    N1 = 180
    therapy_effect = patient['etaE'] * 0.8  # Standard protocol uses 0.8 dose
    genetic_factor = 1 + (1 - 0.95)  # Assume 95% genetic stability
    
    expected_R1_rate = patient['omega_R1'] * therapy_effect * N1 * genetic_factor
    expected_R2_rate = patient['omega_R2'] * therapy_effect * N1 * genetic_factor
    
    print(f'  Expected R1 development: {expected_R1_rate:.2f} cells/day')
    print(f'  Expected R2 development: {expected_R2_rate:.2f} cells/day')
    print(f'  Over 100 days: R1 += {expected_R1_rate * 100:.1f}, R2 += {expected_R2_rate * 100:.1f}')
    
    if expected_R1_rate > 1.0:
        print('   This should produce significant resistance!')
    else:
        print('   Still too low')
    
    # Now test in simulation
    print(f'\nRUNNING SIMULATION WITH FIXED PARAMETERS:')
    
    try:
        runner = CancerModelRunner(output_dir='results/fixed_resistance_test')
        result = runner.simulation_runner.run_single_simulation('average', 'standard', 100)
        
        if result['success']:
            ts = result['time_series']
            final_resistant = ts['resistant_type1'][-1] + ts['resistant_type2'][-1]
            final_total = ts['total_tumor'][-1]
            final_resistance_pct = (final_resistant / final_total * 100) if final_total > 0 else 0
            
            print(f'  Final resistant cells: {final_resistant:.1f}')
            print(f'  Final total tumor: {final_total:.1f}')
            print(f'  Final resistance: {final_resistance_pct:.1f}%')
            
            if final_resistance_pct > 10:
                print(f'   SUCCESS! Realistic resistance achieved!')
            elif final_resistance_pct > 5:
                print(f'   Much better! Getting closer to realistic levels')
            else:
                print(f'   Still low, but improving')
            
            print(f'\nRESISTANCE PROGRESSION:')
            for i in [0, 25, 50, 75, 100]:
                if i < len(ts['time']):
                    day = ts['time'][i]
                    r1 = ts['resistant_type1'][i]
                    r2 = ts['resistant_type2'][i]
                    total = ts['total_tumor'][i]
                    pct = ((r1 + r2) / total * 100) if total > 0 else 0
                    print(f'    Day {day:3.0f}: R1={r1:5.1f}, R2={r2:5.1f}, Total={total:6.1f}, Resistance={pct:4.1f}%')
        else:
            error_msg = result.get('error_message', 'Unknown')
            print(f'   Simulation failed: {error_msg}')
            
    except Exception as e:
        print(f'   Error: {e}')

if __name__ == '__main__':
    test_fixed_resistance()
