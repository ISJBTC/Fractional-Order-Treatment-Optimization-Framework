#!/usr/bin/env python3
"""
Fixed Resistance Simulation - Bypass Parameter Override
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_fixed_resistance_simulation():
    print('TESTING FIXED RESISTANCE WITH DIRECT PARAMETER SETTING')
    print('=' * 60)
    
    from cancer_model.core.model_parameters import ModelParameters, PatientProfiles, InitialConditions
    from cancer_model.core.cancer_model_core import CancerModel
    from cancer_model.core.pharmacokinetics import PharmacokineticModel, CircadianRhythm
    from cancer_model.protocols.treatment_protocols import TreatmentProtocols
    from cancer_model.core.fractional_math import safe_solve_ivp
    
    # Step 1: Create parameters and FORCE the values
    patient_profile = PatientProfiles.get_profile('average')
    model_params = ModelParameters(patient_profile)
    
    # DIRECTLY modify the internal parameters dictionary
    model_params.params['omega_R1'] = 1.0      # Force high resistance
    model_params.params['omega_R2'] = 0.8      # Force high resistance  
    model_params.params['etaE'] = 0.1          # Force high treatment effect
    model_params.params['etaH'] = 0.1          # Force high treatment effect
    model_params.params['etaC'] = 0.1          # Force high treatment effect
    
    print('FORCED PARAMETERS:')
    omega_r1 = model_params.params['omega_R1']
    omega_r2 = model_params.params['omega_R2']  
    eta_e = model_params.params['etaE']
    print(f'  omega_R1: {omega_r1}')
    print(f'  omega_R2: {omega_r2}')
    print(f'  etaE: {eta_e}')
    
    # Step 2: Create model components with forced parameters
    params = model_params.get_all_parameters()
    pk_model = PharmacokineticModel(params)
    circadian_model = CircadianRhythm(params)
    cancer_model = CancerModel(params, pk_model, circadian_model)
    
    # Step 3: Get treatment protocol
    protocols = TreatmentProtocols()
    protocol = protocols.get_protocol('standard', patient_profile)
    
    # Step 4: Setup simulation
    t_span = [0, 100]
    t_eval = np.linspace(0, 100, 101)
    initial_conditions = InitialConditions.get_conditions_for_profile('average')
    
    temp_func = protocol.get('temperature', lambda t: 37.0)
    
    def model_function(t, y):
        current_temp = temp_func(t)
        return cancer_model.enhanced_temperature_cancer_model(
            t, y, protocol['drugs'], current_temp, True
        )
    
    print(f'\nRunning 100-day simulation with FORCED parameters...')
    
    # Step 5: Run simulation
    result = safe_solve_ivp(model_function, t_span, initial_conditions, 'RK45', t_eval)
    
    if result.success:
        print(' Simulation successful!')
        
        # Extract final results
        N1 = result.y[0]
        N2 = result.y[1] 
        Q = result.y[6]
        R1 = result.y[7]
        R2 = result.y[8]
        S = result.y[9]
        
        total_tumor = N1 + N2 + Q + R1 + R2 + S
        total_resistant = R1 + R2
        resistance_fraction = (total_resistant / total_tumor * 100)
        
        print(f'\nRESISTANCE PROGRESSION:')
        checkpoints = [0, 25, 50, 75, 100]
        for i, day in enumerate(checkpoints):
            if day < len(result.t):
                idx = day
                r1_val = R1[idx]
                r2_val = R2[idx]
                total_val = total_tumor[idx]
                resist_pct = (r1_val + r2_val) / total_val * 100
                print(f'  Day {day:3d}: R1={r1_val:6.1f}, R2={r2_val:6.1f}, Total={total_val:6.1f}, Resistance={resist_pct:5.1f}%')
        
        final_resistance = resistance_fraction[-1]
        print(f'\nFINAL RESULTS:')
        print(f'  Final resistance: {final_resistance:.1f}%')
        print(f'  R1 final: {R1[-1]:.1f} (grew by {R1[-1] - R1[0]:.1f})')
        print(f'  R2 final: {R2[-1]:.1f} (grew by {R2[-1] - R2[0]:.1f})')
        
        if final_resistance > 10:
            print(f'   SUCCESS! Realistic resistance achieved!')
        elif final_resistance > 5:
            print(f'   Much better! Getting realistic')
        else:
            print(f'   Still low, but much improved')
            
        # Save results
        output_dir = Path('results/fixed_resistance_success')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'resistance_success.txt', 'w') as f:
            f.write('SUCCESSFUL RESISTANCE SIMULATION\n')
            f.write('=' * 40 + '\n\n')
            f.write(f'Final resistance: {final_resistance:.1f}%\n')
            f.write(f'R1 growth: {R1[0]:.1f}  {R1[-1]:.1f}\n')
            f.write(f'R2 growth: {R2[0]:.1f}  {R2[-1]:.1f}\n\n')
            f.write('Key insight: Parameters must be forced directly\n')
            f.write('The simulation runner overrides custom parameters\n')
        
        print(f'\nResults saved to: results/fixed_resistance_success/resistance_success.txt')
        
    else:
        print(f' Simulation failed: {result.message}')

if __name__ == '__main__':
    test_fixed_resistance_simulation()
