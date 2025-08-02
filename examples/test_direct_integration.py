#!/usr/bin/env python3
"""
Direct Integration Test
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cancer_model import CancerModelRunner, PatientProfiles
from cancer_model.core.fractional_math import safe_solve_ivp

def test_direct_integration():
    print('TESTING DIRECT INTEGRATION OF RESISTANCE')
    print('=' * 50)
    
    # Set up the same conditions as our working test
    patient = PatientProfiles.get_profile('average')
    patient.update({
        'omega_R1': 1.0,
        'omega_R2': 0.8,
        'etaE': 0.1
    })
    
    from cancer_model.core.model_parameters import ModelParameters
    from cancer_model.core.cancer_model_core import CancerModel
    from cancer_model.core.pharmacokinetics import PharmacokineticModel, CircadianRhythm
    from cancer_model.protocols.treatment_protocols import TreatmentProtocols
    
    model_params = ModelParameters(patient)
    params = model_params.get_all_parameters()
    
    pk_model = PharmacokineticModel(params)
    circadian_model = CircadianRhythm(params)
    cancer_model = CancerModel(params, pk_model, circadian_model)
    
    protocols = TreatmentProtocols()
    protocol = protocols.get_protocol('standard', patient)
    
    # Initial conditions
    y0 = np.array([
        180.0,  # N1
        10.0,   # N2
        40.0,   # I1
        10.0,   # I2
        0.1,    # P
        1.0,    # A
        0.1,    # Q
        1.0,    # R1  Watch this!
        1.0,    # R2  Watch this!
        0.1,    # S
        0.5,    # D
        0.0,    # Dm
        0.95,   # G
        1.0,    # M
        0.0     # H
    ])
    
    print('Initial conditions:')
    print(f'  R1: {y0[7]:.1f}')
    print(f'  R2: {y0[8]:.1f}')
    
    # Define model function
    def model_function(t, y):
        return cancer_model.enhanced_temperature_cancer_model(
            t, y, protocol['drugs'], 37.0, True
        )
    
    # Test the derivative at t=0
    initial_derivatives = model_function(0, y0)
    print(f'\nInitial derivatives:')
    print(f'  dR1/dt: {initial_derivatives[7]:.6f}')
    print(f'  dR2/dt: {initial_derivatives[8]:.6f}')
    
    # Run integration for just 10 days to see what happens
    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 11)
    
    print(f'\nRunning 10-day integration...')
    result = safe_solve_ivp(model_function, t_span, y0, 'RK45', t_eval)
    
    if result.success:
        print(f'Integration successful!')
        
        print(f'\nResistant cell progression:')
        for i, t in enumerate(result.t):
            R1 = result.y[7, i]
            R2 = result.y[8, i]
            total = result.y[0, i] + result.y[1, i] + result.y[6, i] + result.y[7, i] + result.y[8, i] + result.y[9, i]
            resistance = (R1 + R2) / total * 100
            print(f'  Day {t:2.0f}: R1={R1:6.3f}, R2={R2:6.3f}, Resistance={resistance:4.2f}%')
        
        # Check if R1 and R2 actually changed
        R1_change = result.y[7, -1] - result.y[7, 0]
        R2_change = result.y[8, -1] - result.y[8, 0]
        
        print(f'\nChanges over 10 days:')
        print(f'  R1 change: {R1_change:.6f}')
        print(f'  R2 change: {R2_change:.6f}')
        
        if abs(R1_change) > 0.01 or abs(R2_change) > 0.01:
            print(' Resistance is actually growing!')
            print('The problem may be with longer simulations or parameter scaling')
        else:
            print(' Resistance still not changing significantly')
            
    else:
        print(f' Integration failed: {result.message}')

if __name__ == '__main__':
    test_direct_integration()
