#!/usr/bin/env python3
"""
Simplified Resistance Test
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_resistance_mechanism():
    print('SIMPLIFIED RESISTANCE MECHANISM TEST')
    print('=' * 50)
    
    # Simulate the resistance development equation manually
    print('Testing resistance development equation manually...')
    
    # Parameters (using massive values)
    omega_R1 = 1.0
    omega_R2 = 0.8
    etaE = 0.1
    uE = 0.8  # Standard dose
    
    # State variables
    N1 = 180.0  # Sensitive cells
    G = 0.95    # Genetic stability
    
    # Calculate therapy effect
    therapy_effect = etaE * uE
    print(f'Therapy effect: {therapy_effect}')
    
    # Calculate resistance development (the actual equation from the model)
    resistance_dev_factor = (1 + (1 - G))
    resistance_dev_R1 = max(omega_R1 * therapy_effect * N1 * resistance_dev_factor, 0.0)
    resistance_dev_R2 = max(omega_R2 * therapy_effect * N1 * resistance_dev_factor, 0.0)
    
    print(f'Resistance development factor: {resistance_dev_factor}')
    print(f'R1 development rate: {resistance_dev_R1:.6f} cells/day')
    print(f'R2 development rate: {resistance_dev_R2:.6f} cells/day')
    
    # Simulate over time
    print(f'\nSimulating resistance growth over time:')
    R1 = 1.0
    R2 = 1.0
    total_tumor = 200.0
    
    for day in [0, 25, 50, 75, 100]:
        if day > 0:
            # Simple Euler integration
            R1 += resistance_dev_R1
            R2 += resistance_dev_R2
        
        resistance_pct = (R1 + R2) / total_tumor * 100
        print(f'Day {day:3d}: R1={R1:6.1f}, R2={R2:6.1f}, Resistance={resistance_pct:4.1f}%')
    
    print(f'\nWith these parameters, resistance should reach {resistance_pct:.1f}% after 100 days')
    if resistance_pct > 10:
        print(' This should produce realistic resistance levels!')
    else:
        print(' Still too low even with manual calculation')

if __name__ == '__main__':
    test_resistance_mechanism()
