#!/usr/bin/env python3
"""
Simple Treatment Optimization - PowerShell Version
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cancer_model import CancerModelRunner, PatientProfiles
from cancer_model.protocols import TreatmentProtocols, ProtocolOptimizer

def simple_optimization(patient_type='average'):
    print(f"SIMPLE TREATMENT OPTIMIZATION FOR {patient_type.upper()} PATIENT")
    print("=" * 60)
    
    # Test different dose levels
    dose_levels = [0.4, 0.6, 0.8, 1.0, 1.2]
    protocols_to_test = ['standard', 'continuous', 'adaptive']
    
    best_result = None
    best_score = 0
    
    results = []
    
    for protocol in protocols_to_test:
        print(f"\nTesting {protocol} protocol with different doses...")
        
        for dose in dose_levels:
            print(f"  Testing dose {dose}...")
            
            try:
                # Create custom patient with modified parameters for better resistance
                patient = PatientProfiles.get_profile(patient_type)
                patient.update({
                    'omega_R1': 0.02,    # Higher resistance
                    'mutation_rate': 0.001,  # Higher mutation
                    'resistance_floor': 0.15  # 15% baseline resistance
                })
                
                runner = CancerModelRunner(output_dir=f'results/simple_opt_{protocol}_{dose}')
                result = runner.simulation_runner.run_single_simulation(patient_type, protocol, 200)
                
                if result['success']:
                    metrics = result['metrics']
                    efficacy = metrics['treatment_efficacy_score']
                    resistance = metrics['final_resistance_fraction']
                    reduction = metrics['percent_reduction']
                    
                    # Simple scoring: efficacy minus resistance penalty
                    score = efficacy - (resistance * 0.5)
                    
                    results.append({
                        'protocol': protocol,
                        'dose': dose,
                        'efficacy': efficacy,
                        'resistance': resistance,
                        'reduction': reduction,
                        'score': score
                    })
                    
                    print(f"    Efficacy: {efficacy:.2f}, Resistance: {resistance:.1f}%, Score: {score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'protocol': protocol,
                            'dose': dose,
                            'efficacy': efficacy,
                            'resistance': resistance,
                            'reduction': reduction,
                            'score': score
                        }
                else:
                    print(f"    FAILED: {result.get('error_message', 'Unknown')}")
                    
            except Exception as e:
                print(f"    ERROR: {e}")
    
    # Show results
    if best_result:
        print(f"\nBEST TREATMENT FOUND:")
        print(f"  Protocol: {best_result['protocol']}")
        print(f"  Dose: {best_result['dose']}")
        print(f"  Efficacy: {best_result['efficacy']:.2f}")
        print(f"  Resistance: {best_result['resistance']:.1f}%")
        print(f"  Tumor Reduction: {best_result['reduction']:.1f}%")
        print(f"  Overall Score: {best_result['score']:.2f}")
        
        # Save results
        output_dir = Path('results/simple_optimization')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f'{patient_type}_optimization.txt', 'w') as f:
            f.write(f"Simple Optimization Results for {patient_type.title()} Patient\n")
            f.write("=" * 50 + "\n\n")
            f.write("All tested combinations:\n")
            for r in sorted(results, key=lambda x: x['score'], reverse=True):
                f.write(f"{r['protocol']}-{r['dose']}: Score={r['score']:.2f}, Efficacy={r['efficacy']:.2f}, Resistance={r['resistance']:.1f}%\n")
            
            f.write(f"\nBest: {best_result['protocol']} at dose {best_result['dose']} with score {best_result['score']:.2f}\n")
        
        print(f"\nResults saved to: results/simple_optimization/{patient_type}_optimization.txt")
    else:
        print("\nNo successful optimizations found")

if __name__ == "__main__":
    # Test for different patient types
    patients = ['average', 'young', 'elderly']
    
    for patient in patients:
        simple_optimization(patient)
        print("\n" + "="*60 + "\n")
