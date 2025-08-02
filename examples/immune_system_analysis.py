#!/usr/bin/env python3
"""
Immune System Focus Analysis - PowerShell Version
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cancer_model import CancerModelRunner, PatientProfiles

def test_immune_strength():
    print("TESTING IMMUNE SYSTEM STRENGTH")
    print("=" * 40)
    
    # Test different beta1 values (your analysis showed 0.010 was best)
    beta1_values = [0.005, 0.008, 0.010, 0.012, 0.015]
    
    results = []
    
    for i, beta1 in enumerate(beta1_values):
        print(f"\nTesting immune strength {i+1}/{len(beta1_values)}: beta1 = {beta1}")
        
        try:
            # Create custom patient with this immune strength
            patient = PatientProfiles.get_profile('average')
            patient['beta1'] = beta1
            
            # Run simulation
            output_dir = f'results/immune_test_beta1_{beta1}'
            runner = CancerModelRunner(output_dir=output_dir, fine_tuning_preset='realistic_resistance')
            result = runner.simulation_runner.run_single_simulation('average', 'standard', 200)
            
            if result['success']:
                metrics = result['metrics']
                print(f"  Efficacy: {metrics['treatment_efficacy_score']:.2f}")
                print(f"  Resistance: {metrics['final_resistance_fraction']:.1f}%")
                print(f"  Tumor reduction: {metrics['percent_reduction']:.1f}%")
                
                results.append({
                    'beta1': beta1,
                    'efficacy': metrics['treatment_efficacy_score'],
                    'resistance': metrics['final_resistance_fraction'],
                    'reduction': metrics['percent_reduction']
                })
            else:
                print(f"  FAILED: {result.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['efficacy'])
        print(f"\nBEST IMMUNE STRENGTH FOUND:")
        print(f"  beta1 = {best['beta1']}")
        print(f"  Efficacy = {best['efficacy']:.2f}")
        print(f"  Resistance = {best['resistance']:.1f}%")
        print(f"  Tumor reduction = {best['reduction']:.1f}%")
        
        # Save results
        output_dir = Path('results/immune_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'immune_test_results.txt', 'w') as f:
            f.write("Immune System Strength Test Results\n")
            f.write("=" * 40 + "\n\n")
            for r in results:
                f.write(f"beta1 = {r['beta1']}: Efficacy = {r['efficacy']:.2f}, Resistance = {r['resistance']:.1f}%\n")
            f.write(f"\nBest: beta1 = {best['beta1']} with efficacy = {best['efficacy']:.2f}\n")
        
        print(f"\nResults saved to: results/immune_analysis/immune_test_results.txt")
    else:
        print("\nNo successful tests - check for errors above")

if __name__ == "__main__":
    test_immune_strength()
