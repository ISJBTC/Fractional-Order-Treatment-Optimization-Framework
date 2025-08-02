#!/usr/bin/env python3
"""
Realistic Analysis - Fixed Version for PowerShell
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cancer_model import CancerModelRunner

def main():
    print("RUNNING REALISTIC CANCER MODEL ANALYSIS")
    print("=" * 50)
    
    # Set random seed for consistent results
    np.random.seed(42)
    
    # Use realistic resistance parameters
    runner = CancerModelRunner(
        output_dir='results/realistic_analysis',
        fine_tuning_preset='realistic_resistance'
    )
    
    print("Running analysis with realistic resistance parameters...")
    print("This will take 5-10 minutes...")
    
    try:
        # Run the analysis
        results = runner.run_basic_analysis()
        
        print("\nANALYSIS COMPLETE!")
        print("Check the results/realistic_analysis/ folder for:")
        print("   Comparison plots")
        print("   Efficacy charts") 
        print("   Analysis summary")
        
        # Show quick summary
        if 'results' in results:
            success_count = 0
            total_count = 0
            
            for patient_profile, protocols in results['results'].items():
                for protocol_name, result in protocols.items():
                    total_count += 1
                    if result.get('success', False):
                        success_count += 1
            
            print(f"\nQuick Summary:")
            print(f"  Successful simulations: {success_count}/{total_count}")
            
            if success_count > 0:
                print(f"  Best protocol: {results.get('best_protocol', 'Unknown')}")
                print(f"  Best patient: {results.get('best_patient', 'Unknown')}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        print("Check that all cancer_model files are properly installed")
        return None

if __name__ == "__main__":
    main()
