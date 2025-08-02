#!/usr/bin/env python3
"""
Basic Analysis Example
======================

This script demonstrates the basic functionality of the cancer model system.
It runs a comparative analysis across different patient profiles and treatment protocols.

Usage:
    python examples/basic_analysis.py

Output:
    - Generated figures in results/basic_analysis/
    - Analysis summary and data files
    - Console output with key findings

Author: Cancer Model Team
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import cancer model components
from cancer_model import CancerModelRunner, PatientProfiles, FineTuningPresets
from cancer_model.protocols import TreatmentProtocols
import pandas as pd


def main():
    """Run basic comparative analysis."""
    
    print("CANCER MODEL: BASIC ANALYSIS")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Setup output directory
    output_dir = project_root / 'results' / 'basic_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Initialize the model runner
    print("\nInitializing Cancer Model Runner...")
    runner = CancerModelRunner(output_dir=str(output_dir))
    
    # Display available options
    print("\nAvailable Patient Profiles:")
    profiles = ['average', 'young', 'elderly', 'compromised']
    for profile in profiles:
        profile_data = PatientProfiles.get_profile(profile)
        key_factors = {
            'age_factor': profile_data.get('age_factor', 1.0),
            'immune_status': profile_data.get('immune_status', 1.0),
            'performance_status': profile_data.get('performance_status', 1.0)
        }
        print(f"  • {profile.title()}: {key_factors}")
    
    print("\nAvailable Treatment Protocols:")
    protocols = TreatmentProtocols()
    protocols.list_protocols()
    
    # Run the analysis
    print("\nRunning comparative analysis...")
    print("   This may take a few minutes...")
    
    patient_profiles = ['average', 'young', 'elderly', 'compromised']
    treatment_protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']
    
    basic_results = runner.run_basic_analysis()
    results = basic_results['results']  # Extract the actual results dictionary

    # Analyze results
    print("\nANALYSIS RESULTS")
    print("=" * 30)

    # Create results summary
    analysis_data = []
    successful_sims = 0
    total_sims = 0

    for patient_profile, protocols_results in results.items():
        for protocol_name, result in protocols_results.items():
            total_sims += 1
            if 'success' in result and result['success']:
                successful_sims += 1
                metrics = result['metrics']
                analysis_data.append({
                    'Patient': patient_profile.title(),
                    'Protocol': protocol_name.replace('_', ' ').title(),
                    'Efficacy_Score': metrics['treatment_efficacy_score'],
                    'Tumor_Reduction_Pct': metrics['percent_reduction'],
                    'Final_Resistance_Pct': metrics['final_resistance_fraction'],
                    'Immune_Activation': metrics['immune_activation'],
                    'Genetic_Instability': metrics['genetic_instability']
                })
    
    print(f"Successful simulations: {successful_sims}/{total_sims}")
    
    if analysis_data:
        # Create DataFrame for analysis
        df = pd.DataFrame(analysis_data)
        
        # Find best results
        best_result = df.loc[df['Efficacy_Score'].idxmax()]
        print(f"\nBest Overall Result:")
        print(f"   Patient: {best_result['Patient']}")
        print(f"   Protocol: {best_result['Protocol']}")
        print(f"   Efficacy Score: {best_result['Efficacy_Score']:.2f}")
        print(f"   Tumor Reduction: {best_result['Tumor_Reduction_Pct']:.2f}%")
        print(f"   Final Resistance: {best_result['Final_Resistance_Pct']:.2f}%")
        
        # Protocol rankings
        print(f"\nProtocol Rankings (by average efficacy):")
        protocol_avg = df.groupby('Protocol')['Efficacy_Score'].mean().sort_values(ascending=False)
        for i, (protocol, score) in enumerate(protocol_avg.items()):
            print(f"   {i+1}. {protocol}: {score:.2f}")
        
        # Patient profile analysis
        print(f"\nPatient Profile Performance:")
        patient_avg = df.groupby('Patient')['Efficacy_Score'].mean().sort_values(ascending=False)
        for patient, score in patient_avg.items():
            print(f"   {patient}: {score:.2f}")
        
        # Resistance analysis
        avg_resistance = df['Final_Resistance_Pct'].mean()
        print(f"\nResistance Analysis:")
        print(f"   Average final resistance: {avg_resistance:.2f}%")
        print(f"   Range: {df['Final_Resistance_Pct'].min():.2f}% - {df['Final_Resistance_Pct'].max():.2f}%")
        
        if avg_resistance < 2.0:
            print(f"   Warning: Resistance levels appear unrealistically low")
            print(f"   Consider using fine-tuning presets to increase resistance")
            print(f"   Available presets: realistic_resistance, high_resistance_scenario")
        
        # Save results
        results_file = output_dir / 'basic_analysis_results.csv'
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        # Save summary
        summary_file = output_dir / 'analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Basic Analysis Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total simulations: {total_sims}\n")
            f.write(f"Successful simulations: {successful_sims}\n")
            f.write(f"Success rate: {100*successful_sims/total_sims:.1f}%\n\n")
            f.write(f"Best Protocol: {best_result['Protocol']}\n")
            f.write(f"Best Patient: {best_result['Patient']}\n")
            f.write(f"Best Efficacy Score: {best_result['Efficacy_Score']:.2f}\n\n")
            f.write("Protocol Rankings:\n")
            for i, (protocol, score) in enumerate(protocol_avg.items()):
                f.write(f"  {i+1}. {protocol}: {score:.2f}\n")
        
        print(f"Summary saved to: {summary_file}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    visualizations = runner.visualizer.create_protocol_comparison_plot(results, 'average')
    if visualizations:
        print(f"Protocol comparison plot: {visualizations}")
    
    efficacy_plot = runner.visualizer.create_efficacy_metrics_chart(results, 'average')
    if efficacy_plot:
        print(f"Efficacy metrics chart: {efficacy_plot}")
    
    heatmap_plot = runner.visualizer.create_heatmaps(results)
    if heatmap_plot:
        print(f"Treatment heatmaps: {heatmap_plot}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS")
    print("=" * 20)
    print("1. Review the generated visualizations in the results directory")
    print("2. Consider running optimization analysis for specific patient types")
    print("3. Perform sensitivity analysis to understand parameter effects")
    print("4. Test custom protocols based on these baseline results")
    
    if avg_resistance < 5.0:
        print("\nFINE-TUNING SUGGESTIONS:")
        print("- Use 'realistic_resistance' preset for more clinical resistance levels")
        print("- Adjust mutation_rate and omega_R1/R2 parameters manually")
        print("- Consider higher baseline resistance floors")
    
    print(f"\nBasic analysis complete!")
    print(f"Check the results directory: {output_dir}")
    print(f"View generated plots and data files")
    print(f"Read analysis_summary.txt for detailed findings")
    
    return results, df if analysis_data else None


def demo_fine_tuning():
    """Demonstrate fine-tuning with realistic resistance parameters."""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: FINE-TUNING FOR REALISTIC RESISTANCE")
    print("="*60)
    
    # Setup for realistic resistance
    output_dir = project_root / 'results' / 'realistic_resistance'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running analysis with realistic resistance parameters...")
    
    # Initialize runner with realistic resistance preset
    runner = CancerModelRunner(
        output_dir=str(output_dir),
        fine_tuning_preset='realistic_resistance'
    )
    
    # Run single simulation for comparison
    result = runner.simulation_runner.run_single_simulation(
        'average', 'standard', simulation_days=300
    )
    
    if result['success']:
        metrics = result['metrics']
        print(f"Realistic resistance simulation complete!")
        print(f"   Final resistance: {metrics['final_resistance_fraction']:.2f}%")
        print(f"   Efficacy score: {metrics['treatment_efficacy_score']:.2f}")
        print(f"   Tumor reduction: {metrics['percent_reduction']:.2f}%")
        
        return result
    else:
        print(f"Simulation failed: {result['error_message']}")
        return None


if __name__ == "__main__":
    # Check if we're in the correct directory
    if not (project_root / 'cancer_model').exists():
        print("Error: Please run this script from the project root directory")
        print("   Expected structure: cancer_model_project/cancer_model/")
        sys.exit(1)
    
    try:
        # Run basic analysis
        results, df = main()
        
        # Optionally run fine-tuning demo
        print("\n" + "?"*50)
        response = input("Would you like to see realistic resistance demo? (y/N): ")
        if response.lower().startswith('y'):
            realistic_result = demo_fine_tuning()
        
        print("\nAll analyses complete!")
        print("Check the results/ directory for outputs")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)