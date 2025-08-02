#!/usr/bin/env python3
"""
Treatment Optimization Example
==============================

This script demonstrates how to optimize treatment protocols for specific 
patient profiles using the cancer model system.

Features:
- Patient-specific treatment optimization
- Dose and scheduling optimization
- Therapeutic index maximization
- Comparison of optimization results

Usage:
    python examples/optimization_example.py

Author: Cancer Model Team
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import cancer model components
from cancer_model import CancerModelRunner, PatientProfiles
from cancer_model.protocols import ProtocolOptimizer, TreatmentProtocols


def optimize_for_patient(patient_profile_name, base_protocol='standard'):
    """
    Optimize treatment for a specific patient profile.
    
    Args:
        patient_profile_name (str): Name of patient profile
        base_protocol (str): Base protocol to optimize
        
    Returns:
        dict: Optimization results
    """
    print(f"\nOPTIMIZING TREATMENT FOR {patient_profile_name.upper()} PATIENT")
    print("=" * 60)
    
    # Setup output directory
    output_dir = project_root / 'results' / 'optimization' / patient_profile_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize runner
    runner = CancerModelRunner(output_dir=str(output_dir))
    
    # Display patient characteristics
    patient_profile = PatientProfiles.get_profile(patient_profile_name)
    print(f"Patient Characteristics:")
    key_factors = ['age_factor', 'performance_status', 'immune_status', 
                   'liver_function', 'kidney_function']
    for factor in key_factors:
        value = patient_profile.get(factor, 1.0)
        status = "↑" if value > 1.0 else "↓" if value < 1.0 else "→"
        print(f"   {factor.replace('_', ' ').title()}: {value:.2f} {status}")
    
    # Run optimization
    print(f"\nRunning optimization algorithm...")
    print(f"   Base protocol: {base_protocol}")
    print(f"   Testing multiple dose/schedule combinations...")
    
    opt_results = runner.optimize_patient_treatment(
        patient_profile_name, 
        base_protocol=base_protocol,
        output_results=False  # We'll handle output ourselves
    )
    
    # Extract best results
    best_config = opt_results['best_configuration']
    all_results = opt_results['optimization_results']['all_results']
    
    print(f"\nOPTIMIZATION RESULTS")
    print("-" * 30)
    print(f"Best Configuration Found:")
    print(f"   Dose: {best_config['dose']:.2f}")
    print(f"   Schedule: {best_config['treatment_days']} days on, {best_config['rest_days']} days off")
    print(f"   Therapeutic Index: {best_config['therapeutic_index']:.2f}")
    print(f"   Estimated Efficacy: {best_config['estimated_efficacy']:.2f}")
    print(f"   Estimated Toxicity: {best_config['estimated_toxicity']:.2f}")
    
    # Compare with standard doses
    standard_configs = [r for r in all_results if abs(r['dose'] - 0.8) < 0.1]
    if standard_configs:
        std_config = max(standard_configs, key=lambda x: x['therapeutic_index'])
        improvement = ((best_config['therapeutic_index'] - std_config['therapeutic_index']) 
                      / std_config['therapeutic_index'] * 100)
        print(f"\nImprovement over standard dose (0.8):")
        print(f"   Standard therapeutic index: {std_config['therapeutic_index']:.2f}")
        print(f"   Optimized therapeutic index: {best_config['therapeutic_index']:.2f}")
        print(f"   Improvement: {improvement:+.1f}%")
    
    # Save optimization results
    results_df = pd.DataFrame(all_results)
    results_file = output_dir / f'{patient_profile_name}_optimization_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nOptimization data saved: {results_file}")
    
    # Create optimization visualization
    create_optimization_plot(all_results, best_config, patient_profile_name, output_dir)
    
    return opt_results


def create_optimization_plot(all_results, best_config, patient_name, output_dir):
    """Create visualization of optimization results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Treatment Optimization: {patient_name.title()} Patient', fontsize=16)
    
    # Extract data
    doses = [r['dose'] for r in all_results]
    efficacies = [r['estimated_efficacy'] for r in all_results]
    toxicities = [r['estimated_toxicity'] for r in all_results]
    therapeutic_indices = [r['therapeutic_index'] for r in all_results]
    
    # Plot 1: Dose vs Therapeutic Index
    axes[0,0].scatter(doses, therapeutic_indices, alpha=0.6, s=50)
    axes[0,0].scatter(best_config['dose'], best_config['therapeutic_index'], 
                     color='red', s=100, marker='*', label='Optimal')
    axes[0,0].set_xlabel('Dose')
    axes[0,0].set_ylabel('Therapeutic Index')
    axes[0,0].set_title('Dose vs Therapeutic Index')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Efficacy vs Toxicity
    scatter = axes[0,1].scatter(toxicities, efficacies, c=therapeutic_indices, 
                               cmap='viridis', alpha=0.7, s=50)
    axes[0,1].scatter(best_config['estimated_toxicity'], best_config['estimated_efficacy'], 
                     color='red', s=100, marker='*', label='Optimal')
    axes[0,1].set_xlabel('Estimated Toxicity')
    axes[0,1].set_ylabel('Estimated Efficacy')
    axes[0,1].set_title('Efficacy vs Toxicity Trade-off')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0,1], label='Therapeutic Index')
    
    # Plot 3: Schedule comparison
    schedule_data = {}
    for r in all_results:
        schedule_key = f"{r['treatment_days']}/{r['rest_days']}"
        if schedule_key not in schedule_data:
            schedule_data[schedule_key] = []
        schedule_data[schedule_key].append(r['therapeutic_index'])
    
    schedules = list(schedule_data.keys())
    avg_indices = [np.mean(schedule_data[s]) for s in schedules]
    
    bars = axes[1,0].bar(range(len(schedules)), avg_indices)
    axes[1,0].set_xticks(range(len(schedules)))
    axes[1,0].set_xticklabels(schedules, rotation=45)
    axes[1,0].set_ylabel('Average Therapeutic Index')
    axes[1,0].set_title('Schedule Comparison (Treatment/Rest Days)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Highlight best schedule
    best_schedule = f"{best_config['treatment_days']}/{best_config['rest_days']}"
    if best_schedule in schedules:
        best_idx = schedules.index(best_schedule)
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(0.8)
    
    # Plot 4: Top configurations
    top_5 = sorted(all_results, key=lambda x: x['therapeutic_index'], reverse=True)[:5]
    
    config_labels = [f"D:{r['dose']:.1f}\n{r['treatment_days']}/{r['rest_days']}" for r in top_5]
    config_scores = [r['therapeutic_index'] for r in top_5]
    
    bars = axes[1,1].bar(range(len(config_labels)), config_scores)
    axes[1,1].set_xticks(range(len(config_labels)))
    axes[1,1].set_xticklabels(config_labels)
    axes[1,1].set_ylabel('Therapeutic Index')
    axes[1,1].set_title('Top 5 Configurations')
    axes[1,1].grid(True, alpha=0.3)
    
    # Highlight best
    bars[0].set_color('red')
    bars[0].set_alpha(0.8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plot_file = output_dir / f'{patient_name}_optimization_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Optimization plot saved: {plot_file}")


def compare_patient_optimizations():
    """Compare optimization results across different patient profiles."""
    
    print(f"\nCOMPARING OPTIMIZATIONS ACROSS PATIENT PROFILES")
    print("=" * 60)
    
    # Patient profiles to compare
    patient_profiles = ['average', 'young', 'elderly', 'compromised']
    
    # Store results for comparison
    comparison_data = []
    
    for patient_profile in patient_profiles:
        print(f"\nProcessing {patient_profile} patient...")
        
        try:
            opt_results = optimize_for_patient(patient_profile)
            best_config = opt_results['best_configuration']
            
            comparison_data.append({
                'Patient': patient_profile.title(),
                'Optimal_Dose': best_config['dose'],
                'Treatment_Days': best_config['treatment_days'],
                'Rest_Days': best_config['rest_days'],
                'Therapeutic_Index': best_config['therapeutic_index'],
                'Estimated_Efficacy': best_config['estimated_efficacy'],
                'Estimated_Toxicity': best_config['estimated_toxicity']
            })
            
        except Exception as e:
            print(f"Error optimizing for {patient_profile}: {e}")
            continue
    
    if comparison_data:
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        print(f"\nOPTIMIZATION COMPARISON SUMMARY")
        print("=" * 50)
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        
        # Save comparison results
        output_dir = project_root / 'results' / 'optimization'
        comparison_file = output_dir / 'patient_optimization_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nComparison saved: {comparison_file}")
        
        # Create comparison visualization
        create_comparison_plot(comparison_df, output_dir)
        
        # Analysis insights
        print(f"\nKEY INSIGHTS:")
        
        # Find patient requiring lowest/highest doses
        min_dose_patient = comparison_df.loc[comparison_df['Optimal_Dose'].idxmin()]
        max_dose_patient = comparison_df.loc[comparison_df['Optimal_Dose'].idxmax()]
        
        print(f"   • Lowest optimal dose: {min_dose_patient['Patient']} ({min_dose_patient['Optimal_Dose']:.2f})")
        print(f"   • Highest optimal dose: {max_dose_patient['Patient']} ({max_dose_patient['Optimal_Dose']:.2f})")
        
        # Best therapeutic index
        best_ti_patient = comparison_df.loc[comparison_df['Therapeutic_Index'].idxmax()]
        print(f"   • Best therapeutic index: {best_ti_patient['Patient']} ({best_ti_patient['Therapeutic_Index']:.2f})")
        
        # Schedule preferences
        gentle_schedules = comparison_df[comparison_df['Rest_Days'] >= 14]
        if not gentle_schedules.empty:
            print(f"   • Patients benefiting from gentler schedules: {', '.join(gentle_schedules['Patient'])}")
        
        return comparison_df
    
    return None


def create_comparison_plot(comparison_df, output_dir):
    """Create comparison visualization across patient profiles."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Treatment Optimization Comparison Across Patient Profiles', fontsize=16)
    
    patients = comparison_df['Patient']
    
    # Plot 1: Optimal doses
    axes[0,0].bar(patients, comparison_df['Optimal_Dose'], color='skyblue', alpha=0.7)
    axes[0,0].set_ylabel('Optimal Dose')
    axes[0,0].set_title('Optimal Dose by Patient Profile')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Therapeutic indices
    axes[0,1].bar(patients, comparison_df['Therapeutic_Index'], color='lightgreen', alpha=0.7)
    axes[0,1].set_ylabel('Therapeutic Index')
    axes[0,1].set_title('Therapeutic Index by Patient Profile')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Efficacy vs Toxicity
    scatter = axes[1,0].scatter(comparison_df['Estimated_Toxicity'], 
                               comparison_df['Estimated_Efficacy'],
                               s=100, alpha=0.7)
    
    # Add patient labels
    for i, patient in enumerate(patients):
        axes[1,0].annotate(patient, 
                          (comparison_df['Estimated_Toxicity'].iloc[i], 
                           comparison_df['Estimated_Efficacy'].iloc[i]),
                          xytext=(5, 5), textcoords='offset points')
    
    axes[1,0].set_xlabel('Estimated Toxicity')
    axes[1,0].set_ylabel('Estimated Efficacy')
    axes[1,0].set_title('Efficacy vs Toxicity Trade-offs')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Treatment schedules
    schedule_labels = [f"{row['Treatment_Days']}/{row['Rest_Days']}" 
                      for _, row in comparison_df.iterrows()]
    
    axes[1,1].bar(patients, comparison_df['Treatment_Days'], 
                  label='Treatment Days', alpha=0.7)
    axes[1,1].bar(patients, comparison_df['Rest_Days'], 
                  bottom=comparison_df['Treatment_Days'],
                  label='Rest Days', alpha=0.7)
    
    axes[1,1].set_ylabel('Days')
    axes[1,1].set_title('Optimal Treatment Schedules')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plot_file = output_dir / 'patient_optimization_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {plot_file}")


def main():
    """Main optimization analysis workflow."""
    
    print("CANCER MODEL: TREATMENT OPTIMIZATION")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    output_dir = project_root / 'results' / 'optimization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Option 1: Single patient optimization
        print("\nChoose analysis type:")
        print("1. Optimize for single patient profile")
        print("2. Compare optimizations across all patient profiles")
        print("3. Both analyses")
        
        choice = input("\nEnter your choice (1-3, default=3): ").strip()
        if not choice:
            choice = "3"
        
        if choice in ["1", "3"]:
            print("\nAvailable patient profiles:")
            profiles = ['average', 'young', 'elderly', 'compromised']
            for i, profile in enumerate(profiles):
                print(f"   {i+1}. {profile.title()}")
            
            profile_choice = input(f"\nSelect patient profile (1-{len(profiles)}, default=3): ").strip()
            if not profile_choice:
                profile_choice = "3"
            
            try:
                profile_idx = int(profile_choice) - 1
                if 0 <= profile_idx < len(profiles):
                    selected_profile = profiles[profile_idx]
                    opt_results = optimize_for_patient(selected_profile)
                    print(f"Single patient optimization complete!")
                else:
                    print("Invalid choice, using 'elderly' as default")
                    opt_results = optimize_for_patient('elderly')
            except ValueError:
                print("Invalid input, using 'elderly' as default")
                opt_results = optimize_for_patient('elderly')
        
        if choice in ["2", "3"]:
            comparison_df = compare_patient_optimizations()
        
        print(f"\nOPTIMIZATION ANALYSIS COMPLETE!")
        print(f"Results saved in: {output_dir}")
        print(f"Check generated plots and CSV files")
        
        print(f"\nNEXT STEPS:")
        print("   • Review optimization plots to understand trade-offs")
        print("   • Test the optimized protocols in full simulations")
        print("   • Consider patient-specific factors in clinical decisions")
        print("   • Validate results with sensitivity analysis")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if we're in the correct directory
    if not (project_root / 'cancer_model').exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    main()