#!/usr/bin/env python3
"""
Fixed Sensitivity Analysis Example
==================================

This script demonstrates parameter sensitivity analysis for the cancer model system.
It systematically varies key parameters to understand their impact on treatment outcomes.

Features:
- Multi-parameter sensitivity analysis
- Visualization of parameter effects
- Identification of critical parameters
- Robustness assessment

Usage:
    python examples/sensitivity_analysis_fixed.py

Author: Cancer Model Team
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import cancer model components
from cancer_model import CancerModelRunner, PatientProfiles, ModelParameters
from cancer_model.protocols import TreatmentProtocols


def single_parameter_sensitivity(parameter_name, parameter_values, base_patient='average', 
                                base_protocol='standard', simulation_days=300):
    """
    Analyze sensitivity to a single parameter.
    
    Args:
        parameter_name (str): Name of parameter to vary
        parameter_values (list): List of values to test
        base_patient (str): Base patient profile
        base_protocol (str): Base treatment protocol
        simulation_days (int): Simulation duration
        
    Returns:
        pd.DataFrame: Results dataframe
    """
    print(f"\nAnalyzing sensitivity to {parameter_name}...")
    print(f"Testing values: {parameter_values}")
    
    results = []
    
    for i, value in enumerate(parameter_values):
        print(f"  Testing {parameter_name} = {value} ({i+1}/{len(parameter_values)})")
        
        try:
            # Create custom patient profile with modified parameter
            patient_profile = PatientProfiles.get_profile(base_patient)
            patient_profile[parameter_name] = value
            
            # Setup output directory
            output_dir = project_root / 'results' / 'sensitivity' / parameter_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize runner
            runner = CancerModelRunner(output_dir=str(output_dir))
            
            # Run simulation
            result = runner.simulation_runner.run_single_simulation(
                base_patient, base_protocol, simulation_days
            )
            
            if result['success']:
                metrics = result['metrics']
                results.append({
                    'parameter': parameter_name,
                    'parameter_value': value,
                    'efficacy_score': metrics['treatment_efficacy_score'],
                    'tumor_reduction': metrics['percent_reduction'],
                    'final_resistance': metrics['final_resistance_fraction'],
                    'immune_activation': metrics['immune_activation'],
                    'genetic_instability': metrics['genetic_instability'],
                    'time_to_control': metrics.get('time_to_control', None),
                    'success': True
                })
            else:
                print(f"    Simulation failed: {result['error_message']}")
                results.append({
                    'parameter': parameter_name,
                    'parameter_value': value,
                    'efficacy_score': 0,
                    'tumor_reduction': 0,
                    'final_resistance': 100,
                    'immune_activation': 0,
                    'genetic_instability': 1,
                    'time_to_control': None,
                    'success': False
                })
                
        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                'parameter': parameter_name,
                'parameter_value': value,
                'efficacy_score': 0,
                'tumor_reduction': 0,
                'final_resistance': 100,
                'immune_activation': 0,
                'genetic_instability': 1,
                'time_to_control': None,
                'success': False
            })
    
    return pd.DataFrame(results)


def multi_parameter_sensitivity():
    """
    Analyze sensitivity to multiple key parameters.
    
    Returns:
        dict: Dictionary of DataFrames for each parameter
    """
    print("MULTI-PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 50)
    
    # Define parameters to analyze with their test ranges
    parameter_configs = {
        'alpha': {
            'values': [0.85, 0.88, 0.90, 0.93, 0.95, 0.97, 0.99],
            'description': 'Fractional order (memory effects)'
        },
        'mutation_rate': {
            'values': [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005],
            'description': 'Base mutation rate'
        },
        'omega_R1': {
            'values': [0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02],
            'description': 'Type 1 resistance development rate'
        },
        'beta1': {
            'values': [0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02],
            'description': 'Immune killing rate'
        },
        'lambda1': {
            'values': [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01],
            'description': 'Sensitive cell growth rate'
        },
        'etaE': {
            'values': [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
            'description': 'Hormone therapy effectiveness'
        }
    }
    
    all_results = {}
    
    for param_name, config in parameter_configs.items():
        print(f"\n{param_name.upper()}: {config['description']}")
        
        # Run sensitivity analysis for this parameter
        param_results = single_parameter_sensitivity(
            param_name, 
            config['values'],
            simulation_days=200  # Shorter for multiple parameters
        )
        
        all_results[param_name] = param_results
        
        # Quick summary
        successful_runs = param_results[param_results['success']].shape[0]
        total_runs = param_results.shape[0]
        print(f"  Successful runs: {successful_runs}/{total_runs}")
        
        if successful_runs > 0:
            best_result = param_results.loc[param_results['efficacy_score'].idxmax()]
            print(f"  Best efficacy: {best_result['efficacy_score']:.2f} at {param_name}={best_result['parameter_value']}")
    
    return all_results


def create_sensitivity_plots(all_results, output_dir):
    """Create comprehensive sensitivity analysis plots."""
    
    print(f"\nCreating sensitivity analysis plots...")
    
    # Determine number of parameters
    n_params = len(all_results)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Create main sensitivity plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
    
    plot_idx = 0
    for param_name, results_df in all_results.items():
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]
        
        # Filter successful results
        success_df = results_df[results_df['success']]
        
        if len(success_df) > 0:
            # Create twin axis for dual metrics
            ax2 = ax.twinx()
            
            # Plot efficacy score
            line1 = ax.plot(success_df['parameter_value'], success_df['efficacy_score'], 
                           'b-o', linewidth=2, markersize=6, label='Efficacy Score')
            ax.set_ylabel('Efficacy Score', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            # Plot resistance fraction
            line2 = ax2.plot(success_df['parameter_value'], success_df['final_resistance'], 
                            'r-s', linewidth=2, markersize=6, label='Final Resistance (%)')
            ax2.set_ylabel('Final Resistance (%)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Formatting
            ax.set_xlabel(param_name)
            ax.set_title(f'Sensitivity to {param_name}')
            ax.grid(True, alpha=0.3)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best')
        else:
            ax.text(0.5, 0.5, 'No successful\nsimulations', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sensitivity to {param_name}')
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save main plot
    main_plot_file = output_dir / 'parameter_sensitivity_analysis.png'
    plt.savefig(main_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Main sensitivity plot: {main_plot_file}")
    
    # Create correlation analysis
    create_correlation_analysis(all_results, output_dir)
    
    # Create parameter ranking plot
    create_parameter_ranking(all_results, output_dir)


def create_correlation_analysis(all_results, output_dir):
    """Create correlation analysis between parameters and outcomes."""
    
    print("  Creating correlation analysis...")
    
    # Combine all successful results
    combined_data = []
    
    for param_name, results_df in all_results.items():
        success_df = results_df[results_df['success']]
        for _, row in success_df.iterrows():
            combined_data.append({
                'parameter': param_name,
                'parameter_value': row['parameter_value'],
                'efficacy_score': row['efficacy_score'],
                'tumor_reduction': row['tumor_reduction'],
                'final_resistance': row['final_resistance'],
                'immune_activation': row['immune_activation']
            })
    
    if not combined_data:
        print("    No successful simulations for correlation analysis")
        return
    
    combined_df = pd.DataFrame(combined_data)
    
    # Create correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Parameter effects on efficacy
    try:
        efficacy_by_param = combined_df.pivot_table(
            values='efficacy_score', 
            index='parameter', 
            columns='parameter_value', 
            aggfunc='mean'
        )
        
        if not efficacy_by_param.empty:
            sns.heatmap(efficacy_by_param, annot=True, fmt='.1f', cmap='viridis', ax=axes[0])
            axes[0].set_title('Efficacy Score by Parameter Value')
            axes[0].set_xlabel('Parameter Value')
            axes[0].set_ylabel('Parameter')
    except Exception as e:
        print(f"    Error creating efficacy heatmap: {e}")
        axes[0].text(0.5, 0.5, 'Could not create\nefficacy heatmap', 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # Plot 2: Parameter effects on resistance
    try:
        resistance_by_param = combined_df.pivot_table(
            values='final_resistance', 
            index='parameter', 
            columns='parameter_value', 
            aggfunc='mean'
        )
        
        if not resistance_by_param.empty:
            sns.heatmap(resistance_by_param, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1])
            axes[1].set_title('Final Resistance by Parameter Value')
            axes[1].set_xlabel('Parameter Value')
            axes[1].set_ylabel('Parameter')
    except Exception as e:
        print(f"    Error creating resistance heatmap: {e}")
        axes[1].text(0.5, 0.5, 'Could not create\nresistance heatmap', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    
    # Save correlation plot
    corr_plot_file = output_dir / 'parameter_correlation_analysis.png'
    plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Correlation analysis: {corr_plot_file}")


def create_parameter_ranking(all_results, output_dir):
    """Create parameter importance ranking."""
    
    print("  Creating parameter ranking...")
    
    # Calculate sensitivity metrics for each parameter
    sensitivity_metrics = []
    
    for param_name, results_df in all_results.items():
        success_df = results_df[results_df['success']]
        
        if len(success_df) > 1:
            # Calculate coefficient of variation for efficacy
            efficacy_mean = success_df['efficacy_score'].mean()
            efficacy_std = success_df['efficacy_score'].std()
            efficacy_cv = efficacy_std / efficacy_mean if efficacy_mean > 0 else 0
            
            # Calculate range of effects
            efficacy_range = success_df['efficacy_score'].max() - success_df['efficacy_score'].min()
            resistance_range = success_df['final_resistance'].max() - success_df['final_resistance'].min()
            
            sensitivity_metrics.append({
                'parameter': param_name,
                'efficacy_cv': efficacy_cv,
                'efficacy_range': efficacy_range,
                'resistance_range': resistance_range,
                'sensitivity_score': efficacy_cv * efficacy_range  # Combined metric
            })
    
    if not sensitivity_metrics:
        print("    No data for parameter ranking")
        return
    
    ranking_df = pd.DataFrame(sensitivity_metrics)
    ranking_df = ranking_df.sort_values('sensitivity_score', ascending=False)
    
    # Create ranking plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Sensitivity scores
    axes[0].barh(ranking_df['parameter'], ranking_df['sensitivity_score'])
    axes[0].set_xlabel('Sensitivity Score')
    axes[0].set_title('Parameter Sensitivity Ranking')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Efficacy vs Resistance sensitivity
    scatter = axes[1].scatter(ranking_df['efficacy_range'], ranking_df['resistance_range'], 
                             s=100, alpha=0.7)
    
    # Add parameter labels
    for _, row in ranking_df.iterrows():
        axes[1].annotate(row['parameter'], 
                        (row['efficacy_range'], row['resistance_range']),
                        xytext=(5, 5), textcoords='offset points')
    
    axes[1].set_xlabel('Efficacy Range')
    axes[1].set_ylabel('Resistance Range')
    axes[1].set_title('Parameter Impact on Efficacy vs Resistance')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save ranking plot
    ranking_plot_file = output_dir / 'parameter_ranking.png'
    plt.savefig(ranking_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Parameter ranking: {ranking_plot_file}")
    
    return ranking_df


def generate_sensitivity_report(all_results, ranking_df, output_dir):
    """Generate comprehensive sensitivity analysis report."""
    
    print("  Generating sensitivity report...")
    
    report_file = output_dir / 'sensitivity_analysis_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("PARAMETER SENSITIVITY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        total_params = len(all_results)
        total_simulations = sum(len(df) for df in all_results.values())
        successful_simulations = sum(df['success'].sum() for df in all_results.values())
        
        f.write(f"Parameters analyzed: {total_params}\n")
        f.write(f"Total simulations: {total_simulations}\n")
        f.write(f"Successful simulations: {successful_simulations}\n")
        f.write(f"Success rate: {100*successful_simulations/total_simulations:.1f}%\n\n")
        
        # Parameter ranking
        if ranking_df is not None and len(ranking_df) > 0:
            f.write("PARAMETER SENSITIVITY RANKING\n")
            f.write("-" * 30 + "\n")
            f.write("(Ranked by impact on treatment efficacy)\n\n")
            
            for i, (_, row) in enumerate(ranking_df.iterrows()):
                f.write(f"{i+1}. {row['parameter'].upper()}\n")
                f.write(f"   Sensitivity Score: {row['sensitivity_score']:.3f}\n")
                f.write(f"   Efficacy Range: {row['efficacy_range']:.2f}\n")
                f.write(f"   Resistance Range: {row['resistance_range']:.2f}\n\n")
        
        # Detailed parameter analysis
        f.write("DETAILED PARAMETER ANALYSIS\n")
        f.write("-" * 35 + "\n\n")
        
        for param_name, results_df in all_results.items():
            success_df = results_df[results_df['success']]
            
            f.write(f"{param_name.upper()}\n")
            f.write("." * len(param_name) + "\n")
            
            if len(success_df) > 0:
                # Best and worst values
                best_idx = success_df['efficacy_score'].idxmax()
                worst_idx = success_df['efficacy_score'].idxmin()
                
                best_result = success_df.loc[best_idx]
                worst_result = success_df.loc[worst_idx]
                
                f.write(f"Best value: {best_result['parameter_value']}\n")
                f.write(f"  Efficacy: {best_result['efficacy_score']:.2f}\n")
                f.write(f"  Resistance: {best_result['final_resistance']:.2f}%\n")
                f.write(f"Worst value: {worst_result['parameter_value']}\n")
                f.write(f"  Efficacy: {worst_result['efficacy_score']:.2f}\n")
                f.write(f"  Resistance: {worst_result['final_resistance']:.2f}%\n")
                
                # Statistics
                f.write(f"Mean efficacy: {success_df['efficacy_score'].mean():.2f}\n")
                f.write(f"Std efficacy: {success_df['efficacy_score'].std():.2f}\n")
                f.write(f"Range: {success_df['efficacy_score'].max() - success_df['efficacy_score'].min():.2f}\n")
                
            else:
                f.write("No successful simulations\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        f.write("1. Focus on the top-ranked parameters for model calibration\n")
        f.write("2. Consider uncertainty ranges for sensitive parameters\n")
        f.write("3. Validate model behavior at parameter extremes\n")
        f.write("4. Use parameter sensitivity for robust design\n")
        f.write("5. Consider multi-parameter optimization for critical applications\n")
    
    print(f"  Sensitivity report: {report_file}")


def main():
    """Main sensitivity analysis workflow."""
    
    print("CANCER MODEL: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Setup output directory
    output_dir = project_root / 'results' / 'sensitivity'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run multi-parameter sensitivity analysis
        print("Starting comprehensive parameter sensitivity analysis...")
        print("This may take 10-15 minutes depending on your system...")
        
        all_results = multi_parameter_sensitivity()
        
        # Save individual parameter results
        for param_name, results_df in all_results.items():
            param_file = output_dir / f'{param_name}_sensitivity_results.csv'
            results_df.to_csv(param_file, index=False)
            print(f"Saved {param_name} results: {param_file}")
        
        # Create visualizations
        create_sensitivity_plots(all_results, output_dir)
        
        # Create parameter ranking
        ranking_df = create_parameter_ranking(all_results, output_dir)
        
        # Generate comprehensive report
        generate_sensitivity_report(all_results, ranking_df, output_dir)
        
        # Summary of findings
        print(f"\nSENSITIVITY ANALYSIS COMPLETE!")
        print("=" * 40)
        print(f"Results saved to: {output_dir}")
        
        if ranking_df is not None and len(ranking_df) > 0:
            print(f"\nMost sensitive parameters:")
            for i, (_, row) in enumerate(ranking_df.head(3).iterrows()):
                print(f"  {i+1}. {row['parameter']} (score: {row['sensitivity_score']:.3f})")
        
        print(f"\nGenerated files:")
        print(f"  • Individual parameter CSV files")
        print(f"  • Sensitivity analysis plots")
        print(f"  • Parameter ranking visualization")
        print(f"  • Comprehensive analysis report")
        
        print(f"\nKey insights:")
        print(f"  • Review parameter_sensitivity_analysis.png for overall trends")
        print(f"  • Check parameter_ranking.png for parameter importance")
        print(f"  • Read sensitivity_analysis_report.txt for detailed findings")
        print(f"  • Use results to guide model calibration and uncertainty analysis")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError during sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if we're in the correct directory
    if not (project_root / 'cancer_model').exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    main()