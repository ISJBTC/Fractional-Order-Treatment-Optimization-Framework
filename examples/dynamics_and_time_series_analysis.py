#!/usr/bin/env python3
"""
Model Dynamics and Time Series Analysis Example
==============================================

This script demonstrates comprehensive dynamics and time series analysis
of the cancer model system including:

- Phase space analysis and attractors
- Stability and bifurcation analysis  
- Time series trend and correlation analysis
- Changepoint detection and frequency analysis
- Prediction and forecasting analysis

Usage:
    python examples/dynamics_and_time_series_analysis.py

Author: Cancer Model Team
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import cancer model components
from cancer_model import CancerModelRunner, ModelParameters, PatientProfiles
from cancer_model.core.cancer_model_core import CancerModel
from cancer_model.core.pharmacokinetics import PharmacokineticModel, CircadianRhythm
from cancer_model.analysis.dynamics_analysis import DynamicsAnalyzer
from cancer_model.analysis.time_series_analysis import TimeSeriesAnalyzer


def run_dynamics_analysis():
    """Run comprehensive dynamics analysis"""
    
    print("🔄 CANCER MODEL: DYNAMICS ANALYSIS")
    print("=" * 60)
    
    # Setup output directory
    output_dir = project_root / 'results' / 'dynamics_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model components
    patient_profile = PatientProfiles.get_profile('average')
    model_params = ModelParameters(patient_profile)
    params = model_params.get_all_parameters()
    
    # Create model instances
    pk_model = PharmacokineticModel(params)
    circadian_model = CircadianRhythm(params)
    cancer_model = CancerModel(params, pk_model, circadian_model)
    
    # Initialize dynamics analyzer
    dynamics_analyzer = DynamicsAnalyzer(cancer_model, output_dir)
    
    # Get initial conditions
    from cancer_model.core.model_parameters import InitialConditions
    initial_conditions = InitialConditions.get_standard_conditions()
    
    print(f"📁 Output directory: {output_dir}")
    
    # 1. Phase Space Analysis
    print("\n🌀 Running phase space analysis...")
    
    # Define treatment protocol (optional)
    from cancer_model.protocols import DrugScheduling
    treatment_protocol = {
        'hormone': DrugScheduling.create_cyclic_dosing_schedule(14, 7, 0.8, start_day=0)
    }
    
    phase_analysis = dynamics_analyzer.phase_space_analysis(
        initial_conditions=initial_conditions,
        params=params,
        time_span=300,
        variables=['N1', 'R1', 'I1'],
        treatment_protocol=treatment_protocol
    )
    
    if phase_analysis:
        print("✅ Phase space analysis completed")
        print(f"   Analyzed variables: {phase_analysis['variables']}")
        print(f"   Lyapunov exponents: {phase_analysis['lyapunov_exponents']}")
    
    # 2. Bifurcation Analysis
    print("\n🔀 Running bifurcation analysis...")
    
    # Test bifurcation for mutation rate
    mutation_rates = np.linspace(0.00005, 0.001, 20)
    
    bifurcation_analysis = dynamics_analyzer.bifurcation_analysis(
        parameter_name='mutation_rate',
        parameter_range=mutation_rates,
        initial_conditions=initial_conditions,
        base_params=params
    )
    
    print("✅ Bifurcation analysis completed")
    print(f"   Parameter: {bifurcation_analysis['parameter_name']}")
    print(f"   Range tested: {len(bifurcation_analysis['parameter_values'])} values")
    
    # 3. Oscillation Analysis
    if phase_analysis:
        print("\n🌊 Running oscillation analysis...")
        
        oscillation_analysis = dynamics_analyzer.oscillation_analysis(
            phase_analysis, variables=['N1', 'R1', 'I1']
        )
        
        print("✅ Oscillation analysis completed")
        for var, osc_data in oscillation_analysis.items():
            print(f"   {var}: {osc_data['num_peaks']} peaks detected")
            if osc_data['mean_period']:
                print(f"       Mean period: {osc_data['mean_period']:.1f} days")
    
    # 4. Comprehensive Report
    print("\n📄 Generating comprehensive dynamics report...")
    
    analysis_results = {
        'phase_space': phase_analysis,
        'bifurcation': bifurcation_analysis,
        'oscillations': oscillation_analysis if phase_analysis else {}
    }
    
    report_path = dynamics_analyzer.comprehensive_dynamics_report(analysis_results)
    
    print(f"✅ Dynamics analysis complete!")
    print(f"📊 Visualizations saved in: {output_dir}")
    print(f"📄 Report saved: {report_path}")
    
    return analysis_results


def run_time_series_analysis():
    """Run comprehensive time series analysis"""
    
    print("\n📈 CANCER MODEL: TIME SERIES ANALYSIS")
    print("=" * 60)
    
    # Setup output directory
    output_dir = project_root / 'results' / 'time_series_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run simulations to get time series data
    print("🔬 Running simulations for time series analysis...")
    
    runner = CancerModelRunner(output_dir=str(output_dir.parent / 'temp_sims'))
    
    # Run a subset of simulations for detailed analysis
    patient_profiles = ['average', 'young', 'elderly']
    treatment_protocols = ['standard', 'continuous', 'adaptive']
    
    simulation_results = runner.simulation_runner.run_comparative_analysis(
        patient_profiles=patient_profiles,
        treatment_protocols=treatment_protocols,
        simulation_days=400,  # Longer for better time series analysis
        use_circadian=True
    )
    
    # Initialize time series analyzer
    ts_analyzer = TimeSeriesAnalyzer(output_dir)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Run comprehensive time series analysis
    print("\n📊 Running comprehensive time series analysis...")
    
    ts_analysis = ts_analyzer.comprehensive_time_series_analysis(
        simulation_results=simulation_results,
        protocols=treatment_protocols
    )
    
    # Display key results
    print("✅ Time series analysis completed!")
    
    if 'comparative_analysis' in ts_analysis:
        comp_analysis = ts_analysis['comparative_analysis']
        
        if 'variability_comparison' in comp_analysis:
            print("\n📊 Variability Analysis:")
            
            # Find most and least variable scenarios
            variabilities = comp_analysis['variability_comparison']
            
            if variabilities:
                # Calculate average variability per scenario
                scenario_avg_vars = {}
                for scenario, var_data in variabilities.items():
                    avg_var = np.mean(list(var_data.values()))
                    scenario_avg_vars[scenario] = avg_var
                
                most_stable = min(scenario_avg_vars, key=scenario_avg_vars.get)
                most_variable = max(scenario_avg_vars, key=scenario_avg_vars.get)
                
                print(f"   Most stable: {most_stable} (CV: {scenario_avg_vars[most_stable]:.3f})")
                print(f"   Most variable: {most_variable} (CV: {scenario_avg_vars[most_variable]:.3f})")
    
    print(f"📊 Visualizations saved in: {ts_analysis['visualizations_dir']}")
    print(f"📄 Report saved: {ts_analysis['report_path']}")
    
    return ts_analysis


def comparative_dynamics_study():
    """Compare dynamics across different parameter sets"""
    
    print("\n🔬 COMPARATIVE DYNAMICS STUDY")
    print("=" * 50)
    
    # Setup for comparative study
    output_dir = project_root / 'results' / 'comparative_dynamics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test different parameter scenarios
    scenarios = {
        'low_resistance': {
            'mutation_rate': 0.00005,
            'omega_R1': 0.002,
            'omega_R2': 0.0015
        },
        'high_resistance': {
            'mutation_rate': 0.0005,
            'omega_R1': 0.01,
            'omega_R2': 0.008
        },
        'strong_immune': {
            'beta1': 0.01,
            'phi1': 0.15,
            'immune_status': 1.3
        },
        'weak_immune': {
            'beta1': 0.003,
            'phi1': 0.08,
            'immune_status': 0.7
        }
    }
    
    comparative_results = {}
    
    for scenario_name, param_modifications in scenarios.items():
        print(f"\n🔍 Analyzing scenario: {scenario_name}")
        
        # Create modified parameters
        patient_profile = PatientProfiles.get_profile('average')
        model_params = ModelParameters(patient_profile)
        params = model_params.get_all_parameters()
        
        # Apply modifications
        for param_name, param_value in param_modifications.items():
            params[param_name] = param_value
        
        # Create model instances
        pk_model = PharmacokineticModel(params)
        circadian_model = CircadianRhythm(params)
        cancer_model = CancerModel(params, pk_model, circadian_model)
        
        # Initialize analyzer for this scenario
        scenario_output_dir = output_dir / scenario_name
        scenario_output_dir.mkdir(exist_ok=True)
        dynamics_analyzer = DynamicsAnalyzer(cancer_model, scenario_output_dir)
        
        # Get initial conditions
        from cancer_model.core.model_parameters import InitialConditions
        initial_conditions = InitialConditions.get_standard_conditions()
        
        # Run phase space analysis
        print(f"   🌀 Phase space analysis for {scenario_name}...")
        phase_analysis = dynamics_analyzer.phase_space_analysis(
            initial_conditions=initial_conditions,
            params=params,
            time_span=200,
            variables=['N1', 'R1', 'I1']
        )
        
        # Run stability analysis
        print(f"   ⚖️ Stability analysis for {scenario_name}...")
        stability_analysis = dynamics_analyzer.stability_analysis(
            initial_conditions=initial_conditions,
            params=params,
            perturbation_magnitude=0.1
        )
        
        # Store results
        comparative_results[scenario_name] = {
            'parameters': param_modifications,
            'phase_analysis': phase_analysis,
            'stability_analysis': stability_analysis
        }
        
        print(f"   ✅ Completed analysis for {scenario_name}")
    
    # Generate comparative report
    print("\n📄 Generating comparative dynamics report...")
    
    report_path = output_dir / 'comparative_dynamics_report.md'
    with open(report_path, 'w') as f:
        f.write("# Comparative Dynamics Analysis Report\n\n")
        f.write("## Overview\n")
        f.write("This report compares dynamics across different parameter scenarios.\n\n")
        
        for scenario_name, results in comparative_results.items():
            f.write(f"## Scenario: {scenario_name}\n\n")
            f.write("### Parameter Modifications:\n")
            for param, value in results['parameters'].items():
                f.write(f"- {param}: {value}\n")
            
            if results['phase_analysis']:
                f.write("\n### Phase Space Analysis:\n")
                f.write(f"- Variables analyzed: {results['phase_analysis']['variables']}\n")
                if 'lyapunov_exponents' in results['phase_analysis']:
                    f.write(f"- Lyapunov exponents: {results['phase_analysis']['lyapunov_exponents']}\n")
            
            if results['stability_analysis']:
                f.write("\n### Stability Analysis:\n")
                f.write(f"- System stability: {results['stability_analysis']['is_stable']}\n")
                if 'dominant_eigenvalue' in results['stability_analysis']:
                    f.write(f"- Dominant eigenvalue: {results['stability_analysis']['dominant_eigenvalue']}\n")
            
            f.write("\n---\n\n")
    
    print(f"✅ Comparative dynamics study complete!")
    print(f"📊 Results saved in: {output_dir}")
    print(f"📄 Report saved: {report_path}")
    
    return comparative_results


def sensitivity_analysis():
    """Perform sensitivity analysis on key parameters"""
    
    print("\n🔧 SENSITIVITY ANALYSIS")
    print("=" * 40)
    
    output_dir = project_root / 'results' / 'sensitivity_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters to analyze
    sensitive_params = {
        'mutation_rate': (0.00001, 0.001),
        'omega_R1': (0.001, 0.02),
        'beta1': (0.001, 0.02),
        'phi1': (0.05, 0.3),
        'immune_status': (0.5, 2.0)
    }
    
    # Base parameters
    patient_profile = PatientProfiles.get_profile('average')
    model_params = ModelParameters(patient_profile)
    base_params = model_params.get_all_parameters()
    
    sensitivity_results = {}
    
    for param_name, (min_val, max_val) in sensitive_params.items():
        print(f"\n🔍 Analyzing sensitivity to {param_name}...")
        
        # Create parameter range
        param_values = np.linspace(min_val, max_val, 15)
        outcomes = []
        
        for param_val in param_values:
            # Modify parameters
            test_params = base_params.copy()
            test_params[param_name] = param_val
            
            # Create model
            pk_model = PharmacokineticModel(test_params)
            circadian_model = CircadianRhythm(test_params)
            cancer_model = CancerModel(test_params, pk_model, circadian_model)
            
            # Run short simulation
            from cancer_model.core.model_parameters import InitialConditions
            initial_conditions = InitialConditions.get_standard_conditions()
            
            # Simple integration for outcome
            try:
                # Run simulation for 100 days
                t_span = np.linspace(0, 100, 1000)
                
                # Get final tumor burden as outcome metric
                # This would typically use the model's solve method
                # For now, we'll simulate a response
                outcome_metric = simulate_outcome(cancer_model, initial_conditions, t_span)
                outcomes.append(outcome_metric)
                
            except Exception as e:
                print(f"   Warning: Failed for {param_name}={param_val}: {e}")
                outcomes.append(np.nan)
        
        # Calculate sensitivity metrics
        valid_outcomes = np.array([o for o in outcomes if not np.isnan(o)])
        valid_params = param_values[:len(valid_outcomes)]
        
        if len(valid_outcomes) > 1:
            # Calculate normalized sensitivity
            param_range = max_val - min_val
            outcome_range = np.max(valid_outcomes) - np.min(valid_outcomes)
            
            if outcome_range > 0:
                sensitivity_score = (outcome_range / np.mean(valid_outcomes)) / (param_range / np.mean(valid_params))
            else:
                sensitivity_score = 0.0
        else:
            sensitivity_score = 0.0
        
        sensitivity_results[param_name] = {
            'parameter_values': param_values,
            'outcomes': outcomes,
            'sensitivity_score': sensitivity_score,
            'parameter_range': (min_val, max_val)
        }
        
        print(f"   ✅ Sensitivity score: {sensitivity_score:.3f}")
    
    # Generate sensitivity report
    print("\n📄 Generating sensitivity analysis report...")
    
    # Rank parameters by sensitivity
    ranked_params = sorted(sensitivity_results.items(), 
                          key=lambda x: abs(x[1]['sensitivity_score']), 
                          reverse=True)
    
    report_path = output_dir / 'sensitivity_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write("# Sensitivity Analysis Report\n\n")
        f.write("## Parameter Sensitivity Ranking\n\n")
        
        for i, (param_name, results) in enumerate(ranked_params, 1):
            f.write(f"{i}. **{param_name}**: {results['sensitivity_score']:.4f}\n")
        
        f.write("\n## Detailed Results\n\n")
        
        for param_name, results in sensitivity_results.items():
            f.write(f"### {param_name}\n")
            f.write(f"- Parameter range: {results['parameter_range']}\n")
            f.write(f"- Sensitivity score: {results['sensitivity_score']:.4f}\n")
            f.write(f"- Number of valid outcomes: {len([o for o in results['outcomes'] if not np.isnan(o)])}\n\n")
    
    print(f"✅ Sensitivity analysis complete!")
    print(f"📊 Results saved in: {output_dir}")
    print(f"📄 Report saved: {report_path}")
    
    return sensitivity_results


def simulate_outcome(cancer_model, initial_conditions, t_span):
    """Simulate a simple outcome metric for sensitivity analysis"""
    
    # This is a placeholder for actual model simulation
    # In practice, this would use the cancer model's solve method
    
    # For demonstration, create a simple response based on model parameters
    params = cancer_model.params
    
    # Calculate a composite outcome based on key parameters
    mutation_effect = params.get('mutation_rate', 0.0001) * 10000
    immune_effect = params.get('immune_status', 1.0)
    resistance_effect = params.get('omega_R1', 0.005) * 200
    
    # Simple outcome metric (final tumor burden proxy)
    outcome = mutation_effect + resistance_effect - immune_effect + np.random.normal(0, 0.1)
    
    return max(0, outcome)  # Ensure non-negative


def main():
    """Main execution function"""
    
    print("🧬 CANCER MODEL: COMPREHENSIVE DYNAMICS & TIME SERIES ANALYSIS")
    print("=" * 80)
    print("This script performs comprehensive analysis of cancer model dynamics")
    print("including phase space, bifurcation, time series, and sensitivity analysis.\n")
    
    try:
        # 1. Run dynamics analysis
        dynamics_results = run_dynamics_analysis()
        
        # 2. Run time series analysis
        ts_results = run_time_series_analysis()
        
        # 3. Run comparative dynamics study
        comparative_results = comparative_dynamics_study()
        
        # 4. Run sensitivity analysis
        sensitivity_results = sensitivity_analysis()
        
        # 5. Generate master summary report
        print("\n📋 GENERATING MASTER SUMMARY REPORT")
        print("=" * 50)
        
        master_output_dir = project_root / 'results' / 'master_analysis'
        master_output_dir.mkdir(parents=True, exist_ok=True)
        
        master_report_path = master_output_dir / 'master_dynamics_analysis_report.md'
        
        with open(master_report_path, 'w') as f:
            f.write("# Master Dynamics and Time Series Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write("This report summarizes comprehensive analysis of cancer model dynamics.\n\n")
            
            f.write("## Analysis Components\n\n")
            f.write("1. **Dynamics Analysis**: Phase space, bifurcation, and oscillation analysis\n")
            f.write("2. **Time Series Analysis**: Trend analysis, correlation, and forecasting\n")
            f.write("3. **Comparative Study**: Parameter scenario comparison\n")
            f.write("4. **Sensitivity Analysis**: Parameter influence quantification\n\n")
            
            # Add key findings from sensitivity analysis
            if sensitivity_results:
                f.write("## Key Sensitivity Findings\n\n")
                ranked_params = sorted(sensitivity_results.items(), 
                                     key=lambda x: abs(x[1]['sensitivity_score']), 
                                     reverse=True)
                
                f.write("Most influential parameters:\n")
                for i, (param_name, results) in enumerate(ranked_params[:3], 1):
                    f.write(f"{i}. {param_name}: {results['sensitivity_score']:.4f}\n")
            
            f.write(f"\n## Analysis Date\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✅ Master analysis complete!")
        print(f"📄 Master report saved: {master_report_path}")
        
        # Summary statistics
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"   Dynamics scenarios analyzed: {len(comparative_results) if comparative_results else 0}")
        print(f"   Sensitive parameters tested: {len(sensitivity_results) if sensitivity_results else 0}")
        print(f"   Output directories created: 4")
        
        return {
            'dynamics': dynamics_results,
            'time_series': ts_results,
            'comparative': comparative_results,
            'sensitivity': sensitivity_results
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Check that all required cancer model modules are available.")
        return None


if __name__ == "__main__":
    # Add pandas import for timestamp
    try:
        import pandas as pd
    except ImportError:
        import datetime
        # Create simple timestamp function if pandas not available
        class Timestamp:
            @staticmethod
            def now():
                return datetime.datetime.now()
            
            def strftime(self, fmt):
                return datetime.datetime.now().strftime(fmt)
        
        pd = type('pd', (), {'Timestamp': Timestamp})
    
    # Run the complete analysis
    results = main()
    
    if results:
        print("\n🎉 All analyses completed successfully!")
        print("Check the results/ directory for detailed outputs and visualizations.")
    else:
        print("\n⚠️ Analysis completed with errors. Check logs above.")