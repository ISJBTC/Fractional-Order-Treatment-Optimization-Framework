"""
Module 8: Main Runner and Examples
=================================
Main execution module with examples of how to use the modular cancer model system.
Provides easy-to-use interfaces for different types of analyses.
"""

import numpy as np
from cancer_model.simulation.simulation_runner import SimulationRunner
from cancer_model.visualization.visualization_module import VisualizationEngine
from cancer_model.core.model_parameters import FineTuningPresets
import os


class CancerModelRunner:
    """Main class for running cancer model analyses"""
    
    def __init__(self, output_dir='cancer_model_results', fine_tuning_preset=None):
        """
        Initialize the cancer model runner
        
        Args:
            output_dir (str): Directory to save results
            fine_tuning_preset (str): Name of fine-tuning preset to apply
        """
        self.output_dir = output_dir
        self.simulation_runner = SimulationRunner(fine_tuning_preset)
        self.visualizer = VisualizationEngine(output_dir)
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Cancer Model Runner initialized")
        print(f"Output directory: {output_dir}")
        if fine_tuning_preset:
            print(f"Fine-tuning preset: {fine_tuning_preset}")
    
    def run_basic_analysis(self):
        """Run basic comparative analysis with default settings"""
        print("\n" + "="*60)
        print("RUNNING BASIC CANCER MODEL ANALYSIS")
        print("="*60)
        
        # Default patient profiles and protocols
        patient_profiles = ['average', 'young', 'elderly', 'compromised']
        treatment_protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']
        
        # Run comparative analysis
        results = self.simulation_runner.run_comparative_analysis(
            patient_profiles, treatment_protocols, simulation_days=500
        )
        
        # Create visualizations
        print("\n" + "-"*40)
        print("GENERATING VISUALIZATIONS")
        print("-"*40)
        
        visualization_paths = []
        
        # Protocol comparison for average patient
        path = self.visualizer.create_protocol_comparison_plot(results, 'average')
        if path: visualization_paths.append(path)
        
        # Efficacy metrics chart
        path = self.visualizer.create_efficacy_metrics_chart(results, 'average')
        if path: visualization_paths.append(path)
        
        # Patient comparison for standard protocol
        path = self.visualizer.create_patient_comparison_plot(results, 'standard')
        if path: visualization_paths.append(path)
        
        # Heatmaps comparing all combinations
        path = self.visualizer.create_heatmaps(results)
        if path: visualization_paths.append(path)
        
        # Detailed analysis for best protocol
        best_protocol, best_patient = self._find_best_protocol(results)
        if best_protocol and best_patient:
            best_result = results[best_patient][best_protocol]
            path = self.visualizer.create_detailed_analysis_plot(
                best_result, f" - {best_protocol.title()} Protocol ({best_patient.title()} Patient)"
            )
            if path: visualization_paths.append(path)
        
        # Generate summary report
        summary = self._generate_summary_report(results)
        
        print(f"\nBasic analysis complete!")
        print(f"Generated {len(visualization_paths)} visualizations")
        print(f"Results saved to: {self.output_dir}")
        
        return {
            'results': results,
            'visualizations': visualization_paths,
            'summary': summary,
            'best_protocol': best_protocol,
            'best_patient': best_patient
        }
    
    def run_optimization_analysis(self, patient_profiles=None):
        """Run treatment optimization for specific patient profiles"""
        print("\n" + "="*60)
        print("RUNNING TREATMENT OPTIMIZATION ANALYSIS")
        print("="*60)
        
        if patient_profiles is None:
            patient_profiles = ['elderly', 'compromised', 'young']
        
        optimization_results = {}
        visualization_paths = []
        
        for patient_profile in patient_profiles:
            print(f"\nOptimizing treatment for {patient_profile} patients...")
            
            # Run optimization
            opt_result = self.simulation_runner.optimize_patient_treatment(
                patient_profile, base_protocol='standard', output_results=True
            )
            
            optimization_results[patient_profile] = opt_result
            
            # Create optimization visualization
            path = self.visualizer.create_optimization_plot(
                opt_result['optimization_results'], patient_profile
            )
            if path: visualization_paths.append(path)
        
        print(f"\nOptimization analysis complete!")
        print(f"Generated {len(visualization_paths)} optimization plots")
        
        return {
            'optimization_results': optimization_results,
            'visualizations': visualization_paths
        }
    
    def run_sensitivity_analysis(self, base_patient='average', base_protocol='standard'):
        """Run parameter sensitivity analysis"""
        print("\n" + "="*60)
        print("RUNNING PARAMETER SENSITIVITY ANALYSIS")
        print("="*60)
        
        # Define parameters to analyze
        parameter_variations = {
            'alpha': [0.85, 0.90, 0.93, 0.95, 0.98],
            'mutation_rate': [0.00005, 0.0001, 0.0002, 0.0005, 0.001],
            'omega_R1': [0.002, 0.004, 0.006, 0.008, 0.01],
            'beta1': [0.003, 0.005, 0.007, 0.009, 0.012]
        }
        
        # Run sensitivity analysis
        sensitivity_results = self.simulation_runner.sensitivity_analysis(
            base_patient, base_protocol, parameter_variations, simulation_days=300
        )
        
        # Create visualization
        path = self.visualizer.create_sensitivity_analysis_plot(sensitivity_results)
        
        print(f"\nSensitivity analysis complete!")
        print(f"Analyzed {len(parameter_variations)} parameters")
        
        return {
            'sensitivity_results': sensitivity_results,
            'visualization': path
        }
    
    def run_custom_protocol_test(self, custom_protocols):
        """Test custom treatment protocols"""
        print("\n" + "="*60)
        print("TESTING CUSTOM TREATMENT PROTOCOLS")
        print("="*60)
        
        # Add custom protocols to the treatment system
        for protocol_name, protocol_config in custom_protocols.items():
            self.simulation_runner.treatment_protocols.create_custom_protocol(
                protocol_name,
                protocol_config['description'],
                protocol_config['drugs'],
                protocol_config.get('temperature', lambda t: 37.0),
                protocol_config.get('parameters', {})
            )
        
        # Test protocols on average patient
        results = {}
        visualization_paths = []
        
        for protocol_name in custom_protocols.keys():
            print(f"\nTesting {protocol_name} protocol...")
            
            result = self.simulation_runner.run_single_simulation(
                'average', protocol_name, simulation_days=500
            )
            
            results[protocol_name] = result
            
            if result['success']:
                # Create detailed analysis
                path = self.visualizer.create_detailed_analysis_plot(
                    result, f" - {protocol_name.title()} Protocol"
                )
                if path: visualization_paths.append(path)
        
        print(f"\nCustom protocol testing complete!")
        print(f"Tested {len(custom_protocols)} custom protocols")
        
        return {
            'results': results,
            'visualizations': visualization_paths
        }
    
    def run_comprehensive_analysis(self):
        """Run all types of analyses"""
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE CANCER MODEL ANALYSIS")
        print("="*80)
        
        all_results = {}
        all_visualizations = []
        
        # 1. Basic analysis
        print("\n1. BASIC COMPARATIVE ANALYSIS")
        basic_results = self.run_basic_analysis()
        all_results['basic'] = basic_results
        all_visualizations.extend(basic_results['visualizations'])
        
        # 2. Optimization analysis
        print("\n2. TREATMENT OPTIMIZATION")
        opt_results = self.run_optimization_analysis()
        all_results['optimization'] = opt_results
        all_visualizations.extend(opt_results['visualizations'])
        
        # 3. Sensitivity analysis
        print("\n3. PARAMETER SENSITIVITY")
        sens_results = self.run_sensitivity_analysis()
        all_results['sensitivity'] = sens_results
        if sens_results['visualization']:
            all_visualizations.append(sens_results['visualization'])
        
        # Generate comprehensive report
        report_path = self.visualizer.generate_comprehensive_report(
            basic_results['results']
        )
        
        print(f"\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total visualizations created: {len(all_visualizations)}")
        print(f"Results directory: {self.output_dir}")
        print(f"Report: {report_path}")
        
        return all_results
    
    def _find_best_protocol(self, results):
        """Find the best performing protocol across all patients"""
        best_efficacy = 0
        best_protocol = None
        best_patient = None
        
        for patient_profile, protocols in results.items():
            for protocol_name, result in protocols.items():
                if result.get('success', False):
                    efficacy = result['metrics']['treatment_efficacy_score']
                    if efficacy > best_efficacy:
                        best_efficacy = efficacy
                        best_protocol = protocol_name
                        best_patient = patient_profile
        
        return best_protocol, best_patient
    
    def _generate_summary_report(self, results):
        """Generate a text summary of results"""
        report_path = os.path.join(self.output_dir, 'analysis_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("Cancer Model Analysis Summary\n")
            f.write("============================\n\n")
            
            # Count simulations
            total_sims = 0
            successful_sims = 0
            
            for patient_profile, protocols in results.items():
                for protocol_name, result in protocols.items():
                    total_sims += 1
                    if result.get('success', False):
                        successful_sims += 1
            
            f.write(f"Total simulations: {total_sims}\n")
            f.write(f"Successful simulations: {successful_sims}\n")
            f.write(f"Success rate: {100*successful_sims/total_sims:.1f}%\n\n")
            
            # Best protocol
            best_protocol, best_patient = self._find_best_protocol(results)
            if best_protocol:
                best_result = results[best_patient][best_protocol]
                f.write(f"Best overall result:\n")
                f.write(f"  Patient: {best_patient.title()}\n")
                f.write(f"  Protocol: {best_protocol.title()}\n")
                f.write(f"  Efficacy Score: {best_result['metrics']['treatment_efficacy_score']:.2f}\n")
                f.write(f"  Tumor Reduction: {best_result['metrics']['percent_reduction']:.2f}%\n")
                f.write(f"  Final Resistance: {best_result['metrics']['final_resistance_fraction']:.2f}%\n\n")
            
            # Protocol rankings
            protocol_scores = {}
            protocol_counts = {}
            
            for patient_profile, protocols in results.items():
                for protocol_name, result in protocols.items():
                    if result.get('success', False):
                        score = result['metrics']['treatment_efficacy_score']
                        if protocol_name not in protocol_scores:
                            protocol_scores[protocol_name] = 0
                            protocol_counts[protocol_name] = 0
                        protocol_scores[protocol_name] += score
                        protocol_counts[protocol_name] += 1
            
            # Calculate averages
            avg_scores = {}
            for protocol in protocol_scores:
                if protocol_counts[protocol] > 0:
                    avg_scores[protocol] = protocol_scores[protocol] / protocol_counts[protocol]
            
            # Sort by average score
            sorted_protocols = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            f.write("Protocol Rankings (by average efficacy):\n")
            f.write("-" * 40 + "\n")
            for i, (protocol, avg_score) in enumerate(sorted_protocols):
                f.write(f"{i+1}. {protocol.title()}: {avg_score:.2f}\n")
        
        return report_path


# Example usage functions
def example_basic_analysis():
    """Example: Run basic comparative analysis"""
    runner = CancerModelRunner(output_dir='basic_analysis_results')
    results = runner.run_basic_analysis()
    return results


def example_with_realistic_resistance():
    """Example: Run analysis with realistic resistance parameters"""
    runner = CancerModelRunner(
        output_dir='realistic_resistance_results',
        fine_tuning_preset='realistic_resistance'
    )
    results = runner.run_basic_analysis()
    return results


def example_optimization_focus():
    """Example: Focus on treatment optimization"""
    runner = CancerModelRunner(output_dir='optimization_results')
    
    # Run optimization for specific patient types
    opt_results = runner.run_optimization_analysis(['elderly', 'compromised'])
    
    # Also run sensitivity analysis
    sens_results = runner.run_sensitivity_analysis()
    
    return {
        'optimization': opt_results,
        'sensitivity': sens_results
    }


def example_custom_protocols():
    """Example: Test custom treatment protocols"""
    from cancer_model.core.pharmacokinetics import DrugScheduling, TemperatureProtocol
    
    # Define custom protocols
    custom_protocols = {
        'ultra_low_dose': {
            'description': 'Ultra-low dose continuous therapy',
            'drugs': {
                'hormone': DrugScheduling.create_continuous_dosing_schedule(0.2),
                'her2': DrugScheduling.create_continuous_dosing_schedule(0.3)
            }
        },
        'pulse_therapy': {
            'description': 'High-dose pulse therapy with long breaks',
            'drugs': {
                'chemo': DrugScheduling.create_cyclic_dosing_schedule(3, 28, 1.2),
                'immuno': DrugScheduling.create_cyclic_dosing_schedule(1, 35, 1.0)
            }
        },
        'temperature_enhanced': {
            'description': 'Standard therapy with aggressive hyperthermia',
            'drugs': {
                'hormone': DrugScheduling.create_cyclic_dosing_schedule(14, 7, 0.8),
                'her2': DrugScheduling.create_cyclic_dosing_schedule(14, 7, 0.8)
            },
            'temperature': TemperatureProtocol.create_cyclic_hyperthermia(14, 5, 39.5, 37.0)
        }
    }
    
    runner = CancerModelRunner(output_dir='custom_protocol_results')
    results = runner.run_custom_protocol_test(custom_protocols)
    return results


def example_comprehensive():
    """Example: Run comprehensive analysis with all features"""
    runner = CancerModelRunner(output_dir='comprehensive_results')
    results = runner.run_comprehensive_analysis()
    return results


# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Cancer Model - Modular System")
    print("=" * 50)
    print("Available examples:")
    print("1. Basic Analysis")
    print("2. Realistic Resistance Analysis")
    print("3. Optimization Focus")
    print("4. Custom Protocols")
    print("5. Comprehensive Analysis")
    
    # Default: Run basic analysis
    print("\nRunning basic analysis...")
    results = example_basic_analysis()
    
    print("\nTo run other examples, call the appropriate function:")
    print("- example_with_realistic_resistance()")
    print("- example_optimization_focus()")
    print("- example_custom_protocols()")
    print("- example_comprehensive()")


# === FINE-TUNING EXAMPLES ===

def fine_tune_resistance_parameters():
    """Example: Fine-tune resistance development parameters"""
    from cancer_model.core.model_parameters import ModelParameters, PatientProfiles
    
    # Create base parameters
    patient_profile = PatientProfiles.get_profile('average')
    params = ModelParameters(patient_profile)
    
    # Test different resistance parameter combinations
    resistance_configs = [
        {'omega_R1': 0.004, 'omega_R2': 0.003, 'mutation_rate': 0.0001},  # Original
        {'omega_R1': 0.008, 'omega_R2': 0.006, 'mutation_rate': 0.0003},  # Higher resistance
        {'omega_R1': 0.012, 'omega_R2': 0.009, 'mutation_rate': 0.0005},  # Very high resistance
        {'omega_R1': 0.002, 'omega_R2': 0.0015, 'mutation_rate': 0.00005}, # Lower resistance
    ]
    
    results = {}
    
    for i, config in enumerate(resistance_configs):
        print(f"\nTesting resistance configuration {i+1}: {config}")
        
        # Update parameters
        params.update_parameters(config)
        
        # Run a quick simulation
        runner = CancerModelRunner(output_dir=f'resistance_tune_{i+1}')
        result = runner.simulation_runner.run_single_simulation(
            'average', 'standard', simulation_days=300
        )
        
        if result['success']:
            final_resistance = result['metrics']['final_resistance_fraction']
            efficacy = result['metrics']['treatment_efficacy_score']
            print(f"  Final resistance: {final_resistance:.2f}%")
            print(f"  Efficacy score: {efficacy:.2f}")
            
            results[f'config_{i+1}'] = {
                'parameters': config,
                'final_resistance': final_resistance,
                'efficacy': efficacy
            }
    
    return results


def fine_tune_immune_parameters():
    """Example: Fine-tune immune system parameters"""
    from cancer_model.core.model_parameters import ModelParameters, PatientProfiles
    
    # Test different immune parameter combinations
    immune_configs = [
        {'beta1': 0.005, 'phi1': 0.1, 'delta_I': 0.04},     # Original
        {'beta1': 0.008, 'phi1': 0.15, 'delta_I': 0.03},    # Strong immune
        {'beta1': 0.012, 'phi1': 0.2, 'delta_I': 0.025},    # Very strong immune
        {'beta1': 0.003, 'phi1': 0.08, 'delta_I': 0.05},    # Weak immune
    ]
    
    results = {}
    
    for i, config in enumerate(immune_configs):
        print(f"\nTesting immune configuration {i+1}: {config}")
        
        # Create parameters with this config
        patient_profile = PatientProfiles.get_profile('average')
        patient_profile.update(config)  # Add immune parameters
        
        # Run simulation
        runner = CancerModelRunner(output_dir=f'immune_tune_{i+1}')
        result = runner.simulation_runner.run_single_simulation(
            'average', 'immuno_combo', simulation_days=300
        )
        
        if result['success']:
            immune_activation = result['metrics']['immune_activation']
            efficacy = result['metrics']['treatment_efficacy_score']
            print(f"  Immune activation: {immune_activation:.2f}x")
            print(f"  Efficacy score: {efficacy:.2f}")
            
            results[f'config_{i+1}'] = {
                'parameters': config,
                'immune_activation': immune_activation,
                'efficacy': efficacy
            }
    
    return results


print("\nModular Cancer Model System loaded successfully!")
print("Use the example functions to run different types of analyses.")
print("For fine-tuning, use fine_tune_resistance_parameters() or fine_tune_immune_parameters()")
