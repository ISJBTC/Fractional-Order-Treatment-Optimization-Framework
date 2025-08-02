"""
Module 6: Simulation Runner
===========================
Orchestrates model simulations with different protocols and patient profiles.
Handles the integration of all components and runs comparative analyses.
"""

import numpy as np
from cancer_model.core.model_parameters import ModelParameters, PatientProfiles, InitialConditions, FineTuningPresets
from cancer_model.core.pharmacokinetics import PharmacokineticModel, CircadianRhythm
from cancer_model.core.cancer_model_core import CancerModel
from cancer_model.protocols.treatment_protocols import TreatmentProtocols, ProtocolOptimizer
from cancer_model.core.fractional_math import safe_solve_ivp


class SimulationRunner:
    """Main simulation orchestrator"""
    
    def __init__(self, fine_tuning_preset=None):
        self.treatment_protocols = TreatmentProtocols()
        self.optimizer = ProtocolOptimizer(self.treatment_protocols)
        self.fine_tuning_preset = fine_tuning_preset
        
        # Apply fine-tuning preset if provided
        if fine_tuning_preset:
            print(f"Applying fine-tuning preset: {fine_tuning_preset}")
    
    def run_single_simulation(self, patient_profile_name, protocol_name, 
                            simulation_days=500, use_circadian=True):
        """
        Run a single simulation with specified patient and protocol
        
        Args:
            patient_profile_name (str): Name of patient profile
            protocol_name (str): Name of treatment protocol
            simulation_days (int): Number of days to simulate
            use_circadian (bool): Whether to use circadian effects
            
        Returns:
            dict: Simulation results and metrics
        """
        print(f"Running simulation: {patient_profile_name} patient with {protocol_name} protocol...")
        
        # Load patient profile and create parameters
        patient_profile = PatientProfiles.get_profile(patient_profile_name)
        model_params = ModelParameters(patient_profile)
        
        # Apply fine-tuning preset if specified
        if self.fine_tuning_preset:
            preset_params = getattr(FineTuningPresets, self.fine_tuning_preset, lambda: {})()
            model_params.update_parameters(preset_params)
        
        # Get treatment protocol
        protocol = self.treatment_protocols.get_protocol(protocol_name, patient_profile)
        
        # Create model components
        pk_model = PharmacokineticModel(model_params.get_all_parameters())
        circadian_model = CircadianRhythm(model_params.get_all_parameters())
        cancer_model = CancerModel(model_params.get_all_parameters(), pk_model, circadian_model)
        
        # Setup simulation
        t_span = [0, simulation_days]
        t_eval = np.linspace(0, simulation_days, simulation_days + 1)
        
        # Get initial conditions
        initial_conditions = InitialConditions.get_conditions_for_profile(patient_profile_name)
        
        # Create temperature function
        temp_func = protocol.get('temperature', lambda t: 37.0)
        
        # Define the model function for the solver
        def model_function(t, y):
            current_temp = temp_func(t)
            return cancer_model.enhanced_temperature_cancer_model(
                t, y, protocol['drugs'], current_temp, use_circadian
            )
        
        # Run simulation
        result = safe_solve_ivp(model_function, t_span, initial_conditions, 'RK45', t_eval)
        
        # Process results if successful
        if result.success:
            metrics = self._calculate_metrics(result, protocol)
            time_series = self._extract_time_series(result)
            
            return {
                'success': True,
                'metrics': metrics,
                'time_series': time_series,
                'protocol': protocol,
                'patient_profile': patient_profile,
                'model_params': model_params.get_all_parameters()
            }
        else:
            return {
                'success': False,
                'error_message': result.message,
                'protocol': protocol,
                'patient_profile': patient_profile,
                'model_params': model_params.get_all_parameters()
            }
    
    def run_comparative_analysis(self, patient_profiles=None, treatment_protocols=None, 
                               simulation_days=500, use_circadian=True):
        """
        Run comparative analysis across multiple patient profiles and treatment protocols
        
        Args:
            patient_profiles (list): List of patient profile names
            treatment_protocols (list): List of treatment protocol names
            simulation_days (int): Number of days to simulate
            use_circadian (bool): Whether to use circadian effects
            
        Returns:
            dict: Results for all simulations
        """
        # Default profiles and protocols if not provided
        if patient_profiles is None:
            patient_profiles = ['average', 'young', 'elderly', 'compromised']
        
        if treatment_protocols is None:
            treatment_protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']
        
        print(f"\nRunning comparative analysis:")
        print(f"Patient profiles: {patient_profiles}")
        print(f"Treatment protocols: {treatment_protocols}")
        print(f"Simulation duration: {simulation_days} days")
        
        # Store all results
        all_results = {}
        total_simulations = len(patient_profiles) * len(treatment_protocols)
        current_simulation = 0
        
        # Run simulations for each combination
        for profile in patient_profiles:
            all_results[profile] = {}
            
            for protocol in treatment_protocols:
                current_simulation += 1
                print(f"\nProgress: {current_simulation}/{total_simulations}")
                
                # Run simulation
                result = self.run_single_simulation(profile, protocol, simulation_days, use_circadian)
                
                # Store result
                all_results[profile][protocol] = result
                
                # Print quick summary
                if result['success']:
                    efficacy = result['metrics']['treatment_efficacy_score']
                    resistance = result['metrics']['final_resistance_fraction']
                    print(f"  Efficacy: {efficacy:.2f}, Resistance: {resistance:.2f}%")
                else:
                    print(f"  FAILED: {result['error_message']}")
        
        return all_results
    
    def optimize_patient_treatment(self, patient_profile_name, base_protocol='standard', 
                                 output_results=True):
        """
        Optimize treatment for a specific patient profile
        
        Args:
            patient_profile_name (str): Name of patient profile
            base_protocol (str): Base protocol to optimize
            output_results (bool): Whether to print optimization results
            
        Returns:
            dict: Optimization results
        """
        print(f"\nOptimizing treatment for {patient_profile_name} patient...")
        
        # Load patient profile
        patient_profile = PatientProfiles.get_profile(patient_profile_name)
        patient_profile['profile_type'] = patient_profile_name  # Add profile type for reference
        
        # Run optimization
        optimization_results = self.optimizer.optimize_dose_schedule(
            base_protocol, patient_profile
        )
        
        if output_results:
            best = optimization_results['best_protocol']
            print(f"\nOptimization Results for {patient_profile_name.title()} Patient:")
            print(f"Best Protocol Configuration:")
            print(f"  Dose: {best['dose']:.2f}")
            print(f"  Schedule: {best['treatment_days']} days on, {best['rest_days']} days off")
            print(f"  Therapeutic Index: {best['therapeutic_index']:.2f}")
            print(f"  Estimated Efficacy: {best['estimated_efficacy']:.2f}")
            print(f"  Estimated Toxicity: {best['estimated_toxicity']:.2f}")
        
        # Create personalized protocol
        personalized_name, best_result = self.optimizer.create_personalized_protocol(
            patient_profile, optimization_results
        )
        
        return {
            'personalized_protocol_name': personalized_name,
            'optimization_results': optimization_results,
            'best_configuration': best_result
        }
    
    def _calculate_metrics(self, result, protocol):
        """Calculate performance metrics from simulation results"""
        # Extract state variables
        N1 = result.y[0]  # Sensitive cells
        N2 = result.y[1]  # Partially resistant cells
        I1 = result.y[2]  # Cytotoxic immune cells
        I2 = result.y[3]  # Regulatory immune cells
        P = result.y[4]   # Metastatic potential
        A = result.y[5]   # Angiogenesis factor
        Q = result.y[6]   # Quiescent cells
        R1 = result.y[7]  # Type 1 resistant cells
        R2 = result.y[8]  # Type 2 resistant cells
        S = result.y[9]   # Senescent cells
        D = result.y[10]  # Drug concentration
        G = result.y[12]  # Genetic stability
        M = result.y[13]  # Metabolism status
        H = result.y[14]  # Hypoxia level
        
        # Calculate derived metrics
        total_tumor = N1 + N2 + Q + R1 + R2 + S
        total_resistant = R1 + R2
        resistance_fraction = (R1 + R2) / total_tumor * 100
        
        # Calculate tumor reduction from initial
        initial_burden = total_tumor[0]
        final_burden = total_tumor[-1]
        percent_reduction = 100 * (1 - final_burden / initial_burden)
        
        # Calculate area under curve (tumor burden over time)
        auc_tumor = np.trapz(total_tumor, result.t)
        
        # Calculate treatment efficacy score
        treatment_efficacy = percent_reduction / (1 + resistance_fraction[-1]/100)
        
        # Calculate other metrics
        immune_activation = I1[-1] / I1[0] if I1[0] > 0 else 1.0
        genetic_instability = 1 - G[-1]
        metabolic_shift = M[-1]
        final_hypoxia = H[-1]
        
        # Calculate time to resistance (when resistance > 5%)
        time_to_resistance = None
        for i, res_frac in enumerate(resistance_fraction):
            if res_frac > 5.0:
                time_to_resistance = result.t[i]
                break
        
        # Calculate time to tumor control (when tumor < 50% of initial)
        time_to_control = None
        control_threshold = initial_burden * 0.5
        for i, tumor_burden in enumerate(total_tumor):
            if tumor_burden < control_threshold:
                time_to_control = result.t[i]
                break
        
        return {
            'initial_burden': initial_burden,
            'final_burden': final_burden,
            'percent_reduction': percent_reduction,
            'final_resistance_fraction': resistance_fraction[-1],
            'treatment_efficacy_score': treatment_efficacy,
            'immune_activation': immune_activation,
            'genetic_instability': genetic_instability,
            'metabolic_shift': metabolic_shift,
            'hypoxia_level': final_hypoxia,
            'auc_tumor_burden': auc_tumor,
            'time_to_resistance': time_to_resistance,
            'time_to_control': time_to_control,
            'max_drug_concentration': np.max(D),
            'final_metastatic_potential': P[-1],
            'final_angiogenesis': A[-1]
        }
    
    def _extract_time_series(self, result):
        """Extract time series data for visualization"""
        # Extract all state variables
        N1, N2, I1, I2, P, A, Q, R1, R2, S, D, Dm, G, M, H = result.y
        
        # Calculate derived time series
        total_tumor = N1 + N2 + Q + R1 + R2 + S
        total_resistant = R1 + R2
        resistance_fraction = (R1 + R2) / total_tumor * 100
        
        return {
            'time': result.t,
            'total_tumor': total_tumor,
            'sensitive_cells': N1,
            'partially_resistant': N2,
            'resistant_type1': R1,
            'resistant_type2': R2,
            'quiescent': Q,
            'senescent': S,
            'cytotoxic_immune': I1,
            'regulatory_immune': I2,
            'resistance_fraction': resistance_fraction,
            'drug_concentration': D,
            'metabolized_drug': Dm,
            'genetic_stability': G,
            'metabolism': M,
            'hypoxia': H,
            'metastatic_potential': P,
            'angiogenesis_factor': A
        }
    
    def sensitivity_analysis(self, base_patient='average', base_protocol='standard', 
                           parameter_variations=None, simulation_days=500):
        """
        Perform sensitivity analysis on key parameters
        
        Args:
            base_patient (str): Base patient profile
            base_protocol (str): Base treatment protocol
            parameter_variations (dict): Parameters to vary and their ranges
            simulation_days (int): Simulation duration
            
        Returns:
            dict: Sensitivity analysis results
        """
        if parameter_variations is None:
            parameter_variations = {
                'alpha': [0.85, 0.90, 0.93, 0.95, 0.98],
                'mutation_rate': [0.00005, 0.0001, 0.0002, 0.0005, 0.001],
                'omega_R1': [0.002, 0.004, 0.006, 0.008, 0.01],
                'beta1': [0.003, 0.005, 0.007, 0.009, 0.012]
            }
        
        print(f"\nRunning sensitivity analysis...")
        print(f"Base configuration: {base_patient} patient, {base_protocol} protocol")
        
        sensitivity_results = {}
        
        for param_name, param_values in parameter_variations.items():
            print(f"\nAnalyzing sensitivity to {param_name}...")
            param_results = []
            
            for value in param_values:
                # Create custom parameter set
                patient_profile = PatientProfiles.get_profile(base_patient)
                patient_profile[param_name] = value  # Override parameter
                
                # Run simulation with modified parameter
                model_params = ModelParameters(patient_profile)
                
                # Quick simulation setup
                protocol = self.treatment_protocols.get_protocol(base_protocol, patient_profile)
                pk_model = PharmacokineticModel(model_params.get_all_parameters())
                circadian_model = CircadianRhythm(model_params.get_all_parameters())
                cancer_model = CancerModel(model_params.get_all_parameters(), pk_model, circadian_model)
                
                t_span = [0, simulation_days]
                t_eval = np.linspace(0, simulation_days, simulation_days + 1)
                initial_conditions = InitialConditions.get_conditions_for_profile(base_patient)
                temp_func = protocol.get('temperature', lambda t: 37.0)
                
                def model_function(t, y):
                    current_temp = temp_func(t)
                    return cancer_model.enhanced_temperature_cancer_model(
                        t, y, protocol['drugs'], current_temp, True
                    )
                
                result = safe_solve_ivp(model_function, t_span, initial_conditions, 'RK45', t_eval)
                
                if result.success:
                    metrics = self._calculate_metrics(result, protocol)
                    param_results.append({
                        'parameter_value': value,
                        'efficacy_score': metrics['treatment_efficacy_score'],
                        'final_resistance': metrics['final_resistance_fraction'],
                        'tumor_reduction': metrics['percent_reduction']
                    })
                else:
                    param_results.append({
                        'parameter_value': value,
                        'efficacy_score': 0,
                        'final_resistance': 100,
                        'tumor_reduction': 0,
                        'failed': True
                    })
            
            sensitivity_results[param_name] = param_results
        
        return sensitivity_results
    
    def get_simulation_summary(self, results):
        """Generate a summary of simulation results"""
        if not results:
            return "No results to summarize."
        
        summary = []
        summary.append("Simulation Summary")
        summary.append("=" * 50)
        
        # Count successful vs failed simulations
        total_sims = 0
        successful_sims = 0
        
        for patient_profile, protocols in results.items():
            for protocol_name, result in protocols.items():
                total_sims += 1
                if result.get('success', False):
                    successful_sims += 1
        
        summary.append(f"Total simulations: {total_sims}")
        summary.append(f"Successful simulations: {successful_sims}")
        summary.append(f"Success rate: {100*successful_sims/total_sims:.1f}%")
        summary.append("")
        
        # Find best overall protocol
        best_efficacy = 0
        best_combo = None
        
        for patient_profile, protocols in results.items():
            for protocol_name, result in protocols.items():
                if result.get('success', False):
                    efficacy = result['metrics']['treatment_efficacy_score']
                    if efficacy > best_efficacy:
                        best_efficacy = efficacy
                        best_combo = (patient_profile, protocol_name)
        
        if best_combo:
            summary.append(f"Best overall result:")
            summary.append(f"  Patient: {best_combo[0]}")
            summary.append(f"  Protocol: {best_combo[1]}")
            summary.append(f"  Efficacy Score: {best_efficacy:.2f}")
        
        return "\n".join(summary)
