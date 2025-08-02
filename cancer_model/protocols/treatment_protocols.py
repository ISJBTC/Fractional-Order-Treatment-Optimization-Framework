"""
Module 5: Treatment Protocols
=============================
Contains treatment protocol definitions and management.
Easily customizable for different therapeutic strategies.
"""

from cancer_model.core.pharmacokinetics import DrugScheduling, TemperatureProtocol, optimized_adaptive_therapy


class TreatmentProtocols:
    """Treatment protocol definitions and management"""
    
    def __init__(self):
        self.protocols = self._define_base_protocols()
    
    def _define_base_protocols(self):
        """Define base treatment protocols"""
        return {
            'standard': {
                'description': 'Standard cyclic hormone/HER2 therapy',
                'drugs': {
                    'hormone': DrugScheduling.create_cyclic_dosing_schedule(14, 7, 0.8, start_day=0),
                    'her2': DrugScheduling.create_cyclic_dosing_schedule(14, 7, 0.8, start_day=0)
                },
                'temperature': lambda t: 37.0,
                'parameters': {}
            },
            
            'continuous': {
                'description': 'Continuous hormone/HER2 therapy',
                'drugs': {
                    'hormone': DrugScheduling.create_continuous_dosing_schedule(0.8, start_day=0),
                    'her2': DrugScheduling.create_continuous_dosing_schedule(0.8, start_day=0)
                },
                'temperature': lambda t: 37.0,
                'parameters': {}
            },
            
            'adaptive': {
                'description': 'Adaptive therapy based on tumor response',
                'drugs': {
                    'hormone': optimized_adaptive_therapy,
                    'her2': optimized_adaptive_therapy
                },
                'temperature': lambda t: 37.0,
                'parameters': {}
            },
            
            'immuno_combo': {
                'description': 'Immuno-chemotherapy combination',
                'drugs': {
                    'chemo': DrugScheduling.create_cyclic_dosing_schedule(7, 14, 0.6, start_day=0),
                    'immuno': DrugScheduling.create_cyclic_dosing_schedule(2, 19, 0.7, start_day=0)
                },
                'temperature': lambda t: 37.0,
                'parameters': {}
            },
            
            'hyperthermia': {
                'description': 'Hormone/HER2 therapy with periodic hyperthermia',
                'drugs': {
                    'hormone': DrugScheduling.create_cyclic_dosing_schedule(14, 7, 0.7, start_day=0),
                    'her2': DrugScheduling.create_cyclic_dosing_schedule(14, 7, 0.7, start_day=0)
                },
                'temperature': TemperatureProtocol.create_cyclic_hyperthermia(21, 2, 38.5, 37.0),
                'parameters': {}
            },
            
            'multi_modal': {
                'description': 'Multi-modal therapy with sequential drug combinations',
                'drugs': {
                    'hormone': DrugScheduling.create_cyclic_dosing_schedule(14, 7, 0.6, start_day=0),
                    'chemo': DrugScheduling.create_cyclic_dosing_schedule(5, 16, 0.5, start_day=30),
                    'immuno': DrugScheduling.create_cyclic_dosing_schedule(1, 20, 0.7, start_day=60)
                },
                'temperature': TemperatureProtocol.create_cyclic_hyperthermia(42, 3, 38.5, 37.0),
                'parameters': {}
            },
            
            'metronomic': {
                'description': 'Metronomic low-dose continuous therapy',
                'drugs': {
                    'hormone': DrugScheduling.create_metronomic_schedule(0.3, 0.6, 14),
                    'chemo': DrugScheduling.create_metronomic_schedule(0.2, 0.4, 7)
                },
                'temperature': lambda t: 37.0,
                'parameters': {}
            }
        }
    
    def get_protocol(self, protocol_name, patient_profile=None):
        """
        Get a specific treatment protocol with optional patient adjustments
        
        Args:
            protocol_name (str): Name of the protocol
            patient_profile (dict): Patient profile for dose adjustments
            
        Returns:
            dict: Complete treatment protocol
        """
        if protocol_name not in self.protocols:
            print(f"Warning: Protocol '{protocol_name}' not found. Using 'standard' protocol.")
            protocol_name = 'standard'
        
        protocol = self.protocols[protocol_name].copy()
        
        # Apply patient-specific adjustments
        if patient_profile:
            protocol = self._apply_patient_adjustments(protocol, patient_profile)
        
        return protocol
    
    def _apply_patient_adjustments(self, protocol, patient_profile):
        """Apply patient-specific dose and timing adjustments"""
        # Extract patient factors
        performance_factor = patient_profile.get('performance_status', 1.0)
        age_factor = patient_profile.get('age_factor', 1.0)
        liver_function = patient_profile.get('liver_function', 1.0)
        kidney_function = patient_profile.get('kidney_function', 1.0)
        
        # Calculate overall dose adjustment
        dose_adjustment = (performance_factor + age_factor) / 2
        metabolism_adjustment = (liver_function + kidney_function) / 2
        
        # Adjust each drug schedule
        adjusted_drugs = {}
        for drug_type, original_schedule in protocol['drugs'].items():
            if callable(original_schedule):
                # Create adjusted wrapper function
                def adjusted_schedule(t, orig_schedule=original_schedule, 
                                    dose_adj=dose_adjustment, 
                                    metab_adj=metabolism_adjustment):
                    base_dose = orig_schedule(t)
                    # Adjust dose based on patient factors
                    adjusted_dose = base_dose * dose_adj
                    # Consider metabolism for timing (simplified)
                    if metab_adj < 0.8:  # Slow metabolizers
                        adjusted_dose *= 0.9  # Slightly reduce dose
                    elif metab_adj > 1.2:  # Fast metabolizers
                        adjusted_dose *= 1.1  # Slightly increase dose
                    
                    return adjusted_dose
                
                adjusted_drugs[drug_type] = adjusted_schedule
            else:
                adjusted_drugs[drug_type] = original_schedule
        
        protocol['drugs'] = adjusted_drugs
        return protocol
    
    def create_custom_protocol(self, name, description, drug_schedules, temperature_func=None, parameters=None):
        """
        Create a custom treatment protocol
        
        Args:
            name (str): Protocol name
            description (str): Protocol description
            drug_schedules (dict): Dictionary of drug type: schedule function
            temperature_func (function): Temperature modulation function
            parameters (dict): Additional parameters
            
        Returns:
            str: Protocol name for reference
        """
        custom_protocol = {
            'description': description,
            'drugs': drug_schedules,
            'temperature': temperature_func or (lambda t: 37.0),
            'parameters': parameters or {}
        }
        
        self.protocols[name] = custom_protocol
        print(f"Custom protocol '{name}' created successfully.")
        return name
    
    def list_protocols(self):
        """List all available protocols with descriptions"""
        print("\nAvailable Treatment Protocols:")
        print("=" * 50)
        for name, protocol in self.protocols.items():
            print(f"{name.upper()}: {protocol['description']}")
            drugs = list(protocol['drugs'].keys())
            print(f"  Drugs: {', '.join(drugs)}")
            has_temp = protocol['temperature'](0) != 37.0
            if has_temp:
                print("  Includes temperature modulation")
            print()
    
    def compare_protocols(self, protocol_names, patient_profile=None):
        """
        Compare multiple protocols for a given patient profile
        
        Args:
            protocol_names (list): List of protocol names to compare
            patient_profile (dict): Patient profile for comparison
            
        Returns:
            dict: Comparison summary
        """
        comparison = {}
        
        for protocol_name in protocol_names:
            if protocol_name in self.protocols:
                protocol = self.get_protocol(protocol_name, patient_profile)
                
                # Extract key characteristics
                drugs = list(protocol['drugs'].keys())
                has_temperature = protocol['temperature'](0) != 37.0
                
                # Simple dose intensity calculation (for first 100 days)
                total_dose_intensity = 0
                sample_times = range(0, 100, 1)
                
                for drug in drugs:
                    drug_schedule = protocol['drugs'][drug]
                    if callable(drug_schedule):
                        for t in sample_times:
                            total_dose_intensity += drug_schedule(t)
                
                comparison[protocol_name] = {
                    'description': protocol['description'],
                    'drugs': drugs,
                    'temperature_modulation': has_temperature,
                    'dose_intensity_100d': total_dose_intensity,
                    'estimated_toxicity': self._estimate_toxicity(protocol, patient_profile)
                }
        
        return comparison
    
    def _estimate_toxicity(self, protocol, patient_profile):
        """Estimate relative toxicity based on protocol and patient factors"""
        base_toxicity = 1.0
        
        # Drug-specific toxicity weights
        drug_toxicity = {
            'hormone': 0.3,
            'her2': 0.5,
            'chemo': 1.0,
            'immuno': 0.7
        }
        
        # Calculate cumulative toxicity
        for drug in protocol['drugs']:
            if drug in drug_toxicity:
                base_toxicity += drug_toxicity[drug]
        
        # Temperature modulation adds toxicity
        if protocol['temperature'](0) != 37.0:
            base_toxicity *= 1.2
        
        # Patient factors modify toxicity
        if patient_profile:
            performance = patient_profile.get('performance_status', 1.0)
            age_factor = patient_profile.get('age_factor', 1.0)
            liver_function = patient_profile.get('liver_function', 1.0)
            
            # Lower performance status = higher toxicity
            toxicity_multiplier = (2 - performance) * (2 - age_factor) * (2 - liver_function) / 8
            base_toxicity *= toxicity_multiplier
        
        return base_toxicity


class ProtocolOptimizer:
    """Optimize treatment protocols for specific patients"""
    
    def __init__(self, treatment_protocols):
        self.protocols = treatment_protocols
    
    def optimize_dose_schedule(self, base_protocol_name, patient_profile, 
                             dose_range=(0.3, 1.0), cycle_options=None):
        """
        Optimize dose and scheduling for a patient
        
        Args:
            base_protocol_name (str): Base protocol to optimize
            patient_profile (dict): Patient characteristics
            dose_range (tuple): Range of doses to test
            cycle_options (list): List of (treatment_days, rest_days) tuples
            
        Returns:
            dict: Optimization results
        """
        if cycle_options is None:
            if patient_profile.get('performance_status', 1.0) < 0.8:
                # For compromised patients, gentler cycles
                cycle_options = [(7, 21), (10, 18), (14, 14)]
            else:
                # For healthier patients, more intensive cycles
                cycle_options = [(7, 14), (10, 11), (14, 7)]
        
        base_protocol = self.protocols.get_protocol(base_protocol_name, patient_profile)
        
        # Generate dose options
        n_doses = 5
        dose_options = [dose_range[0] + i * (dose_range[1] - dose_range[0]) / (n_doses - 1) 
                       for i in range(n_doses)]
        
        optimization_results = []
        
        for dose in dose_options:
            for treatment_days, rest_days in cycle_options:
                # Create optimized schedules
                optimized_drugs = {}
                
                for drug_type in base_protocol['drugs']:
                    if drug_type in ['hormone', 'her2', 'chemo']:
                        optimized_drugs[drug_type] = DrugScheduling.create_cyclic_dosing_schedule(
                            treatment_days, rest_days, dose, start_day=0
                        )
                    else:
                        # Keep original schedule for other drugs
                        optimized_drugs[drug_type] = base_protocol['drugs'][drug_type]
                
                # Calculate metrics
                dose_intensity = self._calculate_dose_intensity(optimized_drugs, 200)
                estimated_efficacy = self._estimate_efficacy(dose_intensity, patient_profile)
                estimated_toxicity = self._estimate_toxicity(dose_intensity, patient_profile)
                
                optimization_results.append({
                    'dose': dose,
                    'treatment_days': treatment_days,
                    'rest_days': rest_days,
                    'dose_intensity': dose_intensity,
                    'estimated_efficacy': estimated_efficacy,
                    'estimated_toxicity': estimated_toxicity,
                    'therapeutic_index': estimated_efficacy / estimated_toxicity,
                    'drugs': optimized_drugs
                })
        
        # Sort by therapeutic index
        optimization_results.sort(key=lambda x: x['therapeutic_index'], reverse=True)
        
        return {
            'best_protocol': optimization_results[0],
            'all_results': optimization_results,
            'base_protocol': base_protocol_name,
            'patient_profile': patient_profile
        }
    
    def _calculate_dose_intensity(self, drug_schedules, duration_days):
        """Calculate total dose intensity over a period"""
        total_intensity = 0
        
        for drug_type, schedule in drug_schedules.items():
            if callable(schedule):
                for day in range(duration_days):
                    total_intensity += schedule(day)
        
        return total_intensity
    
    def _estimate_efficacy(self, dose_intensity, patient_profile):
        """Estimate treatment efficacy based on dose intensity and patient factors"""
        base_efficacy = min(dose_intensity / 100, 1.0)  # Normalize and cap
        
        # Patient factors
        immune_status = patient_profile.get('immune_status', 1.0)
        performance_status = patient_profile.get('performance_status', 1.0)
        age_factor = patient_profile.get('age_factor', 1.0)
        
        # Calculate efficacy multiplier
        efficacy_multiplier = (immune_status + performance_status + age_factor) / 3
        
        return base_efficacy * efficacy_multiplier
    
    def _estimate_toxicity(self, dose_intensity, patient_profile):
        """Estimate treatment toxicity"""
        base_toxicity = dose_intensity / 50  # Higher dose intensity = higher toxicity
        
        # Patient factors that affect toxicity tolerance
        performance_status = patient_profile.get('performance_status', 1.0)
        liver_function = patient_profile.get('liver_function', 1.0)
        kidney_function = patient_profile.get('kidney_function', 1.0)
        
        # Lower function = higher effective toxicity
        toxicity_multiplier = 3 - performance_status - liver_function - kidney_function
        
        return base_toxicity * max(toxicity_multiplier, 0.5)
    
    def create_personalized_protocol(self, patient_profile, optimization_results):
        """Create a personalized protocol based on optimization results"""
        best_result = optimization_results['best_protocol']
        
        protocol_name = f"personalized_{patient_profile.get('profile_type', 'custom')}"
        description = f"Personalized protocol: {best_result['dose']:.2f} dose, " + \
                     f"{best_result['treatment_days']}/{best_result['rest_days']} schedule"
        
        # Create the personalized protocol
        personalized_protocol = self.protocols.create_custom_protocol(
            name=protocol_name,
            description=description,
            drug_schedules=best_result['drugs'],
            temperature_func=lambda t: 37.0,  # Default temperature
            parameters={
                'optimized_dose': best_result['dose'],
                'treatment_days': best_result['treatment_days'],
                'rest_days': best_result['rest_days'],
                'therapeutic_index': best_result['therapeutic_index'],
                'patient_profile': patient_profile
            }
        )
        
        return protocol_name, best_result


# Pre-defined protocol combinations for specific scenarios
PROTOCOL_COMBINATIONS = {
    'neoadjuvant': {
        'description': 'Pre-surgery tumor reduction',
        'sequence': [
            ('chemo_intensive', 0, 90),      # 3 months intensive chemo
            ('hormone', 90, 180),            # 3 months hormone therapy
            ('hyperthermia', 150, 180)       # 1 month hyperthermia boost
        ]
    },
    
    'adjuvant': {
        'description': 'Post-surgery prevention',
        'sequence': [
            ('hormone', 0, 365),             # 1 year hormone therapy
            ('immuno_combo', 180, 270),      # 3 months immune boost
            ('maintenance', 365, 1095)       # 2 years maintenance
        ]
    },
    
    'metastatic': {
        'description': 'Advanced disease management',
        'sequence': [
            ('multi_modal', 0, 120),         # 4 months intensive
            ('adaptive', 120, 480),          # 1 year adaptive
            ('palliative', 480, 720)         # 8 months palliative
        ]
    }
}
