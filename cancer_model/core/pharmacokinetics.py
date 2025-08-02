"""
Module 3: Pharmacokinetics and Drug Dynamics
============================================
Handles drug absorption, distribution, metabolism, and elimination (ADME).
Also includes circadian rhythm effects and drug scheduling.
"""

import numpy as np


class PharmacokineticModel:
    """Drug pharmacokinetics with patient-specific variations"""
    
    def __init__(self, params):
        self.params = params
    
    def drug_pharmacokinetics(self, t, dose_schedule, drug_type='generic'):
        """
        Calculate drug concentration based on timing and PK parameters
        
        Args:
            t (float): Current time
            dose_schedule (function): Function that returns dose given time
            drug_type (str): Type of drug for specific PK properties
            
        Returns:
            float: Current effective drug concentration change
        """
        # Extract PK parameters
        absorption_rate = self.params.get('absorption_rate', 0.5)
        elimination_rate = self.params.get('elimination_rate', 0.1)
        bioavailability = self.params.get('bioavailability', 0.85)
        liver_function = self.params.get('liver_function', 1.0)
        kidney_function = self.params.get('kidney_function', 1.0)
        
        # Drug-specific adjustments
        drug_adjustments = {
            'hormone': {'absorption_rate': 1.0, 'elimination_rate': 0.8},
            'her2': {'absorption_rate': 0.8, 'elimination_rate': 1.2},
            'chemo': {'absorption_rate': 1.2, 'elimination_rate': 1.5},
            'immuno': {'absorption_rate': 0.6, 'elimination_rate': 0.5}
        }
        
        if drug_type in drug_adjustments:
            absorption_rate *= drug_adjustments[drug_type]['absorption_rate']
            elimination_rate *= drug_adjustments[drug_type]['elimination_rate']
        
        # Adjust elimination based on patient factors
        adjusted_elimination = elimination_rate * liver_function * kidney_function
        
        # Get current dose from schedule
        current_dose = dose_schedule(t) if callable(dose_schedule) else 0.0
        
        # Simple one-compartment model
        effective_dose = current_dose * bioavailability
        concentration_change = absorption_rate * effective_dose - adjusted_elimination
        
        return concentration_change
    
    def calculate_drug_effect(self, concentration, drug_type='generic'):
        """
        Calculate drug effect using Hill equation
        
        Args:
            concentration (float): Drug concentration
            drug_type (str): Type of drug for specific PD properties
            
        Returns:
            float: Drug effect (between 0 and max_effect)
        """
        # Extract PD parameters
        max_effect = self.params.get('max_drug_effect', 1.0)
        ec50 = self.params.get('EC50', 0.3)
        hill = self.params.get('hill_coef', 1.5)
        
        # Drug-specific PD adjustments
        drug_pd_adjustments = {
            'hormone': {'max_effect': 0.9, 'ec50': 0.25, 'hill': 1.2},
            'her2': {'max_effect': 1.1, 'ec50': 0.35, 'hill': 1.8},
            'chemo': {'max_effect': 1.3, 'ec50': 0.4, 'hill': 2.0},
            'immuno': {'max_effect': 0.8, 'ec50': 0.2, 'hill': 1.0}
        }
        
        if drug_type in drug_pd_adjustments:
            max_effect *= drug_pd_adjustments[drug_type]['max_effect']
            ec50 *= drug_pd_adjustments[drug_type]['ec50']
            hill *= drug_pd_adjustments[drug_type]['hill']
        
        # Apply Hill equation
        effect = max_effect * (concentration**hill) / (ec50**hill + concentration**hill)
        
        return effect


class CircadianRhythm:
    """Circadian rhythm effects on biological processes"""
    
    def __init__(self, params):
        self.params = params
    
    def calculate_circadian_effect(self, t, process_type='general'):
        """
        Calculate effect of circadian rhythm on biological processes
        
        Args:
            t (float): Current time (in days)
            process_type (str): Type of biological process
            
        Returns:
            float: Modulation factor based on circadian rhythm (centered at 1.0)
        """
        # Extract circadian parameters
        amplitude = self.params.get('circadian_amplitude', 0.2)
        phase = self.params.get('circadian_phase', 0.0)
        period = self.params.get('circadian_period', 24.0) / 24.0  # Convert hours to days
        
        # Process-specific circadian effects
        process_adjustments = {
            'growth': {'amplitude': 0.15, 'phase': 0.25},      # Growth peaks in early morning
            'immune': {'amplitude': 0.25, 'phase': 0.75},      # Immune activity peaks at night
            'metabolism': {'amplitude': 0.3, 'phase': 0.5},    # Metabolism peaks midday
            'dna_repair': {'amplitude': 0.2, 'phase': 0.0}     # DNA repair peaks at midnight
        }
        
        if process_type in process_adjustments:
            amplitude *= process_adjustments[process_type]['amplitude'] / 0.2
            phase = process_adjustments[process_type]['phase']
        
        # Calculate circadian impact using sinusoidal function
        circadian_factor = 1.0 + amplitude * np.sin(2 * np.pi * (t / period - phase))
        
        return circadian_factor


class DrugScheduling:
    """Drug scheduling and dosing protocols"""
    
    @staticmethod
    def create_cyclic_dosing_schedule(treatment_days, rest_days, dose, start_day=0):
        """
        Create a cyclic drug dosing schedule
        
        Args:
            treatment_days (int): Days of treatment per cycle
            rest_days (int): Days of rest per cycle
            dose (float): Dose amount during treatment days
            start_day (int): Day to start treatment
            
        Returns:
            function: Dosing function that returns dose at given time
        """
        def dosing_schedule(t):
            if t < start_day:
                return 0.0
            
            cycle_length = treatment_days + rest_days
            cycle_position = (t - start_day) % cycle_length
            
            if cycle_position < treatment_days:
                return dose
            else:
                return 0.0
                
        return dosing_schedule
    
    @staticmethod
    def create_continuous_dosing_schedule(dose, start_day=0):
        """Create continuous dosing schedule"""
        def dosing_schedule(t):
            return dose if t >= start_day else 0.0
        return dosing_schedule
    
    @staticmethod
    def create_adaptive_dosing_schedule(initial_dose, monitoring_period=30, 
                                      target_ratio=0.8, max_dose=1.0, min_dose=0.1):
        """
        Create an adaptive therapy dosing schedule
        
        Args:
            initial_dose (float): Starting dose
            monitoring_period (int): Days between dose adjustments
            target_ratio (float): Target tumor burden ratio to maintain
            max_dose (float): Maximum dose
            min_dose (float): Minimum dose
            
        Returns:
            function: Adaptive dosing function
        """
        # Closure variables
        current_dose = initial_dose
        last_adjustment_time = 0
        tumor_history = []
        
        def adaptive_schedule(t, current_tumor_burden=None):
            nonlocal current_dose, last_adjustment_time, tumor_history
            
            # Before start, no treatment
            if t < 0:
                return 0.0
            
            # If no tumor burden provided, return current dose
            if current_tumor_burden is None:
                return current_dose
                
            # Add current burden to history
            tumor_history.append((t, current_tumor_burden))
            
            # Keep only recent history
            cutoff_time = t - 2 * monitoring_period
            tumor_history = [item for item in tumor_history if item[0] >= cutoff_time]
            
            # Check if it's time to adjust dose
            if t >= last_adjustment_time + monitoring_period and len(tumor_history) >= 2:
                # Get initial burden in this period
                initial_idx = 0
                while initial_idx < len(tumor_history) and tumor_history[initial_idx][0] < last_adjustment_time:
                    initial_idx += 1
                
                if initial_idx < len(tumor_history):
                    initial_burden = tumor_history[initial_idx][1]
                    current_burden = tumor_history[-1][1]
                    
                    # Calculate ratio
                    burden_ratio = current_burden / initial_burden if initial_burden > 0 else 1.0
                    
                    # Adjust dose based on ratio
                    if burden_ratio > target_ratio + 0.1:
                        # Tumor growing too fast, increase dose
                        current_dose = min(current_dose * 1.2, max_dose)
                    elif burden_ratio < target_ratio - 0.1:
                        # Tumor declining too fast, decrease dose
                        current_dose = max(current_dose * 0.8, min_dose)
                    
                    # Update last adjustment time
                    last_adjustment_time = t
            
            return current_dose
        
        return adaptive_schedule
    
    @staticmethod
    def create_metronomic_schedule(low_dose, high_dose, switch_period=14):
        """Create metronomic (alternating dose) schedule"""
        def dosing_schedule(t):
            if t < 0:
                return 0.0
            cycle_position = (t % (2 * switch_period))
            return high_dose if cycle_position < switch_period else low_dose
        return dosing_schedule


class TemperatureProtocol:
    """Temperature modulation protocols for hyperthermia/hypothermia"""
    
    @staticmethod
    def create_temperature_protocol(baseline_temp=37.0, hyperthermia_days=None, hypothermia_days=None):
        """
        Generate temperature profile based on treatment protocol
        
        Args:
            baseline_temp (float): Baseline body temperature
            hyperthermia_days (list): List of (start_day, end_day, temp) for hyperthermia
            hypothermia_days (list): List of (start_day, end_day, temp) for hypothermia
            
        Returns:
            function: Temperature function that returns temperature at given time
        """
        def temperature_function(t):
            current_temp = baseline_temp
            
            if hyperthermia_days:
                for start, end, temp in hyperthermia_days:
                    if start <= t <= end:
                        current_temp = temp
                        break
                        
            if hypothermia_days:
                for start, end, temp in hypothermia_days:
                    if start <= t <= end:
                        current_temp = temp
                        break
            
            return current_temp
        
        return temperature_function
    
    @staticmethod
    def create_cyclic_hyperthermia(cycle_days=21, treatment_days=2, hyper_temp=38.5, baseline_temp=37.0):
        """Create cyclic hyperthermia protocol"""
        def temperature_function(t):
            if t < 0:
                return baseline_temp
            cycle_position = t % cycle_days
            return hyper_temp if cycle_position < treatment_days else baseline_temp
        return temperature_function
    
    @staticmethod
    def advanced_temperature_modifier(current_temp, baseline_temp=37.0, time_factor=None):
        """
        Enhanced temperature modification with time-dependent variations.
        
        Args:
            current_temp (float): Current temperature
            baseline_temp (float): Baseline temperature
            time_factor (float): Current time for time-dependent effects
            
        Returns:
            dict: Temperature effects on various biological processes
        """
        # Temperature deviation calculations
        temp_deviation = current_temp - baseline_temp
        
        # Advanced sigmoid transformation
        def advanced_sigmoid(x, steepness=1.0, midpoint=0, max_effect=1.5, min_effect=0.5):
            return min_effect + (max_effect - min_effect) / (1 + np.exp(-steepness * (x - midpoint)))
        
        # Time-dependent stress response
        stress_time_modifier = 1.0
        if time_factor is not None:
            stress_time_modifier = np.sin(time_factor / 50) * 0.2 + 1.0
        
        # Enhanced modification factors
        modifications = {
            'metabolism': {
                'factor': advanced_sigmoid(temp_deviation, steepness=0.8, midpoint=1.5),
                'sensitivity': 0.03
            },
            'immune_activation': {
                'factor': advanced_sigmoid(abs(temp_deviation), steepness=0.6, midpoint=1.0, max_effect=1.8, min_effect=0.6),
                'sensitivity': 0.04
            },
            'cellular_stress': {
                'factor': advanced_sigmoid(temp_deviation, steepness=1.0, midpoint=2.0, max_effect=2.0, min_effect=0.5) * stress_time_modifier,
                'sensitivity': 0.05
            },
            'gene_expression': {
                'factor': 1 + 0.15 * np.tanh(temp_deviation / 2),
                'sensitivity': 0.02
            },
            'resistance_development': {
                'factor': advanced_sigmoid(temp_deviation, steepness=0.7, midpoint=1.5, max_effect=1.7, min_effect=0.7),
                'sensitivity': 0.04
            }
        }
        
        # Apply probabilistic threshold effects
        for key, mod in modifications.items():
            mod['factor'] *= (1 + np.random.normal(0, 0.05))
            mod['factor'] = np.clip(mod['factor'], 0.5, 2.0)
        
        return modifications


# Optimized adaptive therapy schedule
def optimized_adaptive_therapy(t):
    """Optimized adaptive therapy with fixed values for reproducibility"""
    if t < 60:
        return 0.9
    else:
        cycle_period = 90
        phase = ((t - 60) % cycle_period) / cycle_period
        if phase < 0.4:
            return 0.8
        elif phase < 0.7:
            return 0.4
        else:
            return 0.1
