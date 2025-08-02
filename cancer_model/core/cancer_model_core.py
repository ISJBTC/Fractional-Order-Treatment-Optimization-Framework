"""
Module 4: Core Cancer Model
===========================
Contains the main differential equation system and biological dynamics.
This is where the core mathematical model is implemented.
"""

import numpy as np
from cancer_model.core.pharmacokinetics import PharmacokineticModel, CircadianRhythm, TemperatureProtocol


class CancerModel:
    """Core cancer model with enhanced biological dynamics"""
    
    def __init__(self, params, pk_model=None, circadian_model=None):
        self.params = params
        self.pk_model = pk_model or PharmacokineticModel(params)
        self.circadian_model = circadian_model or CircadianRhythm(params)
    
    def advanced_cancer_model(self, t, y, drug_schedules=None, use_circadian=True):
        """
        Advanced cancer model with complex cellular dynamics, pharmacokinetics,
        and circadian rhythm effects.
        
        Args:
            t (float): Current time point
            y (array): State vector
            drug_schedules (dict, optional): Drug dosing schedules
            use_circadian (bool): Whether to use circadian effects
            
        Returns:
            array: Derivatives of state variables
        """
        # Unpack state variables (expanded state vector)
        N1, N2, I1, I2, P, A, Q, R1, R2, S, D, Dm, G, M, H = y
        
        # Ensure non-negative values with a small floor
        y = np.maximum(y, 1e-6)
        N1, N2, I1, I2, P, A, Q, R1, R2, S, D, Dm, G, M, H = y
        
        # Extract parameters
        alpha = self.params.get('alpha', 0.93)
        K = self.params.get('K', 1000)
        
        # Growth rates
        lam1 = self.params.get('lambda1', 0.003)
        lam2 = self.params.get('lambda2', 0.002)
        lam_R1 = self.params.get('lambda_R1', 0.006)
        lam_R2 = self.params.get('lambda_R2', 0.005)
        
        # Immune parameters
        beta1 = self.params.get('beta1', 0.005)
        beta2 = self.params.get('beta2', 0.001)
        phi1 = self.params.get('phi1', 0.1)
        phi2 = self.params.get('phi2', 0.001)
        phi3 = self.params.get('phi3', 0.0003)
        delta_I = self.params.get('delta_I', 0.04)
        
        # Resistance and immune interaction
        immune_resist_factor1 = self.params.get('immune_resist_factor1', 0.10)
        immune_resist_factor2 = self.params.get('immune_resist_factor2', 0.05)
        
        # Metastasis parameters
        gamma = self.params.get('gamma', 0.0001)
        delta_P = self.params.get('delta_P', 0.01)
        
        # Angiogenesis parameters
        alpha_A = self.params.get('alpha_A', 0.01)
        delta_A = self.params.get('delta_A', 0.1)
        
        # Quiescence parameters
        kappa_Q = self.params.get('kappa_Q', 0.001)
        lambda_Q = self.params.get('lambda_Q', 0.0005)
        
        # Resistance development
        omega_R1 = self.params.get('omega_R1', 0.004)
        omega_R2 = self.params.get('omega_R2', 0.003)
        resistance_floor = self.params.get('resistance_floor', 0.01)
        
        # Senescence parameters
        kappa_S = self.params.get('kappa_S', 0.0005)
        delta_S = self.params.get('delta_S', 0.005)
        
        # Therapy parameters
        etaE = self.params.get('etaE', 0.01)
        etaH = self.params.get('etaH', 0.01)
        etaC = self.params.get('etaC', 0.01)
        
        # Metabolism and microenvironment parameters
        metabolic_switch_rate = self.params.get('metabolic_switch_rate', 0.02)
        hypoxia_threshold = self.params.get('hypoxia_threshold', 0.3)
        acidosis_factor = self.params.get('acidosis_factor', 0.01)
        
        # Genetic instability
        mutation_rate = self.params.get('mutation_rate', 0.0001)
        epigenetic_silencing = self.params.get('epigenetic_silencing', 0.002)
        genetic_instability = self.params.get('genetic_instability', 1.0)
        
        # Get treatment controls
        uE = self.params.get('uE', 0.0)
        uH = self.params.get('uH', 0.0)
        uC = self.params.get('uC', 0.0)
        uI = self.params.get('uI', 0.0)
        
        # Handle time-dependent controls
        if callable(uE): uE = uE(t)
        if callable(uH): uH = uH(t)
        if callable(uC): uC = uC(t)
        if callable(uI): uI = uI(t)
        
        # Apply PK/PD for each drug if schedules provided
        if drug_schedules:
            # Process each drug schedule
            if 'hormone' in drug_schedules:
                hormone_effect = self.pk_model.calculate_drug_effect(D, 'hormone')
                uE = hormone_effect
            
            if 'her2' in drug_schedules:
                her2_effect = self.pk_model.calculate_drug_effect(D, 'her2')
                uH = her2_effect
                
            if 'chemo' in drug_schedules:
                chemo_effect = self.pk_model.calculate_drug_effect(D, 'chemo')
                uC = chemo_effect
                
            if 'immuno' in drug_schedules:
                immuno_effect = self.pk_model.calculate_drug_effect(D, 'immuno')
                uI = immuno_effect
        
        # Apply circadian rhythm effects if enabled
        if use_circadian:
            # Get circadian effects for different processes
            growth_circ = self.circadian_model.calculate_circadian_effect(t, 'growth')
            immune_circ = self.circadian_model.calculate_circadian_effect(t, 'immune')
            metabolism_circ = self.circadian_model.calculate_circadian_effect(t, 'metabolism')
            
            # Modulate key parameters
            lam1 *= growth_circ
            lam2 *= growth_circ
            delta_I *= immune_circ
            beta1 *= immune_circ
            metabolic_switch_rate *= metabolism_circ
        
        # Total tumor burden
        total_tumor = N1 + N2 + Q + R1 + R2 + S
        
        # Carrying capacity factor
        carrying_capacity_factor = max(0, 1 - total_tumor / K)
        
        # Combined therapy effect
        therapy_effect = etaE * uE + etaH * uH + etaC * uC
        
        # Hypoxia calculation based on tumor size and angiogenesis
        hypoxia_factor = max(0, (total_tumor / K) - hypoxia_threshold) / (1 - hypoxia_threshold) if (1 - hypoxia_threshold) > 0 else 0
        hypoxia_effect = 1.0 + hypoxia_factor * A / (1 + A)
        
        # Metabolic impact based on hypoxia
        metabolic_shift = M * hypoxia_factor * metabolic_switch_rate
        
        # Acidosis effect based on metabolic state and tumor size
        acidosis_effect = 1.0 + acidosis_factor * M * (total_tumor / K)
        
        # Genetic instability based on treatment and hypoxia
        genetic_damage_rate = mutation_rate * genetic_instability * (1 + therapy_effect + 0.5 * hypoxia_factor)
        
        # Immune boost (immunotherapy)
        immuno_boost = 1.0 + uI
        
        # Growth dynamics with carrying capacity, affected by metabolism
        growth_factor = carrying_capacity_factor * (1 + 0.2 * M)
        growth_N1 = lam1 * N1 * growth_factor / acidosis_effect
        growth_N2 = lam2 * N2 * growth_factor / acidosis_effect
        growth_R1 = lam_R1 * R1 * growth_factor / acidosis_effect
        growth_R2 = lam_R2 * R2 * growth_factor / acidosis_effect
        
        # Immune killing effects, modulated by hypoxia
        immune_kill_factor = 1.0 / (1 + 0.5 * hypoxia_factor)
        immune_kill_N1 = beta1 * N1 * I1 / (1 + 0.01 * total_tumor) * immune_kill_factor
        immune_kill_N2 = beta1 * N2 * I1 / (1 + 0.01 * total_tumor) * 0.5 * immune_kill_factor
        
        # Reduced immune killing of resistant cells
        immune_kill_R1 = beta1 * R1 * I1 / (1 + 0.01 * total_tumor) * immune_resist_factor1 * immune_kill_factor
        immune_kill_R2 = beta1 * R2 * I1 / (1 + 0.01 * total_tumor) * immune_resist_factor2 * immune_kill_factor
        
        # Resistance development enhanced by genetic instability
        resistance_dev_factor = (1 + (1 - G))
        resistance_dev_R1 = max(omega_R1 * therapy_effect * N1 * resistance_dev_factor, 0.0)
        resistance_dev_R2 = max(omega_R2 * therapy_effect * N1 * resistance_dev_factor, 0.0)
        
        # Immune cell dynamics
        immune_prod_I1 = phi1 + phi2 * total_tumor / (1 + 0.01 * total_tumor)
        immune_suppression_I1 = beta2 * I1 * I2 / (1 + I1)
        immune_prod_I2 = phi3 * total_tumor / (1 + 0.01 * total_tumor)
        
        # Quiescence dynamics - hypoxia increases quiescence
        quiescence_factor = 1.0 + 0.5 * hypoxia_factor
        quiescence_induction_N1 = kappa_Q * N1 * quiescence_factor
        quiescence_induction_N2 = kappa_Q * N2 * quiescence_factor
        quiescence_reactivation = lambda_Q * Q / quiescence_factor
        
        # Senescence induction - drug and genetic instability
        senescence_induction = kappa_S * therapy_effect * N1 * (1 + 0.3 * (1 - G))
        
        # === DIFFERENTIAL EQUATIONS ===
        
        # Sensitive cells (N1)
        dN1dt = (growth_N1 
                 - immune_kill_N1 * immuno_boost 
                 - therapy_effect * N1 
                 - quiescence_induction_N1 
                 - resistance_dev_R1 
                 - resistance_dev_R2 
                 - senescence_induction)
        
        # Partially resistant cells (N2)
        dN2dt = (growth_N2 
                 - immune_kill_N2 * immuno_boost 
                 - therapy_effect * N2 * 0.5 
                 - quiescence_induction_N2)
        
        # Cytotoxic immune cells (I1)
        dI1dt = (immune_prod_I1 
                 - immune_suppression_I1 
                 - delta_I * I1 
                 + 0.1 * uI * I1)
        
        # Regulatory immune cells (I2)
        dI2dt = (immune_prod_I2 
                 - delta_I * I2 
                 - 0.1 * uI * I2)
        
        # Metastatic potential (P)
        dPdt = gamma * total_tumor * (1 + 0.5 * hypoxia_factor) - delta_P * P
        
        # Angiogenesis factor (A)
        dAdt = alpha_A * total_tumor / (1 + 0.01 * total_tumor) - delta_A * A
        
        # Quiescent cells (Q)
        dQdt = quiescence_induction_N1 + quiescence_induction_N2 - quiescence_reactivation
        
        # Resistant Type 1 cells (R1)
        dR1dt = resistance_dev_R1 + growth_R1 - immune_kill_R1 * immuno_boost
        
        # Resistant Type 2 cells (R2)
        dR2dt = resistance_dev_R2 + growth_R2 - immune_kill_R2 * immuno_boost
        
        # Senescent cells (S)
        dSdt = senescence_induction - delta_S * S
        
        # Drug concentration (D)
        dDdt = 0.0
        if drug_schedules:
            for drug_type, schedule in drug_schedules.items():
                dDdt += self.pk_model.drug_pharmacokinetics(t, schedule, drug_type)
        dDdt -= self.params.get('elimination_rate', 0.1) * D
        
        # Metabolized drug (Dm)
        dDmdt = self.params.get('elimination_rate', 0.1) * D
        
        # Genetic stability (G)
        dGdt = -genetic_damage_rate * G + 0.001 * (1 - G)  # Slow recovery
        
        # Metabolic state (M)
        dMdt = metabolic_shift - 0.05 * M  # Gradual reversion to normal metabolism
        
        # Hypoxia level (H)
        dHdt = 0.1 * hypoxia_factor - 0.1 * A * H  # Angiogenesis reduces hypoxia
        
        # Combine derivatives
        dydt = np.array([dN1dt, dN2dt, dI1dt, dI2dt, dPdt, dAdt, dQdt, dR1dt, dR2dt, dSdt, 
                         dDdt, dDmdt, dGdt, dMdt, dHdt])
        
        # Apply fractional order scaling for memory effects
        if t > 0:
            memory_factor = min(t**(-alpha), 100)  # Clip to avoid overflow
            fractional_factor = 0.01 * (1 + (1-alpha) * memory_factor)
        else:
            fractional_factor = 1.0
        
        return dydt * fractional_factor
    
    def enhanced_temperature_cancer_model(self, t, y, drug_schedules=None, current_temp=37.0, use_circadian=True):
        """
        Temperature-integrated cancer model
        
        Args:
            t (float): Current time point
            y (array): State vector
            drug_schedules (dict): Drug dosing schedules
            current_temp (float): Current temperature
            use_circadian (bool): Whether to use circadian effects
            
        Returns:
            array: Derivatives of state variables
        """
        # Get advanced temperature modification factors
        temp_mods = TemperatureProtocol.advanced_temperature_modifier(current_temp, time_factor=t)
        
        # Call original advanced cancer model
        derivatives = self.advanced_cancer_model(t, y, drug_schedules, use_circadian)
        
        # Modification mapping for temperature effects
        modification_mapping = {
            0: ['metabolism', 'cellular_stress'],      # Sensitive cells (N1)
            1: ['metabolism', 'gene_expression'],      # Partially resistant cells (N2)
            2: ['immune_activation'],                  # Cytotoxic immune cells (I1)
            3: ['immune_activation'],                  # Regulatory immune cells (I2)
            5: ['metabolism', 'cellular_stress'],      # Angiogenesis factor
            7: ['resistance_development'],             # Resistant Type 1
            8: ['resistance_development'],             # Resistant Type 2
            12: ['gene_expression'],                   # Genetic stability
            13: ['metabolism']                         # Metabolism status
        }
        
        # Apply temperature modifications
        for idx, modification_keys in modification_mapping.items():
            if idx < len(derivatives):  # Safety check
                mod_factor = 1.0
                for key in modification_keys:
                    if key in temp_mods:
                        mod = temp_mods[key]
                        mod_factor *= (1 + mod['sensitivity'] * (mod['factor'] - 1))
                
                derivatives[idx] *= mod_factor
        
        return derivatives
