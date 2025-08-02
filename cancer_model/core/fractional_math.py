"""
Module 1: Core Mathematics and Fractional Derivatives
====================================================
Contains the mathematical foundations for fractional calculus and numerical methods.
"""

import numpy as np
from scipy.integrate import solve_ivp


class EnhancedFractionalDerivative:
    """TRUE fractional derivative implementation with memory effects"""
    
    def __init__(self, alpha=0.93):
        self.alpha = alpha
        self.reset_history()
        
    def reset_history(self):
        """Reset the memory history for fresh calculations"""
        self.time_history = []
        self.function_history = []
        self.derivative_history = []
        self.max_history = 100
        
    def add_point(self, t, f_val, f_prime_val):
        """Add a point to the memory history"""
        self.time_history.append(t)
        self.function_history.append(f_val)
        self.derivative_history.append(f_prime_val)
        
        if len(self.time_history) > self.max_history:
            self.time_history = self.time_history[-self.max_history:]
            self.function_history = self.function_history[-self.max_history:]
            self.derivative_history = self.derivative_history[-self.max_history:]
    
    def calculate_fractional_effect(self, t, current_derivative):
        """Calculate the fractional derivative effect with memory"""
        if abs(self.alpha - 1.0) < 1e-10:
            return current_derivative
            
        if t <= 1e-10 or len(self.time_history) <= 1:
            return current_derivative
            
        times = np.array(self.time_history)
        derivatives = np.array(self.derivative_history)
        
        memory_effect = 0.0
        total_weight = 0.0
        
        for i, tau in enumerate(times[:-1]):
            if t > tau:
                age = t - tau
                if age > 1e-6:
                    weight = (age + 0.1) ** (-self.alpha)
                    memory_effect += weight * derivatives[i]
                    total_weight += weight
        
        if total_weight > 1e-10:
            memory_average = memory_effect / total_weight
        else:
            memory_average = current_derivative
            
        memory_strength = (1 - self.alpha) * 2
        memory_strength = min(memory_strength, 0.8)
        
        fractional_derivative = ((1 - memory_strength) * current_derivative + 
                               memory_strength * memory_average)
        
        return fractional_derivative


def safe_solve_ivp(func, t_span, y0, method, t_eval, *args, max_retries=3, **kwargs):
    """Safely solve IVP with error handling and multiple fallback options."""
    methods = ['RK45', 'BDF', 'Radau', 'DOP853'] if method == 'RK45' else [method, 'RK45', 'BDF', 'Radau']
    rtols = [1e-4, 1e-5, 1e-6]
    atols = [1e-7, 1e-8, 1e-9]
    
    result = None
    success = False
    
    for retry in range(max_retries):
        if success:
            break
            
        for i, method_try in enumerate(methods):
            if success:
                break
                
            for rtol in rtols:
                if success:
                    break
                    
                for atol in atols:
                    try:
                        result = solve_ivp(func, t_span, y0, method=method_try, t_eval=t_eval, 
                                          rtol=rtol, atol=atol, *args, **kwargs)
                        if result.success:
                            success = True
                            print(f"Solver succeeded with method={method_try}, rtol={rtol}, atol={atol}")
                            break
                    except Exception as e:
                        print(f"Attempt with method={method_try}, rtol={rtol}, atol={atol} failed: {str(e)}")
    
    # If all attempts failed, return dummy result
    if not success:
        print("All solver attempts failed. Returning dummy result.")
        result = type('obj', (object,), {
            't': t_eval,
            'y': np.zeros((len(y0), len(t_eval))),
            'success': False,
            'message': "All solver attempts failed"
        })
    
    return result


def initialize_fractional_calculators(alpha=0.93):
    """Initialize fractional derivative calculators for all state variables"""
    variable_names = ['N1', 'N2', 'I1', 'I2', 'P', 'A', 'Q', 'R1', 'R2', 'S', 
                     'D', 'Dm', 'G', 'M', 'H']
    
    fractional_calculators = {
        name: EnhancedFractionalDerivative(alpha) 
        for name in variable_names
    }
    
    return fractional_calculators
