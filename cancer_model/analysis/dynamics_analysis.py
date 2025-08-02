"""
Module: Model Dynamics Analysis
===============================
Advanced analysis of cancer model dynamics including phase space analysis,
stability analysis, bifurcation analysis, and dynamic behavior characterization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.signal import find_peaks
import pandas as pd
from pathlib import Path


class DynamicsAnalyzer:
    """Comprehensive dynamics analysis for cancer model"""
    
    def __init__(self, cancer_model, output_dir='results/dynamics'):
        self.cancer_model = cancer_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def phase_space_analysis(self, initial_conditions, params, time_span=500, 
                           variables=None, treatment_protocol=None):
        """
        Analyze phase space trajectories and attractors
        
        Args:
            initial_conditions (array): Initial state
            params (dict): Model parameters
            time_span (float): Simulation time
            variables (list): Variables to analyze (default: ['N1', 'R1', 'I1'])
            treatment_protocol (dict): Treatment schedule
            
        Returns:
            dict: Phase space analysis results
        """
        print("🔄 Running phase space analysis...")
        
        if variables is None:
            variables = ['N1', 'R1', 'I1']  # Sensitive cells, Resistant cells, Immune cells
        
        # Run simulation
        t_eval = np.linspace(0, time_span, int(time_span) + 1)
        
        def model_func(t, y):
            return self.cancer_model.enhanced_temperature_cancer_model(
                t, y, treatment_protocol, 37.0, True
            )
        
        from cancer_model.core.fractional_math import safe_solve_ivp
        result = safe_solve_ivp(model_func, [0, time_span], initial_conditions, 
                               'RK45', t_eval)
        
        if not result.success:
            print("❌ Phase space simulation failed")
            return None
        
        # Extract variable indices
        var_names = ['N1', 'N2', 'I1', 'I2', 'P', 'A', 'Q', 'R1', 'R2', 'S', 
                     'D', 'Dm', 'G', 'M', 'H']
        var_indices = {name: i for i, name in enumerate(var_names)}
        
        # Create phase space plots
        self._create_phase_space_plots(result, variables, var_indices)
        
        # Analyze attractors and fixed points
        attractor_analysis = self._analyze_attractors(result, variables, var_indices)
        
        # Calculate Lyapunov exponents (simplified)
        lyapunov_exp = self._estimate_lyapunov_exponents(result, variables, var_indices)
        
        return {
            'time': result.t,
            'solution': result.y,
            'variables': variables,
            'attractor_analysis': attractor_analysis,
            'lyapunov_exponents': lyapunov_exp,
            'phase_plots': self.output_dir / 'phase_space_analysis.png'
        }
    
    def _create_phase_space_plots(self, result, variables, var_indices):
        """Create comprehensive phase space visualizations"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # 2D phase portraits
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Only upper triangle
                    ax = plt.subplot(3, 4, len(variables) * i + j)
                    
                    idx1, idx2 = var_indices[var1], var_indices[var2]
                    x, y = result.y[idx1], result.y[idx2]
                    
                    # Color by time
                    colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
                    
                    # Plot trajectory
                    for k in range(len(x)-1):
                        ax.plot([x[k], x[k+1]], [y[k], y[k+1]], 
                               color=colors[k], alpha=0.7, linewidth=1)
                    
                    # Mark start and end
                    ax.scatter(x[0], y[0], color='green', s=100, marker='o', 
                              label='Start', zorder=5)
                    ax.scatter(x[-1], y[-1], color='red', s=100, marker='X', 
                              label='End', zorder=5)
                    
                    ax.set_xlabel(f'{var1} (cells)')
                    ax.set_ylabel(f'{var2} (cells)')
                    ax.set_title(f'{var1} vs {var2} Phase Portrait')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
        
        # 3D phase portrait
        if len(variables) >= 3:
            ax_3d = fig.add_subplot(3, 4, 10, projection='3d')
            
            idx1, idx2, idx3 = [var_indices[var] for var in variables[:3]]
            x, y, z = result.y[idx1], result.y[idx2], result.y[idx3]
            
            # Plot 3D trajectory
            ax_3d.plot(x, y, z, color='blue', alpha=0.7, linewidth=2)
            ax_3d.scatter(x[0], y[0], z[0], color='green', s=100, label='Start')
            ax_3d.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End')
            
            ax_3d.set_xlabel(f'{variables[0]} (cells)')
            ax_3d.set_ylabel(f'{variables[1]} (cells)')
            ax_3d.set_zlabel(f'{variables[2]} (cells)')
            ax_3d.set_title('3D Phase Portrait')
            ax_3d.legend()
        
        # Time series with phase indicators
        ax_time = plt.subplot(3, 4, (11, 12))
        
        for i, var in enumerate(variables):
            idx = var_indices[var]
            ax_time.plot(result.t, result.y[idx], label=var, linewidth=2)
        
        ax_time.set_xlabel('Time (days)')
        ax_time.set_ylabel('Cell Count')
        ax_time.set_title('Time Series Evolution')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase_space_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Phase space plots saved to {self.output_dir / 'phase_space_analysis.png'}")
    
    def _analyze_attractors(self, result, variables, var_indices):
        """Analyze attractors and fixed points"""
        
        # Look for steady states (simplified)
        final_values = {}
        convergence_analysis = {}
        
        for var in variables:
            idx = var_indices[var]
            final_values[var] = result.y[idx, -1]
            
            # Check for convergence (last 10% of simulation)
            end_portion = int(0.9 * len(result.t))
            end_values = result.y[idx, end_portion:]
            
            convergence_analysis[var] = {
                'final_value': final_values[var],
                'std_dev_end': np.std(end_values),
                'mean_end': np.mean(end_values),
                'is_converged': np.std(end_values) < 0.01 * np.mean(end_values)
            }
        
        return {
            'final_values': final_values,
            'convergence_analysis': convergence_analysis
        }
    
    def _estimate_lyapunov_exponents(self, result, variables, var_indices):
        """Estimate largest Lyapunov exponent (simplified method)"""
        
        lyapunov_estimates = {}
        
        for var in variables:
            idx = var_indices[var]
            trajectory = result.y[idx]
            
            # Simple finite difference approximation
            if len(trajectory) > 100:
                # Calculate log divergence of nearby trajectories (simplified)
                dt = result.t[1] - result.t[0]
                d_traj = np.diff(trajectory)
                
                # Estimate exponential growth/decay rate
                positive_changes = d_traj[d_traj > 0]
                if len(positive_changes) > 10:
                    lyap_est = np.mean(np.log(np.abs(positive_changes))) / dt
                    lyapunov_estimates[var] = lyap_est
                else:
                    lyapunov_estimates[var] = 0.0
            else:
                lyapunov_estimates[var] = 0.0
        
        return lyapunov_estimates
    
    def stability_analysis(self, equilibrium_finder=None, perturbation_size=0.01):
        """
        Analyze stability of equilibrium points
        
        Args:
            equilibrium_finder (function): Function to find equilibria
            perturbation_size (float): Size of perturbations for stability testing
            
        Returns:
            dict: Stability analysis results
        """
        print("⚖️  Running stability analysis...")
        
        # This would require numerical methods to find equilibria
        # For now, we'll analyze stability around the final state
        
        stability_results = {
            'analysis_type': 'local_stability',
            'perturbation_size': perturbation_size,
            'note': 'Full stability analysis requires equilibrium point calculation'
        }
        
        return stability_results
    
    def bifurcation_analysis(self, parameter_name, parameter_range, 
                           initial_conditions, base_params):
        """
        Analyze bifurcations as parameters change
        
        Args:
            parameter_name (str): Parameter to vary
            parameter_range (array): Range of parameter values
            initial_conditions (array): Initial state
            base_params (dict): Base parameter set
            
        Returns:
            dict: Bifurcation analysis results
        """
        print(f"🔀 Running bifurcation analysis for {parameter_name}...")
        
        final_states = []
        parameter_values = []
        
        for param_value in parameter_range:
            print(f"  Testing {parameter_name} = {param_value:.4f}")
            
            # Modify parameters
            test_params = base_params.copy()
            test_params[parameter_name] = param_value
            
            # Run simulation
            t_eval = np.linspace(0, 200, 201)  # Shorter for bifurcation analysis
            
            def model_func(t, y):
                return self.cancer_model.enhanced_temperature_cancer_model(
                    t, y, None, 37.0, True
                )
            
            from cancer_model.core.fractional_math import safe_solve_ivp
            result = safe_solve_ivp(model_func, [0, 200], initial_conditions, 
                                   'RK45', t_eval)
            
            if result.success:
                # Store final state
                final_states.append(result.y[:, -1])
                parameter_values.append(param_value)
        
        # Create bifurcation diagram
        self._create_bifurcation_plot(parameter_name, parameter_values, 
                                     final_states)
        
        return {
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'final_states': final_states,
            'bifurcation_plot': self.output_dir / f'bifurcation_{parameter_name}.png'
        }
    
    def _create_bifurcation_plot(self, parameter_name, parameter_values, final_states):
        """Create bifurcation diagram"""
        
        if not final_states:
            print("⚠️  No successful simulations for bifurcation plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Bifurcation Analysis: {parameter_name}', fontsize=16)
        
        # Plot key variables
        var_names = ['N1', 'R1', 'I1', 'Total_Tumor']
        var_indices = [0, 7, 2, None]  # Indices for N1, R1, I1
        
        final_states_array = np.array(final_states)
        
        for i, (var_name, var_idx) in enumerate(zip(var_names, var_indices)):
            ax = axes[i//2, i%2]
            
            if var_name == 'Total_Tumor':
                # Calculate total tumor burden
                y_values = (final_states_array[:, 0] +  # N1
                           final_states_array[:, 1] +   # N2
                           final_states_array[:, 6] +   # Q
                           final_states_array[:, 7] +   # R1
                           final_states_array[:, 8] +   # R2
                           final_states_array[:, 9])    # S
            else:
                y_values = final_states_array[:, var_idx]
            
            ax.plot(parameter_values, y_values, 'bo-', markersize=4, linewidth=1)
            ax.set_xlabel(parameter_name)
            ax.set_ylabel(f'{var_name} (final)')
            ax.set_title(f'{var_name} vs {parameter_name}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / f'bifurcation_{parameter_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Bifurcation plot saved to {self.output_dir / f'bifurcation_{parameter_name}.png'}")
    
    def oscillation_analysis(self, result_data, variables=None):
        """
        Analyze oscillatory behavior in the system
        
        Args:
            result_data (dict): Simulation results
            variables (list): Variables to analyze for oscillations
            
        Returns:
            dict: Oscillation analysis results
        """
        print("🌊 Analyzing oscillatory behavior...")
        
        if variables is None:
            variables = ['N1', 'R1', 'I1']
        
        var_names = ['N1', 'N2', 'I1', 'I2', 'P', 'A', 'Q', 'R1', 'R2', 'S', 
                     'D', 'Dm', 'G', 'M', 'H']
        var_indices = {name: i for i, name in enumerate(var_names)}
        
        oscillation_results = {}
        
        # Create oscillation analysis plot
        fig, axes = plt.subplots(len(variables), 2, figsize=(15, 4*len(variables)))
        if len(variables) == 1:
            axes = axes.reshape(1, -1)
        
        for i, var in enumerate(variables):
            if var not in var_indices:
                continue
                
            idx = var_indices[var]
            time_series = result_data['solution'][idx]
            time = result_data['time']
            
            # Find peaks and troughs
            peaks, peak_properties = find_peaks(time_series, height=np.mean(time_series))
            troughs, trough_properties = find_peaks(-time_series, height=-np.mean(time_series))
            
            # Calculate periods if oscillations exist
            periods = []
            if len(peaks) > 1:
                peak_times = time[peaks]
                periods = np.diff(peak_times)
            
            # Plot time series with peaks and troughs
            ax1 = axes[i, 0]
            ax1.plot(time, time_series, 'b-', linewidth=2, label=var)
            ax1.plot(time[peaks], time_series[peaks], 'ro', markersize=8, label='Peaks')
            ax1.plot(time[troughs], time_series[troughs], 'go', markersize=8, label='Troughs')
            ax1.axhline(y=np.mean(time_series), color='gray', linestyle='--', alpha=0.7, label='Mean')
            
            ax1.set_xlabel('Time (days)')
            ax1.set_ylabel(f'{var} (cells)')
            ax1.set_title(f'{var} Oscillation Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot frequency analysis (simple FFT)
            ax2 = axes[i, 1]
            
            # Remove trend and apply FFT
            detrended = time_series - np.mean(time_series)
            fft = np.fft.fft(detrended)
            freqs = np.fft.fftfreq(len(time_series), d=time[1]-time[0])
            
            # Plot power spectrum (positive frequencies only)
            positive_freqs = freqs[:len(freqs)//2]
            power_spectrum = np.abs(fft[:len(fft)//2])**2
            
            ax2.plot(positive_freqs, power_spectrum, 'b-', linewidth=2)
            ax2.set_xlabel('Frequency (1/days)')
            ax2.set_ylabel('Power')
            ax2.set_title(f'{var} Frequency Spectrum')
            ax2.grid(True, alpha=0.3)
            
            # Store results
            oscillation_results[var] = {
                'num_peaks': len(peaks),
                'num_troughs': len(troughs),
                'periods': periods,
                'mean_period': np.mean(periods) if periods else None,
                'amplitude': np.std(time_series),
                'mean_value': np.mean(time_series),
                'dominant_frequency': positive_freqs[np.argmax(power_spectrum)] if len(power_spectrum) > 0 else 0
            }
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'oscillation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Oscillation analysis saved to {self.output_dir / 'oscillation_analysis.png'}")
        
        return oscillation_results
    
    def comprehensive_dynamics_report(self, analysis_results):
        """Generate comprehensive dynamics analysis report"""
        
        report_path = self.output_dir / 'dynamics_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("CANCER MODEL DYNAMICS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Phase space analysis
            if 'phase_space' in analysis_results:
                f.write("PHASE SPACE ANALYSIS\n")
                f.write("-" * 20 + "\n")
                phase_data = analysis_results['phase_space']
                
                if phase_data and 'attractor_analysis' in phase_data:
                    f.write("Attractor Analysis:\n")
                    for var, conv_data in phase_data['attractor_analysis']['convergence_analysis'].items():
                        f.write(f"  {var}:\n")
                        f.write(f"    Final Value: {conv_data['final_value']:.2f}\n")
                        f.write(f"    Converged: {conv_data['is_converged']}\n")
                        f.write(f"    End Standard Deviation: {conv_data['std_dev_end']:.4f}\n")
                    f.write("\n")
                
                if 'lyapunov_exponents' in phase_data:
                    f.write("Lyapunov Exponents (estimates):\n")
                    for var, lyap in phase_data['lyapunov_exponents'].items():
                        f.write(f"  {var}: {lyap:.4f}\n")
                    f.write("\n")
            
            # Oscillation analysis
            if 'oscillations' in analysis_results:
                f.write("OSCILLATION ANALYSIS\n")
                f.write("-" * 20 + "\n")
                osc_data = analysis_results['oscillations']
                
                for var, osc_results in osc_data.items():
                    f.write(f"{var}:\n")
                    f.write(f"  Number of peaks: {osc_results['num_peaks']}\n")
                    f.write(f"  Mean period: {osc_results['mean_period']:.2f} days\n" if osc_results['mean_period'] else "  No clear periodicity\n")
                    f.write(f"  Amplitude (std): {osc_results['amplitude']:.2f}\n")
                    f.write(f"  Dominant frequency: {osc_results['dominant_frequency']:.4f} 1/days\n")
                    f.write("\n")
            
            # Bifurcation analysis
            if 'bifurcation' in analysis_results:
                f.write("BIFURCATION ANALYSIS\n")
                f.write("-" * 20 + "\n")
                bif_data = analysis_results['bifurcation']
                f.write(f"Parameter analyzed: {bif_data['parameter_name']}\n")
                f.write(f"Parameter range: {min(bif_data['parameter_values']):.4f} to {max(bif_data['parameter_values']):.4f}\n")
                f.write(f"Number of parameter values tested: {len(bif_data['parameter_values'])}\n\n")
            
            f.write("SUMMARY AND INSIGHTS\n")
            f.write("-" * 20 + "\n")
            f.write("1. Check phase space plots for attractor behavior\n")
            f.write("2. Review oscillation analysis for periodic behavior\n")
            f.write("3. Examine bifurcation diagrams for parameter sensitivity\n")
            f.write("4. Use Lyapunov exponents to assess stability\n")
        
        print(f"📄 Comprehensive dynamics report saved to {report_path}")
        return report_path