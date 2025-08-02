"""
Module 7: Visualization and Analysis
===================================
Handles all visualization, plotting, and results analysis.
Creates publication-quality figures and comprehensive reports.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


class VisualizationEngine:
    """Main visualization engine for cancer model results"""
    
    def __init__(self, output_dir='cancer_model_results'):
        self.output_dir = output_dir
        self.colors = self._setup_plotting_style()
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _setup_plotting_style(self):
        """Configure plotting style for consistent, publication-quality figures"""
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'Palatino', 'serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        # Create a custom colorblind-friendly palette
        colors = sns.color_palette("colorblind", 8)
        sns.set_palette(colors)
        
        return colors
    
    def safe_save_figure(self, filename, dpi=300):
        """Safely save matplotlib figure with error handling"""
        try:
            # Ensure directory exists
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Use a sanitized filename
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
            
            # Fallback to current directory if full path fails
            if not os.path.exists(os.path.dirname(safe_filename or '.')):
                safe_filename = os.path.basename(filename)
            
            plt.savefig(safe_filename, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved successfully: {safe_filename}")
            return safe_filename
        except Exception as e:
            print(f"Error saving figure: {e}")
            try:
                # Fallback: save in current directory
                fallback_filename = os.path.basename(filename)
                plt.savefig(fallback_filename, dpi=dpi, bbox_inches='tight')
                print(f"Fallback figure saved: {fallback_filename}")
                return fallback_filename
            except Exception as fallback_error:
                print(f"Fallback save failed: {fallback_error}")
                return None
    
    def create_protocol_comparison_plot(self, results, patient_profile='average'):
        """Create 4-panel comparison of treatment protocols"""
        if patient_profile not in results:
            print(f"Patient profile '{patient_profile}' not found in results")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Treatment Protocol Comparisons ({patient_profile.title()} Patient)', fontsize=18)
        
        # Plot tumor burden
        ax = axes[0, 0]
        for protocol, sim_result in results[patient_profile].items():
            if sim_result['success']:
                time_series = sim_result['time_series']
                ax.plot(time_series['time'], time_series['total_tumor'], 
                       label=protocol.replace('_', ' ').title(), linewidth=2)
        
        ax.set_title('Tumor Burden Over Time')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Total Cell Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot resistance fraction
        ax = axes[0, 1]
        for protocol, sim_result in results[patient_profile].items():
            if sim_result['success']:
                time_series = sim_result['time_series']
                ax.plot(time_series['time'], time_series['resistance_fraction'], 
                       label=protocol.replace('_', ' ').title(), linewidth=2)
        
        ax.set_title('Resistance Fraction Over Time')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Resistance Percentage (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot immune response
        ax = axes[1, 0]
        for protocol, sim_result in results[patient_profile].items():
            if sim_result['success']:
                time_series = sim_result['time_series']
                ax.plot(time_series['time'], time_series['cytotoxic_immune'], 
                       label=protocol.replace('_', ' ').title(), linewidth=2)
        
        ax.set_title('Cytotoxic Immune Response')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Immune Cell Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot genetic stability
        ax = axes[1, 1]
        for protocol, sim_result in results[patient_profile].items():
            if sim_result['success']:
                time_series = sim_result['time_series']
                ax.plot(time_series['time'], time_series['genetic_stability'], 
                       label=protocol.replace('_', ' ').title(), linewidth=2)
        
        ax.set_title('Genetic Stability')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Stability Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        filename = os.path.join(self.output_dir, f'protocol_comparison_{patient_profile}.png')
        saved_path = self.safe_save_figure(filename)
        plt.close()
        
        return saved_path
    
    def create_efficacy_metrics_chart(self, results, patient_profile='average'):
        """Create bar chart comparing treatment efficacy metrics"""
        if patient_profile not in results:
            print(f"Patient profile '{patient_profile}' not found in results")
            return None
        
        plt.figure(figsize=(14, 8))
        
        protocols = []
        efficacy_scores = []
        percent_reductions = []
        resistance_fractions = []
        
        for protocol, sim_result in results[patient_profile].items():
            if sim_result['success']:
                protocols.append(protocol.replace('_', ' ').title())
                metrics = sim_result['metrics']
                efficacy_scores.append(metrics['treatment_efficacy_score'])
                percent_reductions.append(metrics['percent_reduction'])
                resistance_fractions.append(metrics['final_resistance_fraction'])
        
        x = np.arange(len(protocols))
        width = 0.25
        
        plt.bar(x - width, percent_reductions, width, label='Tumor Reduction (%)', 
               color=self.colors[0], alpha=0.8)
        plt.bar(x, resistance_fractions, width, label='Final Resistance (%)', 
               color=self.colors[1], alpha=0.8)
        plt.bar(x + width, efficacy_scores, width, label='Efficacy Score', 
               color=self.colors[2], alpha=0.8)
        
        plt.xlabel('Treatment Protocol')
        plt.ylabel('Percentage / Score')
        plt.title(f'Treatment Efficacy Metrics Comparison ({patient_profile.title()} Patient)')
        plt.xticks(x, protocols, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (red, res, eff) in enumerate(zip(percent_reductions, resistance_fractions, efficacy_scores)):
            plt.text(i - width, red + 0.5, f'{red:.1f}', ha='center', va='bottom', fontsize=10)
            plt.text(i, res + 0.5, f'{res:.1f}', ha='center', va='bottom', fontsize=10)
            plt.text(i + width, eff + 0.5, f'{eff:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.output_dir, f'efficacy_metrics_{patient_profile}.png')
        saved_path = self.safe_save_figure(filename)
        plt.close()
        
        return saved_path
    
    def create_patient_comparison_plot(self, results, protocol_name='standard'):
        """Create 4-panel comparison across patient profiles"""
        patient_profiles = list(results.keys())
        
        # Check if protocol exists for patients
        valid_profiles = []
        for profile in patient_profiles:
            if protocol_name in results[profile] and results[profile][protocol_name]['success']:
                valid_profiles.append(profile)
        
        if not valid_profiles:
            print(f"Protocol '{protocol_name}' not found or failed for all patient profiles")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Patient Profile Comparisons ({protocol_name.title()} Protocol)', fontsize=18)
        
        # Plot tumor burden
        ax = axes[0, 0]
        for profile in valid_profiles:
            time_series = results[profile][protocol_name]['time_series']
            ax.plot(time_series['time'], time_series['total_tumor'], 
                   label=profile.title(), linewidth=2)
        
        ax.set_title('Tumor Burden')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cell Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot resistance fraction
        ax = axes[0, 1]
        for profile in valid_profiles:
            time_series = results[profile][protocol_name]['time_series']
            ax.plot(time_series['time'], time_series['resistance_fraction'], 
                   label=profile.title(), linewidth=2)
        
        ax.set_title('Resistance Fraction')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Percent (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot immune response
        ax = axes[1, 0]
        for profile in valid_profiles:
            time_series = results[profile][protocol_name]['time_series']
            ax.plot(time_series['time'], time_series['cytotoxic_immune'], 
                   label=profile.title(), linewidth=2)
        
        ax.set_title('Cytotoxic Immune Response')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cell Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot drug concentration
        ax = axes[1, 1]
        for profile in valid_profiles:
            time_series = results[profile][protocol_name]['time_series']
            ax.plot(time_series['time'], time_series['drug_concentration'], 
                   label=profile.title(), linewidth=2)
        
        ax.set_title('Drug Concentration')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Concentration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        filename = os.path.join(self.output_dir, f'patient_comparison_{protocol_name}.png')
        saved_path = self.safe_save_figure(filename)
        plt.close()
        
        return saved_path
    
    def create_heatmaps(self, results):
        """Create heatmaps comparing all protocols and patients"""
        # Get all unique patient profiles and protocols
        patient_profiles = list(results.keys())
        protocols = set()
        
        for profile in patient_profiles:
            for protocol in results[profile].keys():
                protocols.add(protocol)
        protocols = list(protocols)
        
        # Create metrics matrices
        efficacy_matrix = np.zeros((len(patient_profiles), len(protocols)))
        reduction_matrix = np.zeros((len(patient_profiles), len(protocols)))
        resistance_matrix = np.zeros((len(patient_profiles), len(protocols)))
        
        # Fill matrices
        for i, profile in enumerate(patient_profiles):
            for j, protocol in enumerate(protocols):
                if protocol in results[profile] and results[profile][protocol]['success']:
                    metrics = results[profile][protocol]['metrics']
                    efficacy_matrix[i, j] = metrics['treatment_efficacy_score']
                    reduction_matrix[i, j] = metrics['percent_reduction']
                    resistance_matrix[i, j] = metrics['final_resistance_fraction']
                else:
                    # Fill with NaN for failed simulations
                    efficacy_matrix[i, j] = np.nan
                    reduction_matrix[i, j] = np.nan
                    resistance_matrix[i, j] = np.nan
        
        # Plot heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Format protocol names for display
        protocol_labels = [p.replace('_', ' ').title() for p in protocols]
        patient_labels = [p.title() for p in patient_profiles]
        
        # Efficacy heatmap
        sns.heatmap(efficacy_matrix, annot=True, fmt=".1f", cmap="viridis", 
                  xticklabels=protocol_labels, yticklabels=patient_labels, ax=axes[0],
                  cbar_kws={'label': 'Efficacy Score'})
        axes[0].set_title('Treatment Efficacy Score')
        axes[0].set_xlabel('Protocol')
        axes[0].set_ylabel('Patient Profile')
        
        # Reduction heatmap
        sns.heatmap(reduction_matrix, annot=True, fmt=".1f", cmap="YlGnBu", 
                  xticklabels=protocol_labels, yticklabels=patient_labels, ax=axes[1],
                  cbar_kws={'label': 'Reduction (%)'})
        axes[1].set_title('Tumor Reduction (%)')
        axes[1].set_xlabel('Protocol')
        axes[1].set_ylabel('')
        
        # Resistance heatmap
        sns.heatmap(resistance_matrix, annot=True, fmt=".1f", cmap="YlOrRd", 
                  xticklabels=protocol_labels, yticklabels=patient_labels, ax=axes[2],
                  cbar_kws={'label': 'Resistance (%)'})
        axes[2].set_title('Final Resistance (%)')
        axes[2].set_xlabel('Protocol')
        axes[2].set_ylabel('')
        
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.output_dir, 'treatment_heatmaps.png')
        saved_path = self.safe_save_figure(filename)
        plt.close()
        
        return saved_path
    
    def create_detailed_analysis_plot(self, result, title_suffix=""):
        """Create detailed 9-panel analysis for a single simulation"""
        if not result['success']:
            print("Cannot create detailed analysis for failed simulation")
            return None
        
        time_series = result['time_series']
        
        # Create detailed visualization
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 3, figure=fig)
        
        # Main title
        fig.suptitle(f'Detailed Cancer Model Analysis{title_suffix}', fontsize=20)
        
        # Tumor dynamics (large plot)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(time_series['time'], time_series['total_tumor'], 'b-', linewidth=3, label='Total Tumor')
        ax1.plot(time_series['time'], time_series['sensitive_cells'], 'g--', linewidth=2, label='Sensitive')
        ax1.plot(time_series['time'], time_series['partially_resistant'], 'y--', linewidth=2, label='Partially Resistant')
        ax1.plot(time_series['time'], time_series['resistant_type1'] + time_series['resistant_type2'], 
                'r--', linewidth=2, label='Total Resistant')
        ax1.set_title('Tumor Cell Population Dynamics')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Cell Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Immune dynamics
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(time_series['time'], time_series['cytotoxic_immune'], 'g-', linewidth=2, label='Cytotoxic')
        ax2.plot(time_series['time'], time_series['regulatory_immune'], 'r-', linewidth=2, label='Regulatory')
        ax2.set_title('Immune Cell Dynamics')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Cell Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Resistance fraction
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_series['time'], time_series['resistance_fraction'], 'r-', linewidth=2)
        ax3.set_title('Resistance Development')
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Resistance (%)')
        ax3.grid(True, alpha=0.3)
        
        # Drug concentration
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_series['time'], time_series['drug_concentration'], 'b-', linewidth=2)
        ax4.set_title('Drug Concentration')
        ax4.set_xlabel('Time (days)')
        ax4.set_ylabel('Concentration')
        ax4.grid(True, alpha=0.3)
        
        # Genetic stability
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(time_series['time'], time_series['genetic_stability'], 'g-', linewidth=2)
        ax5.set_title('Genetic Stability')
        ax5.set_xlabel('Time (days)')
        ax5.set_ylabel('Stability Index')
        ax5.grid(True, alpha=0.3)
        
        # Microenvironment factors
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.plot(time_series['time'], time_series['hypoxia'], 'purple', linewidth=2, label='Hypoxia')
        ax6.plot(time_series['time'], time_series['metabolism'], 'orange', linewidth=2, label='Metabolism')
        ax6.set_title('Microenvironment')
        ax6.set_xlabel('Time (days)')
        ax6.set_ylabel('Level')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Tumor composition evolution (stacked area)
        ax7 = fig.add_subplot(gs[2, 1:])
        
        # Prepare composition data
        time = time_series['time']
        sensitive = time_series['sensitive_cells']
        partial = time_series['partially_resistant']
        resist1 = time_series['resistant_type1']
        resist2 = time_series['resistant_type2']
        quiescent = time_series['quiescent']
        senescent = time_series['senescent']
        
        # Create stacked area chart
        stack_data = np.vstack([sensitive, partial, resist1, resist2, quiescent, senescent])
        
        # Calculate percentage
        stack_sum = np.sum(stack_data, axis=0)
        stack_percent = np.zeros_like(stack_data)
        for i in range(stack_data.shape[0]):
            # Avoid division by zero
            mask = stack_sum > 0
            stack_percent[i, mask] = stack_data[i, mask] / stack_sum[mask] * 100
        
        # Plot stacked area
        labels = ['Sensitive', 'Partially Resistant', 'Resistant Type 1', 
                 'Resistant Type 2', 'Quiescent', 'Senescent']
        ax7.stackplot(time, stack_percent, labels=labels, alpha=0.8)
        
        ax7.set_title('Tumor Composition Evolution')
        ax7.set_xlabel('Time (days)')
        ax7.set_ylabel('Percentage (%)')
        ax7.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        safe_title = title_suffix.replace(' ', '_').replace('(', '').replace(')', '')
        filename = os.path.join(self.output_dir, f'detailed_analysis{safe_title}.png')
        saved_path = self.safe_save_figure(filename)
        plt.close()
        
        return saved_path
    
    def create_sensitivity_analysis_plot(self, sensitivity_results):
        """Create visualization for sensitivity analysis results"""
        n_params = len(sensitivity_results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=18)
        
        for idx, (param_name, param_results) in enumerate(sensitivity_results.items()):
            if idx >= 4:  # Limit to 4 parameters for this layout
                break
                
            ax = axes[idx]
            
            # Extract data
            param_values = [r['parameter_value'] for r in param_results if not r.get('failed', False)]
            efficacy_scores = [r['efficacy_score'] for r in param_results if not r.get('failed', False)]
            resistance_values = [r['final_resistance'] for r in param_results if not r.get('failed', False)]
            
            # Plot efficacy vs parameter value
            ax2 = ax.twinx()
            
            line1 = ax.plot(param_values, efficacy_scores, 'b-o', linewidth=2, label='Efficacy Score')
            line2 = ax2.plot(param_values, resistance_values, 'r-s', linewidth=2, label='Final Resistance (%)')
            
            ax.set_xlabel(f'{param_name}')
            ax.set_ylabel('Efficacy Score', color='b')
            ax2.set_ylabel('Final Resistance (%)', color='r')
            ax.set_title(f'Sensitivity to {param_name}')
            ax.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best')
        
        # Hide unused subplots
        for idx in range(n_params, 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        filename = os.path.join(self.output_dir, 'sensitivity_analysis.png')
        saved_path = self.safe_save_figure(filename)
        plt.close()
        
        return saved_path
    
    def create_optimization_plot(self, optimization_results, patient_name):
        """Create visualization for treatment optimization results"""
        best_result = optimization_results['best_protocol']
        all_results = optimization_results['all_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Treatment Optimization Results ({patient_name.title()} Patient)', fontsize=16)
        
        # Extract data for plotting
        doses = [r['dose'] for r in all_results]
        efficacies = [r['estimated_efficacy'] for r in all_results]
        toxicities = [r['estimated_toxicity'] for r in all_results]
        therapeutic_indices = [r['therapeutic_index'] for r in all_results]
        
        # Dose vs Efficacy
        ax1 = axes[0, 0]
        scatter = ax1.scatter(doses, efficacies, c=therapeutic_indices, cmap='viridis', s=50, alpha=0.7)
        ax1.scatter(best_result['dose'], best_result['estimated_efficacy'], 
                   c='red', s=100, marker='*', label='Best Protocol')
        ax1.set_xlabel('Dose')
        ax1.set_ylabel('Estimated Efficacy')
        ax1.set_title('Dose vs Efficacy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Dose vs Toxicity
        ax2 = axes[0, 1]
        ax2.scatter(doses, toxicities, c=therapeutic_indices, cmap='viridis', s=50, alpha=0.7)
        ax2.scatter(best_result['dose'], best_result['estimated_toxicity'], 
                   c='red', s=100, marker='*', label='Best Protocol')
        ax2.set_xlabel('Dose')
        ax2.set_ylabel('Estimated Toxicity')
        ax2.set_title('Dose vs Toxicity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Efficacy vs Toxicity
        ax3 = axes[1, 0]
        ax3.scatter(toxicities, efficacies, c=therapeutic_indices, cmap='viridis', s=50, alpha=0.7)
        ax3.scatter(best_result['estimated_toxicity'], best_result['estimated_efficacy'], 
                   c='red', s=100, marker='*', label='Best Protocol')
        ax3.set_xlabel('Estimated Toxicity')
        ax3.set_ylabel('Estimated Efficacy')
        ax3.set_title('Efficacy vs Toxicity Trade-off')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Top protocols comparison
        ax4 = axes[1, 1]
        top_5 = sorted(all_results, key=lambda x: x['therapeutic_index'], reverse=True)[:5]
        
        labels = [f"D:{r['dose']:.1f}\n{r['treatment_days']}/{r['rest_days']}" for r in top_5]
        indices = [r['therapeutic_index'] for r in top_5]
        
        bars = ax4.bar(range(len(labels)), indices, color=self.colors[:len(labels)])
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_ylabel('Therapeutic Index')
        ax4.set_title('Top 5 Protocol Configurations')
        ax4.grid(True, alpha=0.3)
        
        # Highlight best protocol
        bars[0].set_color('red')
        bars[0].set_alpha(0.8)
        
        # Add colorbar for scatter plots
        cbar = fig.colorbar(scatter, ax=axes[:, :2].ravel().tolist(), shrink=0.8)
        cbar.set_label('Therapeutic Index')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        filename = os.path.join(self.output_dir, f'optimization_{patient_name}.png')
        saved_path = self.safe_save_figure(filename)
        plt.close()
        
        return saved_path
    
    def generate_comprehensive_report(self, results, output_filename='comprehensive_report.html'):
        """Generate comprehensive HTML report with all visualizations"""
        # This would create an HTML report combining all visualizations
        # For now, return a simple text summary
        report_path = os.path.join(self.output_dir, 'visualization_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("Cancer Model Visualization Summary\n")
            f.write("===================================\n\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Generated visualizations for {len(results)} patient profiles\n")
            
            for patient_profile in results:
                f.write(f"\nPatient Profile: {patient_profile.title()}\n")
                successful_protocols = sum(1 for p in results[patient_profile].values() if p['success'])
                f.write(f"  Successful simulations: {successful_protocols}\n")
        
        return report_path
