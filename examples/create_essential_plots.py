#!/usr/bin/env python3
"""
Comprehensive Essential Time-Series Plots for All Patient-Protocol Combinations
==============================================================================
Creates plots for all combinations of patient profiles and treatment protocols
with high-quality formatting, bold styling, and saves all data as CSV files.

Features:
- All 4 patient profiles: average, young, elderly, compromised
- All 5 treatment protocols: standard, continuous, adaptive, immuno_combo, hyperthermia
- High DPI (300) publication-quality plots
- Bold fonts and enhanced styling
- CSV data export for all scenarios
- Clean presentation without interpretation labels

Run this to generate comprehensive reviewer-required figures with data.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import itertools

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.ultimate_realistic_model import RealisticCancerModelRunner

def create_comprehensive_reviewer_plots():
    """Create comprehensive plots for all patient-protocol combinations"""
    
    print("🎨 Creating Comprehensive Time-Series Plots for All Combinations")
    print("=" * 80)
    
    # Define all combinations
    patient_profiles = ['average', 'young', 'elderly', 'compromised']
    treatment_protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']
    resistance_levels = ['low', 'medium', 'high']
    
    total_combinations = len(patient_profiles) * len(treatment_protocols) * len(resistance_levels)
    print(f"Total simulations to run: {total_combinations}")
    print(f"Patient profiles: {patient_profiles}")
    print(f"Treatment protocols: {treatment_protocols}")
    print(f"Resistance levels: {resistance_levels}")
    
    # Create output directories
    output_dir = Path('results/comprehensive_plots')
    data_dir = output_dir / 'data'
    plots_dir = output_dir / 'plots'
    
    for dir_path in [output_dir, data_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set up high-quality plotting style
    setup_plot_style()
    
    # Run all simulations
    print("\nRunning comprehensive simulations...")
    all_scenarios = run_all_combinations(patient_profiles, treatment_protocols, resistance_levels)
    
    if not all_scenarios:
        print("❌ No successful simulations - check model configuration")
        return
    
    # Export all data to CSV
    export_comprehensive_data(all_scenarios, data_dir)
    
    # Create comprehensive plots
    create_resistance_comparison_plots(all_scenarios, plots_dir, data_dir)
    create_efficacy_comparison_plots(all_scenarios, plots_dir, data_dir)
    create_patient_protocol_heatmaps(all_scenarios, plots_dir, data_dir)
    create_detailed_time_series_plots(all_scenarios, plots_dir, data_dir)
    
    # Create summary analysis
    create_comprehensive_summary(all_scenarios, data_dir)
    
    print(f"\n🎉 Comprehensive plots and data created successfully!")
    print(f"📁 Plots saved to: {plots_dir}")
    print(f"📊 Data saved to: {data_dir}")
    print(f"📋 Total scenarios analyzed: {len(all_scenarios)}")

def setup_plot_style():
    """Set up high-quality plotting style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (16, 10),
        'font.weight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'grid.alpha': 0.4,
        'lines.linewidth': 2.5,
        'lines.markersize': 6
    })

def run_all_combinations(patient_profiles, treatment_protocols, resistance_levels):
    """Run simulations for all combinations"""
    
    all_scenarios = {}
    simulation_count = 0
    
    for resistance_level in resistance_levels:
        print(f"\n  Running {resistance_level} resistance scenarios...")
        runner = RealisticCancerModelRunner(resistance_level=resistance_level)
        
        for patient_profile in patient_profiles:
            for protocol in treatment_protocols:
                simulation_count += 1
                scenario_name = f"{resistance_level}_{patient_profile}_{protocol}"
                
                print(f"    {simulation_count:2d}. {scenario_name}...")
                
                try:
                    result = runner.run_single_simulation(patient_profile, protocol, 500)
                    
                    if result['success']:
                        resistance = result['metrics']['final_resistance_fraction']
                        efficacy = result['metrics']['treatment_efficacy_score']
                        all_scenarios[scenario_name] = result
                        print(f"        ✅ Success: {resistance:.1f}% resistance, {efficacy:.2f} efficacy")
                    else:
                        print(f"        ❌ Failed: {result.get('error_message', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"        ⚠️ Error: {str(e)}")
    
    print(f"\n📊 Successful simulations: {len(all_scenarios)}/{simulation_count}")
    return all_scenarios

def export_comprehensive_data(all_scenarios, data_dir):
    """Export all comprehensive data to CSV files"""
    
    print("📊 Exporting comprehensive data to CSV...")
    
    # Create master dataset
    master_data = []
    time_series_data = []
    
    for scenario_name, result in all_scenarios.items():
        # Parse scenario name
        parts = scenario_name.split('_')
        resistance_level = parts[0]
        patient_profile = parts[1]
        protocol = '_'.join(parts[2:])  # Handle protocols with underscores
        
        # Extract metrics
        metrics = result['metrics']
        time_series = result['time_series']
        
        # Add to master data
        master_data.append({
            'scenario': scenario_name,
            'resistance_level': resistance_level,
            'patient_profile': patient_profile,
            'treatment_protocol': protocol,
            'final_resistance_pct': metrics['final_resistance_fraction'],
            'treatment_efficacy_score': metrics['treatment_efficacy_score'],
            'tumor_reduction_pct': metrics['percent_reduction'],
            'initial_burden': metrics['initial_burden'],
            'final_burden': metrics['final_burden'],
            'max_drug_concentration': metrics['max_drug_concentration'],
            'immune_activation': metrics['immune_activation']
        })
        
        # Add time series data
        for i, time_point in enumerate(time_series['time']):
            time_series_data.append({
                'scenario': scenario_name,
                'resistance_level': resistance_level,
                'patient_profile': patient_profile,
                'treatment_protocol': protocol,
                'time_days': time_point,
                'total_tumor': time_series['total_tumor'][i],
                'sensitive_cells': time_series['sensitive_cells'][i],
                'resistant_type1': time_series['resistant_type1'][i],
                'resistant_type2': time_series['resistant_type2'][i],
                'total_resistant': time_series['resistant_type1'][i] + time_series['resistant_type2'][i],
                'immune_cells': time_series['immune_cells'][i],
                'drug_concentration': time_series['drug_concentration'][i],
                'resistance_fraction': time_series['resistance_fraction'][i]
            })
    
    # Save master datasets
    master_df = pd.DataFrame(master_data)
    master_df.to_csv(data_dir / 'master_summary.csv', index=False)
    
    time_series_df = pd.DataFrame(time_series_data)
    time_series_df.to_csv(data_dir / 'master_time_series.csv', index=False)
    
    print(f"  ✅ Master summary saved: master_summary.csv ({len(master_df)} scenarios)")
    print(f"  ✅ Master time series saved: master_time_series.csv ({len(time_series_df)} data points)")

def create_resistance_comparison_plots(all_scenarios, plots_dir, data_dir):
    """Create comprehensive resistance comparison plots"""
    
    print("📈 Creating resistance comparison plots...")
    
    # Prepare data for plotting
    resistance_data = []
    for scenario_name, result in all_scenarios.items():
        parts = scenario_name.split('_')
        resistance_level = parts[0]
        patient_profile = parts[1]
        protocol = '_'.join(parts[2:])
        
        resistance_data.append({
            'resistance_level': resistance_level,
            'patient_profile': patient_profile,
            'treatment_protocol': protocol,
            'final_resistance': result['metrics']['final_resistance_fraction']
        })
    
    df = pd.DataFrame(resistance_data)
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive Resistance Analysis Across All Combinations', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Plot A: By Patient Profile
    ax = axes[0, 0]
    resistance_levels = ['low', 'medium', 'high']
    patient_profiles = ['average', 'young', 'elderly', 'compromised']
    colors = {'low': '#228B22', 'medium': '#FF8C00', 'high': '#DC143C'}
    
    x = np.arange(len(patient_profiles))
    width = 0.25
    
    for i, level in enumerate(resistance_levels):
        level_data = df[df['resistance_level'] == level]
        means = [level_data[level_data['patient_profile'] == profile]['final_resistance'].mean() 
                for profile in patient_profiles]
        
        ax.bar(x + i*width, means, width, label=f'{level.capitalize()} Resistance', 
               color=colors[level], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Patient Profile', fontweight='bold')
    ax.set_ylabel('Final Resistance (%)', fontweight='bold')
    ax.set_title('A) Resistance by Patient Profile', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([p.capitalize() for p in patient_profiles])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot B: By Treatment Protocol
    ax = axes[0, 1]
    protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']
    
    x = np.arange(len(protocols))
    
    for i, level in enumerate(resistance_levels):
        level_data = df[df['resistance_level'] == level]
        means = [level_data[level_data['treatment_protocol'] == protocol]['final_resistance'].mean() 
                for protocol in protocols]
        
        ax.bar(x + i*width, means, width, label=f'{level.capitalize()} Resistance', 
               color=colors[level], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Treatment Protocol', fontweight='bold')
    ax.set_ylabel('Final Resistance (%)', fontweight='bold')
    ax.set_title('B) Resistance by Treatment Protocol', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([p.replace('_', '\n').title() for p in protocols], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot C: Distribution of resistance values
    ax = axes[1, 0]
    for level in resistance_levels:
        level_data = df[df['resistance_level'] == level]['final_resistance']
        ax.hist(level_data, bins=15, alpha=0.6, label=f'{level.capitalize()}', 
                color=colors[level], edgecolor='black')
    
    ax.set_xlabel('Final Resistance (%)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('C) Distribution of Final Resistance Values', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot D: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_stats = []
    for level in resistance_levels:
        level_data = df[df['resistance_level'] == level]['final_resistance']
        summary_stats.append([
            level.capitalize(),
            f"{level_data.mean():.1f}%",
            f"{level_data.std():.1f}%",
            f"{level_data.min():.1f}%",
            f"{level_data.max():.1f}%"
        ])
    
    table = ax.table(cellText=summary_stats,
                    colLabels=['Resistance\nLevel', 'Mean', 'Std Dev', 'Min', 'Max'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0.1, 0.3, 0.8, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i, level in enumerate(resistance_levels):
        for j in range(5):
            table[(i+1, j)].set_facecolor(colors[level])
            table[(i+1, j)].set_alpha(0.3)
            table[(i+1, j)].set_text_props(weight='bold')
    
    ax.set_title('D) Resistance Summary Statistics', fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'Figure_1_Comprehensive_Resistance_Analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Save resistance data
    df.to_csv(data_dir / 'resistance_analysis.csv', index=False)
    print("  ✅ Figure 1: Comprehensive Resistance Analysis created")

def create_efficacy_comparison_plots(all_scenarios, plots_dir, data_dir):
    """Create comprehensive efficacy comparison plots"""
    
    print("📈 Creating efficacy comparison plots...")
    
    # Prepare data for plotting
    efficacy_data = []
    for scenario_name, result in all_scenarios.items():
        parts = scenario_name.split('_')
        resistance_level = parts[0]
        patient_profile = parts[1]
        protocol = '_'.join(parts[2:])
        
        efficacy_data.append({
            'resistance_level': resistance_level,
            'patient_profile': patient_profile,
            'treatment_protocol': protocol,
            'efficacy_score': result['metrics']['treatment_efficacy_score'],
            'tumor_reduction': result['metrics']['percent_reduction']
        })
    
    df = pd.DataFrame(efficacy_data)
    
    # Create comprehensive efficacy plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive Treatment Efficacy Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    colors = {'low': '#228B22', 'medium': '#FF8C00', 'high': '#DC143C'}
    
    # Plot A: Efficacy by Patient Profile
    ax = axes[0, 0]
    patient_profiles = ['average', 'young', 'elderly', 'compromised']
    resistance_levels = ['low', 'medium', 'high']
    
    x = np.arange(len(patient_profiles))
    width = 0.25
    
    for i, level in enumerate(resistance_levels):
        level_data = df[df['resistance_level'] == level]
        means = [level_data[level_data['patient_profile'] == profile]['efficacy_score'].mean() 
                for profile in patient_profiles]
        
        ax.bar(x + i*width, means, width, label=f'{level.capitalize()}', 
               color=colors[level], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Patient Profile', fontweight='bold')
    ax.set_ylabel('Treatment Efficacy Score', fontweight='bold')
    ax.set_title('A) Efficacy by Patient Profile', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([p.capitalize() for p in patient_profiles])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot B: Efficacy by Treatment Protocol
    ax = axes[0, 1]
    protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']
    
    x = np.arange(len(protocols))
    
    for i, level in enumerate(resistance_levels):
        level_data = df[df['resistance_level'] == level]
        means = [level_data[level_data['treatment_protocol'] == protocol]['efficacy_score'].mean() 
                for protocol in protocols]
        
        ax.bar(x + i*width, means, width, label=f'{level.capitalize()}', 
               color=colors[level], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Treatment Protocol', fontweight='bold')
    ax.set_ylabel('Treatment Efficacy Score', fontweight='bold')
    ax.set_title('B) Efficacy by Treatment Protocol', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([p.replace('_', '\n').title() for p in protocols], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot C: Efficacy vs Resistance scatter
    ax = axes[1, 0]
    for level in resistance_levels:
        level_data = df[df['resistance_level'] == level]
        ax.scatter(level_data['efficacy_score'], level_data['tumor_reduction'], 
                  c=colors[level], alpha=0.6, s=60, label=f'{level.capitalize()}',
                  edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Treatment Efficacy Score', fontweight='bold')
    ax.set_ylabel('Tumor Reduction (%)', fontweight='bold')
    ax.set_title('C) Efficacy Score vs Tumor Reduction', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot D: Best combinations table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Find top 5 combinations by efficacy
    top_combinations = df.nlargest(5, 'efficacy_score')
    
    table_data = []
    for _, row in top_combinations.iterrows():
        table_data.append([
            row['patient_profile'].capitalize(),
            row['treatment_protocol'].replace('_', ' ').title(),
            row['resistance_level'].capitalize(),
            f"{row['efficacy_score']:.2f}",
            f"{row['tumor_reduction']:.1f}%"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Patient', 'Protocol', 'Resistance', 'Efficacy', 'Reduction'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0.05, 0.3, 0.9, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style the table
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('D) Top 5 Treatment Combinations by Efficacy', fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'Figure_2_Comprehensive_Efficacy_Analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Save efficacy data
    df.to_csv(data_dir / 'efficacy_analysis.csv', index=False)
    print("  ✅ Figure 2: Comprehensive Efficacy Analysis created")

def create_patient_protocol_heatmaps(all_scenarios, plots_dir, data_dir):
    """Create heatmaps showing patient-protocol combinations"""
    
    print("📈 Creating patient-protocol heatmaps...")
    
    # Prepare data for heatmaps
    patient_profiles = ['average', 'young', 'elderly', 'compromised']
    protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']
    resistance_levels = ['low', 'medium', 'high']
    
    # Create separate heatmaps for each resistance level
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Patient-Protocol Combination Heatmaps by Resistance Level', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    for idx, resistance_level in enumerate(resistance_levels):
        ax = axes[idx]
        
        # Create matrix for this resistance level
        efficacy_matrix = np.zeros((len(patient_profiles), len(protocols)))
        
        for i, patient in enumerate(patient_profiles):
            for j, protocol in enumerate(protocols):
                scenario_name = f"{resistance_level}_{patient}_{protocol}"
                if scenario_name in all_scenarios:
                    efficacy_matrix[i, j] = all_scenarios[scenario_name]['metrics']['treatment_efficacy_score']
                else:
                    efficacy_matrix[i, j] = np.nan
        
        # Create heatmap
        im = ax.imshow(efficacy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=50)
        
        # Add text annotations
        for i in range(len(patient_profiles)):
            for j in range(len(protocols)):
                if not np.isnan(efficacy_matrix[i, j]):
                    text = ax.text(j, i, f'{efficacy_matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        # Customize axes
        ax.set_xticks(np.arange(len(protocols)))
        ax.set_yticks(np.arange(len(patient_profiles)))
        ax.set_xticklabels([p.replace('_', '\n').title() for p in protocols], rotation=45)
        ax.set_yticklabels([p.capitalize() for p in patient_profiles])
        ax.set_title(f'{resistance_level.capitalize()} Resistance\nTreatment Efficacy', fontweight='bold')
        
        # Add gridlines
        ax.set_xticks(np.arange(len(protocols)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(patient_profiles)+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Treatment Efficacy Score', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'Figure_3_Patient_Protocol_Heatmaps.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print("  ✅ Figure 3: Patient-Protocol Heatmaps created")

def create_detailed_time_series_plots(all_scenarios, plots_dir, data_dir):
    """Create detailed time series plots for selected scenarios"""
    
    print("📈 Creating detailed time series plots...")
    
    # Select representative scenarios for detailed plotting
    selected_scenarios = []
    
    # Get one example from each resistance level with different patient-protocol combinations
    scenario_examples = [
        'low_young_immuno_combo',      # Best case
        'medium_average_standard',      # Typical case  
        'high_elderly_adaptive'         # Challenging case
    ]
    
    # Find actual scenarios that match or are similar
    for example in scenario_examples:
        if example in all_scenarios:
            selected_scenarios.append((example, all_scenarios[example]))
        else:
            # Find a similar scenario
            for scenario_name in all_scenarios:
                parts = scenario_name.split('_')
                if parts[0] in example and len(selected_scenarios) < 3:
                    selected_scenarios.append((scenario_name, all_scenarios[scenario_name]))
                    break
    
    if not selected_scenarios:
        # Fallback: take first 3 scenarios
        items = list(all_scenarios.items())[:3]
        selected_scenarios = items
    
    # Create time series plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Detailed Time Series Analysis - Representative Scenarios', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot A: Tumor burden over time
    ax = axes[0, 0]
    for i, (scenario_name, result) in enumerate(selected_scenarios):
        time_series = result['time_series']
        ax.plot(time_series['time'], time_series['total_tumor'], 
               color=colors[i], linewidth=3, label=scenario_name.replace('_', ' ').title())
    
    ax.set_xlabel('Time (days)', fontweight='bold')
    ax.set_ylabel('Total Tumor Burden', fontweight='bold')
    ax.set_title('A) Tumor Burden Dynamics', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot B: Resistance development
    ax = axes[0, 1]
    for i, (scenario_name, result) in enumerate(selected_scenarios):
        time_series = result['time_series']
        ax.plot(time_series['time'], time_series['resistance_fraction'], 
               color=colors[i], linewidth=3, label=scenario_name.replace('_', ' ').title())
    
    ax.set_xlabel('Time (days)', fontweight='bold')
    ax.set_ylabel('Resistance Fraction (%)', fontweight='bold')
    ax.set_title('B) Resistance Development', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot C: Drug concentration
    ax = axes[1, 0]
    for i, (scenario_name, result) in enumerate(selected_scenarios):
        time_series = result['time_series']
        ax.plot(time_series['time'], time_series['drug_concentration'], 
               color=colors[i], linewidth=3, label=scenario_name.replace('_', ' ').title())
    
    ax.set_xlabel('Time (days)', fontweight='bold')
    ax.set_ylabel('Drug Concentration', fontweight='bold')
    ax.set_title('C) Drug Concentration Profiles', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot D: Immune response
    ax = axes[1, 1]
    for i, (scenario_name, result) in enumerate(selected_scenarios):
        time_series = result['time_series']
        ax.plot(time_series['time'], time_series['immune_cells'], 
               color=colors[i], linewidth=3, label=scenario_name.replace('_', ' ').title())
    
    ax.set_xlabel('Time (days)', fontweight='bold')
    ax.set_ylabel('Immune Cells', fontweight='bold')
    ax.set_title('D) Immune Cell Dynamics', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'Figure_4_Detailed_Time_Series.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print("  ✅ Figure 4: Detailed Time Series created")

def create_comprehensive_summary(all_scenarios, data_dir):
    """Create comprehensive summary analysis"""
    
    print("📋 Creating comprehensive summary...")
    
    # Analyze results by different dimensions
    summary_stats = {}
    
    # By resistance level
    resistance_analysis = {}
    for resistance_level in ['low', 'medium', 'high']:
        scenarios = [s for name, s in all_scenarios.items() if name.startswith(resistance_level)]
        if scenarios:
            resistances = [s['metrics']['final_resistance_fraction'] for s in scenarios]
            efficacies = [s['metrics']['treatment_efficacy_score'] for s in scenarios]
            
            resistance_analysis[resistance_level] = {
                'count': len(scenarios),
                'mean_resistance': np.mean(resistances),
                'std_resistance': np.std(resistances),
                'mean_efficacy': np.mean(efficacies),
                'std_efficacy': np.std(efficacies)
            }
    
    # By patient profile
    patient_analysis = {}
    for patient_profile in ['average', 'young', 'elderly', 'compromised']:
        scenarios = [s for name, s in all_scenarios.items() if f'_{patient_profile}_' in name]
        if scenarios:
            resistances = [s['metrics']['final_resistance_fraction'] for s in scenarios]
            efficacies = [s['metrics']['treatment_efficacy_score'] for s in scenarios]
            
            patient_analysis[patient_profile] = {
                'count': len(scenarios),
                'mean_resistance': np.mean(resistances),
                'std_resistance': np.std(resistances),
                'mean_efficacy': np.mean(efficacies),
                'std_efficacy': np.std(efficacies)
            }
    
    # By treatment protocol
    protocol_analysis = {}
    for protocol in ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']:
        scenarios = [s for name, s in all_scenarios.items() if name.endswith(f'_{protocol}')]
        if scenarios:
            resistances = [s['metrics']['final_resistance_fraction'] for s in scenarios]
            efficacies = [s['metrics']['treatment_efficacy_score'] for s in scenarios]
            
            protocol_analysis[protocol] = {
                'count': len(scenarios),
                'mean_resistance': np.mean(resistances),
                'std_resistance': np.std(resistances),
                'mean_efficacy': np.mean(efficacies),
                'std_efficacy': np.std(efficacies)
            }
    
    # Find best and worst performers
    all_results = [(name, s['metrics']['treatment_efficacy_score'], 
                   s['metrics']['final_resistance_fraction']) for name, s in all_scenarios.items()]
    
    best_efficacy = max(all_results, key=lambda x: x[1])
    worst_efficacy = min(all_results, key=lambda x: x[1])
    lowest_resistance = min(all_results, key=lambda x: x[2])
    highest_resistance = max(all_results, key=lambda x: x[2])
    
    # Create summary DataFrames
    resistance_df = pd.DataFrame(resistance_analysis).T
    resistance_df.to_csv(data_dir / 'summary_by_resistance_level.csv')
    
    patient_df = pd.DataFrame(patient_analysis).T
    patient_df.to_csv(data_dir / 'summary_by_patient_profile.csv')
    
    protocol_df = pd.DataFrame(protocol_analysis).T
    protocol_df.to_csv(data_dir / 'summary_by_treatment_protocol.csv')
    
    # Create comprehensive report
    report_path = data_dir / 'comprehensive_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("COMPREHENSIVE CANCER MODEL ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"OVERVIEW:\n")
        f.write(f"Total scenarios analyzed: {len(all_scenarios)}\n")
        f.write(f"Patient profiles: 4 (average, young, elderly, compromised)\n")
        f.write(f"Treatment protocols: 5 (standard, continuous, adaptive, immuno_combo, hyperthermia)\n")
        f.write(f"Resistance levels: 3 (low, medium, high)\n")
        f.write(f"Simulation duration: 500 days\n\n")
        
        f.write("RESISTANCE LEVEL ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        for level, stats in resistance_analysis.items():
            f.write(f"{level.upper()} Resistance ({stats['count']} scenarios):\n")
            f.write(f"  Mean resistance: {stats['mean_resistance']:.1f}% ± {stats['std_resistance']:.1f}%\n")
            f.write(f"  Mean efficacy: {stats['mean_efficacy']:.2f} ± {stats['std_efficacy']:.2f}\n\n")
        
        f.write("PATIENT PROFILE ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        for profile, stats in patient_analysis.items():
            f.write(f"{profile.upper()} Patient ({stats['count']} scenarios):\n")
            f.write(f"  Mean resistance: {stats['mean_resistance']:.1f}% ± {stats['std_resistance']:.1f}%\n")
            f.write(f"  Mean efficacy: {stats['mean_efficacy']:.2f} ± {stats['std_efficacy']:.2f}\n\n")
        
        f.write("TREATMENT PROTOCOL ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        for protocol, stats in protocol_analysis.items():
            f.write(f"{protocol.upper().replace('_', ' ')} Protocol ({stats['count']} scenarios):\n")
            f.write(f"  Mean resistance: {stats['mean_resistance']:.1f}% ± {stats['std_resistance']:.1f}%\n")
            f.write(f"  Mean efficacy: {stats['mean_efficacy']:.2f} ± {stats['std_efficacy']:.2f}\n\n")
        
        f.write("EXTREME CASES:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Best efficacy: {best_efficacy[0]} (Efficacy: {best_efficacy[1]:.2f})\n")
        f.write(f"Worst efficacy: {worst_efficacy[0]} (Efficacy: {worst_efficacy[1]:.2f})\n")
        f.write(f"Lowest resistance: {lowest_resistance[0]} (Resistance: {lowest_resistance[2]:.1f}%)\n")
        f.write(f"Highest resistance: {highest_resistance[0]} (Resistance: {highest_resistance[2]:.1f}%)\n\n")
        
        f.write("KEY INSIGHTS:\n")
        f.write("-" * 15 + "\n")
        
        # Best patient profile
        best_patient = min(patient_analysis.items(), key=lambda x: x[1]['mean_resistance'])
        f.write(f"Best patient profile (lowest resistance): {best_patient[0].capitalize()}\n")
        
        # Best protocol
        best_protocol = max(protocol_analysis.items(), key=lambda x: x[1]['mean_efficacy'])
        f.write(f"Best treatment protocol (highest efficacy): {best_protocol[0].replace('_', ' ').title()}\n")
        
        # Resistance progression
        low_resistance = resistance_analysis['low']['mean_resistance']
        high_resistance = resistance_analysis['high']['mean_resistance']
        f.write(f"Resistance progression: {low_resistance:.1f}% → {high_resistance:.1f}% ({high_resistance-low_resistance:.1f}% increase)\n")
        
        f.write(f"\nDATA FILES GENERATED:\n")
        f.write("-" * 20 + "\n")
        f.write("• master_summary.csv - All scenario results\n")
        f.write("• master_time_series.csv - Complete time series data\n")
        f.write("• resistance_analysis.csv - Resistance analysis data\n")
        f.write("• efficacy_analysis.csv - Efficacy analysis data\n")
        f.write("• summary_by_resistance_level.csv - Statistics by resistance\n")
        f.write("• summary_by_patient_profile.csv - Statistics by patient\n")
        f.write("• summary_by_treatment_protocol.csv - Statistics by protocol\n")
        
        f.write(f"\nFIGURES GENERATED:\n")
        f.write("-" * 18 + "\n")
        f.write("• Figure_1_Comprehensive_Resistance_Analysis.png\n")
        f.write("• Figure_2_Comprehensive_Efficacy_Analysis.png\n")
        f.write("• Figure_3_Patient_Protocol_Heatmaps.png\n")
        f.write("• Figure_4_Detailed_Time_Series.png\n")
    
    print(f"  ✅ Comprehensive summary saved: {report_path}")
    print(f"  ✅ Summary statistics saved to CSV files")

def create_data_dictionary():
    """Create comprehensive data dictionary for all files"""
    
    data_dict = {
        'file_name': [
            'master_summary.csv',
            'master_time_series.csv',
            'resistance_analysis.csv',
            'efficacy_analysis.csv',
            'summary_by_resistance_level.csv',
            'summary_by_patient_profile.csv',
            'summary_by_treatment_protocol.csv',
            'comprehensive_analysis_report.txt'
        ],
        'description': [
            'Complete summary of all scenario results with metrics',
            'Complete time series data for all scenarios',
            'Resistance analysis data used for plotting',
            'Efficacy analysis data used for plotting',
            'Statistical summary by resistance level (low/medium/high)',
            'Statistical summary by patient profile (average/young/elderly/compromised)',
            'Statistical summary by treatment protocol (standard/continuous/adaptive/immuno_combo/hyperthermia)',
            'Comprehensive text report with insights and key findings'
        ],
        'key_variables': [
            'scenario, resistance_level, patient_profile, treatment_protocol, final_resistance_pct, treatment_efficacy_score, tumor_reduction_pct',
            'scenario, resistance_level, patient_profile, treatment_protocol, time_days, total_tumor, sensitive_cells, resistant_cells, immune_cells, drug_concentration, resistance_fraction',
            'resistance_level, patient_profile, treatment_protocol, final_resistance',
            'resistance_level, patient_profile, treatment_protocol, efficacy_score, tumor_reduction',
            'count, mean_resistance, std_resistance, mean_efficacy, std_efficacy (by resistance level)',
            'count, mean_resistance, std_resistance, mean_efficacy, std_efficacy (by patient profile)',
            'count, mean_resistance, std_resistance, mean_efficacy, std_efficacy (by treatment protocol)',
            'Text report with statistics, insights, and data file descriptions'
        ]
    }
    
    return pd.DataFrame(data_dict)

if __name__ == "__main__":
    create_comprehensive_reviewer_plots()
    
    # Create data dictionary
    data_dir = Path('results/comprehensive_plots/data')
    if data_dir.exists():
        data_dict_df = create_data_dictionary()
        data_dict_df.to_csv(data_dir / 'data_dictionary.csv', index=False)
        print("  ✅ Data dictionary created")

# Additional utility functions for analysis

def get_best_combinations(all_scenarios, metric='efficacy', top_n=10):
    """Get top N combinations by specified metric"""
    
    results = []
    for scenario_name, result in all_scenarios.items():
        parts = scenario_name.split('_')
        resistance_level = parts[0]
        patient_profile = parts[1]
        protocol = '_'.join(parts[2:])
        
        if metric == 'efficacy':
            score = result['metrics']['treatment_efficacy_score']
        elif metric == 'resistance':
            score = -result['metrics']['final_resistance_fraction']  # Negative for ascending order
        elif metric == 'reduction':
            score = result['metrics']['percent_reduction']
        else:
            score = result['metrics']['treatment_efficacy_score']
        
        results.append({
            'scenario': scenario_name,
            'resistance_level': resistance_level,
            'patient_profile': patient_profile,
            'treatment_protocol': protocol,
            'score': score,
            'efficacy': result['metrics']['treatment_efficacy_score'],
            'resistance': result['metrics']['final_resistance_fraction'],
            'tumor_reduction': result['metrics']['percent_reduction']
        })
    
    # Sort by score and return top N
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]

def compare_protocols_for_patient(all_scenarios, patient_profile, resistance_level='medium'):
    """Compare all protocols for a specific patient profile"""
    
    protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']
    comparison = []
    
    for protocol in protocols:
        scenario_name = f"{resistance_level}_{patient_profile}_{protocol}"
        if scenario_name in all_scenarios:
            result = all_scenarios[scenario_name]
            comparison.append({
                'protocol': protocol,
                'efficacy': result['metrics']['treatment_efficacy_score'],
                'resistance': result['metrics']['final_resistance_fraction'],
                'tumor_reduction': result['metrics']['percent_reduction']
            })
    
    return sorted(comparison, key=lambda x: x['efficacy'], reverse=True)

def analyze_patient_differences(all_scenarios, treatment_protocol='standard', resistance_level='medium'):
    """Analyze differences between patient profiles for a specific treatment"""
    
    patients = ['average', 'young', 'elderly', 'compromised']
    comparison = []
    
    for patient in patients:
        scenario_name = f"{resistance_level}_{patient}_{treatment_protocol}"
        if scenario_name in all_scenarios:
            result = all_scenarios[scenario_name]
            comparison.append({
                'patient': patient,
                'efficacy': result['metrics']['treatment_efficacy_score'],
                'resistance': result['metrics']['final_resistance_fraction'],
                'tumor_reduction': result['metrics']['percent_reduction']
            })
    
    return sorted(comparison, key=lambda x: x['efficacy'], reverse=True)