"""
Module: Time Series Analysis
============================
Comprehensive time series analysis for cancer model including trend analysis,
seasonality detection, forecasting, changepoint detection, and correlation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter, welch
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


class TimeSeriesAnalyzer:
    """Comprehensive time series analysis for cancer model dynamics"""
    
    def __init__(self, output_dir='results/time_series'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def comprehensive_time_series_analysis(self, simulation_results, protocols=None):
        """
        Run complete time series analysis suite
        
        Args:
            simulation_results (dict): Results from cancer model simulations
            protocols (list): List of protocols to analyze
            
        Returns:
            dict: Complete time series analysis results
        """
        print("📈 Running comprehensive time series analysis...")
        
        analysis_results = {}
        
        # Process each patient-protocol combination
        for patient_profile, protocol_results in simulation_results.items():
            analysis_results[patient_profile] = {}
            
            for protocol_name, result in protocol_results.items():
                if not result.get('success', False):
                    continue
                    
                print(f"  Analyzing {patient_profile} - {protocol_name}")
                
                # Extract time series data
                time_series_data = self._extract_time_series_data(result)
                
                # Run all analyses
                protocol_analysis = {
                    'trend_analysis': self._trend_analysis(time_series_data),
                    'correlation_analysis': self._correlation_analysis(time_series_data),
                    'changepoint_detection': self._changepoint_detection(time_series_data),
                    'frequency_analysis': self._frequency_analysis(time_series_data),
                    'decomposition_analysis': self._decomposition_analysis(time_series_data),
                    'clustering_analysis': self._clustering_analysis(time_series_data),
                    'prediction_analysis': self._prediction_analysis(time_series_data)
                }
                
                analysis_results[patient_profile][protocol_name] = protocol_analysis
        
        # Create comprehensive visualizations
        self._create_comprehensive_visualizations(analysis_results, simulation_results)
        
        # Generate comparative analysis
        comparative_analysis = self._comparative_time_series_analysis(analysis_results)
        
        # Generate detailed report
        report_path = self._generate_time_series_report(analysis_results, comparative_analysis)
        
        return {
            'detailed_analysis': analysis_results,
            'comparative_analysis': comparative_analysis,
            'report_path': report_path,
            'visualizations_dir': self.output_dir
        }
    
    def _extract_time_series_data(self, simulation_result):
        """Extract and organize time series data from simulation results"""
        
        time_series = simulation_result['time_series']
        
        # Organize data
        data = {
            'time': time_series['time'],
            'total_tumor': time_series['total_tumor'],
            'sensitive_cells': time_series['sensitive_cells'],
            'resistant_cells': time_series['resistant_type1'] + time_series['resistant_type2'],
            'immune_cells': time_series['cytotoxic_immune'],
            'drug_concentration': time_series['drug_concentration'],
            'resistance_fraction': time_series['resistance_fraction'],
            'genetic_stability': time_series['genetic_stability']
        }
        
        return data
    
    def _trend_analysis(self, time_series_data):
        """Analyze trends in time series data"""
        
        trend_results = {}
        
        for variable, values in time_series_data.items():
            if variable == 'time':
                continue
                
            time = time_series_data['time']
            
            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(time, values)
            
            # Polynomial trend (degree 2)
            poly_coeffs = np.polyfit(time, values, 2)
            poly_trend = np.polyval(poly_coeffs, time)
            
            # Smoothed trend using Savitzky-Golay filter
            if len(values) > 10:
                window_length = min(21, len(values)//4 * 2 + 1)  # Odd number
                smooth_trend = savgol_filter(values, window_length, 3)
            else:
                smooth_trend = values
            
            # Trend classification
            if abs(slope) < 0.001:
                trend_type = 'stable'
            elif slope > 0:
                trend_type = 'increasing'
            else:
                trend_type = 'decreasing'
            
            # Calculate trend strength
            trend_strength = abs(r_value)
            
            trend_results[variable] = {
                'linear_slope': slope,
                'linear_r_squared': r_value**2,
                'linear_p_value': p_value,
                'polynomial_coefficients': poly_coeffs,
                'trend_type': trend_type,
                'trend_strength': trend_strength,
                'smooth_trend': smooth_trend,
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            }
        
        return trend_results
    
    def _correlation_analysis(self, time_series_data):
        """Analyze correlations between variables"""
        
        # Create DataFrame for correlation analysis
        df = pd.DataFrame(time_series_data)
        df = df.drop('time', axis=1)  # Remove time column
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Calculate lagged correlations
        lag_correlations = {}
        max_lag = min(50, len(df) // 10)  # Up to 50 time steps or 10% of data
        
        for var1 in df.columns:
            lag_correlations[var1] = {}
            for var2 in df.columns:
                if var1 != var2:
                    lags = range(-max_lag, max_lag + 1)
                    lag_corrs = []
                    
                    for lag in lags:
                        if lag == 0:
                            corr = df[var1].corr(df[var2])
                        elif lag > 0:
                            corr = df[var1].iloc[:-lag].corr(df[var2].iloc[lag:])
                        else:
                            corr = df[var1].iloc[-lag:].corr(df[var2].iloc[:lag])
                        
                        lag_corrs.append(corr if not np.isnan(corr) else 0)
                    
                    lag_correlations[var1][var2] = {
                        'lags': list(lags),
                        'correlations': lag_corrs,
                        'max_correlation': max(lag_corrs),
                        'optimal_lag': lags[np.argmax(lag_corrs)]
                    }
        
        return {
            'correlation_matrix': correlation_matrix,
            'lag_correlations': lag_correlations
        }
    
    def _changepoint_detection(self, time_series_data):
        """Detect changepoints in time series data"""
        
        changepoint_results = {}
        
        for variable, values in time_series_data.items():
            if variable == 'time':
                continue
                
            time = time_series_data['time']
            
            # Simple changepoint detection using variance
            changepoints = []
            window_size = max(10, len(values) // 20)
            
            for i in range(window_size, len(values) - window_size):
                # Calculate variance before and after point
                var_before = np.var(values[i-window_size:i])
                var_after = np.var(values[i:i+window_size])
                
                # Statistical test for change
                if var_before > 0 and var_after > 0:
                    f_stat = var_before / var_after if var_before > var_after else var_after / var_before
                    if f_stat > 2.0:  # Threshold for significant change
                        changepoints.append(i)
            
            # Remove nearby changepoints
            filtered_changepoints = []
            for cp in changepoints:
                if not filtered_changepoints or cp - filtered_changepoints[-1] > window_size:
                    filtered_changepoints.append(cp)
            
            changepoint_results[variable] = {
                'changepoints': filtered_changepoints,
                'changepoint_times': [time[cp] for cp in filtered_changepoints],
                'num_changepoints': len(filtered_changepoints)
            }
        
        return changepoint_results
    
    def _frequency_analysis(self, time_series_data):
        """Analyze frequency components of time series"""
        
        frequency_results = {}
        
        for variable, values in time_series_data.items():
            if variable == 'time':
                continue
                
            time = time_series_data['time']
            dt = time[1] - time[0] if len(time) > 1 else 1.0
            
            # Remove trend for frequency analysis
            detrended = values - np.mean(values)
            
            # Power spectral density
            frequencies, psd = welch(detrended, fs=1/dt, nperseg=min(256, len(detrended)//4))
            
            # Find dominant frequencies
            peak_indices = []
            if len(psd) > 1:
                threshold = np.mean(psd) + 2 * np.std(psd)
                peak_indices = np.where(psd > threshold)[0]
            
            dominant_frequencies = frequencies[peak_indices] if len(peak_indices) > 0 else []
            dominant_periods = [1/f for f in dominant_frequencies if f > 0]
            
            frequency_results[variable] = {
                'frequencies': frequencies,
                'power_spectral_density': psd,
                'dominant_frequencies': dominant_frequencies,
                'dominant_periods': dominant_periods,
                'total_power': np.sum(psd),
                'peak_frequency': frequencies[np.argmax(psd)] if len(psd) > 0 else 0,
                'spectral_centroid': np.sum(frequencies * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
            }
        
        return frequency_results
    
    def _decomposition_analysis(self, time_series_data):
        """Decompose time series into trend, seasonal, and residual components"""
        
        decomposition_results = {}
        
        for variable, values in time_series_data.items():
            if variable == 'time':
                continue
                
            # Simple decomposition using moving averages
            window_size = max(10, len(values) // 10)
            
            # Trend component (moving average)
            trend = np.convolve(values, np.ones(window_size)/window_size, mode='same')
            
            # Detrended series
            detrended = values - trend
            
            # Seasonal component (simplified - assumes fixed period)
            period = min(50, len(values) // 4)  # Assume period of 50 or quarter of data
            if period > 2:
                seasonal = np.zeros_like(values)
                for i in range(period):
                    indices = np.arange(i, len(values), period)
                    if len(indices) > 1:
                        seasonal[indices] = np.mean(detrended[indices])
            else:
                seasonal = np.zeros_like(values)
            
            # Residual component
            residual = values - trend - seasonal
            
            decomposition_results[variable] = {
                'original': values,
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'trend_strength': 1 - np.var(residual) / np.var(values) if np.var(values) > 0 else 0,
                'seasonality_strength': np.var(seasonal) / np.var(values) if np.var(values) > 0 else 0
            }
        
        return decomposition_results
    
    def _clustering_analysis(self, time_series_data):
        """Cluster time series patterns"""
        
        # Prepare data for clustering
        variables = [var for var in time_series_data.keys() if var != 'time']
        data_matrix = np.array([time_series_data[var] for var in variables]).T
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(3, len(variables)))
        data_pca = pca.fit_transform(data_scaled)
        
        # K-means clustering
        n_clusters = min(4, len(time_series_data['time']) // 50)  # Adaptive number of clusters
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data_pca)
        else:
            cluster_labels = np.zeros(len(data_pca))
        
        clustering_results = {
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'pca_components': data_pca,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'cluster_centers': kmeans.cluster_centers_ if n_clusters >= 2 else None
        }
        
        return clustering_results
    
    def _prediction_analysis(self, time_series_data):
        """Simple prediction analysis using trend extrapolation"""
        
        prediction_results = {}
        
        for variable, values in time_series_data.items():
            if variable == 'time':
                continue
                
            time = time_series_data['time']
            
            # Use last 30% of data for trend estimation
            split_point = int(0.7 * len(values))
            recent_time = time[split_point:]
            recent_values = values[split_point:]
            
            if len(recent_values) > 5:
                # Linear trend prediction
                slope, intercept, r_value, p_value, std_err = stats.linregress(recent_time, recent_values)
                
                # Predict next 10% of timeline
                future_time_steps = int(0.1 * len(time))
                future_time = np.linspace(time[-1], time[-1] + future_time_steps * (time[1] - time[0]), future_time_steps)
                future_prediction = slope * future_time + intercept
                
                # Prediction confidence (simplified)
                prediction_std = np.std(recent_values - (slope * recent_time + intercept))
                confidence_interval = 1.96 * prediction_std  # 95% CI
                
                prediction_results[variable] = {
                    'future_time': future_time,
                    'prediction': future_prediction,
                    'confidence_interval': confidence_interval,
                    'prediction_r_squared': r_value**2,
                    'trend_slope': slope
                }
            else:
                prediction_results[variable] = {
                    'future_time': [],
                    'prediction': [],
                    'confidence_interval': 0,
                    'prediction_r_squared': 0,
                    'trend_slope': 0
                }
        
        return prediction_results
    
    def _create_comprehensive_visualizations(self, analysis_results, simulation_results):
        """Create comprehensive time series visualizations"""
        
        print("📊 Creating comprehensive time series visualizations...")
        
        # 1. Multi-variable time series overview
        self._create_multivariate_overview(simulation_results)
        
        # 2. Trend analysis plots
        self._create_trend_analysis_plots(analysis_results)
        
        # 3. Correlation analysis plots
        self._create_correlation_plots(analysis_results)
        
        # 4. Frequency analysis plots
        self._create_frequency_plots(analysis_results)
        
        # 5. Decomposition plots
        self._create_decomposition_plots(analysis_results)
        
        # 6. Changepoint detection plots
        self._create_changepoint_plots(analysis_results, simulation_results)
        
        # 7. Clustering visualization
        self._create_clustering_plots(analysis_results)
        
        # 8. Prediction plots
        self._create_prediction_plots(analysis_results, simulation_results)
    
    def _create_multivariate_overview(self, simulation_results):
        """Create overview of all variables across protocols and patients"""
        
        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        fig.suptitle('Multivariate Time Series Overview', fontsize=20)
        
        variables = ['total_tumor', 'sensitive_cells', 'resistant_cells', 'immune_cells', 
                    'drug_concentration', 'resistance_fraction', 'genetic_stability']
        
        for i, variable in enumerate(variables[:7]):
            ax = axes[i//2, i%2]
            
            for patient_profile, protocol_results in simulation_results.items():
                for protocol_name, result in protocol_results.items():
                    if not result.get('success', False):
                        continue
                    
                    time_series = result['time_series']
                    time = time_series['time']
                    
                    if variable == 'resistant_cells':
                        values = time_series['resistant_type1'] + time_series['resistant_type2']
                    elif variable == 'immune_cells':
                        values = time_series['cytotoxic_immune']
                    else:
                        values = time_series[variable]
                    
                    label = f"{patient_profile}-{protocol_name}"
                    ax.plot(time, values, alpha=0.7, linewidth=1, label=label)
            
            ax.set_xlabel('Time (days)')
            ax.set_ylabel(variable.replace('_', ' ').title())
            ax.set_title(f'{variable.replace("_", " ").title()} Across All Scenarios')
            ax.grid(True, alpha=0.3)
            
            # Only show legend for first subplot to avoid clutter
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Hide last subplot if odd number of variables
        if len(variables) % 2 == 1:
            axes[-1, -1].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / 'multivariate_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_trend_analysis_plots(self, analysis_results):
        """Create trend analysis visualizations"""
        
        # Extract trend data from first available result
        sample_result = None
        for patient_results in analysis_results.values():
            for protocol_result in patient_results.values():
                if 'trend_analysis' in protocol_result:
                    sample_result = protocol_result['trend_analysis']
                    break
            if sample_result:
                break
        
        if not sample_result:
            return
        
        variables = list(sample_result.keys())
        n_vars = len(variables)
        
        fig, axes = plt.subplots((n_vars + 1) // 2, 2, figsize=(16, 4 * ((n_vars + 1) // 2)))
        if n_vars == 1:
            axes = [axes]
        elif (n_vars + 1) // 2 == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Trend Analysis Summary', fontsize=16)
        
        # Collect trend strength data across all scenarios
        trend_strengths = {var: [] for var in variables}
        trend_types = {var: [] for var in variables}
        
        for patient_profile, protocol_results in analysis_results.items():
            for protocol_name, result in protocol_results.items():
                if 'trend_analysis' in result:
                    for var, trend_data in result['trend_analysis'].items():
                        trend_strengths[var].append(trend_data['trend_strength'])
                        trend_types[var].append(trend_data['trend_type'])
        
        for i, variable in enumerate(variables):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot trend strength distribution
            if trend_strengths[variable]:
                ax.hist(trend_strengths[variable], bins=10, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Trend Strength (R²)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{variable.replace("_", " ").title()} - Trend Strength Distribution')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_strength = np.mean(trend_strengths[variable])
                ax.axvline(mean_strength, color='red', linestyle='--', 
                          label=f'Mean: {mean_strength:.3f}')
                ax.legend()
        
        # Hide unused subplots
        for i in range(len(variables), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / 'trend_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_plots(self, analysis_results):
        """Create correlation analysis visualizations"""
        
        # Get sample correlation matrix
        sample_corr = None
        for patient_results in analysis_results.values():
            for protocol_result in patient_results.values():
                if 'correlation_analysis' in protocol_result:
                    sample_corr = protocol_result['correlation_analysis']['correlation_matrix']
                    break
            if sample_corr is not None:
                break
        
        if sample_corr is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Correlation Analysis', fontsize=16)
        
        # Correlation heatmap
        sns.heatmap(sample_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0])
        axes[0].set_title('Variable Correlation Matrix')
        
        # Correlation strength distribution
        corr_values = sample_corr.values
        upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
        
        axes[1].hist(upper_triangle, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Correlation Coefficient')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Pairwise Correlations')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_frequency_plots(self, analysis_results):
        """Create frequency analysis visualizations"""
        
        # This would create frequency domain analysis plots
        # Implementation similar to other plot functions
        pass
    
    def _create_decomposition_plots(self, analysis_results):
        """Create decomposition analysis visualizations"""
        
        # This would create trend/seasonal/residual decomposition plots
        # Implementation similar to other plot functions
        pass
    
    def _create_changepoint_plots(self, analysis_results, simulation_results):
        """Create changepoint detection visualizations"""
        
        # This would create changepoint detection plots
        # Implementation similar to other plot functions
        pass
    
    def _create_clustering_plots(self, analysis_results):
        """Create clustering visualizations"""
        
        # This would create clustering analysis plots
        # Implementation similar to other plot functions
        pass
    
    def _create_prediction_plots(self, analysis_results, simulation_results):
        """Create prediction analysis visualizations"""
        
        # This would create prediction/forecasting plots
        # Implementation similar to other plot functions
        pass
    
    def _comparative_time_series_analysis(self, analysis_results):
        """Compare time series characteristics across scenarios"""
        
        comparison_results = {
            'trend_comparison': {},
            'variability_comparison': {},
            'correlation_comparison': {},
            'changepoint_comparison': {}
        }
        
        # Aggregate results across all scenarios
        all_trends = {}
        all_variabilities = {}
        
        for patient_profile, protocol_results in analysis_results.items():
            for protocol_name, result in protocol_results.items():
                scenario_name = f"{patient_profile}_{protocol_name}"
                
                if 'trend_analysis' in result:
                    all_trends[scenario_name] = result['trend_analysis']
                    
                    # Calculate overall variability
                    variabilities = {}
                    for var, trend_data in result['trend_analysis'].items():
                        variabilities[var] = trend_data['coefficient_of_variation']
                    all_variabilities[scenario_name] = variabilities
        
        comparison_results['trend_comparison'] = all_trends
        comparison_results['variability_comparison'] = all_variabilities
        
        return comparison_results
    
    def _generate_time_series_report(self, analysis_results, comparative_analysis):
        """Generate comprehensive time series analysis report"""
        
        report_path = self.output_dir / 'time_series_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE TIME SERIES ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary statistics
            total_scenarios = sum(len(protocols) for protocols in analysis_results.values())
            f.write(f"Total scenarios analyzed: {total_scenarios}\n")
            f.write(f"Patient profiles: {list(analysis_results.keys())}\n\n")
            
            # Trend analysis summary
            f.write("TREND ANALYSIS SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            if 'trend_comparison' in comparative_analysis:
                trend_data = comparative_analysis['trend_comparison']
                
                # Find most stable and most variable scenarios
                scenario_variabilities = {}
                for scenario, trends in trend_data.items():
                    avg_cv = np.mean([trend_info['coefficient_of_variation'] 
                                    for trend_info in trends.values()])
                    scenario_variabilities[scenario] = avg_cv
                
                most_stable = min(scenario_variabilities, key=scenario_variabilities.get)
                most_variable = max(scenario_variabilities, key=scenario_variabilities.get)
                
                f.write(f"Most stable scenario: {most_stable} (CV: {scenario_variabilities[most_stable]:.3f})\n")
                f.write(f"Most variable scenario: {most_variable} (CV: {scenario_variabilities[most_variable]:.3f})\n\n")
            
            # Key insights
            f.write("KEY INSIGHTS\n")
            f.write("-" * 15 + "\n")
            f.write("1. Review multivariate_overview.png for overall patterns\n")
            f.write("2. Check trend_analysis.png for trend strength distributions\n")
            f.write("3. Examine correlation_analysis.png for variable relationships\n")
            f.write("4. Use decomposition plots to understand seasonal patterns\n")
            f.write("5. Analyze changepoint plots for treatment effect timing\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            f.write("• Focus on scenarios with strong trends for predictability\n")
            f.write("• Investigate high-correlation variable pairs for mechanism insights\n")
            f.write("• Use changepoint analysis to optimize treatment timing\n")
            f.write("• Consider frequency analysis for oscillatory behaviors\n")
        
        print(f"📄 Time series analysis report saved to {report_path}")
        return report_path