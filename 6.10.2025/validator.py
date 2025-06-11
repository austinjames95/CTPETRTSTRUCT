import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import interpolate
from scipy.stats import pearsonr
import seaborn as sns

class DVHValidator:
    def __init__(self, velocity_dvh_path, computed_dvh_path, output_dir="dvh_validation"):
        """
        Initialize DVH validator
        
        Args:
            velocity_dvh_path: Path to Velocity-exported DVH CSV file
            computed_dvh_path: Path to your computed DVH CSV file
            output_dir: Directory to save validation results
        """
        self.velocity_dvh_path = velocity_dvh_path
        self.computed_dvh_path = computed_dvh_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load DVH data
        self.velocity_data = None
        self.computed_data = None
        self.validation_results = {}
        
    def load_velocity_dvh(self):
        """Load Velocity DVH data - adapt this based on Velocity's export format"""
        try:
            # Common Velocity DVH export formats - adjust as needed
            if self.velocity_dvh_path.endswith('.csv'):
                # Try different separators and skip rows
                for sep in [',', '\t', ';']:
                    for skip in [0, 1, 2, 3, 4, 5]:
                        try:
                            df = pd.read_csv(self.velocity_dvh_path, sep=sep, skiprows=skip)
                            if df.shape[1] > 1 and df.shape[0] > 10:  # Reasonable data
                                self.velocity_data = df
                                print(f"Successfully loaded Velocity data with sep='{sep}', skiprows={skip}")
                                print(f"Velocity DVH shape: {df.shape}")
                                print(f"Velocity columns: {list(df.columns)}")
                                return True
                        except:
                            continue
                            
            # If CSV loading failed, try as Excel
            elif self.velocity_dvh_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(self.velocity_dvh_path)
                self.velocity_data = df
                print(f"Successfully loaded Velocity Excel data")
                print(f"Velocity DVH shape: {df.shape}")
                
            if self.velocity_data is None:
                print("Failed to load Velocity DVH data. Please check file format.")
                return False
                
        except Exception as e:
            print(f"Error loading Velocity DVH: {e}")
            return False
            
        return True
    
    def load_computed_dvh(self):
        """Load your computed DVH data"""
        try:
            # Skip comment lines starting with #
            self.computed_data = pd.read_csv(self.computed_dvh_path, comment='#')
            print(f"Successfully loaded computed DVH data")
            print(f"Computed DVH shape: {self.computed_data.shape}")
            print(f"Computed columns: {list(self.computed_data.columns)}")
            return True
        except Exception as e:
            print(f"Error loading computed DVH: {e}")
            return False
    
    def standardize_structure_names(self):
        """Standardize structure names between datasets for matching"""
        velocity_cols = list(self.velocity_data.columns)
        computed_cols = list(self.computed_data.columns)
        
        # Remove dose column
        velocity_structures = [col for col in velocity_cols if not any(x in col.lower() for x in ['dose', 'gy', 'cgy'])]
        computed_structures = [col for col in computed_cols if col != 'Dose_Gy']
        
        print("\nVelocity structures:")
        for i, struct in enumerate(velocity_structures):
            print(f"  {i}: {struct}")
            
        print("\nComputed structures:")
        for i, struct in enumerate(computed_structures):
            print(f"  {i}: {struct}")
        
        # Create matching pairs - simple name matching
        matched_pairs = []
        for comp_struct in computed_structures:
            # Clean computed structure name
            comp_clean = comp_struct.replace('_Volume_cm3', '').replace('_', ' ').lower()
            
            # Find best match in velocity structures
            best_match = None
            best_score = 0
            
            for vel_struct in velocity_structures:
                vel_clean = vel_struct.replace('_', ' ').lower()
                
                # Simple string similarity
                if comp_clean in vel_clean or vel_clean in comp_clean:
                    score = min(len(comp_clean), len(vel_clean)) / max(len(comp_clean), len(vel_clean))
                    if score > best_score:
                        best_score = score
                        best_match = vel_struct
            
            if best_match and best_score > 0.5:  # Threshold for matching
                matched_pairs.append((comp_struct, best_match))
                print(f"Matched: '{comp_struct}' <-> '{best_match}' (score: {best_score:.2f})")
        
        return matched_pairs
    
    def interpolate_dvh(self, dose_values, volume_values, target_dose_points):
        """Interpolate DVH to common dose points"""
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(dose_values) & np.isfinite(volume_values)
        dose_clean = np.array(dose_values)[valid_mask]
        volume_clean = np.array(volume_values)[valid_mask]
        
        if len(dose_clean) < 2:
            return np.full_like(target_dose_points, np.nan)
        
        # Sort by dose
        sort_idx = np.argsort(dose_clean)
        dose_clean = dose_clean[sort_idx]
        volume_clean = volume_clean[sort_idx]
        
        # Interpolate
        try:
            f = interpolate.interp1d(dose_clean, volume_clean, 
                                   kind='linear', 
                                   bounds_error=False, 
                                   fill_value=(volume_clean[0], volume_clean[-1]))
            return f(target_dose_points)
        except:
            return np.full_like(target_dose_points, np.nan)
    
    def calculate_dvh_metrics(self, dose_points, volume_values, structure_name):
        """Calculate standard DVH metrics"""
        metrics = {}
        
        # Remove NaN values
        valid_mask = np.isfinite(volume_values)
        if not np.any(valid_mask):
            return metrics
            
        dose_clean = dose_points[valid_mask]
        volume_clean = volume_values[valid_mask]
        
        # Total volume (volume at 0 Gy)
        if len(volume_clean) > 0:
            metrics['V_total'] = volume_clean[0] if dose_clean[0] <= 0.1 else np.max(volume_clean)
        
        # Dose metrics (dose to X% of volume)
        if metrics.get('V_total', 0) > 0:
            relative_volumes = volume_clean / metrics['V_total']
            
            for pct in [95, 50, 5]:
                target_vol = pct / 100.0
                # Find dose where relative volume drops to target
                idx = np.where(relative_volumes <= target_vol)[0]
                if len(idx) > 0:
                    metrics[f'D{pct}'] = dose_clean[idx[0]]
        
        # Volume metrics (volume receiving X Gy)
        for dose_level in [10, 20, 30, 40, 50]:
            if np.max(dose_clean) >= dose_level:
                idx = np.where(dose_clean >= dose_level)[0]
                if len(idx) > 0:
                    metrics[f'V{dose_level}'] = volume_clean[idx[0]]
        
        return metrics
    
    def validate_structure_pair(self, computed_struct, velocity_struct):
        """Validate DVH for a single structure pair"""
        print(f"\nValidating: {computed_struct} vs {velocity_struct}")
        
        # Get dose columns
        computed_dose = self.computed_data['Dose_Gy'].values
        computed_volume = self.computed_data[computed_struct].values
        
        # Find dose column in velocity data
        velocity_dose_col = None
        for col in self.velocity_data.columns:
            if any(x in col.lower() for x in ['dose', 'gy', 'cgy']):
                velocity_dose_col = col
                break
        
        if velocity_dose_col is None:
            print("Could not find dose column in Velocity data")
            return None
        
        velocity_dose = self.velocity_data[velocity_dose_col].values
        velocity_volume = self.velocity_data[velocity_struct].values
        
        # Convert cGy to Gy if needed
        if 'cgy' in velocity_dose_col.lower():
            velocity_dose = velocity_dose / 100.0
            print("Converted Velocity dose from cGy to Gy")
        
        # Create common dose grid for comparison
        min_dose = max(np.min(computed_dose), np.min(velocity_dose))
        max_dose = min(np.max(computed_dose), np.max(velocity_dose))
        common_dose = np.linspace(min_dose, max_dose, 200)
        
        # Interpolate both DVHs to common grid
        computed_interp = self.interpolate_dvh(computed_dose, computed_volume, common_dose)
        velocity_interp = self.interpolate_dvh(velocity_dose, velocity_volume, common_dose)
        
        # Calculate validation metrics
        valid_mask = np.isfinite(computed_interp) & np.isfinite(velocity_interp)
        if not np.any(valid_mask):
            print("No valid data points for comparison")
            return None
        
        computed_valid = computed_interp[valid_mask]
        velocity_valid = velocity_interp[valid_mask]
        dose_valid = common_dose[valid_mask]
        
        # Statistical comparisons
        correlation, p_value = pearsonr(computed_valid, velocity_valid)
        
        # Mean absolute error
        mae = np.mean(np.abs(computed_valid - velocity_valid))
        
        # Root mean square error
        rmse = np.sqrt(np.mean((computed_valid - velocity_valid)**2))
        
        # Relative error (percentage)
        velocity_nonzero = velocity_valid[velocity_valid > 0.1]  # Avoid division by very small numbers
        computed_nonzero = computed_valid[velocity_valid > 0.1]
        if len(velocity_nonzero) > 0:
            rel_error = np.mean(np.abs((computed_nonzero - velocity_nonzero) / velocity_nonzero)) * 100
        else:
            rel_error = np.nan
        
        # DVH metrics comparison
        computed_metrics = self.calculate_dvh_metrics(computed_dose, computed_volume, computed_struct)
        velocity_metrics = self.calculate_dvh_metrics(velocity_dose, velocity_volume, velocity_struct)
        
        results = {
            'structure_computed': computed_struct,
            'structure_velocity': velocity_struct,
            'correlation': correlation,
            'p_value': p_value,
            'mae': mae,
            'rmse': rmse,
            'relative_error_pct': rel_error,
            'computed_metrics': computed_metrics,
            'velocity_metrics': velocity_metrics,
            'dose_grid': dose_valid,
            'computed_volumes': computed_valid,
            'velocity_volumes': velocity_valid
        }
        
        print(f"  Correlation: {correlation:.4f} (p={p_value:.4f})")
        print(f"  MAE: {mae:.3f} cm³")
        print(f"  RMSE: {rmse:.3f} cm³")
        print(f"  Relative Error: {rel_error:.2f}%")
        
        return results
    
    def create_validation_plots(self, validation_results):
        """Create comprehensive validation plots"""
        n_structures = len(validation_results)
        
        if n_structures == 0:
            print("No validation results to plot")
            return
        
        # 1. Individual DVH comparisons
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (struct_name, results) in enumerate(validation_results.items()):
            if i >= 4:  # Only plot first 4 structures
                break
                
            ax = axes[i]
            
            ax.plot(results['dose_grid'], results['velocity_volumes'], 
                   'b-', linewidth=2, label='Velocity', alpha=0.8)
            ax.plot(results['dose_grid'], results['computed_volumes'], 
                   'r--', linewidth=2, label='Computed', alpha=0.8)
            
            ax.set_xlabel('Dose (Gy)')
            ax.set_ylabel('Volume (cm³)')
            ax.set_title(f"{struct_name}\nCorr: {results['correlation']:.3f}, "
                        f"RMSE: {results['rmse']:.2f} cm³")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(validation_results), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "DVH_Individual_Comparisons.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Individual DVH comparisons saved to: {plot_path}")
        plt.show()
        
        # 2. Correlation scatter plot
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(validation_results)))
        
        for i, (struct_name, results) in enumerate(validation_results.items()):
            plt.scatter(results['velocity_volumes'], results['computed_volumes'],
                       alpha=0.6, s=20, c=[colors[i]], label=struct_name)
        
        # Perfect correlation line
        all_volumes = []
        for results in validation_results.values():
            all_volumes.extend(results['velocity_volumes'])
            all_volumes.extend(results['computed_volumes'])
        
        min_vol, max_vol = np.min(all_volumes), np.max(all_volumes)
        plt.plot([min_vol, max_vol], [min_vol, max_vol], 'k--', alpha=0.5, label='Perfect correlation')
        
        plt.xlabel('Velocity Volume (cm³)')
        plt.ylabel('Computed Volume (cm³)')
        plt.title('DVH Volume Correlation: Velocity vs Computed')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        corr_path = os.path.join(self.output_dir, "DVH_Correlation_Scatter.png")
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"Correlation scatter plot saved to: {corr_path}")
        plt.show()
        
        # 3. Statistical summary heatmap
        metrics_data = []
        struct_names = []
        
        for struct_name, results in validation_results.items():
            struct_names.append(struct_name[:20])  # Truncate long names
            metrics_data.append([
                results['correlation'],
                results['mae'],
                results['rmse'],
                results['relative_error_pct']
            ])
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data, 
                                    index=struct_names,
                                    columns=['Correlation', 'MAE (cm³)', 'RMSE (cm³)', 'Rel Error (%)'])
            
            plt.figure(figsize=(10, max(6, len(struct_names) * 0.5)))
            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0)
            plt.title('DVH Validation Metrics Heatmap')
            plt.tight_layout()
            
            heatmap_path = os.path.join(self.output_dir, "DVH_Validation_Heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            print(f"Validation metrics heatmap saved to: {heatmap_path}")
            plt.show()
    
    def generate_validation_report(self, validation_results):
        """Generate comprehensive validation report"""
        report_path = os.path.join(self.output_dir, "DVH_Validation_Report.txt")
        
        with open(report_path, 'w') as f:
            f.write("DVH VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Velocity DVH file: {self.velocity_dvh_path}\n")
            f.write(f"Computed DVH file: {self.computed_dvh_path}\n")
            f.write(f"Number of structures validated: {len(validation_results)}\n\n")
            
            # Overall statistics
            all_correlations = [r['correlation'] for r in validation_results.values() if not np.isnan(r['correlation'])]
            all_mae = [r['mae'] for r in validation_results.values() if not np.isnan(r['mae'])]
            all_rmse = [r['rmse'] for r in validation_results.values() if not np.isnan(r['rmse'])]
            all_rel_error = [r['relative_error_pct'] for r in validation_results.values() if not np.isnan(r['relative_error_pct'])]
            
            f.write("OVERALL VALIDATION STATISTICS:\n")
            f.write("-" * 35 + "\n")
            if all_correlations:
                f.write(f"Mean Correlation: {np.mean(all_correlations):.4f} ± {np.std(all_correlations):.4f}\n")
                f.write(f"Min Correlation: {np.min(all_correlations):.4f}\n")
                f.write(f"Max Correlation: {np.max(all_correlations):.4f}\n")
            
            if all_mae:
                f.write(f"Mean MAE: {np.mean(all_mae):.3f} ± {np.std(all_mae):.3f} cm³\n")
            
            if all_rmse:
                f.write(f"Mean RMSE: {np.mean(all_rmse):.3f} ± {np.std(all_rmse):.3f} cm³\n")
            
            if all_rel_error:
                f.write(f"Mean Relative Error: {np.mean(all_rel_error):.2f} ± {np.std(all_rel_error):.2f}%\n")
            
            f.write("\nSTRUCTURE-BY-STRUCTURE RESULTS:\n")
            f.write("-" * 40 + "\n\n")
            
            for struct_name, results in validation_results.items():
                f.write(f"Structure: {struct_name}\n")
                f.write(f"  Velocity name: {results['structure_velocity']}\n")
                f.write(f"  Computed name: {results['structure_computed']}\n")
                f.write(f"  Correlation: {results['correlation']:.4f}\n")
                f.write(f"  MAE: {results['mae']:.3f} cm³\n")
                f.write(f"  RMSE: {results['rmse']:.3f} cm³\n")
                f.write(f"  Relative Error: {results['relative_error_pct']:.2f}%\n")
                
                # DVH metrics comparison
                comp_metrics = results['computed_metrics']
                vel_metrics = results['velocity_metrics']
                
                f.write("  DVH Metrics Comparison:\n")
                for metric in ['V_total', 'D95', 'D50', 'D5']:
                    if metric in comp_metrics and metric in vel_metrics:
                        comp_val = comp_metrics[metric]
                        vel_val = vel_metrics[metric]
                        diff = abs(comp_val - vel_val)
                        rel_diff = (diff / vel_val * 100) if vel_val > 0 else 0
                        f.write(f"    {metric}: Computed={comp_val:.2f}, Velocity={vel_val:.2f}, "
                               f"Diff={diff:.2f} ({rel_diff:.1f}%)\n")
                
                f.write("\n")
            
            # Validation criteria
            f.write("VALIDATION CRITERIA:\n")
            f.write("-" * 25 + "\n")
            f.write("Excellent agreement: Correlation > 0.99, RMSE < 1.0 cm³, Rel Error < 5%\n")
            f.write("Good agreement: Correlation > 0.95, RMSE < 2.0 cm³, Rel Error < 10%\n")
            f.write("Acceptable agreement: Correlation > 0.90, RMSE < 5.0 cm³, Rel Error < 20%\n")
            f.write("Poor agreement: Below acceptable thresholds\n\n")
            
            # Assessment for each structure
            f.write("VALIDATION ASSESSMENT:\n")
            f.write("-" * 25 + "\n")
            for struct_name, results in validation_results.items():
                corr = results['correlation']
                rmse = results['rmse']
                rel_err = results['relative_error_pct']
                
                if corr > 0.99 and rmse < 1.0 and rel_err < 5:
                    assessment = "EXCELLENT"
                elif corr > 0.95 and rmse < 2.0 and rel_err < 10:
                    assessment = "GOOD"
                elif corr > 0.90 and rmse < 5.0 and rel_err < 20:
                    assessment = "ACCEPTABLE"
                else:
                    assessment = "POOR"
                
                f.write(f"{struct_name}: {assessment}\n")
        
        print(f"Validation report saved to: {report_path}")
    
    def run_validation(self):
        """Run complete DVH validation"""
        print("Starting DVH validation...")
        
        # Load data
        if not self.load_velocity_dvh():
            return False
        
        if not self.load_computed_dvh():
            return False
        
        # Match structures
        matched_pairs = self.standardize_structure_names()
        
        if not matched_pairs:
            print("No matching structures found between datasets!")
            return False
        
        print(f"\nFound {len(matched_pairs)} matching structure pairs")
        
        # Validate each pair
        validation_results = {}
        
        for computed_struct, velocity_struct in matched_pairs:
            results = self.validate_structure_pair(computed_struct, velocity_struct)
            if results:
                # Use a clean name for the key
                clean_name = computed_struct.replace('_Volume_cm3', '').replace('_', ' ')
                validation_results[clean_name] = results
        
        if not validation_results:
            print("No successful validations performed!")
            return False
        
        # Generate outputs
        self.create_validation_plots(validation_results)
        self.generate_validation_report(validation_results)
        
        print(f"\nValidation complete! Results saved to: {self.output_dir}")
        return True


def validate_dvh_files(velocity_path, computed_path, output_dir="dvh_validation"):
    """
    Main function to validate DVH files
    
    Args:
        velocity_path: Path to Velocity DVH export file
        computed_path: Path to computed DVH file (from your code)
        output_dir: Output directory for validation results
    """
    validator = DVHValidator(velocity_path, computed_path, output_dir)
    return validator.run_validation()

if __name__ == "__main__":
    # Update these paths to your actual files
    velocity_dvh_file = "GivenData/absolute_dvh_export.csv"
    computed_dvh_file = "dvh_data_combined\CumulativeDVH_AllStructures_AbsoluteUnits.csv"
    
    # Run validation
    success = validate_dvh_files(velocity_dvh_file, computed_dvh_file)
    
    if success:
        print("DVH validation completed successfully!")
    else:
        print("DVH validation failed. Please check your input files.")
