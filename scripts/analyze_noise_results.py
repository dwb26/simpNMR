"""
Analyze noise sensitivity experiment results.

Usage:
    python analyze_noise_results.py <results_file.pkl>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path

from analyze_results import (
    load_results,
    plot_noise_vs_chi_error,
    export_noise_summary
)


def export_noise_table(results: dict, output_dir: Path):
    """
    Export structured noise sensitivity results to CSV and Excel.
    
    Creates a table with columns:
    - Noise level %
    - best_chi_init (as formatted string)
    - true_chi (as formatted string)
    - best_chi (as formatted string)
    - best_chi_error_log10
    - best_shift_error_log10
    
    Parameters
    ----------
    results : dict
        Noise sensitivity experiment results
    output_dir : Path
        Directory to save output files
    """
    chi_true = results['chi_true']
    
    rows = []
    for level_result in results['level_results']:
        # Format chi matrices as strings
        chi_init_str = np.array2string(np.diag(level_result['best_chi_init']), 
                                       precision=5, separator=', ',
                                       suppress_small=True)
        chi_true_str = np.array2string(np.diag(chi_true), 
                                      precision=5, separator=', ',
                                      suppress_small=True)
        best_chi_str = np.array2string(np.diag(level_result['best_chi']), 
                                       precision=5, separator=', ',
                                       suppress_small=True)
        
        row = {
            'Noise level (%)': level_result['noise_level'] * 100,
            'Best chi init': chi_init_str,
            'True chi': chi_true_str,
            'Best chi est': best_chi_str,
            'Best chi est error (log10)': np.round(np.log10(level_result['best_chi_error']), 5),
            'Best shift error (log10)': np.round(np.log10(level_result['best_loss']), 5),
            'Best n_iterations': level_result['best_n_iterations'],
            'N successful': level_result['n_success']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Export to CSV
    csv_file = output_dir / 'noise_results_table.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nStructured results exported to: {csv_file}")
    
    # Export to Excel
    xlsx_file = output_dir / 'noise_results_table.xlsx'
    df.to_excel(xlsx_file, index=False, engine='openpyxl')
    print(f"Excel file exported to: {xlsx_file}")
    
    # Also save pickle for programmatic access
    pkl_file = output_dir / 'noise_results_dataframe.pkl'
    df.to_pickle(pkl_file)
    print(f"Pickle file exported to: {pkl_file}")
    
    # Print preview
    print("\nTable preview:")
    print(df[['Noise level (%)', 'Best chi est error (log10)', 'Best shift error (log10)']].to_string(index=False))
    
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_noise_results.py <results_file.pkl>")
        return
    
    results_file = sys.argv[1]
    print(f"Loading noise sensitivity results from {results_file}...")
    results = load_results(results_file)
    
    # Check if this is a noise experiment
    if 'level_results' not in results:
        print("Error: This does not appear to be a noise sensitivity experiment result.")
        print("Use analyze_results.py for standard optimization results.")
        return
    
    # Create output directory
    output_dir = Path("experiments/noise_figures")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nExperiment: {results['config']['experiment']['name']}")
    print(f"Noise levels tested: {len(results['noise_levels'])}")
    print(f"Trials per level: {results['trials_per_level']}")
    
    # Export structured table (CSV, Excel, pickle)
    print("\nExporting structured results table...")
    df = export_noise_table(results, output_dir=output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_noise_vs_chi_error(results, save_path=output_dir / "chi_error_vs_noise.png")
    
    # Export summary
    print("\nExporting summary...")
    export_noise_summary(results, output_file=output_dir / "noise_summary.csv")
    
    # Find breaking point
    print("\n" + "="*60)
    print("ROBUSTNESS ANALYSIS")
    print("="*60)
    
    first_best_error = results['level_results'][0]['best_chi_error']
    print(f"\nBaseline best χ error (at {results['level_results'][0]['noise_level']*100:.1f}% noise): {first_best_error:.2e}")
    
    for level_result in results['level_results']:
        noise_pct = level_result['noise_level'] * 100
        best_error = level_result['best_chi_error']
        best_loss = level_result['best_loss']
        
        if best_error > first_best_error * 10:  # 10x degradation
            print(f"\n⚠️  Best χ error degrades >10x at {noise_pct:.1f}% noise")
            print(f"    Best χ error: {best_error:.2e}")
            print(f"    Best loss: {best_loss:.2e}")
            break
    else:
        max_noise = results['level_results'][-1]['noise_level'] * 100
        last_error = results['level_results'][-1]['best_chi_error']
        print(f"\n✓  Best χ error remains within 10x of baseline up to {max_noise:.1f}% noise")
        print(f"    Final best χ error: {last_error:.2e} ({last_error/first_best_error:.1f}x baseline)")
    
    print(f"\nAll figures saved to: {output_dir}/")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
