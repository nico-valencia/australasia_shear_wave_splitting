"""
diagnostics.py
Functions to diagnose why SI values are small or fits avoid outliers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import copy
from obspy import read
from scipy.stats import pearsonr


def diagnose_si_calculation(pair, baz, sample_interval):
    """
    Diagnose factors affecting SI calculation for a single measurement.
    
    Parameters
    ----------
    pair : splitwavepy.Pair
        Waveform pair (already windowed/chopped)
    baz : float
        Back-azimuth or polarization azimuth
    sample_interval : float
        Sampling interval in seconds
        
    Returns
    -------
    dict with diagnostic metrics
    """
    data = copy.deepcopy(pair)
    data.rotateto(baz)
    
    R = data.x
    T = data.y
    rd = np.diff(R) / sample_interval
    T_trimmed = T[:-1]
    
    # Calculate SI components
    Pop = np.dot(T_trimmed, rd)  # Numerator
    NPop = np.linalg.norm(rd)**2  # Denominator
    SI = -2 * Pop / NPop if NPop > 0 else np.nan
    
    # Diagnostics
    max_R = np.max(np.abs(R))
    max_T = np.max(np.abs(T))
    max_rd = np.max(np.abs(rd))
    
    # Correlation between T and Ṙ
    correlation, p_value = pearsonr(T_trimmed, rd)
    
    # Energy ratio
    energy_R = np.sum(R**2)
    energy_T = np.sum(T**2)
    energy_ratio = energy_T / energy_R if energy_R > 0 else np.nan
    
    # Impulsiveness measure (kurtosis of derivative)
    from scipy.stats import kurtosis
    impulsiveness = kurtosis(rd)
    
    # SNR estimate (rough)
    signal_power = np.mean(R**2)
    noise_estimate = np.std(R[:int(len(R)*0.1)])**2  # First 10% as noise
    snr_estimate = signal_power / noise_estimate if noise_estimate > 0 else np.inf
    
    return {
        'SI': SI,
        'max_R': max_R,
        'max_T': max_T,
        'max_rd': max_rd,
        'norm_rd_squared': NPop,
        'correlation_T_rd': correlation,
        'correlation_p_value': p_value,
        'energy_ratio_T_R': energy_ratio,
        'impulsiveness': impulsiveness,
        'snr_estimate': snr_estimate,
        'R': R,
        'T': T,
        'rd': rd
    }


def diagnose_station_si_characteristics(event_data, station_code, n_samples=20):
    """
    Diagnose SI calculation issues across multiple measurements for a station.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog
    station_code : str
        Station to analyze
    n_samples : int
        Number of waveforms to analyze
        
    Returns
    -------
    pd.DataFrame with diagnostics for each measurement
    """
    import splitwavepy as sw
    
    station_events = event_data[event_data['evstation'] == station_code]
    sample_events = station_events.sample(min(n_samples, len(station_events)))
    
    results = []
    
    for idx, event in sample_events.iterrows():
        try:
            # Load and process waveform
            wf = read(event['waveform_path'])
            wf.detrend('demean')
            wf.detrend('linear')
            wf.filter('bandpass', freqmin=0.01, freqmax=0.125)
            
            north = wf.select(component="N", channel='BH*')[0].data
            east = wf.select(component="E", channel='BH*')[0].data
            sample_interval = wf[0].stats.delta
            
            if len(north) % 2 == 0: north = north[:-1]
            if len(east) % 2 == 0: east = east[:-1]
            
            # Create windowed pair
            start_idx = int(event['windowstart'] / sample_interval)
            end_idx = int(event['windowend'] / sample_interval)
            
            north_chopped = north[start_idx:end_idx]
            east_chopped = east[start_idx:end_idx]
            
            if len(north_chopped) % 2 == 0:
                north_chopped = north_chopped[:-1]
                east_chopped = east_chopped[:-1]
            
            pair = sw.Pair(north_chopped, east_chopped, delta=sample_interval)
            
            # Diagnose
            diag = diagnose_si_calculation(pair, event['calcbaz'], sample_interval)
            
            # Add event info
            diag['evstation'] = event['evstation']
            diag['evphase'] = event['evphase']
            diag['calcbaz'] = event['calcbaz']
            diag['SI_measured'] = event['wiggleSI_calc']
            diag['SI_error'] = event['SI_calc_error']
            
            results.append(diag)
            
        except Exception as e:
            print(f"Failed on event {idx}: {e}")
            continue
    
    # Remove waveform data for DataFrame
    for r in results:
        r.pop('R', None)
        r.pop('T', None)
        r.pop('rd', None)
    
    return pd.DataFrame(results)


def diagnose_azimuthal_coverage(result):
    """
    Check if measurements sample high-SI regions (45° from fast axis).
    
    Parameters
    ----------
    result : dict
        Output from analyze_station()
        
    Returns
    -------
    dict with coverage metrics and figure
    """
    phi = result['phi']
    event_data = result['event_data']
    
    # Calculate angle from fast axis
    angles_from_fast = (event_data['calcbaz'] - phi) % 180
    
    # Classify into regions
    near_null = (angles_from_fast < 20) | (angles_from_fast > 160) | \
                ((angles_from_fast > 70) & (angles_from_fast < 110))
    near_peak = ((angles_from_fast > 25) & (angles_from_fast < 65)) | \
                ((angles_from_fast > 115) & (angles_from_fast < 155))
    
    # Expected SI at each measurement
    expected_si = result['dt'] * np.sin(2 * np.deg2rad(angles_from_fast))
    
    # Metrics
    coverage = {
        'n_total': len(event_data),
        'n_near_null': near_null.sum(),
        'n_near_peak': near_peak.sum(),
        'fraction_near_null': near_null.sum() / len(event_data),
        'fraction_near_peak': near_peak.sum() / len(event_data),
        'mean_expected_si': np.mean(np.abs(expected_si)),
        'median_observed_si': np.median(np.abs(event_data['wiggleSI_calc']))
    }
    
    # Create diagnostic plot
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    # Panel 1: Histogram of angles
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(angles_from_fast, bins=18, range=(0, 180), alpha=0.7, edgecolor='black')
    ax1.axvline(45, color='red', linestyle='--', linewidth=2, label='Max SI (45°)')
    ax1.axvline(135, color='red', linestyle='--', linewidth=2, label='Max SI (135°)')
    ax1.axvspan(0, 20, alpha=0.2, color='gray', label='Null regions')
    ax1.axvspan(70, 110, alpha=0.2, color='gray')
    ax1.axvspan(160, 180, alpha=0.2, color='gray')
    ax1.set_xlabel('Angle from Fast Axis (°)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Azimuthal Coverage\n{coverage["n_near_peak"]} near peaks, {coverage["n_near_null"]} near nulls')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Expected vs Observed SI
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(expected_si, event_data['wiggleSI_calc'], alpha=0.6, s=30)
    ax2.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='1:1 line')
    ax2.set_xlabel('Expected SI (from geometry)')
    ax2.set_ylabel('Observed SI')
    ax2.set_title('Expected vs Observed SI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Panel 3: SI vs Angle
    ax3 = fig.add_subplot(gs[2])
    
    # Plot theoretical curve
    angle_range = np.linspace(0, 180, 181)
    si_theory = result['dt'] * np.sin(2 * np.deg2rad(angle_range))
    ax3.plot(angle_range, si_theory, 'r-', linewidth=2, label=f'Theory (δt={result["dt"]:.2f}s)')
    
    # Plot observations
    ax3.scatter(angles_from_fast, event_data['wiggleSI_calc'], 
               alpha=0.6, s=30, label='Observations')
    ax3.errorbar(angles_from_fast, event_data['wiggleSI_calc'],
                yerr=event_data['SI_calc_error'], fmt='none', alpha=0.3)
    
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Angle from Fast Axis (°)')
    ax3.set_ylabel('Splitting Intensity')
    ax3.set_title('SI vs Geometry')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return coverage, fig


def plot_si_diagnostic_report(diag_df, station_code):
    """
    Create comprehensive diagnostic report from diagnose_station_si_characteristics().
    
    Parameters
    ----------
    diag_df : pd.DataFrame
        Output from diagnose_station_si_characteristics()
    station_code : str
        Station name
        
    Returns
    -------
    fig
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # Panel 1: SI vs ||Ṙ||²
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(diag_df['norm_rd_squared'], np.abs(diag_df['SI']), 
               c=diag_df['snr_estimate'], alpha=0.7, s=50, cmap='viridis')
    ax1.set_xlabel('||Ṙ||² (Denominator)')
    ax1.set_ylabel('|SI|')
    ax1.set_xscale('log')
    ax1.set_title('SI vs Radial Derivative Norm')
    cbar = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar.set_label('SNR Estimate')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Correlation T-Ṙ
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(diag_df['correlation_T_rd'], np.abs(diag_df['SI']), alpha=0.7, s=50)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Correlation(T, dR/dt)')
    ax2.set_ylabel('|SI|')
    ax2.set_title('SI vs T-Ṙ Correlation')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Energy ratio
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(diag_df['energy_ratio_T_R'], np.abs(diag_df['SI']), alpha=0.7, s=50)
    ax3.set_xlabel('Energy(T) / Energy(R)')
    ax3.set_ylabel('|SI|')
    ax3.set_xscale('log')
    ax3.set_title('SI vs Transverse/Radial Energy Ratio')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Impulsiveness
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(diag_df['impulsiveness'], np.abs(diag_df['SI']), alpha=0.7, s=50)
    ax4.set_xlabel('Impulsiveness (kurtosis of dR/dt)')
    ax4.set_ylabel('|SI|')
    ax4.axvline(3, color='red', linestyle='--', label='Gaussian reference', alpha=0.5)
    ax4.set_title('SI vs Waveform Impulsiveness')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: SNR
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(diag_df['snr_estimate'], np.abs(diag_df['SI']), alpha=0.7, s=50)
    ax5.set_xlabel('SNR Estimate')
    ax5.set_ylabel('|SI|')
    ax5.set_xscale('log')
    ax5.set_title('SI vs SNR')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: SI measured vs calculated
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(diag_df['SI_measured'], diag_df['SI'], alpha=0.7, s=50)
    ax6.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='1:1')
    ax6.set_xlabel('SI (from your SplittingIntensity function)')
    ax6.set_ylabel('SI (recalculated in diagnostic)')
    ax6.set_title('SI Consistency Check')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    
    # Panel 7-9: Histograms
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(diag_df['norm_rd_squared'], bins=20, alpha=0.7, edgecolor='black')
    ax7.set_xlabel('||Ṙ||²')
    ax7.set_ylabel('Count')
    ax7.set_title('Distribution of ||Ṙ||²')
    ax7.axvline(diag_df['norm_rd_squared'].median(), color='red', 
               linestyle='--', label=f'Median: {diag_df["norm_rd_squared"].median():.1f}')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(diag_df['correlation_T_rd'], bins=20, alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Correlation(T, dR/dt)')
    ax8.set_ylabel('Count')
    ax8.set_title('Distribution of T-Ṙ Correlation')
    ax8.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax8.axvline(diag_df['correlation_T_rd'].median(), color='orange', 
               linestyle='--', label=f'Median: {diag_df["correlation_T_rd"].median():.2f}')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.hist(diag_df['impulsiveness'], bins=20, alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Impulsiveness')
    ax9.set_ylabel('Count')
    ax9.set_title('Distribution of Impulsiveness')
    ax9.axvline(3, color='red', linestyle='--', label='Gaussian', alpha=0.5)
    ax9.axvline(diag_df['impulsiveness'].median(), color='orange', 
               linestyle='--', label=f'Median: {diag_df["impulsiveness"].median():.1f}')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    fig.suptitle(f'{station_code} - SI Calculation Diagnostics', fontsize=16, y=0.998)
    plt.tight_layout()
    
    return fig


def summarize_diagnostics(diag_df):
    """
    Print summary of diagnostic findings.
    
    Parameters
    ----------
    diag_df : pd.DataFrame
        Output from diagnose_station_si_characteristics()
    """
    print("="*70)
    print("SI CALCULATION DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print(f"\nSample size: {len(diag_df)} measurements")
    
    print(f"\n--- SI Magnitude ---")
    print(f"Median |SI|: {np.median(np.abs(diag_df['SI'])):.3f}")
    print(f"Mean |SI|: {np.mean(np.abs(diag_df['SI'])):.3f}")
    print(f"Range: [{np.min(diag_df['SI']):.3f}, {np.max(diag_df['SI']):.3f}]")
    
    print(f"\n--- Radial Derivative (||Ṙ||²) ---")
    print(f"Median: {diag_df['norm_rd_squared'].median():.1f}")
    high_rd = (diag_df['norm_rd_squared'] > 1000).sum()
    if high_rd > len(diag_df) * 0.3:
        print(f"⚠️  WARNING: {high_rd}/{len(diag_df)} measurements have ||Ṙ||² > 1000")
        print(f"   → Impulsive waveforms making denominator large → small SI")
        print(f"   → Consider more aggressive low-pass filtering")
    
    print(f"\n--- T-Ṙ Correlation ---")
    print(f"Median correlation: {diag_df['correlation_T_rd'].median():.3f}")
    weak_corr = (np.abs(diag_df['correlation_T_rd']) < 0.3).sum()
    if weak_corr > len(diag_df) * 0.3:
        print(f"⚠️  WARNING: {weak_corr}/{len(diag_df)} measurements have weak correlation (<0.3)")
        print(f"   → T and Ṙ not well-related → unreliable SI")
        print(f"   → May indicate noise or complex anisotropy")
    
    print(f"\n--- Waveform Impulsiveness ---")
    print(f"Median kurtosis: {diag_df['impulsiveness'].median():.1f} (Gaussian = 3)")
    impulsive = (diag_df['impulsiveness'] > 10).sum()
    if impulsive > len(diag_df) * 0.2:
        print(f"⚠️  WARNING: {impulsive}/{len(diag_df)} measurements are highly impulsive")
        print(f"   → Sharp waveforms → large derivatives → small SI")
    
    print(f"\n--- SNR ---")
    print(f"Median SNR: {diag_df['snr_estimate'].median():.1f}")
    low_snr = (diag_df['snr_estimate'] < 3).sum()
    if low_snr > len(diag_df) * 0.2:
        print(f"⚠️  WARNING: {low_snr}/{len(diag_df)} measurements have low SNR (<3)")
    
    print("\n" + "="*70)