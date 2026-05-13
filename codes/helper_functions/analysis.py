"""
analysis.py
Core analysis functions: binning measurements and fitting splitting intensity curves.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable


def weighted_mean_and_std(values, errors):
    """
    Compute inverse-variance weighted mean and weighted standard deviation.

    Parameters
    ----------
    values : array-like
        SI measurements
    errors : array-like
        Measurement uncertainties (SI_calc_error)

    Returns
    -------
    mean_w : float
        Weighted mean
    std_w : float
        Weighted standard deviation
    """

    values = np.asarray(values)
    errors = np.asarray(errors)

    # Remove invalid entries
    mask = np.isfinite(values) & np.isfinite(errors) & (errors > 0)

    values = values[mask]
    errors = errors[mask]

    if len(values) == 0:
        return np.nan, np.nan

    # Inverse-variance weights
    weights = 1 / (errors**2)

    # Weighted mean
    mean_w = np.sum(weights * values) / np.sum(weights)

    # Weighted variance
    variance_w = np.sum(weights * (values - mean_w)**2) / np.sum(weights)

    # Weighted std
    std_w = np.sqrt(variance_w)

    return mean_w, std_w


'''
May 7 2026 
Trying out new way to calculate the measurement uncertainty for these measurements
- SI err calculates measurement uncertainty, used as filter for measurement quality 
- inverse variance weighted mean per bin calculates best estimate of SI at that azimuth 
- weighted variance / weighted SD calculates coherence of anisotropic signal within bin

Individual splitting intensity (SI) measurements are first filtered using their associated SI errors 
to remove low-quality waveform measurements and ensure that only reliable estimates contribute to the 
analysis. Within each azimuthal bin, a weighted mean is then calculated using inverse-variance weighting 
so that measurements with smaller SI uncertainties contribute more strongly to the estimated bin-average 
SI. Because variability within a bin reflects not only random measurement noise but also unresolved 
effects such as source-side anisotropy and structural heterogeneity, a weighted standard deviation is 
used to quantify the coherence of the measurements within the bin rather than the statistical uncertainty
of the mean alone. This approach preserves physically meaningful scatter in the data and provides a more 
realistic estimate of how well a single anisotropic signal represents the measurements within each azimuthal range.
'''

'''
May 8 2026 
- yet ANOTHER change to my binning schema
- bin ALL phases together, since filtering is done upstream via qualitative assessment (I looked at it)
'''

def bin_measurements(event_data, bin_width=5, bin_minimum=2, s_depth=400, phase_weights=None):
    """
    Bin measurements by azimuth, combining ALL phases within each bin.
    
    Phase filtering should be done BEFORE calling this function.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event measurements (pre-filtered by caller for phase selection)
        Required columns: 'calcbaz', 'wiggleSI_calc', 'SI_calc_error'
    bin_width : float
        Azimuth bin width in degrees (default: 5°)
    bin_minimum : int
        Minimum number of measurements per bin (default: 2)
    s_depth : float
        Not used (kept for compatibility)
    phase_weights : dict, optional
        Not used (kept for compatibility)
        
    Returns
    -------
    pd.DataFrame
        Binned results with columns: ['chunk', 'mean', 'std', 'variance', 'azimuth', 'n']
    """
    
    # Create azimuth bins
    array = np.arange(0, 360, bin_width)
    
    # Assign measurements to bins
    df = event_data.copy()
    df = df.sort_values(by='calcbaz', ascending=True)
    df['chunk'] = np.digitize(df['calcbaz'], array, right=True)
    
    # Initialize results list
    results = []
    
    # Loop over bins
    for chunk in range(len(array)):
        bin_data = df[df['chunk'] == chunk + 1]
        
        if len(bin_data) < bin_minimum:
            continue
        
        # ALL measurements in this bin (regardless of phase type)
        si_values = bin_data['wiggleSI_calc'].values
        si_errors = bin_data['SI_calc_error'].values
        
        # Weighted statistics
        mean_w, std_w = weighted_mean_and_std(si_values, si_errors)
        
        azimuth_center = array[chunk] + bin_width / 2
        
        # Store results
        results.append({
            'chunk': chunk,
            'mean': mean_w,
            'std': std_w,
            'variance': std_w**2,
            'azimuth': azimuth_center,
            'n': len(bin_data)
        })
    
    return pd.DataFrame(results)

# def bin_measurements(event_data,
#                      bin_width=5,
#                      bin_minimum=2,
#                      s_depth=400,
#                      phase_weights=None):
#     """
#     Bin SI measurements by back-azimuth using weighted statistics.

#     Returns
#     -------
#     pd.DataFrame
#         Columns:
#         ['chunk', 'mean', 'variance', 'weight',
#          'azimuth', 'std', 'name', 'n']
#     """


#     if phase_weights is None:
#         phase_weights = {
#             'sks': 1,
#             's': 1,
#             's_depth': 1,
#             's_shallow': 1
#         }

#     # Azimuth bins
#     array = np.arange(0, 360, bin_width)

#     # Prepare dataframe
#     df = event_data.copy()
#     df = df.sort_values(by='calcbaz', ascending=True)

#     df['chunk'] = np.digitize(df['calcbaz'], array, right=True)

#     # Result arrays
#     binned_sks = np.zeros((len(array), 8))
#     binned_s = np.zeros((len(array), 8))
#     binned_s_depth = np.zeros((len(array), 8))
#     binned_s_shallow = np.zeros((len(array), 8))


#     # Loop over bins
#     for chunk in range(len(array)):

#         bin_data = df[df['chunk'] == chunk + 1]

#         # Phase selections
#         phase_cases = [

#             (
#                 bin_data[bin_data['evphase'] == 'SKS'],
#                 binned_sks,
#                 phase_weights['sks'],
#                 1
#             ),

#             (
#                 bin_data[bin_data['evphase'] == 'S'],
#                 binned_s,
#                 phase_weights['s'],
#                 2
#             ),

#             (
#                 bin_data[
#                     (bin_data['evphase'] == 'S') &
#                     (bin_data['evdp'] >= s_depth)
#                 ],
#                 binned_s_depth,
#                 phase_weights['s_depth'],
#                 3
#             ),

#             (
#                 bin_data[
#                     (bin_data['evphase'] == 'S') &
#                     (bin_data['evdp'] < s_depth)
#                 ],
#                 binned_s_shallow,
#                 phase_weights['s_shallow'],
#                 4
#             )
#         ]

#         for phase_df, results, phase_weight, name in phase_cases:
#             si_values = phase_df['wiggleSI_calc'].values
#             si_errors = phase_df['SI_calc_error'].values

#             n = len(si_values)

#             if n < bin_minimum:
#                 results[chunk, :] = 0
#                 continue

#             # Weighted statistics
#             mean_w, std_w = weighted_mean_and_std(
#                 si_values,
#                 si_errors
#             )

#             variance_w = std_w**2

#             # Inverse-variance curve-fit weight
#             if variance_w > 0:
#                 fit_weight = std_w
#             else:
#                 fit_weight = 0

#             azimuth_center = array[chunk] + bin_width / 2

#             # Store:
#             # chunk, mean, variance, weight,
#             # azimuth, std, name, n
#             results[chunk, :] = [
#                 chunk,
#                 mean_w,
#                 variance_w,
#                 fit_weight,
#                 azimuth_center,
#                 std_w,
#                 name,
#                 n
#             ]

#     # Remove empty rows
#     binned_sks = binned_sks[binned_sks[:, 3] != 0]
#     binned_s = binned_s[binned_s[:, 3] != 0]
#     binned_s_depth = binned_s_depth[binned_s_depth[:, 3] != 0]
#     binned_s_shallow = binned_s_shallow[binned_s_shallow[:, 3] != 0]

#     # Combine desired datasets
#     binned = np.vstack((
#         binned_sks,
#         binned_s_depth,
#         binned_s_shallow
#     ))

#     # Final dataframe
#     binned_df = pd.DataFrame(
#         binned,
#         columns=[
#             'chunk',
#             'mean',
#             'variance',
#             'weight',
#             'azimuth',
#             'std',
#             'name',
#             'n'
#         ]
#     )

#     return binned_df


# def bin_measurements(event_data, 
#                      bin_width=5, 
#                      bin_minimum=2,
#                      s_depth=400,
#                      weights=None):
#     """
#     Bin measurements by back azimuth and calculate statistics per phase type.
    
#     Parameters
#     ----------
#     event_data : pd.DataFrame
#         Event measurements with columns: 'calcbaz', 'evphase', 'evdp', 'wiggleSI_calc', 'SI_calc_error'
#     bin_width : float
#         Azimuth bin width in degrees
#     bin_minimum : int
#         Minimum number of measurements per bin to calculate statistics
#     s_depth : float
#         Depth threshold (km) separating shallow and deep S phases
#     weights : dict, optional
#         Phase-specific weights: {'sks': 1, 's': 1, 's_depth': 1, 's_shallow': 1}
        
#     Returns
#     -------
#     pd.DataFrame
#         Binned results with columns: ['chunk', 'mean', 'weight', 'azimuth', 'margin', 'name']
#         where 'name' codes: 1=SKS, 2=S (all), 3=S deep, 4=S shallow
#     """
#     if weights is None:
#         weights = {'sks': 1, 's': 1, 's_depth': 1, 's_shallow': 1}
    
#     # Create bins
#     array = np.arange(0, 360, bin_width)
    
#     # Assign each measurement to a bin
#     df = event_data.copy()
#     df = df.sort_values(by='calcbaz', ascending=True)
#     df['chunk'] = np.digitize(df['calcbaz'], array, right=True)
    
#     # Initialize result arrays
#     binned_sks = np.zeros((len(array), 6))
#     binned_s = np.zeros((len(array), 6))
#     binned_s_depth = np.zeros((len(array), 6))
#     binned_s_shallow = np.zeros((len(array), 6))
    
#     # Loop over bins
#     for chunk in range(len(array)):
#         bin_data = df[df['chunk'] == chunk + 1]
        
#         # Separate by phase
#         si_sks = bin_data[bin_data['evphase'] == 'SKS']['wiggleSI_calc']
#         si_s = bin_data[bin_data['evphase'] == 'S']['wiggleSI_calc']
#         si_s_depth = bin_data[(bin_data['evphase'] == 'S') & 
#                                (bin_data['evdp'] >= s_depth)]['wiggleSI_calc']
#         si_s_shallow = bin_data[(bin_data['evphase'] == 'S') & 
#                                  (bin_data['evdp'] < s_depth)]['wiggleSI_calc']
        
#         # Compute statistics for each phase category
#         si_cases = [
#             (si_sks, binned_sks, weights['sks'], 1),
#             (si_s, binned_s, weights['s'], 2),
#             (si_s_depth, binned_s_depth, weights['s_depth'], 3),
#             (si_s_shallow, binned_s_shallow, weights['s_shallow'], 4)
#         ]
        
#         for si_values, results, phase_weight, name in si_cases:
#             n = len(si_values)
            
#             if n < bin_minimum:
#                 results[chunk, :] = chunk, 0, 0, 0, 0, 0
#                 continue
            
#             mean = np.mean(si_values)
#             std = np.std(si_values, ddof=1)
            
#             # 95% confidence interval
#             t_crit = stats.t.ppf(0.975, df=n - 1)
#             margin = t_crit * (std / np.sqrt(n))
            
#             # Weight inversely by margin (scaled by phase weight)
#             weight = margin / phase_weight
            
#             # Store: chunk, mean, weight, azimuth (bin center), margin, name
#             results[chunk, :] = chunk, mean, weight, array[chunk] + bin_width / 2, margin, name

#     # Remove empty bins (where weight == 0)
#     binned_sks = binned_sks[binned_sks[:, 2] != 0, :]
#     binned_s = binned_s[binned_s[:, 2] != 0, :]
#     binned_s_depth = binned_s_depth[binned_s_depth[:, 2] != 0, :]
#     binned_s_shallow = binned_s_shallow[binned_s_shallow[:, 2] != 0, :]
    
#     # Combine and convert to DataFrame
#     # binned = np.vstack((binned_sks, binned_s))  # or include all four
#     binned = np.vstack((binned_sks, binned_s_depth, binned_s_shallow))  # Include SKS and both S categories

#     binned_df = pd.DataFrame(binned, columns=['chunk', 'mean', 'weight', 'azimuth', 'margin', 'name'])
    
#     # Optional: filter out bins with large margin
#     # binned_df = binned_df[binned_df['margin'] <= 1]
    
#     return binned_df


def si_function_simple(baz, fast, lag):
    """
    Simple splitting intensity function (single horizontal layer).
    
    SI(θ) = lag * 2 * sin(θ - φ) * cos(θ - φ)
    
    Parameters
    ----------
    baz : float or array
        Back azimuth (degrees)
    fast : float
        Fast axis orientation (degrees)
    lag : float
        Delay time (s)
        
    Returns
    -------
    float or array
        Splitting intensity
    """
    # return lag * 2 * np.sin(np.deg2rad(baz - fast)) * np.cos(np.deg2rad(baz - fast))
    return lag *  np.sin( 2* np.deg2rad(baz - fast))



def si_function_tilt(baz_incident, fast, lag, tilt):
    """
    Splitting intensity function with dipping symmetry axis.
    
    Parameters
    ----------
    baz_incident : tuple of (baz, incident)
        baz : array, back azimuth (degrees)
        incident : array, incident angle (degrees)
    fast : float
        Fast axis orientation (degrees)
    lag : float
        Delay time (s)
    tilt : float
        Tilt of symmetry axis (degrees)
        
    Returns
    -------
    array
        Splitting intensity
    """
    baz, incident = baz_incident
    term1 = (np.cos(np.deg2rad(tilt))**2 * 
             np.sin(2 * np.deg2rad(baz - fast)))
    term2 = (np.sin(2 * np.deg2rad(tilt)) * 
             np.sin(np.deg2rad(baz - fast)) * 
             np.tan(np.deg2rad(incident)))
    return lag * (term1 + term2)

def fit_splitting_intensity(binned, incident_angle=25, use_tilt=False):
    """
    Fit splitting intensity using LINEAR least squares.
    
    Parameters
    ----------
    binned : pd.DataFrame
        Output from bin_measurements() with columns: 
        ['azimuth', 'mean', 'std', 'n', ...]
    incident_angle : float
        Incident angle (degrees) - NOT USED in linear approach but kept for compatibility
    use_tilt : bool
        If True, includes tilt term (NOT RECOMMENDED - set to False)
        
    Returns
    -------
    dict
        Results dictionary with keys:
        - 'phi': fast direction (degrees)
        - 'dt': delay time (seconds)
        - 'phi_err': uncertainty in phi
        - 'dt_err': uncertainty in dt
        - 'covariance': 2x2 covariance matrix
        - 'correlation': 2x2 correlation matrix
        - 'perr': array of uncertainties [phi_err, dt_err]
        - 'A', 'B': linear coefficients
    """
    
    # Extract data
    baz = np.radians(binned['azimuth'].values)
    SI = binned['mean'].values
    sigma = binned['std'].values
    
    # Build design matrix: SI = A*sin(2*baz) + B*cos(2*baz)
    X = np.column_stack([
        np.sin(2 * baz),   # Column for A
        np.cos(2 * baz)    # Column for B
    ])
    
    # Weight matrix (inverse variance weighting)
    W = np.diag(1 / sigma**2)
    
    # Weighted normal equations: (X^T W X) p = X^T W y
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ SI
    
    # Solve for linear parameters [A, B]
    params = np.linalg.solve(XtWX, XtWy)
    A, B = params
    
    # Convert to physical parameters
    dt = np.sqrt(A**2 + B**2)
    phi = 0.5 * np.degrees(np.arctan2(-B, A))
    
    # Ensure phi in [0, 180)
    if phi < 0:
        phi += 180
    
    # Covariance matrix of linear parameters (A, B)
    pcov_AB = np.linalg.inv(XtWX)
    
    # # Jacobian for transforming (A,B) uncertainties to (phi, dt)
    # # J[i,j] = ∂(phi, dt)[i] / ∂(A, B)[j]
    # denom = A**2 + B**2
    # J = np.array([
    #     [-B/(2*denom), A/(2*denom)],  # [∂phi/∂A, ∂phi/∂B]
    #     [A/dt, B/dt]                   # [∂dt/∂A,  ∂dt/∂B]
    # ])
    
    # Jacobian with CORRECT degrees conversion
    denom = A**2 + B**2
    J = np.array([
        # ∂phi/∂A, ∂phi/∂B  (include 0.5 × 180/π factor!)
        [np.degrees(-B/(2*denom)), np.degrees(A/(2*denom))],
        # ∂dt/∂A,  ∂dt/∂B
        [A/dt, B/dt]
    ])

    # Propagate covariance to physical parameters
    pcov_physical = J @ pcov_AB @ J.T
    
    # Extract uncertainties
    perr = np.sqrt(np.diag(pcov_physical))
    phi_err, dt_err = perr
    
    # Calculate correlation matrix
    corr = pcov_physical / np.outer(perr, perr)
    
    # Package results (same format as before for compatibility)
    result = {
        'phi': phi,
        'dt': dt,
        'phi_err': phi_err,
        'dt_err': dt_err,
        'covariance': pcov_physical,
        'correlation': corr,
        'perr': perr,
        'A': A,  # Linear coefficients (bonus info)
        'B': B,
        'pcov_AB': pcov_AB  # Covariance in linear space (bonus)
    }

    return result

# def fit_splitting_intensity(binned_results, 
#                              incident_angle=25,
#                              use_tilt=True,
#                              absolute_sigma=True):
#     """
#     Fit splitting intensity curve to binned measurements.
    
#     Parameters
#     ----------
#     binned_results : pd.DataFrame
#         Output from bin_measurements()
#     incident_angle : float
#         Assumed incident angle (degrees) for all measurements
#     use_tilt : bool
#         If True, fit 3-parameter model (fast, lag, tilt)
#         If False, fit 2-parameter model (fast, lag)
        
#     Returns
#     -------
#     dict
#         Fit results: {'phi': fast_axis, 'dt': delay_time, 'tilt': tilt_angle (if use_tilt),
#                       'params': fitted params, 'cov': covariance matrix}
#     """
#     azimuths = binned_results['azimuth'].values
#     si_obs = binned_results['mean'].values
#     weights = binned_results['weight'].values
    
#     if use_tilt:
#         # 3-parameter fit
#         incident = np.full_like(azimuths, incident_angle)
#         baz_incident = (azimuths, incident)
#         bounds = ([-90, 0.1, 0], [90, 2, 45])
        
#         params, cov = curve_fit(
#             si_function_tilt, 
#             baz_incident, 
#             si_obs, 
#             sigma=weights,
#             absolute_sigma=absolute_sigma,
#             bounds=bounds,
#             nan_policy='omit',
#             method='dogbox',
#             p0=[45,1,0]
#         )
#         phi, dt, tilt = params
        
#         return {
#             'phi': phi,
#             'dt': dt,
#             'tilt': tilt,
#             'params': params,
#             'cov': cov
#         }
#     else:
#         # 2-parameter fit
#         bounds = ([-90, 0.1], [90, 2])
        
#         params, cov = curve_fit(
#             si_function_simple,
#             azimuths,
#             si_obs,
#             sigma=weights,
#             absolute_sigma=absolute_sigma,
#             bounds=bounds,
#             nan_policy='omit',
#             method='dogbox',
#             p0=[45,1]
#         )
#         phi, dt = params
        
#         return {
#             'phi': phi,
#             'dt': dt,
#             'params': params,
#             'cov': cov
#         }


def analyze_station(event_data, 
                    station_code,
                    bin_width=5,
                    bin_minimum=2,
                    s_depth=400,
                    weights=None,
                    incident_angle=25,
                    use_tilt=True):
    """
    Complete analysis for a single station: filter, bin, fit.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Filtered event catalog
    station_code : str
        Station to analyze
    bin_width, bin_minimum, s_depth, weights, incident_angle, use_tilt :
        See bin_measurements() and fit_splitting_intensity()
        
    Returns
    -------
    dict
        Results dictionary with keys:
        - 'station': station code
        - 'n_events': number of measurements used
        - 'event_data': filtered DataFrame for this station
        - 'binned': binned DataFrame
        - 'phi', 'dt', 'tilt' (if use_tilt): fitted parameters
        - 'fit': full fit dict from fit_splitting_intensity()
    """
    from . import filtering
    
    # Filter to station
    station_events = filtering.filter_station_events(event_data, station_code)

    
    if len(station_events) == 0:
        return None
    

    # Bin measurements
    binned = bin_measurements(
        station_events,
        bin_width=bin_width,
        bin_minimum=bin_minimum,
        s_depth=s_depth,
        phase_weights=weights
    )

    print('do we at least bin these')
    if len(binned) == 0:
        return None
    
    # Fit splitting
    fit_result = fit_splitting_intensity(
        binned,
        incident_angle=incident_angle,
        use_tilt=use_tilt
    )
    
    # Package results
    result = {
        'station': station_code,
        'n_events': len(station_events),
        'event_data': station_events,
        'binned': binned,
        'fit': fit_result,
        'phi': fit_result['phi'],
        'dt': fit_result['dt'],
        'cov': fit_result['covariance']
    }
    
    if use_tilt:
        result['tilt'] = fit_result['tilt']
    
    return result


def group_stations_by_grid(event_data, 
                           grid_size=5.0,
                           min_stations_per_group=1):
    """
    Group stations into geographic grid cells.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog with 'evstation', 'evstationlat', 'evstationlon'
    grid_size : float
        Grid cell size in degrees (default 5° x 5°)
    min_stations_per_group : int
        Minimum number of stations required in a group
        
    Returns
    -------
    dict
        Keys: (lat_bin, lon_bin) tuples
        Values: dict with 'stations', 'center_lat', 'center_lon', 'event_data'
    """
    import numpy as np
    
    # Get unique stations with their coordinates
    stations = event_data.groupby('evstation').agg({
        'evstationlat': 'first',
        'evstationlon': 'first'
    }).reset_index()
    
    # Assign each station to a grid cell
    stations['lat_bin'] = (stations['evstationlat'] // grid_size) * grid_size
    stations['lon_bin'] = (stations['evstationlon'] // grid_size) * grid_size
    
    # Group stations by grid cell
    grouped = {}
    
    for (lat_bin, lon_bin), group in stations.groupby(['lat_bin', 'lon_bin']):
        station_codes = group['evstation'].tolist()
        
        if len(station_codes) < min_stations_per_group:
            continue
        
        # Get all events from these stations
        group_events = event_data[event_data['evstation'].isin(station_codes)].copy()
        
        # Calculate group center
        center_lat = lat_bin + grid_size / 2
        center_lon = lon_bin + grid_size / 2
        
        grouped[(lat_bin, lon_bin)] = {
            'stations': station_codes,
            'n_stations': len(station_codes),
            'center_lat': center_lat,
            'center_lon': center_lon,
            'lat_range': (lat_bin, lat_bin + grid_size),
            'lon_range': (lon_bin, lon_bin + grid_size),
            'event_data': group_events,
            'n_measurements': len(group_events)
        }
    
    print(f"\nCreated {len(grouped)} groups from {len(stations)} stations:")
    for key, val in grouped.items():
        print(f"  Grid {key}: {val['n_stations']} stations, "
              f"{val['n_measurements']} measurements")
        print(f"    Stations: {', '.join(val['stations'][:5])}"
              f"{'...' if len(val['stations']) > 5 else ''}")
    
    return grouped


def analyze_station_group(group_data,
                          bin_width=5,
                          bin_minimum=2,
                          s_depth=400,
                          weights=None,
                          incident_angle=25,
                          use_tilt=True):
    """
    Analyze combined measurements from a group of stations.
    
    Parameters
    ----------
    group_data : dict
        Single group from group_stations_by_grid()
    ... (other params same as analyze_station)
    
    Returns
    -------
    dict
        Analysis results for the group
    """
    event_data = group_data['event_data']
    
    if len(event_data) == 0:
        return None
    
    # Bin measurements (same as single station)
    binned = bin_measurements(
        event_data,
        bin_width=bin_width,
        bin_minimum=bin_minimum,
        s_depth=s_depth,
        weights=weights
    )
    
    if len(binned) == 0:
        return None
    
    # Fit splitting
    fit_result = fit_splitting_intensity(
        binned,
        incident_angle=incident_angle,
        use_tilt=use_tilt
    )
    
    result = {
        'group_id': (group_data['lat_range'], group_data['lon_range']),
        'stations': group_data['stations'],
        'n_stations': group_data['n_stations'],
        'center_lat': group_data['center_lat'],
        'center_lon': group_data['center_lon'],
        'n_events': len(event_data),
        'event_data': event_data,
        'binned': binned,
        'fit': fit_result,
        'phi': fit_result['phi'],
        'dt': fit_result['dt']
    }
    
    if use_tilt:
        result['tilt'] = fit_result['tilt']
    
    return result

def fit_splitting_intensity_individual(event_data, 
                                       incident_angle=25,
                                       use_tilt=True):
    """
    Fit splitting intensity curve to individual (unbinned) measurements.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event measurements with 'calcbaz', 'wiggleSI_calc', 'SI_calc_error'
    incident_angle : float
        Assumed incident angle (degrees)
    use_tilt : bool
        If True, fit 3-parameter model (fast, lag, tilt)
        
    Returns
    -------
    dict
        Fit results: {'phi', 'dt', 'tilt' (if use_tilt), 'params', 'cov'}
    """
    azimuths = event_data['calcbaz'].values
    si_obs = event_data['wiggleSI_calc'].values
    weights = event_data['SI_calc_error'].values
    
    # Remove any NaN/inf
    valid = np.isfinite(azimuths) & np.isfinite(si_obs) & np.isfinite(weights) & (weights > 0)
    azimuths = azimuths[valid]
    si_obs = si_obs[valid]
    weights = weights[valid]
    
    if len(azimuths) < 3:
        raise ValueError("Not enough valid measurements for fitting")
    
    if use_tilt:
        # 3-parameter fit
        incident = np.full_like(azimuths, incident_angle)
        baz_incident = (azimuths, incident)
        bounds = ([-90, 0.1, 0], [90, 2, 45])
        
        params, cov = curve_fit(
            si_function_tilt, 
            baz_incident, 
            si_obs, 
            sigma=weights,
            bounds=bounds,
            nan_policy='omit',
            method='dogbox',
            p0=[45,1,0]
        )
        phi, dt, tilt = params
        
        return {
            'phi': phi,
            'dt': dt,
            'tilt': tilt,
            'params': params,
            'cov': cov
        }
    else:
        # 2-parameter fit
        bounds = ([-90, 0.1], [90, 2])
        
        params, cov = curve_fit(
            si_function_simple,
            azimuths,
            si_obs,
            sigma=weights,
            bounds=bounds,
            nan_policy='omit',
            method='dogbox',
            p0=[45,1]
        )
        phi, dt = params
        
        return {
            'phi': phi,
            'dt': dt,
            'params': params,
            'cov': cov
        }

def fit_SI_linear_explicit(binned):
    """
    Explicit linear least squares - clear for thesis/papers
    """
    # Extract data
    baz = np.radians(binned['azimuth'].values)
    SI = binned['mean'].values
    sigma = binned['std'].values
    
    # Design matrix
    X = np.column_stack([np.sin(2*baz), np.cos(2*baz)])
    
    # Weight matrix
    W = np.diag(1 / sigma**2)
    
    # Normal equations: (X^T W X) p = X^T W y
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ SI
    
    # Solve
    p = np.linalg.solve(XtWX, XtWy)
    A, B = p
    
    # Physical parameters
    dt = np.sqrt(A**2 + B**2)
    phi = 0.5 * np.degrees(np.arctan2(B, A))
    if phi < 0:
        phi += 180
    
    # Covariance
    pcov_AB = np.linalg.inv(XtWX)
    
    # Jacobian for uncertainty propagation
    denom = A**2 + B**2
    J = np.array([
        [-B/(2*(A**2 + B**2)), A/(2*(A**2 + B**2))],  # d(phi)/d(A,B)
        [A/dt, B/dt]                                   # d(dt)/d(A,B)
    ])
    pcov = J @ pcov_AB @ J.T
    
    return {
        'phi': phi, 'dt': dt,
        'phi_err': np.sqrt(pcov[0,0]),
        'dt_err': np.sqrt(pcov[1,1]),
        'covariance': pcov,
        'A': A, 'B': B
    }

def compute_SI_model(baz_deg, phi, dt):
    """
    Compute splitting intensity from physical parameters.
    
    Parameters
    ----------
    baz_deg : array
        Back azimuth in degrees
    phi : float
        Fast direction in degrees
    dt : float
        Delay time in seconds
        
    Returns
    -------
    array
        Splitting intensity values
    """
    baz_rad = np.radians(baz_deg)
    phi_rad = np.radians(phi)
    return dt * np.sin(2 * (baz_rad - phi_rad))
    
# commented out since it does not have full tilt capabilities
# def analyze_station_comparison(eventData, station_code, bin_width=5, bin_minimum=2, 
#                                 s_depth=400, incident_angle=20, use_tilt=False):
#     """
#     Analyze a station with three phase selections:
#     1. SKS only
#     2. SKS + Deep S (>= s_depth)
#     3. All phases
    
#     For each selection, compute:
#     - Binned fit
#     - Individual fit
    
#     Returns
#     -------
#     dict with keys: 'sks', 'sks_deep', 'all'
#     Each contains: 'result_binned', 'fit_individual', 'station_events'
#     """
#     # from helper_functions import analysis, filtering
    
#     # Prepare filtered datasets
#     eventData_sks = eventData[eventData['evphase'] == 'SKS']
#     eventData_sks_deep = eventData[
#         (eventData['evphase'] == 'SKS') | 
#         ((eventData['evphase'] == 'S') & (eventData['evdp'] >= s_depth))
#     ]
#     eventData_all = eventData.copy()
    
#     results = {}
    
#     for name, data in [('sks', eventData_sks), 
#                        ('sks_deep', eventData_sks_deep), 
#                        ('all', eventData_all)]:
        
#         # Get station events
#         station_events = data[data['evstation'] == station_code]
        
#         if len(station_events) < 3:
#             results[name] = None
#             continue
        
#         # Binned analysis
#         try:
#             result_binned = analyze_station(
#                 event_data=data,
#                 station_code=station_code,
#                 bin_width=bin_width,
#                 bin_minimum=bin_minimum,
#                 s_depth=s_depth,
#                 incident_angle=incident_angle,
#                 use_tilt=use_tilt
#             )
#         except:
#             result_binned = None
        
#         # Individual analysis
#         try:
#             fit_individual = fit_splitting_intensity_individual(
#                 station_events,
#                 incident_angle=incident_angle,
#                 use_tilt=use_tilt
#             )
#         except:
#             fit_individual = None
        
#         results[name] = {
#             'result_binned': result_binned,
#             'fit_individual': fit_individual,
#             'station_events': station_events
#         }
    
#     return results

def analyze_station_comparison(eventData, station_code, bin_width=5, bin_minimum=2, 
                                s_depth=400, incident_angle=25, use_tilt=False):  # <-- Add use_tilt
    """
    Analyze a station with three phase selections.
    """
    from helper_functions import analysis, filtering
    
    # Prepare filtered datasets
    eventData_sks = eventData[eventData['evphase'] == 'SKS']
    eventData_sks_deep = eventData[
        (eventData['evphase'] == 'SKS') | 
        ((eventData['evphase'] == 'S') & (eventData['evdp'] >= s_depth))
    ]
    eventData_all = eventData.copy()
    
    results = {}
    
    for name, data in [('sks', eventData_sks), 
                       ('sks_deep', eventData_sks_deep), 
                       ('all', eventData_all)]:
        
        station_events = data[data['evstation'] == station_code]
        
        if len(station_events) < 3:
            results[name] = None
            continue
        
        # Binned analysis with use_tilt
        try:
            print('gonna try and analyze a station')
            result_binned = analysis.analyze_station(
                event_data=data,
                station_code=station_code,
                bin_width=bin_width,
                bin_minimum=bin_minimum,
                s_depth=s_depth,
                incident_angle=incident_angle,
                use_tilt=use_tilt  # <-- Pass through
            )
        except Exception as e:
            print("it did not work...")
            print(e)
            result_binned = None
        
        # Individual analysis with use_tilt
        try:
            fit_individual = analysis.fit_splitting_intensity_individual(
                station_events,
                incident_angle=incident_angle,
                use_tilt=use_tilt  # <-- Pass through
            )
        except:
            fit_individual = None
        
        results[name] = {
            'result_binned': result_binned,
            'fit_individual': fit_individual,
            'station_events': station_events
        }
    
    return results

# commented out since this does not have full tilt capabilities
# def plot_station_comparison(results, station_code, s_depth=400, figsize=(20, 12)):
#     """
#     Create 2×3 grid comparing binned vs individual fits across phase selections.
    
#     Layout:
#     Row 1: Individual measurements with both fits
#     Row 2: Binned measurements with both fits
    
#     Columns: SKS | SKS+Deep S | All
#     """
#     fig = plt.figure(figsize=figsize)
#     gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)
    
#     # Color schemes for each column
#     colors_sks = {1: 'blue'}  # Only SKS
#     colors_sks_deep = {1: 'blue', 2: 'green', 3: 'green'}  # SKS + any S = green
#     colors_all = {1: 'blue', 2: 'red', 3: 'red', 4: 'red'}  # All S phases red
    
#     color_schemes = [colors_sks, colors_sks_deep, colors_all]
#     phase_labels = ['SKS Only', 'SKS + Deep S', 'All Phases']
#     summary_stats = {}
    
#     for col, (phase_key, label, colors) in enumerate(zip(['sks', 'sks_deep', 'all'], 
#                                                           phase_labels, 
#                                                           color_schemes)):
        
#         data = results.get(phase_key)
#         if data is None or data['result_binned'] is None or data['fit_individual'] is None:
#             # Empty panels if no data
#             ax_top = fig.add_subplot(gs[0, col])
#             ax_bottom = fig.add_subplot(gs[1, col])
#             ax_top.text(0.5, 0.5, f'{label}\nInsufficient Data', 
#                        ha='center', va='center', fontsize=14)
#             ax_top.axis('off')
#             ax_bottom.axis('off')
#             continue
        
#         result_binned = data['result_binned']
#         fit_individual = data['fit_individual']
#         station_events = data['station_events']
        
#         # Extract parameters
#         phi_binned = result_binned['phi']
#         dt_binned = result_binned['dt']
#         phi_indiv = fit_individual['phi']
#         dt_indiv = fit_individual['dt']
        
#         # Calculate fit curves
#         degs = np.arange(0, 360, 1)
#         si_fit_binned = si_function_simple(degs, phi_binned, dt_binned)
#         si_fit_indiv = si_function_simple(degs, phi_indiv, dt_indiv)
        
#         # Store stats
#         summary_stats[phase_key] = {
#             'phi_binned': phi_binned,
#             'dt_binned': dt_binned,
#             'phi_indiv': phi_indiv,
#             'dt_indiv': dt_indiv,
#             'delta_phi': abs(phi_binned - phi_indiv),
#             'delta_dt': abs(dt_binned - dt_indiv),
#             'n_measurements': len(station_events),
#             'n_bins': len(result_binned['binned'])
#         }
        
#         # ==================== TOP PANEL: Individual Measurements ====================
#         ax_top = fig.add_subplot(gs[0, col])
        
#         # Plot individual measurements
#         for _, row in station_events.iterrows():
#             if row['evphase'] == 'SKS':
#                 color = 'blue'
#             elif col == 1:  # Middle column (SKS + Deep S)
#                 color = 'green'  # All S here are deep
#             elif col == 2:  # Right column (All phases)
#                 # Color by depth
#                 if row['evdp'] >= s_depth:
#                     color = 'green'  # Deep S
#                 else:
#                     color = 'red'    # Shallow S
#             else:
#                 color = 'blue'  # Left column, only SKS
            
#             ax_top.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
#                        color=color, alpha=0.4, markersize=6)
#             ax_top.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
#                            yerr=row['SI_calc_error'], 
#                            ecolor=color, alpha=0.2, fmt='none')
        
#         # Plot both fit curves
#         ax_top.plot(degs, si_fit_binned, '-', color='darkblue', linewidth=2.5, 
#                    label=f'Binned Fit: φ={phi_binned:.1f}°, δt={dt_binned:.2f}s')
#         ax_top.plot(degs, si_fit_indiv, '--', color='darkorange', linewidth=2.5,
#                    label=f'Individual Fit: φ={phi_indiv:.1f}°, δt={dt_indiv:.2f}s')
        
#         ax_top.set_title(f'{label}\nIndividual Measurements (N={len(station_events)})', 
#                         fontsize=13)
#         ax_top.set_ylabel('Splitting Intensity', fontsize=11)
#         ax_top.set_ylim(-3, 3)
#         ax_top.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
#         ax_top.legend(loc='upper right', fontsize=9)
#         ax_top.grid(True, alpha=0.3)
        
#         # ==================== BOTTOM PANEL: Binned Measurements ====================
#         ax_bottom = fig.add_subplot(gs[1, col])
        
#         # Plot binned measurements with column-specific colors
#         binned = result_binned['binned']
#         for _, row in binned.iterrows():
#             name = int(row['name'])
#             # Use column-specific color scheme
#             bin_color = colors.get(name, 'gray')
#             ax_bottom.plot(row['azimuth'], row['mean'], 'o', 
#                           color=bin_color, markersize=10)
#             ax_bottom.errorbar(row['azimuth'], row['mean'], yerr=row['margin'], 
#                               color=bin_color, fmt='none', alpha=0.6)
        
#         # Plot both fit curves (same as top)
#         ax_bottom.plot(degs, si_fit_binned, '-', color='darkblue', linewidth=2.5,
#                       label=f'Binned Fit: φ={phi_binned:.1f}°, δt={dt_binned:.2f}s')
#         ax_bottom.plot(degs, si_fit_indiv, '--', color='darkorange', linewidth=2.5,
#                       label=f'Individual Fit: φ={phi_indiv:.1f}°, δt={dt_indiv:.2f}s')
        
#         ax_bottom.set_title(f'Binned Measurements (N bins={len(binned)})', 
#                            fontsize=13)
#         ax_bottom.set_xlabel('Polarization Azimuth (°)', fontsize=11)
#         ax_bottom.set_ylabel('Splitting Intensity', fontsize=11)
#         ax_bottom.set_ylim(-3, 3)
#         ax_bottom.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
#         ax_bottom.legend(loc='upper right', fontsize=9)
#         ax_bottom.grid(True, alpha=0.3)
    
#     # Overall title
#     fig.suptitle(f'{station_code} - Binned vs Individual Fitting Comparison', 
#                 fontsize=18, y=0.995)
    
#     return fig, summary_stats

def plot_station_comparison(results, station_code, s_depth=400, figsize=(30, 16)):
    """
    Create 2×3 grid comparing binned vs individual fits across phase selections.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)
    
    # Color schemes for each column
    colors_sks = {1: 'blue'}
    colors_sks_deep = {1: 'blue', 2: 'green', 3: 'green'}
    colors_all = {1: 'blue', 2: 'red', 3: 'red', 4: 'red'}
    
    color_schemes = [colors_sks, colors_sks_deep, colors_all]
    phase_labels = ['SKS Only', 'SKS + Deep S', 'All Phases']
    summary_stats = {}
    
    for col, (phase_key, label, colors) in enumerate(zip(['sks', 'sks_deep', 'all'], 
                                                          phase_labels, 
                                                          color_schemes)):
        
        data = results.get(phase_key)
        if data is None or data['result_binned'] is None or data['fit_individual'] is None:
            # Empty panels if no data
            ax_top = fig.add_subplot(gs[0, col])
            ax_bottom = fig.add_subplot(gs[1, col])
            ax_top.text(0.5, 0.5, f'{label}\nInsufficient Data', 
                       ha='center', va='center', fontsize=14)
            ax_top.axis('off')
            ax_bottom.axis('off')
            continue
        
        result_binned = data['result_binned']
        fit_individual = data['fit_individual']
        station_events = data['station_events']
        
        # Extract parameters
        phi_binned = result_binned['phi']
        dt_binned = result_binned['dt']
        tilt_binned = result_binned.get('tilt', None)  # Get tilt if present
        
        phi_indiv = fit_individual['phi']
        dt_indiv = fit_individual['dt']
        tilt_indiv = fit_individual.get('tilt', None)  # Get tilt if present
        
        # Calculate fit curves - USE TILT IF PRESENT
        degs = np.arange(0, 360, 1)
        incident = np.full_like(degs, 20.0)  # Assumed incident angle
        
        if tilt_binned is not None:
            # Use tilted model
            si_fit_binned = si_function_tilt((degs, incident), 
                                                      phi_binned, dt_binned, tilt_binned)
        else:
            # Use simple model
            si_fit_binned = si_function_simple(degs, phi_binned, dt_binned)
        
        if tilt_indiv is not None:
            si_fit_indiv = si_function_tilt((degs, incident),
                                                     phi_indiv, dt_indiv, tilt_indiv)
        else:
            si_fit_indiv = si_function_simple(degs, phi_indiv, dt_indiv)
        
        # Store stats (including tilt)
        summary_stats[phase_key] = {
            'phi_binned': phi_binned,
            'dt_binned': dt_binned,
            'tilt_binned': tilt_binned,
            'phi_indiv': phi_indiv,
            'dt_indiv': dt_indiv,
            'tilt_indiv': tilt_indiv,
            'delta_phi': abs(phi_binned - phi_indiv),
            'delta_dt': abs(dt_binned - dt_indiv),
            'n_measurements': len(station_events),
            'n_bins': len(result_binned['binned'])
        }
        
        # ==================== TOP PANEL: Individual Measurements ====================
        ax_top = fig.add_subplot(gs[0, col])
        
        # Plot individual measurements
        for _, row in station_events.iterrows():
            if row['evphase'] == 'SKS':
                color = 'blue'
            elif col == 1:  # Middle column (SKS + Deep S)
                color = 'green'
            elif col == 2:  # Right column (All)
                if row['evdp'] >= s_depth:
                    color = 'green'
                else:
                    color = 'red'
            else:
                color = 'blue'
            
            ax_top.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                       color=color, alpha=0.4, markersize=6)
            ax_top.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                           yerr=row['SI_calc_error'], 
                           ecolor=color, alpha=0.2, fmt='none')
        
        # Build legend labels with tilt if present
        binned_label = f'Binned: φ={phi_binned:.1f}°, δt={dt_binned:.2f}s'
        if tilt_binned is not None:
            binned_label += f', tilt={tilt_binned:.1f}°'
        
        indiv_label = f'Individual: φ={phi_indiv:.1f}°, δt={dt_indiv:.2f}s'
        if tilt_indiv is not None:
            indiv_label += f', tilt={tilt_indiv:.1f}°'
        
        # Plot both fit curves
        ax_top.plot(degs, si_fit_binned, '-', color='darkblue', linewidth=2.5, 
                   label=binned_label)
        ax_top.plot(degs, si_fit_indiv, '--', color='darkorange', linewidth=2.5,
                   label=indiv_label)
        
        ax_top.set_title(f'{label}\nIndividual Measurements (N={len(station_events)})', 
                        fontsize=13)
        ax_top.set_ylabel('Splitting Intensity', fontsize=11)
        ax_top.set_ylim(-3, 3)
        ax_top.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_top.legend(loc='upper right', fontsize=9)
        ax_top.grid(True, alpha=0.3)
        
        # ==================== BOTTOM PANEL: Binned Measurements ====================
        ax_bottom = fig.add_subplot(gs[1, col])
        
        # Plot binned measurements with column-specific colors
        binned = result_binned['binned']
        for _, row in binned.iterrows():
            name = int(row['name'])
            bin_color = colors.get(name, 'gray')
            ax_bottom.plot(row['azimuth'], row['mean'], 'o', 
                          color=bin_color, markersize=10)
            ax_bottom.errorbar(row['azimuth'], row['mean'], yerr=row['std'], 
                              color=bin_color, fmt='none', alpha=0.6)
        
        # Plot both fit curves with tilt in legend
        ax_bottom.plot(degs, si_fit_binned, '-', color='darkblue', linewidth=2.5,
                      label=binned_label)
        ax_bottom.plot(degs, si_fit_indiv, '--', color='darkorange', linewidth=2.5,
                      label=indiv_label)
        
        ax_bottom.set_title(f'Binned Measurements (N bins={len(binned)})', 
                           fontsize=13)
        ax_bottom.set_xlabel('Polarization Azimuth (°)', fontsize=11)
        ax_bottom.set_ylabel('Splitting Intensity', fontsize=11)
        ax_bottom.set_ylim(-3, 3)
        ax_bottom.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_bottom.legend(loc='upper right', fontsize=9)
        ax_bottom.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{station_code} - Binned vs Individual Fitting Comparison', 
                fontsize=18, y=0.995)
    
    return fig, summary_stats

def plot_station_sks_deep(results, station_code, s_depth=400, figsize=(16, 10)):
    """
    Single panel plot: SKS + Deep S individual measurements with fits.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use SKS + Deep S only
    data = results.get('sks_deep')
    
    if data is None or data['result_binned'] is None or data['fit_individual'] is None:
        ax.text(0.5, 0.5, 'SKS + Deep S\nInsufficient Data', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, {}
    
    result_binned = data['result_binned']
    fit_individual = data['fit_individual']
    station_events = data['station_events']
    
    # Extract parameters
    phi_binned = result_binned['phi']
    dt_binned = result_binned['dt']
    tilt_binned = result_binned.get('tilt', None)
    
    phi_indiv = fit_individual['phi']
    dt_indiv = fit_individual['dt']
    tilt_indiv = fit_individual.get('tilt', None)
    
    # Fit curves
    degs = np.arange(0, 360, 1)
    incident = np.full_like(degs, 20.0)
    
    if tilt_binned is not None:
        si_fit_binned = si_function_tilt((degs, incident),
                                        phi_binned, dt_binned, tilt_binned)
    else:
        si_fit_binned = si_function_simple(degs, phi_binned, dt_binned)
    
    if tilt_indiv is not None:
        si_fit_indiv = si_function_tilt((degs, incident),
                                       phi_indiv, dt_indiv, tilt_indiv)
    else:
        si_fit_indiv = si_function_simple(degs, phi_indiv, dt_indiv)
    
    # ==================== PLOT INDIVIDUAL MEASUREMENTS ====================
    for _, row in station_events.iterrows():
        if row['evphase'] == 'SKS':
            color = 'blue'
        else:
            color = 'green'  # Deep S
        
        ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                color=color, alpha=0.4, markersize=6)
        ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                    yerr=row['SI_calc_error'], 
                    ecolor=color, alpha=0.2, fmt='none')
    
    # Labels with tilt if present
    binned_label = f'Binned: φ={phi_binned:.1f}°, δt={dt_binned:.2f}s'
    if tilt_binned is not None:
        binned_label += f', tilt={tilt_binned:.1f}°'
    
    # indiv_label = f'Individual: φ={phi_indiv:.1f}°, δt={dt_indiv:.2f}s'
    # if tilt_indiv is not None:
    #     indiv_label += f', tilt={tilt_indiv:.1f}°'
    
    # Plot fits
    ax.plot(degs, si_fit_binned, '-', color='darkblue', linewidth=3.5,
            label=binned_label)
    # ax.plot(degs, si_fit_indiv, '--', color='darkorange', linewidth=2.5,
    #         label=indiv_label)
    
    # Formatting (same as before)
    ax.set_title(f'SKS + Deep S\nIndividual Measurements (N={len(station_events)})',
                 fontsize=10)
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=11)
    ax.set_ylabel('Splitting Intensity', fontsize=11)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{station_code} - Splitting Measurements', fontsize=16, y=0.98)
    
    # Summary stats (kept consistent)
    summary_stats = {
        'phi_binned': phi_binned,
        'dt_binned': dt_binned,
        'tilt_binned': tilt_binned,
        'phi_indiv': phi_indiv,
        'dt_indiv': dt_indiv,
        'tilt_indiv': tilt_indiv,
        'delta_phi': abs(phi_binned - phi_indiv),
        'delta_dt': abs(dt_binned - dt_indiv),
        'n_measurements': len(station_events),
        'n_bins': len(result_binned['binned'])
    }
    
    return fig, summary_stats

def plot_station_s_only(results, station_code, s_depth=400, figsize=(16, 10), dpi=300, savepath=None):
    """
    Single panel plot: Deep S (green) vs Shallow S (red) individual measurements with fit.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use ALL phases but only plot S phases
    data = results.get('all')
    
    if data is None or data['result_binned'] is None:
        ax.text(0.5, 0.5, 'S Phases\nInsufficient Data', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, {}
    
    result_binned = data['result_binned']
    station_events = data['station_events']
    
    # Extract parameters
    phi_binned = result_binned['phi']
    dt_binned = result_binned['dt']
    tilt_binned = result_binned.get('tilt', None)
    
    # Fit curve
    degs = np.arange(0, 360, 1)
    incident = np.full_like(degs, 20.0)
    
    if tilt_binned is not None:
        si_fit_binned = si_function_tilt((degs, incident),
                                        phi_binned, dt_binned, tilt_binned)
    else:
        si_fit_binned = si_function_simple(degs, phi_binned, dt_binned)
    
    # ==================== PLOT S PHASES ONLY ====================
    n_used = 0
    for _, row in station_events.iterrows():
        if row['evphase'] == 'SKS':
            continue  # skip SKS
        
        # Color by depth
        if row['evdp'] >= s_depth:
            color = 'green'   # Deep S
        else:
            color = 'red'     # Shallow S
        
        ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                color=color, alpha=0.4, markersize=6)
        ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                    yerr=row['SI_calc_error'], 
                    ecolor=color, alpha=0.2, fmt='none')
        n_used += 1
    
    # Label
    binned_label = f'Fit: φ={phi_binned:.1f}°, δt={dt_binned:.2f}s'
    if tilt_binned is not None:
        binned_label += f', tilt={tilt_binned:.1f}°'
    
    # Plot fit
    ax.plot(degs, si_fit_binned, '-', color='darkblue', linewidth=3.5,
            label=binned_label)
    
    # Formatting
    ax.set_title(f'S Phases Only (Deep + Shallow)\nN={n_used}', fontsize=10)
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=11)
    ax.set_ylabel('Splitting Intensity', fontsize=11)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{station_code} - S Phase Splitting', fontsize=16, y=0.98)
    
    # ===== SAVE WITH HIGH RESOLUTION =====
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    
    return fig, {
        'phi_binned': phi_binned,
        'dt_binned': dt_binned,
        'tilt_binned': tilt_binned,
        'n_measurements': n_used
    }

def plot_all_phases(results, station_code, s_depth=400, figsize=(16, 10), dpi=300, savepath=None):
    """
    Single panel plot: All phases (SKS, Deep S, Shallow S) individual measurements with fit.
    
    Parameters
    ----------
    results : dict
        Output from analyze_station_comparison()
    station_code : str
        Station name
    s_depth : float
        Depth threshold for S-phase coloring
    figsize : tuple
        Figure size
    dpi : int
        Resolution for saving
    savepath : str, optional
        Path to save figure
        
    Returns
    -------
    fig, stats_dict
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use ALL phases
    data = results.get('all')
    
    if data is None or data['result_binned'] is None:
        ax.text(0.5, 0.5, 'All Phases\nInsufficient Data', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, {}
    
    result_binned = data['result_binned']
    station_events = data['station_events']
    
    # Extract parameters
    phi_binned = result_binned['phi']
    dt_binned = result_binned['dt']
    tilt_binned = result_binned.get('tilt', None)
    
    # Fit curve
    degs = np.arange(0, 360, 1)
    incident = np.full_like(degs, 20.0)
    
    if tilt_binned is not None:
        si_fit_binned = si_function_tilt((degs, incident),
                                        phi_binned, dt_binned, tilt_binned)
    else:
        si_fit_binned = si_function_simple(degs, phi_binned, dt_binned)
    
    # ==================== PLOT ALL PHASES ====================
    sks_plotted = s_deep_plotted = s_shallow_plotted = False
    n_sks = n_deep = n_shallow = 0
    
    for _, row in station_events.iterrows():
        if row['evphase'] == 'SKS':
            label = 'SKS phase' if not sks_plotted else None
            sks_plotted = True
            color = 'blue'
            n_sks += 1
            
            ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                   color=color, alpha=0.5, markersize=6, label=label)
            ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                       yerr=row['SI_calc_error'], 
                       ecolor=color, alpha=0.3, fmt='none')
        
        elif row['evphase'] == 'S':
            # Color by depth
            if row['evdp'] >= s_depth:
                label = 'S deep' if not s_deep_plotted else None
                s_deep_plotted = True
                color = 'green'
                n_deep += 1
            else:
                label = 'S shallow' if not s_shallow_plotted else None
                s_shallow_plotted = True
                color = 'red'
                n_shallow += 1
            
            ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                   color=color, alpha=0.5, markersize=6, label=label)
            ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                       yerr=row['SI_calc_error'], 
                       ecolor=color, alpha=0.3, fmt='none')
    
    # Label
    binned_label = f'Fit: φ={phi_binned:.1f}°, δt={dt_binned:.2f}s'
    if tilt_binned is not None:
        binned_label += f', tilt={tilt_binned:.1f}°'
    
    # Plot fit
    ax.plot(degs, si_fit_binned, '-', color='black', linewidth=3,
            label=binned_label)
    
    # Formatting
    n_total = n_sks + n_deep + n_shallow
    title_str = f'All Phases: SKS={n_sks}, S deep={n_deep}, S shallow={n_shallow} (N={n_total})'
    
    ax.set_title(title_str, fontsize=11)
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=11)
    ax.set_ylabel('Splitting Intensity', fontsize=11)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{station_code} - All Phase Splitting', fontsize=16, y=0.98)
    
    # ===== SAVE WITH HIGH RESOLUTION =====
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    
    return fig, {
        'phi_binned': phi_binned,
        'dt_binned': dt_binned,
        'tilt_binned': tilt_binned,
        'n_sks': n_sks,
        'n_deep': n_deep,
        'n_shallow': n_shallow,
        'n_total': n_total
    }

def bootstrap_splitting_parameters(event_data, 
                                   station_code=None,
                                   bin_width=20,
                                   bin_minimum=2,
                                   s_depth=400,
                                   n_bootstrap=1000,
                                   use_tilt=False,
                                   subsample_fraction=1.0,
                                   confidence_level=0.95):
    """
    Bootstrap resampling to estimate parameter uncertainties.
    
    Repeatedly:
    1. Resample measurements with replacement
    2. Bin and fit
    3. Store parameters
    
    Returns distribution of φ and δt for uncertainty estimation.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event measurements (pre-filtered to station if station_code=None)
    station_code : str, optional
        Station to analyze (if None, assumes data already filtered)
    bin_width, bin_minimum, s_depth : float
        Binning parameters
    n_bootstrap : int
        Number of bootstrap samples (default 1000)
    use_tilt : bool
        If True, fit 3-parameter model (φ, δt, tilt)
    confidence_level : float
        Confidence level for percentile intervals (default 0.95 = 95%)
        
    Returns
    -------
    dict with:
        'phi_samples', 'dt_samples' : arrays of bootstrap parameters
        'tilt_samples' : array (if use_tilt=True)
        'phi_median', 'dt_median' : median values
        'phi_ci', 'dt_ci' : confidence intervals (tuple: lower, upper)
        'original_fit' : fit from full dataset
    """
    from . import filtering
    
    # Filter to station if needed
    if station_code is not None:
        station_events = filtering.filter_station_events(event_data, station_code)
    else:
        station_events = event_data.copy()
    
    n_events = len(station_events)
    
    if n_events < 5:
        raise ValueError(f"Not enough data for bootstrapping ({n_events} events)")
    
    # Fit original (full dataset)
    original_binned = bin_measurements(station_events, bin_width, bin_minimum, s_depth)
    original_fit = fit_splitting_intensity(original_binned, use_tilt=use_tilt)
    
    # Bootstrap
    phi_samples = []
    dt_samples = []
    tilt_samples = [] if use_tilt else None
    
    print(f"Running {n_bootstrap} bootstrap samples...")

    n_events = len(station_events)
    n_subsample = int(n_events * subsample_fraction)  # ← Calculate subsample size
    
    for i in range(n_bootstrap):
        # Resample with replacement, but use only subsample_fraction of data
        bootstrap_sample = station_events.sample(n=n_subsample, replace=True)
        
        try:
            # Bin and fit
            binned = bin_measurements(bootstrap_sample, bin_width, bin_minimum, s_depth)
            
            if len(binned) < 3:  # Need minimum bins to fit
                continue
            
            fit = fit_splitting_intensity(binned, use_tilt=use_tilt)
            
            # Store parameters
            phi_samples.append(fit['phi'])
            dt_samples.append(fit['dt'])
            
            if use_tilt:
                tilt_samples.append(fit['tilt'])
            
        except Exception as e:
            # Skip failed fits
            continue
    
    # Convert to arrays
    phi_samples = np.array(phi_samples)
    dt_samples = np.array(dt_samples)
    if use_tilt:
        tilt_samples = np.array(tilt_samples)
    
    # Calculate percentiles for confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    # Handle circular statistics for φ (directional data)
    phi_samples = np.array(phi_samples)  # Already in -90 to 90 range
    phi_median = np.median(phi_samples) 
    
    phi_ci = (np.percentile(phi_samples, lower_percentile),
              np.percentile(phi_samples, upper_percentile))
    
    # δt is linear
    dt_median = np.median(dt_samples)
    dt_ci = (np.percentile(dt_samples, lower_percentile),
             np.percentile(dt_samples, upper_percentile))
    
    result = {
        'phi_samples': phi_samples,
        'dt_samples': dt_samples,
        'phi_median': phi_median,
        'dt_median': dt_median,
        'phi_ci': phi_ci,
        'dt_ci': dt_ci,
        'phi_std': np.std(phi_samples),
        'dt_std': np.std(dt_samples),
        'n_successful': len(phi_samples),
        'n_attempted': n_bootstrap,
        'success_rate': len(phi_samples) / n_bootstrap,
        'original_fit': original_fit,
        'use_tilt': use_tilt
    }
    
    if use_tilt:
        tilt_median = np.median(tilt_samples)
        tilt_ci = (np.percentile(tilt_samples, lower_percentile),
                   np.percentile(tilt_samples, upper_percentile))
        
        result['tilt_samples'] = tilt_samples
        result['tilt_median'] = tilt_median
        result['tilt_ci'] = tilt_ci
        result['tilt_std'] = np.std(tilt_samples)
    
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP RESULTS (n={len(phi_samples)} successful)")
    print('='*70)
    print(f"φ: {phi_median:.1f}° [{phi_ci[0]:.1f}°, {phi_ci[1]:.1f}°] (95% CI)")
    print(f"δt: {dt_median:.3f}s [{dt_ci[0]:.3f}s, {dt_ci[1]:.3f}s] (95% CI)")
    
    if use_tilt:
        print(f"tilt: {tilt_median:.1f}° [{tilt_ci[0]:.1f}°, {tilt_ci[1]:.1f}°] (95% CI)")

    result['binned'] = original_binned
    
    return result

'''
Publication Quality Figures of Individual Splitting Measurements and SI Curve
Drafted May 7 2026
'''

def publication_SI_curve(results, station_code, phases_allowed, s_depth=400, figsize=(24, 10), dpi=300, savepath=None):
   


    fig, ax = plt.subplots(figsize=figsize)
    
    # Use phase from station phase mapping
    data = results.get(phases_allowed)
    
    result_binned = data['result_binned']
    station_events = data['station_events']
    
    # Extract parameters
    phi_binned = result_binned['phi']
    dt_binned = result_binned['dt']

    # phi_binned = 56 this was done just to show a different fast axis angle for ARMA, but it is not correct 
    
    # Fit curve
    degs = np.arange(0, 360, 1)
    si_fit_binned = si_function_simple(degs, phi_binned, dt_binned)


    if station_code in ["AS31","SNZO","XMIS"]:
        ylim = math.ceil(abs(max(station_events['wiggleSI_calc'])))
    else:
        ylim = math.ceil(abs(max(station_events['wiggleSI_calc']))*2) / 2


    # Cyclic colormap for azimuth
    cmap = plt.cm.twilight  # twilight is cyclic (wraps at 360°)
    
    # ==================== PLOT ALL PHASES ====================
    sks_plotted = s_deep_plotted = s_shallow_plotted = False
    n_sks = n_deep = n_shallow = 0
    sks_scatter = None
    s_scatter = None
    
    for _, row in station_events.iterrows():
        if row['evphase'] == 'SKS':
            label = 'SKS phase' if not sks_plotted else None
            sks_plotted = True
            color = 'black'
            n_sks += 1
            
            # ax.plot(row['calcbaz'], row['wiggleSI_calc'], 'x', 
            #        color=color, alpha=0.7, markersize=10,markeredgewidth=3, label=label)

            sks_scatter = ax.scatter(
                row['calcbaz'],
                row['wiggleSI_calc'],
                c = color,
                # cmap=cmap,
                # vmin=0,
                # vmax=360,
                marker='o',
                s=100,
                edgecolors='black',
                linewidths=1,
                alpha=0.8,
                zorder=5
            )
            # ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
            #            yerr=row['SI_calc_error'], 
            #            ecolor=color, alpha=0.3, fmt='none')
            
        
        elif row['evphase'] == 'S':

            label = 'S Phase' if not s_deep_plotted else None
            s_deep_plotted = True
            color = 'white'

            # Color by depth
            if row['evdp'] >= s_depth:
                # label = 'Shallow' if not s_deep_plotted else None
                # s_deep_plotted = True
                # color = 'green'
                n_deep += 1
            else:
                # label = 'S shallow' if not s_shallow_plotted else None
                # s_shallow_plotted = True
                # color = 'red'
                n_shallow += 1
            
            # ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
            #        color=color, alpha=0.5, markersize=6, label=label)
            # ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
            #            yerr=row['SI_calc_error'], 
            #            ecolor=color, alpha=0.3, fmt='none')
            s_scatter = ax.scatter(
                row['calcbaz'],
                row['wiggleSI_calc'],
                c = row['evbaz'],
                cmap=cmap,
                vmin=0,
                vmax=360,
                marker='o',
                s=100,
                edgecolors='black',
                linewidths=1,
                alpha=0.8,
                zorder=5
            )

    if s_scatter is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.15)
        # cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar = plt.colorbar(s_scatter, cax=cax, orientation='vertical', 
                        pad=0.5, shrink=0.9, aspect=25)
        # cbar.ax.xaxis.set_label_position('bottom')
        cbar.set_label('Backazimuth (°)', fontsize=14,labelpad=10)
        cbar.set_ticks([0, 90, 180, 270, 360])

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0],
            marker='o',
            color='w',
            label='SKS phase',
            markerfacecolor='black',
            markeredgecolor='black',
            markersize=10),

        Line2D([0], [0],
            marker='o',
            color='w',
            label='S phase \n(color = BAZ)',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=10)
    ]

    ax.legend(handles=legend_elements,loc='upper right', fontsize=10)        
    
    # Label
    # binned_label = f'Fit: φ={phi_binned:.1f}°, δt={dt_binned:.2f}s'
    
    # Plot fit
    ax.plot(degs, si_fit_binned, '-', color='black', linewidth=3)
    
    binned = result_binned['binned']
    for _, row in binned.iterrows():
        # name = int(row['name'])
        bin_color = 'gray'
        # ax.plot(row['azimuth'], row['mean'], 'o', 
        #                 color=bin_color, markersize=10)
        ax.errorbar(row['azimuth'], row['mean'], yerr=1*row['std'], 
                            color=bin_color, fmt='none', alpha=0.75)
    
    # # Formatting
    n_total = n_sks + n_deep + n_shallow
    title_str = fr'$\phi$ = {round(phi_binned)}°, $\delta t$ = {round(dt_binned,1)}s'
    
    ax.set_title(title_str, fontsize=14)
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=14)
    ax.set_xticks(np.linspace(0,360,7))
    ax.set_ylabel('Splitting Intensity', fontsize=14)
    ax.set_ylim(-ylim,ylim)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.axhline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(-1, color='gray', linestyle='--', linewidth=0.8, alpha=0.3)

    # ax.legend(loc='upper right', fontsize=10)
    # ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{station_code} Splitting Intensity', fontsize=20, y=0.98)

    print(f"Fast Axis: {phi_binned}, Delay Time: {dt_binned}")
    
    # ===== SAVE WITH HIGH RESOLUTION =====
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    
    return fig, {
        'phi_binned': phi_binned,
        'dt_binned': dt_binned,
        'n_sks': n_sks,
        'n_deep': n_deep,
        'n_shallow': n_shallow,
        'n_total': n_total
    }