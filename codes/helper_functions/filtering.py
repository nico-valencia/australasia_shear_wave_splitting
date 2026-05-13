"""
filtering.py
Functions for filtering and preparing seismic event data.
"""
import numpy as np
import pandas as pd


def calculate_si_errors(event_data):
    """
    Calculate splitting intensity errors and add to dataframe.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event measurements with 'wiggleSI_calc' and 'wiggleSIlow_calc' columns
        
    Returns
    -------
    pd.DataFrame
        Copy of event_data with 'SI_calc_error' column added
    """
    df = event_data.copy()
    df['SI_calc_error'] = abs(df['wiggleSI_calc'] - df['wiggleSIlow_calc'])
    return df


def filter_events(event_data, 
                  epicentral_max=None,
                  epicentral_min=None,
                  depth_min=None,
                  depth_max=None,
                  snr_min=None,
                  snrbefore_min=None,
                  snrafter_min=None,
                  sta_lta_min=None,
                  mag_max=None,
                  ellipticity_min=None,
                  ellipticity_max=None,
                  si_error_max=None,
                  incident_max=None,
                  pierce_depth_min=None,
                  delta_travel_max=None):
    """
    Filter event data by quality criteria.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Raw event measurements
    epicentral_max : float, optional
        Maximum epicentral distance (degrees)
    depth_min, depth_max : float, optional
        Event depth range (km)
    snr_min : float, optional
        Minimum signal-to-noise ratio
    si_error_max : float, optional
        Maximum splitting intensity error
    ... (etc for other filters)
    
    Returns
    -------
    pd.DataFrame
        Filtered copy of event_data
    """
    df = event_data.copy()
    
    # Apply each filter if specified
    if epicentral_max is not None:
        df = df[df['evepic'] < epicentral_max]

    if epicentral_min is not None:
        df = df[df['evepic'] > epicentral_min]
    
    if depth_min is not None:
        df = df[df['evdp'] > depth_min]
        
    if depth_max is not None:
        df = df[df['evdp'] < depth_max]
    
    if snr_min is not None:
        df = df[df['snr'] > snr_min]
    
    if snrbefore_min is not None:
        df = df[df['snrbefore'] > snrbefore_min]
        
    if snrafter_min is not None:
        df = df[df['snrafter'] > snrafter_min]
    
    if sta_lta_min is not None:
        df = df[df['sta_lta'] > sta_lta_min]
    
    if mag_max is not None:
        df = df[df['evmg'] < mag_max]
    
    if ellipticity_min is not None:
        df = df[df['ellipticity'] > ellipticity_min]

    if ellipticity_max is not None:
        df = df[df['ellipticity'] < ellipticity_max]
    
    if si_error_max is not None:
        df = df[df['SI_calc_error'] < si_error_max]
        df = df[df['SI_calc_error'] != 0]
    
    if incident_max is not None:
        df = df[df['evincident'] < incident_max]
        
    if pierce_depth_min is not None:
        df = df[df['evpiercedepth'] >= pierce_depth_min]

    if delta_travel_max is not None:
        df = df[df['delta_travel'] <= delta_travel_max]
    
    return df


def filter_station_events(event_data, station_code):
    """
    Extract events for a single station and remove duplicates.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Full event catalog
    station_code : str
        Station code (e.g., 'BBOO')
        
    Returns
    -------
    pd.DataFrame
        Events for this station only, duplicates removed
    """
    df = event_data[event_data['evstation'] == station_code].copy()
    df = df.drop_duplicates(subset=['evtime', 'evphase'])
    return df


def prepare_azimuth_range(event_data, ravel=False):
    """
    Prepare back-azimuth columns and degree arrays.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event data with 'calcbaz' column
    ravel : bool
        If True, wrap azimuths to 0-180; if False, use 0-360
        
    Returns
    -------
    event_data : pd.DataFrame
        Modified copy with azimuth adjusted
    degs : np.ndarray
        Degree array for plotting (0-180 or 0-360)
    """
    df = event_data.copy()
    
    if ravel:
        df['calcbaz'] = df['calcbaz'] % 180
        degs = np.arange(0, 180, 1)
    else:
        degs = np.arange(0, 360, 1)
    
    return df, degs

def prepare_wolf_data(wolf_data):
    """
    Calculate back azimuths for Wolf et al. measurements.
    
    Parameters
    ----------
    wolf_data : pd.DataFrame
        Wolf measurements with columns: 'station_lat', 'station_lon', 
        'event_lat', 'event_lon', 'station_code'
        
    Returns
    -------
    pd.DataFrame
        Copy with 'backaz' column added
    """
    from obspy.geodetics import gps2dist_azimuth
    
    df = wolf_data.copy()
    
    df["backaz"] = df.apply(
        lambda row: gps2dist_azimuth(
            row["station_lat"], row["station_lon"],
            row["event_lat"], row["event_lon"]
        )[2],  # index 2 = back azimuth
        axis=1
    )
    
    return df