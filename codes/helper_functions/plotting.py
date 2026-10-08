"""
plotting.py
Plotting functions for splitting intensity analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from support import plot_great_circle_ray_path
from pyproj import Geod
from .analysis import si_function_simple,si_function_tilt
import matplotlib.path as mpath
import matplotlib.colors as mcolors



def plot_si_measurements(result, 
                         wolf_data=None,
                         s_depth=400,
                         figsize=(20, 8),
                         style='standard'):
    """
    Plot individual SI measurements with error bars and fitted curve.
    
    Parameters
    ----------
    result : dict
        Output from analyze_station()
    wolf_data : pd.DataFrame, optional
        Wolf et al. reference measurements for comparison
    s_depth : float
        Depth threshold for S-phase coloring
    figsize : tuple
        Figure size
    style : str
        'standard', 'publication', 'presentation'
        
    Returns
    -------
    fig, ax
    """
    station = result['station']
    event_data = result['event_data']
    phi = result['phi']
    dt = result['dt']
    tilt = result.get('tilt', None)
    fit = result['fit']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual measurements
    sks_plotted = False
    s_deep_plotted = False
    s_shallow_plotted = False
    
    for _, row in event_data.iterrows():
        if row['evphase'] == 'SKS':
            label = 'SKS phase' if not sks_plotted else None
            sks_plotted = True
            ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                   color='blue', alpha=0.5, label=label)
            ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                       yerr=row['SI_calc_error'], 
                       ecolor='blue', alpha=0.5, fmt='none')
        
        elif row['evphase'] == 'S':
            if row['evdp'] >= s_depth:
                label = 'S deep' if not s_deep_plotted else None
                s_deep_plotted = True
                ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                       color='green', alpha=0.5, label=label)
                ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                           yerr=row['SI_calc_error'], 
                           ecolor='green', alpha=0.5, fmt='none')
            else:
                label = 'S shallow' if not s_shallow_plotted else None
                s_shallow_plotted = True
                ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                       color='red', alpha=0.5, label=label)
                ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                           yerr=row['SI_calc_error'], 
                           ecolor='red', alpha=0.5, fmt='none')
    
    # Plot fitted curve
    degs = np.arange(0, 360, 1)
    if tilt is not None:
        from .analysis import si_function_tilt
        incident = np.full_like(degs, 20)  # assume 20° incident
        si_fit = si_function_tilt((degs, incident), *fit['params'])
    else:
        from .analysis import si_function_simple
        si_fit = si_function_simple(degs, *fit['params'])
    
    ax.plot(degs, si_fit, 'k-', linewidth=2, label='Fitted curve')
    
    # Wolf data if provided
    if wolf_data is not None:
        wolf_stat = wolf_data[wolf_data['station_code'] == station]
        if len(wolf_stat) > 0:
            ax.plot(wolf_stat['backaz'], wolf_stat['SI'], 'o', 
                   color='pink', label='Wolf et al.', markersize=8)
    
    # Styling
    title_str = f"{station}: φ={phi:.1f}°, δt={dt:.2f}s"
    if tilt is not None:
        title_str += f", tilt={tilt:.1f}°"
    
    ax.set_title(title_str, fontsize=14 if style == 'standard' else 20)
    ax.set_xlabel('Back Azimuth (°)', fontsize=12 if style == 'standard' else 16)
    ax.set_ylabel('Splitting Intensity', fontsize=12 if style == 'standard' else 16)
    ax.set_ylim(-3, 3)
    ax.legend(loc='upper right', fontsize=10 if style == 'standard' else 14)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_binned_measurements(result, 
                             wolf_data=None,
                             figsize=(20, 8)):
    """
    Plot binned SI measurements with error bars and fitted curve.
    
    Parameters
    ----------
    result : dict
        Output from analyze_station()
    wolf_data : pd.DataFrame, optional
        Wolf et al. reference measurements
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, ax
    """
    station = result['station']
    binned = result['binned']
    phi = result['phi']
    dt = result['dt']
    tilt = result.get('tilt', None)
    fit = result['fit']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot binned points by phase type
    colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}
    labels = {1: 'SKS', 2: 'S (all)', 3: 'S deep', 4: 'S shallow'}
    
    for _, row in binned.iterrows():
        name = int(row['name'])
        ax.plot(row['azimuth'], row['mean'], 'o', color=colors[name], 
               markersize=8, label=labels.get(name, ''))
        ax.errorbar(row['azimuth'], row['mean'], yerr=row['margin'], 
                   color=colors[name], fmt='none')
    
    # Remove duplicate legend entries
    handles, labels_list = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_list, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # Plot fitted curve
    degs = np.arange(0, 360, 1)
    if tilt is not None:
        from .analysis import si_function_tilt
        incident = np.full_like(degs, 20)
        si_fit = si_function_tilt((degs, incident), *fit['params'])
    else:
        from .analysis import si_function_simple
        si_fit = si_function_simple(degs, *fit['params'])
    
    ax.plot(degs, si_fit, 'k-', linewidth=3)
    
    # Wolf data
    if wolf_data is not None:
        wolf_stat = wolf_data[wolf_data['station_code'] == station]
        if len(wolf_stat) > 0:
            ax.plot(wolf_stat['backaz'], wolf_stat['SI'], 'o', 
                   color='pink', markersize=10, label='Wolf et al.')
    
    title_str = f"{station} Binned: φ={phi:.1f}°, δt={dt:.2f}s"
    if tilt is not None:
        title_str += f", tilt={tilt:.1f}°"
    
    ax.set_title(title_str, fontsize=14)
    ax.set_xlabel('Back Azimuth (°)', fontsize=12)
    ax.set_ylabel('Splitting Intensity', fontsize=12)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_station_map(result,s_depth=None, figsize=(10, 10)):
    """
    Plot ray paths on azimuthal equidistant map centered on station.
    
    Parameters
    ----------
    result : dict
        Output from analyze_station()
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, ax
    """
    event_data = result['event_data']
    station = result['station']
    
    # Station location (from first event)
    stat_lat = event_data['evstationlat'].iloc[0]
    stat_lon = event_data['evstationlon'].iloc[0]
    
    # Create map
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.AzimuthalEquidistant(
        central_longitude=stat_lon, 
        central_latitude=stat_lat
    ))
    
    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.gridlines(draw_labels=True)
    
    # Plot station
    ax.plot(stat_lon, stat_lat, '^', color='red', markersize=15, 
           transform=ccrs.PlateCarree(), label=station)
    
    # Plot events and ray paths
    # (You'll need to implement plot_great_circle_ray_path or similar)
    # For now, just plot event locations:
    for _, event in event_data.iterrows():
        color = 'blue' if event['evphase'] == 'SKS' else 'red'
        ax.plot(event['evlon'], event['evlat'], 'o', color=color, 
               markersize=4, alpha=0.6, transform=ccrs.PlateCarree())
        
    # # Plot ray paths for each event
    # for _, event in event_data.iterrows():
    #     ax = plot_great_circle_ray_path(
    #         event['evlat'], event['evlon'],
    #         stat_lat, stat_lon,
    #         phase_type=event['evphase'],
    #         ev_depth=event['evdp'],
    #         s_depth=s_depth,
    #         ax=ax
    #     )

    for _, event in event_data.iterrows():
        ax = plot_great_circle_ray_path(
            eq_lat=event['evlat'],
            eq_lon=event['evlon'],
            st_lat=stat_lat,
            st_lon=stat_lon,
            phase_type=event['evphase'],
            ev_depth=event['evdp'],
            s_depth=s_depth,  # <-- pass s_depth here
            ax=ax
        )
    
    ax.legend()
    plt.title(f"{station} Event Distribution", fontsize=14)
    
    return fig, ax


def plot_full_analysis(result, 
                      wolf_data=None, 
                      s_depth=400,
                      figsize=(20, 16)):
    """
    Create comprehensive multi-panel plot: measurements + binned + map.
    
    Parameters
    ----------
    result : dict
        Output from analyze_station()
    wolf_data : pd.DataFrame, optional
        Reference data
    s_depth : float
        Depth threshold
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig
    """
    station = result['station']
    event_data = result['event_data']
    phi = result['phi']
    dt = result['dt']
    tilt = result.get('tilt', None)
    
    # Create layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 8, figure=fig)
    
    ax1 = fig.add_subplot(gs[0:2, :4])  # Individual measurements
    ax2 = fig.add_subplot(gs[2:4, :4])  # Binned measurements
    
    # Station location
    stat_lat = event_data['evstationlat'].iloc[0]
    stat_lon = event_data['evstationlon'].iloc[0]
    
    ax3 = fig.add_subplot(gs[:, 4:], projection=ccrs.AzimuthalEquidistant(
        central_longitude=stat_lon, central_latitude=stat_lat
    ))
    
    # Panel 1: Individual measurements
    plot_si_on_axis(ax1, result, wolf_data, s_depth, show_individuals=True)
    
    # Panel 2: Binned measurements
    plot_binned_on_axis(ax2, result, wolf_data)
    
    # Panel 3: Map
    plot_map_on_axis(ax3, result)
    
    fig.tight_layout()
    
    return fig


def plot_si_on_axis(ax, result, wolf_data, s_depth, show_individuals=True):
    """Helper to plot SI measurements on given axis."""
    # (Similar to plot_si_measurements but draws on provided ax)
    # Implementation left as exercise or copy logic from plot_si_measurements
    pass


def plot_binned_on_axis(ax, result, wolf_data):
    """Helper to plot binned measurements on given axis."""
    pass


def plot_map_on_axis(ax, result):
    """Helper to plot map on given axis."""
    pass

def plot_si_by_distance_groups(event_data,
                               distance_threshold=60,
                               s_depth=400,
                               station_code=None,
                               figsize=(16, 6)):
    """
    Plot splitting intensity measurements split by epicentral distance.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog with columns: 'evepic', 'calcbaz', 'wiggleSI_calc', 
        'SI_calc_error', 'evphase', 'evdp'
    distance_threshold : float
        Epicentral distance threshold (degrees) to split near/far
    s_depth : float
        Depth threshold for S-phase coloring
    station_code : str, optional
        Station name for title
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, (ax1, ax2)
    """
    # Split data
    near = event_data[event_data['evepic'] < distance_threshold]
    far = event_data[event_data['evepic'] >= distance_threshold]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    # Helper to plot on an axis
    def plot_si_on_ax(ax, data, title_suffix):
        sks_plotted = s_deep_plotted = s_shallow_plotted = False
        
        for _, row in data.iterrows():
            if row['evphase'] == 'SKS':
                label = 'SKS' if not sks_plotted else None
                sks_plotted = True
                ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.',
                       color='blue', alpha=0.6, markersize=8, label=label)
                ax.errorbar(row['calcbaz'], row['wiggleSI_calc'],
                           yerr=row['SI_calc_error'],
                           ecolor='blue', alpha=0.4, fmt='none')
            
            elif row['evphase'] == 'S':
                if row['evdp'] >= s_depth:
                    label = 'S deep' if not s_deep_plotted else None
                    s_deep_plotted = True
                    color = 'green'
                else:
                    label = 'S shallow' if not s_shallow_plotted else None
                    s_shallow_plotted = True
                    color = 'red'
                
                ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.',
                       color=color, alpha=0.6, markersize=8, label=label)
                ax.errorbar(row['calcbaz'], row['wiggleSI_calc'],
                           yerr=row['SI_calc_error'],
                           ecolor=color, alpha=0.4, fmt='none')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('Back Azimuth (°)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        title = title_suffix
        if station_code:
            title = f"{station_code} - {title}"
        ax.set_title(title, fontsize=12)
    
    # Plot near events
    plot_si_on_ax(ax1, near, 
                  f'Near (< {distance_threshold}°, n={len(near)})')
    ax1.set_ylabel('Splitting Intensity', fontsize=11)
    
    # Plot far events
    plot_si_on_ax(ax2, far,
                  f'Far (≥ {distance_threshold}°, n={len(far)})')
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)



def plot_grouped_si_measurements(grouped_results,
                                wolf_data=None,
                                s_depth=400,
                                ncols=3,
                                figsize=(20, 14)):
    """
    Plot SI measurements for multiple station groups in a grid.
    
    Parameters
    ----------
    grouped_results : list of dict
        List of results from analyze_station_group()
    wolf_data : pd.DataFrame, optional
        Wolf measurements
    s_depth : float
        Depth threshold
    ncols : int
        Number of columns in subplot grid
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, axes
    """
    n_groups = len(grouped_results)
    nrows = int(np.ceil(n_groups / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for idx, result in enumerate(grouped_results):
        ax = axes[idx]
        
        event_data = result['event_data']
        phi = result['phi']
        dt = result['dt']
        tilt = result.get('tilt', None)
        fit = result['fit']

        # Plot individual measurements with error bars
        sks_plotted = s_deep_plotted = s_shallow_plotted = False
        
        for _, row in event_data.iterrows():
            if row['evphase'] == 'SKS':
                label = 'SKS' if not sks_plotted else None
                sks_plotted = True
                ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.',
                       color='blue', alpha=0.5, markersize=4, label=label)
                ax.errorbar(row['calcbaz'], row['wiggleSI_calc'],
                           yerr=row['SI_calc_error'],
                           ecolor='blue', alpha=0.3, fmt='none')
            
            elif row['evphase'] == 'S':
                if row['evdp'] >= s_depth:
                    label = 'S deep' if not s_deep_plotted else None
                    s_deep_plotted = True
                    color = 'green'
                else:
                    label = 'S shallow' if not s_shallow_plotted else None
                    s_shallow_plotted = True
                    color = 'red'
                
                ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.',
                       color=color, alpha=0.5, markersize=4, label=label)
                ax.errorbar(row['calcbaz'], row['wiggleSI_calc'],
                           yerr=row['SI_calc_error'],
                           ecolor=color, alpha=0.3, fmt='none')
        
        # # Plot individual measurements
        # sks_plotted = s_deep_plotted = s_shallow_plotted = False
        
        # for _, row in event_data.iterrows():
        #     if row['evphase'] == 'SKS':
        #         label = 'SKS' if not sks_plotted else None
        #         sks_plotted = True
        #         ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.',
        #                color='blue', alpha=0.4, markersize=4, label=label)
            
        #     elif row['evphase'] == 'S':
        #         if row['evdp'] >= s_depth:
        #             label = 'S deep' if not s_deep_plotted else None
        #             s_deep_plotted = True
        #             color = 'green'
        #         else:
        #             label = 'S shallow' if not s_shallow_plotted else None
        #             s_shallow_plotted = True
        #             color = 'red'
                
        #         ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.',
        #                color=color, alpha=0.4, markersize=4, label=label)
        
        # Plot fitted curve
        degs = np.arange(0, 360, 1)
        if tilt is not None:
            incident = np.full_like(degs, 20)
            si_fit = si_function_tilt((degs, incident), *fit['params'])
        else:
            si_fit = si_function_simple(degs, *fit['params'])
        
        ax.plot(degs, si_fit, 'k-', linewidth=2)
        
        # Styling
        title = (f"{result['n_stations']} stations\n"
                f"φ={phi:.1f}°, δt={dt:.2f}s")
        if tilt is not None:
            title += f", tilt={tilt:.1f}°"
        
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        if idx % ncols == 0:  # first column
            ax.set_ylabel('SI', fontsize=9)
        if idx >= (nrows - 1) * ncols:  # bottom row
            ax.set_xlabel('Azimuth (°)', fontsize=9)
        
        if idx == 0:  # legend only on first plot
            ax.legend(fontsize=7, loc='upper right')
    
    # Hide unused subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Regional Splitting Intensity (Grouped Stations)', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    return fig, axes


def plot_station_group_map(grouped_results, figsize=(14, 10)):
    """
    Plot map showing station groups and their splitting results.
    
    Parameters
    ----------
    grouped_results : list of dict
        Results from analyze_station_group()
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, ax
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    # Plot each group
    for result in grouped_results:
        center_lat = result['center_lat']
        center_lon = result['center_lon']
        phi = result['phi']
        dt = result['dt']
        n_stations = result['n_stations']
        
        # Plot group center as a circle sized by number of stations
        size = 100 + n_stations * 20
        ax.plot(center_lon, center_lat, 'o', 
               markersize=np.sqrt(size), 
               color='red', alpha=0.6,
               transform=ccrs.PlateCarree())
        
        # Annotate with phi
        ax.text(center_lon, center_lat, f"{phi:.0f}°",
               fontsize=8, ha='center', va='center',
               transform=ccrs.PlateCarree(),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Draw grid box
        lat_range = result['group_id'][0]
        lon_range = result['group_id'][1]
        
        box_lons = [lon_range[0], lon_range[1], lon_range[1], lon_range[0], lon_range[0]]
        box_lats = [lat_range[0], lat_range[0], lat_range[1], lat_range[1], lat_range[0]]
        
        ax.plot(box_lons, box_lats, 'k-', linewidth=1, alpha=0.3,
               transform=ccrs.PlateCarree())
    
    ax.set_title('Station Groups and Fast Directions', fontsize=14)
    
    return fig, ax

def plot_si_individual_measurements(event_data, 
                                   station_code=None,
                                   incident_angle=20,
                                   use_tilt=False,
                                   s_depth=400,
                                   figsize=(20, 8),
                                   wolf_data=None):
    """
    Fit and plot individual SI measurements (no binning).
    
    Automatically performs SI fit to individual measurements and plots.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Individual measurements (already filtered to station if needed)
    station_code : str, optional
        Station name for title
    incident_angle : float
        Assumed incident angle (degrees)
    use_tilt : bool
        If True, fit 3-parameter model
    s_depth : float
        Depth threshold for S-phase coloring
    figsize : tuple
        Figure size
    wolf_data : pd.DataFrame, optional
        Wolf et al. reference data
        
    Returns
    -------
    fig, ax
    """
    from .analysis import fit_splitting_intensity_individual, si_function_simple, si_function_tilt
    
    # Perform the fit
    fit_result = fit_splitting_intensity_individual(
        event_data,
        incident_angle=incident_angle,
        use_tilt=use_tilt
    )
    
    phi = fit_result['phi']
    dt = fit_result['dt']
    tilt = fit_result.get('tilt', None)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual measurements by phase type
    sks_plotted = s_deep_plotted = s_shallow_plotted = False
    
    for _, row in event_data.iterrows():
        if row['evphase'] == 'SKS':
            label = 'SKS phase' if not sks_plotted else None
            sks_plotted = True
            ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                   color='blue', alpha=0.5, markersize=6, label=label)
            ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                       yerr=row['SI_calc_error'], 
                       ecolor='blue', alpha=0.5, fmt='none')
        
        elif row['evphase'] == 'S':
            if row['evdp'] >= s_depth:
                label = 'S deep' if not s_deep_plotted else None
                s_deep_plotted = True
                color = 'green'
            else:
                label = 'S shallow' if not s_shallow_plotted else None
                s_shallow_plotted = True
                color = 'red'
            
            ax.plot(row['calcbaz'], row['wiggleSI_calc'], '.', 
                   color=color, alpha=0.5, markersize=6, label=label)
            ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
                       yerr=row['SI_calc_error'], 
                       ecolor=color, alpha=0.5, fmt='none')
    
    # Plot fitted curve
    degs = np.arange(0, 360, 1)
    if tilt is not None:
        incident = np.full_like(degs, incident_angle)
        si_fit = si_function_tilt((degs, incident), *fit_result['params'])
    else:
        si_fit = si_function_simple(degs, *fit_result['params'])
    
    ax.plot(degs, si_fit, 'k-', linewidth=2, label='Fitted curve')
    
    # Wolf data if provided
    if wolf_data is not None and station_code is not None:
        wolf_stat = wolf_data[wolf_data['station_code'] == station_code]
        if len(wolf_stat) > 0:
            ax.plot(wolf_stat['backaz'], wolf_stat['SI'], 'o', 
                   color='pink', label='Wolf et al.', markersize=8)
    
    # Styling
    title_str = f"Individual Measurements: φ={phi:.1f}°, δt={dt:.2f}s"
    if tilt is not None:
        title_str += f", tilt={tilt:.1f}°"
    if station_code:
        title_str = f"{station_code} - {title_str}"
    
    ax.set_title(title_str, fontsize=14)
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=12)
    ax.set_ylabel('Splitting Intensity', fontsize=12)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def plot_s_phases_by_depth(event_data, station_code=None, s_depth=400, 
                          figsize=(20, 8), cmap='viridis'):
    """
    Plot S-phase SI measurements colored by event depth.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog (can be pre-filtered to station)
    station_code : str, optional
        Station name for title
    s_depth : float
        Reference depth line
    figsize : tuple
        Figure size
    cmap : str
        Colormap for depth ('viridis', 'plasma', 'cool', 'RdYlGn_r')
        
    Returns
    -------
    fig, ax
    """
    # Filter to S phases only
    s_data = event_data[event_data['evphase'] == 'S'].copy()
    
    if len(s_data) == 0:
        print("No S-phase data found")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot colored by depth
    scatter = ax.scatter(s_data['calcbaz'], s_data['wiggleSI_calc'],
                        c=s_data['evdp'], 
                        cmap=cmap,
                        s=60, 
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5)
    
    # Error bars
    ax.errorbar(s_data['calcbaz'], s_data['wiggleSI_calc'],
               yerr=s_data['SI_calc_error'],
               fmt='none', ecolor='gray', alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Event Depth (km)', fontsize=12)
    
    # Reference line for s_depth
    # Add text annotation instead of trying to draw on colorbar
    ax.text(0.98, 0.02, f'Deep S threshold: {s_depth} km', 
           transform=ax.transAxes, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
           fontsize=10)
    
    # Styling
    title = 'S-Phase Measurements Colored by Event Depth'
    if station_code:
        title = f'{station_code} - {title}'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=12)
    ax.set_ylabel('Splitting Intensity', fontsize=12)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # Add stats
    n_deep = (s_data['evdp'] >= s_depth).sum()
    n_shallow = (s_data['evdp'] < s_depth).sum()
    ax.text(0.02, 0.98, f'N deep (≥{s_depth}km): {n_deep}\nN shallow (<{s_depth}km): {n_shallow}',
           transform=ax.transAxes, va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax


def plot_s_phases_by_backazimuth(event_data, station_code=None, s_depth=400,
                                 figsize=(20, 8), cmap='hsv'):
    """
    Plot S-phase SI measurements colored by TRUE back-azimuth (source direction).
    
    This helps visualize how polarization azimuth differs from back-azimuth.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog
    station_code : str, optional
        Station name
    s_depth : float
        Depth threshold for filtering
    cmap : str
        Colormap ('hsv' wraps at 360°, 'twilight' also good)
        
    Returns
    -------
    fig, ax
    """
    # Filter to S phases only
    s_data = event_data[event_data['evphase'] == 'S'].copy()
    
    if len(s_data) == 0:
        print("No S-phase data found")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot colored by TRUE back-azimuth
    scatter = ax.scatter(s_data['calcbaz'], s_data['wiggleSI_calc'],
                        c=s_data['evbaz'],  # Color by TRUE back-azimuth
                        cmap=cmap,
                        vmin=0, vmax=360,
                        s=60, 
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5)
    
    # Error bars
    ax.errorbar(s_data['calcbaz'], s_data['wiggleSI_calc'],
               yerr=s_data['SI_calc_error'],
               fmt='none', ecolor='gray', alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('True Back-Azimuth (°)', fontsize=12)
    
    # Styling
    title = 'S-Phase: Polarization Azimuth vs SI (Colored by True Back-Azimuth)'
    if station_code:
        title = f'{station_code} - {title}'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Polarization Azimuth (calcbaz, °)', fontsize=12)
    ax.set_ylabel('Splitting Intensity', fontsize=12)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # Show depth breakdown
    n_deep = (s_data['evdp'] >= s_depth).sum()
    n_shallow = (s_data['evdp'] < s_depth).sum()
    ax.text(0.02, 0.98, f'Total S phases: {len(s_data)}\nDeep (≥{s_depth}km): {n_deep}\nShallow: {n_shallow}',
           transform=ax.transAxes, va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax


def plot_s_phases_dual_view(event_data, station_code=None, s_depth=400, figsize=(20, 16)):
    """
    Two-panel plot: top = depth coloring, bottom = back-azimuth coloring.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog
    station_code : str, optional
        Station name
    s_depth : float
        Depth threshold
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, (ax1, ax2)
    """
    # Filter to S phases
    s_data = event_data[event_data['evphase'] == 'S'].copy()
    
    if len(s_data) == 0:
        print("No S-phase data found")
        return None, None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # ========== Panel 1: Colored by Depth ==========
    scatter1 = ax1.scatter(s_data['calcbaz'], s_data['wiggleSI_calc'],
                          c=s_data['evdp'],
                          cmap='viridis',
                          s=60,
                          alpha=0.7,
                          edgecolors='black',
                          linewidths=0.5)
    
    ax1.errorbar(s_data['calcbaz'], s_data['wiggleSI_calc'],
                yerr=s_data['SI_calc_error'],
                fmt='none', ecolor='gray', alpha=0.3)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Event Depth (km)', fontsize=12)
    
    ax1.set_ylabel('Splitting Intensity', fontsize=12)
    ax1.set_ylim(-3, 3)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 360)
    
    title1 = 'S-Phases Colored by Event Depth'
    if station_code:
        title1 = f'{station_code} - {title1}'
    ax1.set_title(title1, fontsize=14)
    
    # ========== Panel 2: Colored by True Back-Azimuth ==========
    scatter2 = ax2.scatter(s_data['calcbaz'], s_data['wiggleSI_calc'],
                          c=s_data['evbaz'],
                          cmap='hsv',
                          vmin=0, vmax=360,
                          s=60,
                          alpha=0.7,
                          edgecolors='black',
                          linewidths=0.5)
    
    ax2.errorbar(s_data['calcbaz'], s_data['wiggleSI_calc'],
                yerr=s_data['SI_calc_error'],
                fmt='none', ecolor='gray', alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('True Back-Azimuth (°)', fontsize=12)
    
    ax2.set_xlabel('Polarization Azimuth (calcbaz, °)', fontsize=12)
    ax2.set_ylabel('Splitting Intensity', fontsize=12)
    ax2.set_ylim(-3, 3)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)
    
    ax2.set_title('S-Phases Colored by True Back-Azimuth (Source Direction)', fontsize=14)
    
    # Overall stats
    n_deep = (s_data['evdp'] >= s_depth).sum()
    fig.suptitle(f'S-Phase Analysis: {len(s_data)} measurements ({n_deep} deep, {len(s_data)-n_deep} shallow)',
                fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def plot_polarization_vs_backazimuth_offset(event_data, station_code=None, 
                                            s_depth=400, figsize=(16, 6)):
    """
    Show the difference between polarization azimuth and back-azimuth for S phases.
    This is the key to your method - S phases have calcbaz ≠ evbaz!
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog
    station_code : str, optional
        Station name
    s_depth : float
        Depth threshold
        
    Returns
    -------
    fig, (ax1, ax2)
    """
    # Get S phases
    s_data = event_data[event_data['evphase'] == 'S'].copy()
    
    if len(s_data) == 0:
        return None, None
    
    # Calculate offset
    s_data['baz_offset'] = (s_data['calcbaz'] - s_data['evbaz']) % 360
    # Wrap to -180 to 180
    s_data['baz_offset'] = np.where(s_data['baz_offset'] > 180, 
                                     s_data['baz_offset'] - 360, 
                                     s_data['baz_offset'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== Panel 1: Offset colored by depth ==========
    deep = s_data[s_data['evdp'] >= s_depth]
    shallow = s_data[s_data['evdp'] < s_depth]
    
    scatter1 = ax1.scatter(shallow['evbaz'], shallow['baz_offset'],
                          c=shallow['evdp'],
                          cmap='Reds',
                          s=50, alpha=0.7,
                          edgecolors='black', linewidths=0.5,
                          vmin=0, vmax=s_depth,
                          label=f'Shallow (<{s_depth}km)')
    
    scatter2 = ax1.scatter(deep['evbaz'], deep['baz_offset'],
                          c=deep['evdp'],
                          cmap='Greens',
                          s=50, alpha=0.7,
                          edgecolors='black', linewidths=0.5,
                          vmin=s_depth, vmax=deep['evdp'].max(),
                          label=f'Deep (≥{s_depth}km)')
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('True Back-Azimuth (°)', fontsize=12)
    ax1.set_ylabel('Polarization - Back-Azimuth Offset (°)', fontsize=12)
    ax1.set_title('S-Phase: How Much Does Polarization Differ from Back-Azimuth?', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Colorbar for depth
    cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.02, aspect=30)
    cbar1.set_label('Shallow Depth (km)', fontsize=10)
    if len(deep) > 0:
        cbar2 = plt.colorbar(scatter2, ax=ax1, pad=0.08, aspect=30)
        cbar2.set_label('Deep Depth (km)', fontsize=10)
    
    # ========== Panel 2: Histogram of offsets ==========
    ax2.hist(shallow['baz_offset'], bins=36, range=(-180, 180), 
            alpha=0.6, color='red', edgecolor='black', label=f'Shallow (n={len(shallow)})')
    ax2.hist(deep['baz_offset'], bins=36, range=(-180, 180),
            alpha=0.6, color='green', edgecolor='black', label=f'Deep (n={len(deep)})')
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Polarization - Back-Azimuth Offset (°)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Polarization Offsets', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    title = f'S-Phase Polarization Analysis (N={len(s_data)})'
    if station_code:
        title = f'{station_code} - {title}'
    fig.suptitle(title, fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def plot_s_phases_deep_only(event_data, station_code=None, s_depth=400, 
                            color_by='depth', figsize=(20, 8)):
    """
    Plot ONLY deep S-phases (>= s_depth) with color coding.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog
    station_code : str, optional
        Station name
    s_depth : float
        Minimum depth to include
    color_by : str
        'depth' - color by event depth
        'backazimuth' - color by true back-azimuth
        'distance' - color by epicentral distance
        
    Returns
    -------
    fig, ax
    """
    # Filter to deep S phases only
    deep_s = event_data[(event_data['evphase'] == 'S') & (event_data['evdp'] >= s_depth)].copy()
    
    if len(deep_s) == 0:
        print(f"No deep S-phase data (>={s_depth}km) found")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose color variable and colormap
    if color_by == 'depth':
        color_var = deep_s['evdp']
        cmap = 'viridis'
        cbar_label = 'Event Depth (km)'
        vmin, vmax = deep_s['evdp'].min(), deep_s['evdp'].max()
    elif color_by == 'backazimuth':
        color_var = deep_s['evbaz']
        cmap = 'hsv'
        cbar_label = 'True Back-Azimuth (°)'
        vmin, vmax = 0, 360
    elif color_by == 'distance':
        color_var = deep_s['evepic']
        cmap = 'plasma'
        cbar_label = 'Epicentral Distance (°)'
        vmin, vmax = deep_s['evepic'].min(), deep_s['evepic'].max()
    else:
        raise ValueError("color_by must be 'depth', 'backazimuth', or 'distance'")
    
    # Scatter plot
    scatter = ax.scatter(deep_s['calcbaz'], deep_s['wiggleSI_calc'],
                        c=color_var,
                        cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        s=80,
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5)
    
    # Error bars
    ax.errorbar(deep_s['calcbaz'], deep_s['wiggleSI_calc'],
               yerr=deep_s['SI_calc_error'],
               fmt='none', ecolor='gray', alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    
    # Styling
    title = f'Deep S-Phases Only (≥{s_depth}km, N={len(deep_s)}) - Colored by {color_by.title()}'
    if station_code:
        title = f'{station_code} - {title}'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=12)
    ax.set_ylabel('Splitting Intensity', fontsize=12)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # Stats
    depth_range = f"{deep_s['evdp'].min():.0f}-{deep_s['evdp'].max():.0f} km"
    ax.text(0.98, 0.02, f'Depth range: {depth_range}\nMedian depth: {deep_s["evdp"].median():.0f} km',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax


def plot_s_phase_source_distribution(event_data, station_code=None, s_depth=400, 
                                     figsize=(18, 8)):
    """
    3-panel view: SI vs polarization, polarization vs back-az, polar histogram.
    
    Shows source distribution for S phases.
    
    Returns
    -------
    fig, (ax1, ax2, ax3)
    """
    from matplotlib.projections import polar
    
    s_data = event_data[event_data['evphase'] == 'S'].copy()
    
    if len(s_data) == 0:
        return None, None
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1])
    
    # ========== Panel 1: SI vs Polarization (colored by depth) ==========
    ax1 = fig.add_subplot(gs[0])
    
    deep = s_data[s_data['evdp'] >= s_depth]
    shallow = s_data[s_data['evdp'] < s_depth]
    
    if len(shallow) > 0:
        ax1.scatter(shallow['calcbaz'], shallow['wiggleSI_calc'],
                   c=shallow['evdp'], cmap='Reds',
                   s=50, alpha=0.6, edgecolors='black', linewidths=0.5,
                   vmin=0, vmax=s_depth, label=f'Shallow (<{s_depth}km)')
    
    if len(deep) > 0:
        ax1.scatter(deep['calcbaz'], deep['wiggleSI_calc'],
                   c=deep['evdp'], cmap='Greens',
                   s=50, alpha=0.6, edgecolors='black', linewidths=0.5,
                   vmin=s_depth, vmax=deep['evdp'].max(), label=f'Deep (≥{s_depth}km)')
    
    ax1.set_xlabel('Polarization Azimuth (°)', fontsize=11)
    ax1.set_ylabel('Splitting Intensity', fontsize=11)
    ax1.set_ylim(-3, 3)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 360)
    
    # ========== Panel 2: Polarization vs Back-Azimuth ==========
    ax2 = fig.add_subplot(gs[1])
    
    # 1:1 line
    ax2.plot([0, 360], [0, 360], 'k--', linewidth=1, alpha=0.5, label='Pol = BAz (SKS)')
    
    if len(shallow) > 0:
        ax2.scatter(shallow['evbaz'], shallow['calcbaz'],
                   c='red', s=50, alpha=0.6, edgecolors='black', linewidths=0.5,
                   label=f'Shallow')
    
    if len(deep) > 0:
        ax2.scatter(deep['evbaz'], deep['calcbaz'],
                   c='green', s=50, alpha=0.6, edgecolors='black', linewidths=0.5,
                   label=f'Deep')
    
    ax2.set_xlabel('True Back-Azimuth (°)', fontsize=11)
    ax2.set_ylabel('Polarization Azimuth (°)', fontsize=11)
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, 360)
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Polarization vs Back-Azimuth\n(Scatter = focal mechanism effects)', fontsize=12)
    
    # ========== Panel 3: Polar histogram of back-azimuths ==========
    ax3 = fig.add_subplot(gs[2], projection='polar')
    
    # Convert to radians
    baz_rad_shallow = np.deg2rad(shallow['evbaz']) if len(shallow) > 0 else []
    baz_rad_deep = np.deg2rad(deep['evbaz']) if len(deep) > 0 else []
    
    bins = np.linspace(0, 2*np.pi, 37)
    
    if len(shallow) > 0:
        ax3.hist(baz_rad_shallow, bins=bins, alpha=0.6, color='red', label='Shallow')
    if len(deep) > 0:
        ax3.hist(baz_rad_deep, bins=bins, alpha=0.6, color='green', label='Deep')
    
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_title('Source Distribution\n(Back-Azimuth)', fontsize=12, pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Overall
    title = f'S-Phase Source Analysis'
    if station_code:
        title = f'{station_code} - {title}'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3)

def plot_s_phases_by_incidence(event_data, station_code=None, s_depth=400,
                               figsize=(20, 8), cmap='coolwarm'):
    """
    Plot S-phase SI measurements colored by incidence angle.
    
    Steeper rays (smaller incidence) sample deeper structure differently.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog
    station_code : str, optional
        Station name
    s_depth : float
        Depth threshold for annotation
    cmap : str
        Colormap ('coolwarm', 'RdYlBu_r', 'viridis')
        
    Returns
    -------
    fig, ax
    """
    # Filter to S phases with incidence data
    s_data = event_data[(event_data['evphase'] == 'S') & 
                        (event_data['evincident'].notna())].copy()
    
    if len(s_data) == 0:
        print("No S-phase data with incidence angle found")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot colored by incidence angle
    scatter = ax.scatter(s_data['calcbaz'], s_data['wiggleSI_calc'],
                        c=s_data['evincident'],
                        cmap=cmap,
                        s=60,
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5,
                        vmin=0, vmax=40)  # Typical range for teleseismic
    
    # Error bars
    ax.errorbar(s_data['calcbaz'], s_data['wiggleSI_calc'],
               yerr=s_data['SI_calc_error'],
               fmt='none', ecolor='gray', alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Incidence Angle (°)', fontsize=12)
    
    # Styling
    title = 'S-Phases Colored by Incidence Angle (steeper rays = smaller angle)'
    if station_code:
        title = f'{station_code} - {title}'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=12)
    ax.set_ylabel('Splitting Intensity', fontsize=12)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # Stats
    median_inc = s_data['evincident'].median()
    ax.text(0.02, 0.98, 
           f'N S phases: {len(s_data)}\n'
           f'Incidence range: {s_data["evincident"].min():.1f}°-{s_data["evincident"].max():.1f}°\n'
           f'Median: {median_inc:.1f}°',
           transform=ax.transAxes, va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax


def plot_s_phases_by_pierce_depth(event_data, station_code=None, s_depth=400,
                                  figsize=(20, 8), cmap='viridis_r'):
    """
    Plot S-phase SI measurements colored by pierce point depth.
    
    Shows which depth range the anisotropy measurement samples.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog with 'evpiercedepth' column
    station_code : str, optional
        Station name
    s_depth : float
        Reference depth
    cmap : str
        Colormap ('viridis_r' = shallow=yellow, deep=purple)
        
    Returns
    -------
    fig, ax
    """
    # Filter to S phases with pierce depth data
    s_data = event_data[(event_data['evphase'] == 'S') & 
                        (event_data['evpiercedepth'].notna())].copy()
    
    if len(s_data) == 0:
        print("No S-phase data with pierce depth found")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot colored by pierce depth
    scatter = ax.scatter(s_data['calcbaz'], s_data['wiggleSI_calc'],
                        c=s_data['evpiercedepth'],
                        cmap=cmap,
                        s=60,
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5)
    
    # Error bars
    ax.errorbar(s_data['calcbaz'], s_data['wiggleSI_calc'],
               yerr=s_data['SI_calc_error'],
               fmt='none', ecolor='gray', alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pierce Point Depth (km)', fontsize=12)
    
    # Mark common discontinuity depths
    for depth_label, depth_km in [('410', 410), ('660', 660)]:
        if s_data['evpiercedepth'].min() < depth_km < s_data['evpiercedepth'].max():
            cbar.ax.axhline(depth_km, color='red', linestyle='--', linewidth=1, alpha=0.7)
            cbar.ax.text(0.5, depth_km, depth_label, color='red', fontsize=8)
    
    # Styling
    title = 'S-Phases Colored by Pierce Point Depth (deepest sampling point)'
    if station_code:
        title = f'{station_code} - {title}'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=12)
    ax.set_ylabel('Splitting Intensity', fontsize=12)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # Stats
    ax.text(0.02, 0.98,
           f'N S phases: {len(s_data)}\n'
           f'Pierce depth range: {s_data["evpiercedepth"].min():.0f}-{s_data["evpiercedepth"].max():.0f} km\n'
           f'Median: {s_data["evpiercedepth"].median():.0f} km',
           transform=ax.transAxes, va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax


def plot_s_phase_parameter_comparison(event_data, station_code=None, s_depth=400,
                                     figsize=(24, 16)):
    """
    6-panel comprehensive view of S-phase parameters.
    
    Shows: depth, back-azimuth, incidence, pierce depth, distance, and SI_error.
    
    Returns
    -------
    fig, axes
    """
    s_data = event_data[event_data['evphase'] == 'S'].copy()
    
    if len(s_data) == 0:
        return None, None
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Define parameters to plot
    params = [
        ('evdp', 'Event Depth (km)', 'viridis', None, None),
        ('evbaz', 'True Back-Azimuth (°)', 'hsv', 0, 360),
        ('evincident', 'Incidence Angle (°)', 'coolwarm', 0, 40),
        ('evpiercedepth', 'Pierce Depth (km)', 'viridis_r', None, None),
        ('evepic', 'Epicentral Distance (°)', 'plasma', None, None),
        ('SI_calc_error', 'SI Error', 'Reds', 0, 1)
    ]
    
    for idx, (param, label, cmap, vmin, vmax) in enumerate(params):
        ax = axes[idx]
        
        # Skip if column doesn't exist or all NaN
        if param not in s_data.columns or s_data[param].isna().all():
            ax.text(0.5, 0.5, f'{label}\nNo Data', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        # Filter out NaN
        plot_data = s_data[s_data[param].notna()]
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, f'{label}\nNo Data', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        # Auto-set vmin/vmax if not specified
        if vmin is None:
            vmin = plot_data[param].min()
        if vmax is None:
            vmax = plot_data[param].max()
        
        # Scatter plot
        scatter = ax.scatter(plot_data['calcbaz'], plot_data['wiggleSI_calc'],
                           c=plot_data[param],
                           cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           s=50,
                           alpha=0.7,
                           edgecolors='black',
                           linewidths=0.5)
        
        # Error bars
        ax.errorbar(plot_data['calcbaz'], plot_data['wiggleSI_calc'],
                   yerr=plot_data['SI_calc_error'],
                   fmt='none', ecolor='gray', alpha=0.2)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(label, fontsize=10)
        
        # Mark s_depth threshold if showing depth
        if param == 'evdp':
            cbar.ax.axhline(s_depth, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7)
            cbar.ax.text(0.5, s_depth, f'{s_depth}km', 
                        color='red', fontsize=8, ha='center')
        
        # Mark discontinuities if showing pierce depth
        if param == 'evpiercedepth':
            for depth_name, depth_val in [('410', 410), ('660', 660)]:
                if vmin < depth_val < vmax:
                    cbar.ax.axhline(depth_val, color='red', linestyle='--', 
                                   linewidth=1, alpha=0.7)
                    cbar.ax.text(0.5, depth_val, depth_name, 
                                color='red', fontsize=7, ha='center')
        
        # Styling
        ax.set_xlabel('Polarization Azimuth (°)', fontsize=11)
        ax.set_ylabel('Splitting Intensity', fontsize=11)
        ax.set_ylim(-3, 3)
        ax.set_xlim(0, 360)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Colored by {label}', fontsize=12)
    
    # Overall title
    title = f'S-Phase Parameter Comparison (N={len(s_data)})'
    if station_code:
        title = f'{station_code} - {title}'
    fig.suptitle(title, fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    return fig, axes


def plot_parameter_vs_si_quality(event_data, parameter='evincident', 
                                 station_code=None, figsize=(16, 12)):
    """
    Detailed analysis of how one parameter relates to SI quality.
    
    4-panel view: parameter vs |SI|, parameter vs SI_error, 
                  parameter distribution, parameter vs correlation
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog
    parameter : str
        Column to analyze: 'evincident', 'evpiercedepth', 'evdp', 'evepic'
    station_code : str, optional
        Station name
        
    Returns
    -------
    fig, axes
    """
    # Filter to S phases with valid data
    s_data = event_data[(event_data['evphase'] == 'S') & 
                        (event_data[parameter].notna())].copy()
    
    if len(s_data) == 0:
        print(f"No S-phase data with {parameter}")
        return None, None
    
    # Get parameter label
    param_labels = {
        'evincident': 'Incidence Angle (°)',
        'evpiercedepth': 'Pierce Point Depth (km)',
        'evdp': 'Event Depth (km)',
        'evepic': 'Epicentral Distance (°)',
        'evtakeoff': 'Takeoff Angle (°)',
        'evrayparam': 'Ray Parameter (s/°)'
    }
    param_label = param_labels.get(parameter, parameter)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ========== Panel 1: Parameter vs |SI| ==========
    ax = axes[0, 0]
    
    # Color by depth
    deep = s_data[s_data['evdp'] >= 400]
    shallow = s_data[s_data['evdp'] < 400]
    
    if len(shallow) > 0:
        ax.scatter(shallow[parameter], np.abs(shallow['wiggleSI_calc']),
                  c='red', s=40, alpha=0.6, edgecolors='black', 
                  linewidths=0.5, label='Shallow S')
    
    if len(deep) > 0:
        ax.scatter(deep[parameter], np.abs(deep['wiggleSI_calc']),
                  c='green', s=40, alpha=0.6, edgecolors='black',
                  linewidths=0.5, label='Deep S')
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('|SI|', fontsize=12)
    ax.set_title(f'{param_label} vs SI Magnitude', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(s_data) > 5:
        z = np.polyfit(s_data[parameter], np.abs(s_data['wiggleSI_calc']), 1)
        p = np.poly1d(z)
        x_range = np.linspace(s_data[parameter].min(), s_data[parameter].max(), 100)
        ax.plot(x_range, p(x_range), 'k--', linewidth=2, alpha=0.5, label='Trend')
        
        corr = np.corrcoef(s_data[parameter], np.abs(s_data['wiggleSI_calc']))[0, 1]
        ax.text(0.98, 0.02, f'r = {corr:.3f}',
               transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # ========== Panel 2: Parameter vs SI_error ==========
    ax = axes[0, 1]
    
    ax.scatter(s_data[parameter], s_data['SI_calc_error'],
              c=s_data['evdp'], cmap='RdYlGn_r',
              s=40, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Quality threshold')
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('SI_calc_error', fontsize=12)
    ax.set_title(f'{param_label} vs Measurement Uncertainty', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Trend
    if len(s_data) > 5:
        z = np.polyfit(s_data[parameter], s_data['SI_calc_error'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(s_data[parameter].min(), s_data[parameter].max(), 100)
        ax.plot(x_range, p(x_range), 'k--', linewidth=2, alpha=0.5)
        
        corr = np.corrcoef(s_data[parameter], s_data['SI_calc_error'])[0, 1]
        ax.text(0.98, 0.98, f'r = {corr:.3f}',
               transform=ax.transAxes, ha='right', va='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # ========== Panel 3: Distribution ==========
    ax = axes[1, 0]
    
    # Separate histograms for good vs bad SI
    good_si = s_data[s_data['SI_calc_error'] < 0.5]
    bad_si = s_data[s_data['SI_calc_error'] >= 0.5]
    
    ax.hist(good_si[parameter], bins=20, alpha=0.6, color='green',
           edgecolor='black', label=f'Good SI (error<0.5, n={len(good_si)})')
    ax.hist(bad_si[parameter], bins=20, alpha=0.6, color='red',
           edgecolor='black', label=f'Bad SI (error≥0.5, n={len(bad_si)})')
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{param_label} Distribution by Quality', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Stats
    ax.text(0.02, 0.98,
           f'Good median: {good_si[parameter].median():.1f}\n'
           f'Bad median: {bad_si[parameter].median():.1f}',
           transform=ax.transAxes, va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== Panel 4: Binned average SI vs parameter ==========
    ax = axes[1, 1]
    
    # Bin parameter into groups
    param_bins = pd.qcut(s_data[parameter], q=min(10, len(s_data)//5), duplicates='drop')
    binned_stats = s_data.groupby(param_bins).agg({
        'wiggleSI_calc': ['mean', 'std', 'count'],
        parameter: 'mean'
    })
    
    bin_centers = binned_stats[parameter]['mean'].values
    mean_si = binned_stats['wiggleSI_calc']['mean'].values
    std_si = binned_stats['wiggleSI_calc']['std'].values
    counts = binned_stats['wiggleSI_calc']['count'].values
    
    # Plot with error bars
    ax.errorbar(bin_centers, np.abs(mean_si), yerr=std_si,
               fmt='o-', markersize=8, linewidth=2, capsize=5,
               label='Binned mean |SI|')
    
    # Size of markers by count
    ax.scatter(bin_centers, np.abs(mean_si), s=counts*10, alpha=0.3, color='blue')
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Mean |SI| (± std)', fontsize=12)
    ax.set_title(f'Binned: {param_label} vs SI', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Overall title
    title = f'S-Phase {param_label} Analysis'
    if station_code:
        title = f'{station_code} - {title}'
    fig.suptitle(title, fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    return fig, axes


def plot_incidence_pierce_depth_grid(event_data, station_code=None,
                                     figsize=(18, 6)):
    """
    Side-by-side comparison: incidence angle and pierce depth effects.
    
    Returns
    -------
    fig, (ax1, ax2, ax3)
    """
    s_data = event_data[(event_data['evphase'] == 'S') & 
                        (event_data['evincident'].notna()) &
                        (event_data['evpiercedepth'].notna())].copy()
    
    if len(s_data) == 0:
        return None, None
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # ========== Panel 1: Incidence vs |SI| ==========
    scatter1 = ax1.scatter(s_data['evincident'], np.abs(s_data['wiggleSI_calc']),
                          c=s_data['evdp'],
                          cmap='RdYlGn_r',
                          s=60, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    
    ax1.set_xlabel('Incidence Angle (°)', fontsize=12)
    ax1.set_ylabel('|SI|', fontsize=12)
    ax1.set_title('Incidence Angle vs SI\n(color = event depth)', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Event Depth (km)', fontsize=10)
    
    # ========== Panel 2: Pierce Depth vs |SI| ==========
    scatter2 = ax2.scatter(s_data['evpiercedepth'], np.abs(s_data['wiggleSI_calc']),
                          c=s_data['evincident'],
                          cmap='coolwarm',
                          s=60, alpha=0.7,
                          edgecolors='black', linewidths=0.5,
                          vmin=0, vmax=40)
    
    ax2.set_xlabel('Pierce Point Depth (km)', fontsize=12)
    ax2.set_ylabel('|SI|', fontsize=12)
    ax2.set_title('Pierce Depth vs SI\n(color = incidence angle)', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # Mark discontinuities
    for depth in [410, 660]:
        if s_data['evpiercedepth'].min() < depth < s_data['evpiercedepth'].max():
            ax2.axvline(depth, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Incidence (°)', fontsize=10)
    
    # ========== Panel 3: Incidence vs Pierce Depth (colored by SI) ==========
    scatter3 = ax3.scatter(s_data['evincident'], s_data['evpiercedepth'],
                          c=s_data['wiggleSI_calc'],
                          cmap='RdBu_r',
                          vmin=-2, vmax=2,
                          s=60, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    
    ax3.set_xlabel('Incidence Angle (°)', fontsize=12)
    ax3.set_ylabel('Pierce Point Depth (km)', fontsize=12)
    ax3.set_title('Incidence vs Pierce Depth\n(color = SI)', fontsize=13)
    ax3.grid(True, alpha=0.3)
    
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('SI', fontsize=10)
    
    # Overall
    title = f'S-Phase Geometric Parameters'
    if station_code:
        title = f'{station_code} - {title}'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3)


def plot_s_phase_geometry_grid(event_data, station_code=None, s_depth=400,
                               figsize=(24, 6)):
    """
    Three-panel view: incidence vs |SI|, pierce depth vs |SI|, epicentral distance vs |SI|.
    
    Shows how ray geometry affects S-phase measurements.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Event catalog
    station_code : str, optional
        Station name
    s_depth : float
        Depth threshold for coloring
        
    Returns
    -------
    fig, (ax1, ax2, ax3)
    """
    s_data = event_data[(event_data['evphase'] == 'S') & 
                        (event_data['evincident'].notna()) &
                        (event_data['evpiercedepth'].notna())].copy()
    
    if len(s_data) == 0:
        print("No S-phase data with geometric parameters")
        return None, None
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # ========== Panel 1: Incidence Angle vs |SI| ==========
    scatter1 = ax1.scatter(s_data['evincident'], np.abs(s_data['wiggleSI_calc']),
                          c=s_data['evdp'],
                          cmap='RdYlGn_r',
                          s=60, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    
    ax1.set_xlabel('Incidence Angle (°)', fontsize=12)
    ax1.set_ylabel('|SI|', fontsize=12)
    ax1.set_title('Incidence Angle vs SI\n(color = event depth)', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Event Depth (km)', fontsize=10)
    
    # ========== Panel 2: Pierce Depth vs |SI| ==========
    scatter2 = ax2.scatter(s_data['evpiercedepth'], np.abs(s_data['wiggleSI_calc']),
                          c=s_data['evincident'],
                          cmap='coolwarm',
                          s=60, alpha=0.7,
                          edgecolors='black', linewidths=0.5,
                          vmin=0, vmax=40)
    
    ax2.set_xlabel('Pierce Point Depth (km)', fontsize=12)
    ax2.set_ylabel('|SI|', fontsize=12)
    ax2.set_title('Pierce Depth vs SI\n(color = incidence angle)', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # Mark discontinuities
    for depth in [410, 660]:
        if s_data['evpiercedepth'].min() < depth < s_data['evpiercedepth'].max():
            ax2.axvline(depth, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Incidence (°)', fontsize=10)
    
    # ========== Panel 3: Epicentral Distance vs |SI| ==========
    scatter3 = ax3.scatter(s_data['evepic'], np.abs(s_data['wiggleSI_calc']),
                          c=s_data['evdp'],
                          cmap='viridis',
                          s=60, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    
    ax3.set_xlabel('Epicentral Distance (°)', fontsize=12)
    ax3.set_ylabel('|SI|', fontsize=12)
    ax3.set_title('Epicentral Distance vs SI\n(color = event depth)', fontsize=13)
    ax3.grid(True, alpha=0.3)
    
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Event Depth (km)', fontsize=10)
    
    # Overall
    title = f'S-Phase Geometric Parameters (N={len(s_data)})'
    if station_code:
        title = f'{station_code} - {title}'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3)

def plot_bootstrap_results(bootstrap_result, binned_data, station_code=None,
                          figsize=(20, 12)):
    """
    Plot bootstrap uncertainty: median fit with confidence envelopes.
    
    Parameters
    ----------
    bootstrap_result : dict
        Output from bootstrap_splitting_parameters()
    binned_data : pd.DataFrame
        Binned measurements to overlay
    station_code : str, optional
        Station name
        
    Returns
    -------
    fig, axes
    """
    use_tilt = bootstrap_result['use_tilt']
    
    # Determine grid size
    if use_tilt:
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
    
    phi_samples = bootstrap_result['phi_samples']
    dt_samples = bootstrap_result['dt_samples']
    phi_median = bootstrap_result['phi_median']
    dt_median = bootstrap_result['dt_median']
    phi_ci = bootstrap_result['phi_ci']
    dt_ci = bootstrap_result['dt_ci']
    
    if use_tilt:
        tilt_samples = bootstrap_result['tilt_samples']
        tilt_median = bootstrap_result['tilt_median']
        tilt_ci = bootstrap_result['tilt_ci']
    
    # ========== Panel 0: SI with median fit and confidence envelope ==========
    ax = axes[0]
    
    # Plot data
    ax.scatter(binned_data['azimuth'], binned_data['mean'],
              s=100, c='black', edgecolors='black', linewidths=1,
              zorder=10, label='Binned data')
    ax.errorbar(binned_data['azimuth'], binned_data['mean'],
               yerr=binned_data['std'],
               fmt='none', ecolor='gray', alpha=0.5)
    
    # Calculate fit curves
    degs = np.arange(0, 360, 1)
    
    # Median fit
    if use_tilt:
        incident = np.full_like(degs, 20.0)
        si_median = si_function_tilt((degs, incident), phi_median, dt_median, tilt_median)
        
        # Bootstrap envelope
        si_bootstrap_curves = np.array([
            si_function_tilt((degs, incident), phi, dt, tilt)
            for phi, dt, tilt in zip(phi_samples[::10], dt_samples[::10], tilt_samples[::10])
        ])
    else:
        si_median = si_function_simple(degs, phi_median, dt_median)
        
        # Bootstrap envelope
        si_bootstrap_curves = np.array([
            si_function_simple(degs, phi, dt)
            for phi, dt in zip(phi_samples[::10], dt_samples[::10])
        ])
    
    si_lower = np.percentile(si_bootstrap_curves, 2.5, axis=0)
    si_upper = np.percentile(si_bootstrap_curves, 97.5, axis=0)
    
    # Plot median curve
    param_str = f'φ={phi_median:.1f}°, δt={dt_median:.2f}s'
    if use_tilt:
        param_str += f', tilt={tilt_median:.1f}°'
    
    ax.plot(degs, si_median, 'b-', linewidth=3, label=f'Median: {param_str}')
    
    # Plot confidence envelope
    ax.fill_between(degs, si_lower, si_upper, 
                    alpha=0.3, color='blue', label='95% CI envelope')
    
    ax.set_xlabel('Polarization Azimuth (°)', fontsize=12)
    ax.set_ylabel('Splitting Intensity', fontsize=12)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Bootstrap Median Fit with 95% CI', fontsize=13)
    
    # ========== Panel 1: φ distribution ==========
    ax = axes[1]
    
    ax.hist(phi_samples, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(phi_median, color='red', linestyle='-', linewidth=2, 
              label=f'Median: {phi_median:.1f}°')
    ax.axvline(phi_ci[0], color='red', linestyle='--', linewidth=1.5)
    ax.axvline(phi_ci[1], color='red', linestyle='--', linewidth=1.5,
              label=f'95% CI: [{phi_ci[0]:.1f}°, {phi_ci[1]:.1f}°]')
    
    ax.set_xlabel('Fast Axis φ (°)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'φ Distribution (n={len(phi_samples)})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========== Panel 2: δt distribution ==========
    ax = axes[2]
    
    ax.hist(dt_samples, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(dt_median, color='red', linestyle='-', linewidth=2,
              label=f'Median: {dt_median:.2f}s')
    ax.axvline(dt_ci[0], color='red', linestyle='--', linewidth=1.5)
    ax.axvline(dt_ci[1], color='red', linestyle='--', linewidth=1.5,
              label=f'95% CI: [{dt_ci[0]:.2f}s, {dt_ci[1]:.2f}s]')
    
    ax.set_xlabel('Delay Time δt (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'δt Distribution (n={len(dt_samples)})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========== Panel 3: φ vs δt scatter ==========
    ax = axes[3]
    
    ax.scatter(phi_samples, dt_samples, alpha=0.3, s=10, c='gray')
    ax.plot(phi_median, dt_median, 'ro', markersize=15,
           markeredgecolor='black', markeredgewidth=2, label='Median')
    
    ax.set_xlabel('Fast Axis φ (°)', fontsize=12)
    ax.set_ylabel('Delay Time δt (s)', fontsize=12)
    ax.set_title('Parameter Correlation', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ========== Panel 4: Tilt distribution (if applicable) ==========
    if use_tilt:
        ax = axes[4]
        
        ax.hist(tilt_samples, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(tilt_median, color='red', linestyle='-', linewidth=2,
                  label=f'Median: {tilt_median:.1f}°')
        ax.axvline(tilt_ci[0], color='red', linestyle='--', linewidth=1.5)
        ax.axvline(tilt_ci[1], color='red', linestyle='--', linewidth=1.5,
                  label=f'95% CI: [{tilt_ci[0]:.1f}°, {tilt_ci[1]:.1f}°]')
        
        ax.set_xlabel('Tilt (°)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Tilt Distribution (n={len(tilt_samples)})', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Panel 5: Leave empty or add other diagnostic
        axes[5].axis('off')
    
    # Overall title
    n_attempted = bootstrap_result['n_attempted']
    n_successful = bootstrap_result['n_successful']

    title = f'Bootstrap Uncertainty Analysis ({n_successful}/{n_attempted} samples)'
    # title = f'Bootstrap Uncertainty Analysis ({n_bootstrap} samples)'
    if station_code:
        title = f'{station_code} - {title}'
    fig.suptitle(title, fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    return fig, axes


def plot_station_stereographic_map(results, station_code, phases_allowed, s_depth=400, figsize=(10, 10)):
    """
    Stereographic map centered on station with events colored by back azimuth.
    Phase types distinguished by marker edge color.
    
    Parameters
    ----------
    results : dict
        Output from analyze_station() with keys containing phase-specific data
    station_code : str
        Station identifier
    phases_allowed : str
        Key to access specific phase data in results dict
    s_depth : float
        Depth threshold for S-phase coloring (default: 400 km)
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, ax
    """

    data = results.get(phases_allowed)
    station_events = data['station_events']
    
    if len(station_events) == 0:
        print("No event data provided")
        return None, None
    
    # Station location
    stat_lat = station_events['evstationlat'].iloc[0]
    stat_lon = station_events['evstationlon'].iloc[0]
    
    # ===== AZIMUTHAL EQUIDISTANT projection =====
    proj = ccrs.AzimuthalEquidistant(
        central_longitude=stat_lon, 
        central_latitude=stat_lat
    )
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)

    # Make circular boundary
    theta = np.linspace(0, 2*np.pi, 100)
    center = [0.5, 0.5]
    radius = 0.5

    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    # ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_global()
        
    # ===== Map styling =====
    # ax.stock_img()
    ax.coastlines()
    
    # Limit view
    # ax.set_extent([stat_lon - 160, stat_lon + 160,
    #                stat_lat - 90, stat_lat + 90],
    #               crs=ccrs.PlateCarree())
    
    # ===== Cyclic colormap for azimuth =====
    cmap = plt.cm.twilight  # Cyclic colormap (wraps at 360°)
    norm = plt.Normalize(vmin=0, vmax=360)
    
    # Track legend labels to avoid duplicates
    existing_labels = []

    sks_plotted = s_deep_plotted = s_shallow_plotted = False
    n_sks = n_deep = n_shallow = 0

    for _, row in station_events.iterrows():
        if row['evphase'] == 'SKS':
            label = 'SKS phase' if not sks_plotted else None
            sks_plotted = True
            color = 'black'
            n_sks += 1
            
            ax.plot(row['evlon'], row['evlat'], '*', 
                   color=color, alpha=0.6, markersize=20,markeredgewidth=1.5,markeredgecolor='black', label=label,
                   transform=ccrs.PlateCarree(),zorder=5)
            # ax.errorbar(row['calcbaz'], row['wiggleSI_calc'], 
            #            yerr=row['SI_calc_error'], 
            #            ecolor=color, alpha=0.3, fmt='none')
            
        
        elif row['evphase'] == 'S':
            # Color by depth
            if row['evdp'] >= s_depth:
                label = 'Deep S' if not s_deep_plotted else None
                s_deep_plotted = True
                # color = 'tab:blue'
                color = 'tab:green'

                n_deep += 1
            else:
                label = 'Shallow S' if not s_shallow_plotted else None
                s_shallow_plotted = True
                color = 'tab:orange'
                n_shallow += 1
            
            ax.plot(row['evlon'], row['evlat'], '*',
                   color=color, alpha=0.6, markersize=20,markeredgewidth=1.5,markeredgecolor=color, label=label,
                   transform=ccrs.PlateCarree(),zorder=5)
            # ax.errorbar(row['evlon'], row['evlat'], 
            #            yerr=row['SI_calc_error'], 
            #            ecolor=color, alpha=0.3, fmt='none')
            # scatter = ax.scatter(
            #     row['evlon'],
            #     row['evlat'],
            #     c = row['evbaz'],
            #     cmap=cmap,
            #     vmin=0,
            #     vmax=360,
            #     marker='*',
            #     s=150,
            #     edgecolors='black',
            #     linewidths=1,
            #     alpha=0.7,
            #     zorder=5,
            #     transform=ccrs.PlateCarree()
            # )
    
    station_events_sorted = station_events.sort_values('evbaz')
    min_azimuth_spacing = 5
    last_azimuth = None
    
    # ===== Plot ray paths =====
    for idx, (_, event) in enumerate(station_events_sorted.iterrows()):

        if last_azimuth is None: 
            should_plot = True
        else:   
            azimuth_diff = abs(event['evbaz'] - last_azimuth)
            should_plot = azimuth_diff >= min_azimuth_spacing
            
        if not should_plot:
            continue

        # if abs(event['evbaz'] - last_azimuth) < min_azimuth_spacing:
        #     continue
        # Initialize geod object for WGS84 ellipsoid
        geod = Geod(ellps="WGS84")
        
        # Compute intermediate points along great circle
        npts = 100
        intermediate_points = geod.npts(
            event['evlon'], event['evlat'],
            stat_lon, stat_lat,
            npts
        )
        
        # Add endpoints
        lons = [event['evlon']] + [pt[0] for pt in intermediate_points] + [stat_lon]
        lats = [event['evlat']] + [pt[1] for pt in intermediate_points] + [stat_lat]
        
        # Plot great circle path
        ax.plot(lons, lats, 
               'k--', linewidth=1.5, alpha=0.6,
               transform=ccrs.Geodetic(),zorder=4)
        
        # Update last plotted azimuth
        last_azimuth = event['evbaz']


    # Simple version using tissot
    for dist_deg in [30, 60, 90, 120]:
        ax.tissot(rad_km=dist_deg*111.32, lons=[stat_lon], lats=[stat_lat],
                facecolor='none', edgecolor='gray', linewidth=1, alpha=0.5,zorder=4)

        # Add distance label
        angle = 0  # degrees (NE direction)
        label_lon = stat_lon + dist_deg * 0.95 * np.cos(np.radians(90 - angle))
        label_lat = stat_lat + dist_deg * 1.0 * np.sin(np.radians(90 - angle))
        ax.text(label_lon, label_lat, f'{dist_deg}°',
                fontsize=9, color='gray', ha='center', va='bottom',
                transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='none', alpha=0.7),zorder=3)
        




        
    # ===== Plot station (yellow triangle) =====
    ax.plot(stat_lon, stat_lat, 
        'yellow', marker='v', markersize=12,
        linestyle='None',
        transform=ccrs.PlateCarree(),
        markeredgecolor='black',
        markeredgewidth=1.5,
        zorder=10,
        label='Station')
    
    # # ===== Add colorbar for azimuth =====
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
    #                    pad=0.05, shrink=0.8, aspect=25)
    # cbar.set_label('Back Azimuth (°)', fontsize=11)
    
    # ===== Title & legend =====
    ax.set_title(f'{station_code} Event Distribution', 
                fontsize=14,)
    ax.legend(fontsize=15, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    return fig, ax



### FUNCTION TO READ GMT FILES FOR PLATE BOUNDARIES
def read_gmt_file(filename):
    """
    Read GMT file with segment separators (>)
    Returns list of dataframes, one per segment
    """
    segments = []
    current_segment = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines or comments
            if not line or line.startswith('#'):
                continue
            
            # ">" indicates new segment
            if line.startswith('>'):
                if current_segment:
                    segments.append(pd.DataFrame(current_segment, 
                                                columns=['lon', 'lat']))
                    current_segment = []
                continue
            
            # Parse coordinate line
            try:
                lon, lat = map(float, line.split()[:2])
                current_segment.append([lon, lat])
            except:
                continue
        
        # Add last segment
        if current_segment:
            segments.append(pd.DataFrame(current_segment, 
                                        columns=['lon', 'lat']))
    
    return segments

def read_and_filter_gmt(filename, minlon, maxlon, minlat, maxlat, max_gap=20):
    """
    Read GMT file, handle dateline, and filter to map extent
    """
    segments = []
    current_segment = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('>'):
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                continue
            
            try:
                lon, lat = map(float, line.split()[:2])
                lon = lon % 360
                
                # Check if point is in map bounds
                if not (minlon <= lon <= maxlon and minlat <= lat <= maxlat):
                    # Point outside bounds - end current segment
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = []
                    continue
                
                # Check for dateline jump
                if current_segment:
                    last_lon = current_segment[-1][0]
                    if abs(lon - last_lon) > max_gap:
                        segments.append(current_segment)
                        current_segment = []
                
                current_segment.append([lon, lat])
                
            except:
                continue
        
        if current_segment:
            segments.append(current_segment)
    
    return segments

# def plot_station_splitting(
#     statname,
#     statlat,
#     statlon,
#     phi,
#     dt,
#     ax,
#     version=None,
#     annotate=True,
#     plotted_points=None
# ):
#     """
#     Plot shear-wave splitting bar and optional station annotation.
#     Automatically offsets nearby labels to reduce overlap.
#     """

#     import numpy as np
#     import cartopy.crs as ccrs

#     # initialize tracking list if needed
#     if plotted_points is None:
#         plotted_points = []

#     # ----------------------------------------
#     # splitting line geometry
#     # ----------------------------------------

#     phi_rad = np.radians(90 - phi)

#     half_dt = dt / 2

#     dx = half_dt * np.cos(phi_rad)
#     dy = half_dt * np.sin(phi_rad)

#     x0 = statlon - dx
#     x1 = statlon + dx

#     y0 = statlat - dy
#     y1 = statlat + dy

#     # ----------------------------------------
#     # plot splitting line
#     # ----------------------------------------

#     if version == 'New':

#         ax.plot(
#             [x0, x1],
#             [y0, y1],
#             color='black',
#             linewidth=3,
#             transform=ccrs.PlateCarree(),
#             zorder=10
#         )

#     elif version == 'Legend':

#         ax.plot(
#             [x0, x1],
#             [y0, y1],
#             color='black',
#             linewidth=3,
#             transform=ccrs.PlateCarree(),
#             zorder=10
#         )

#     elif version == 'Old':

#         ax.plot(
#             [x0, x1],
#             [y0, y1],
#             color="#FF4400",
#             linewidth=3,
#             transform=ccrs.PlateCarree(),
#             zorder=9
#         )

#     # ----------------------------------------
#     # label offset logic
#     # ----------------------------------------

#     label_dx = 10
#     label_dy = 10

#     nearby_count = 0

#     for other_lon, other_lat in plotted_points:

#         # check for nearby stations
#         if (
#             abs(statlon - other_lon) < 2
#             and
#             abs(statlat - other_lat) < 2
#         ):

#             nearby_count += 1

#     # progressively move labels farther away
#     offset_patterns = [
#         (10, 10),
#         (-10, 10),
#         (10, -10),
#         (-10, -10),
#         (15, 0),
#         (-15, 0),
#         (0, 15),
#         (0, -15),
#     ]

#     label_dx, label_dy = offset_patterns[
#         nearby_count % len(offset_patterns)
#     ]

#     # store current point AFTER overlap check
#     plotted_points.append((statlon, statlat))

#     # ----------------------------------------
#     # annotate station
#     # ----------------------------------------

#     if annotate:

#         ax.annotate(
#             statname,

#             xy=(statlon, statlat),

#             xytext=(label_dx, label_dy),

#             xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),

#             textcoords='offset points',

#             fontsize=16,
#             fontweight='bold',

#             color='black',

#             bbox=dict(
#                 facecolor='white',
#                 alpha=0.7,
#                 edgecolor='none',
#                 pad=1
#             ),

#             zorder=8
#         )

def plot_station_splitting(statname,statlat,statlon,phi,dt,ax,version=None,annotate=True):
    phi = np.radians(90 - phi)
    half_dt = dt / 2 

    #halfway points of each line
    dx = half_dt * np.cos(phi)
    dy = half_dt * np.sin(phi)

    x0 = statlon - dx
    x1 = statlon + dx
    y0 = statlat - dy
    y1 = statlat + dy

    if version == 'New':
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=3, transform=ccrs.PlateCarree(),zorder=10)
    if version == 'New_lowalpha':
            ax.plot([x0, x1], [y0, y1], color='black', alpha=0.8,linewidth=3, transform=ccrs.PlateCarree(),zorder=10)
    if version == 'Legend':
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=3, transform=ccrs.PlateCarree())
    if version == 'SurfaceLegend':
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=3, transform=ccrs.PlateCarree())
    
    if version == 'Prediction1':
        ax.plot([x0, x1], [y0, y1], color='red',alpha=0.5, linewidth=3, transform=ccrs.PlateCarree(),zorder=10)

    if version == 'Prediction2':
        ax.plot([x0, x1], [y0, y1], color='blue',alpha=0.5, linewidth=3, transform=ccrs.PlateCarree(),zorder=10)

    if version == 'CMB_single_station':
        ax.plot(statlon,
                statlat,
                marker='v',
                markersize=12,
                markerfacecolor='yellow',
                markeredgecolor='black',
                markeredgewidth=1.5,
                linestyle='None',
                transform=ccrs.PlateCarree(),
                zorder=10)

    
    if version == 'Old':
            ax.plot([x0, x1], [y0, y1], color="tomato", linewidth=2, transform=ccrs.PlateCarree())
    if version == 'BoldNew':
            ax.plot([x0, x1], [y0, y1], color="#FF4400", linewidth=5, transform=ccrs.PlateCarree(),zorder=10)
    if version == 'Surface':
        if (dt < 100) & (dt > 20):
            color = 'green'
        else:
            color= 'black'
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=3, transform=ccrs.PlateCarree(),zorder=10,alpha=0.7)
        
    dx,dy = 10,10
    ha = 'left'
    va = 'bottom'

    if statname == "CAN":
        dx,dy = 10,-20

    if statname == "CNB":
        dx,dy = 10,0

    if statname == "STKA":
        dx,dy = 0,20

    if statname == "TOO":
        # dx,dy = -10,-10 #OLD VALUES BEFORE SHIFTING FOR FIG 4
        # ha = 'right'
        # va = 'top'
        dx,dy = 40,-10
        ha = 'right'
        va = 'top'

    if statname == "MUN":
        dx,dy = -10,-10
        ha = 'right'
        va = 'top'

    if statname == "AS31":
        dx,dy = -10,-10
        ha = 'right'
        va = 'top'

    if statname == "FORT":
        dx,dy = -10,-10
        ha = 'right'
        va = 'top'
    
    if statname == "BBOO":
        dx,dy = -10,-10
        ha = 'right'
        va = 'top'

    if statname == "YNG":
        dx,dy = -8,10
        ha = 'right'
        va = 'top'

    if statname == "SNZO":
        dx,dy = -10,-45
        ha = 'left'
        va = 'bottom'

    if annotate:
        ax.annotate(
        statname,
        xy=(statlon,statlat),
        xytext=(dx,dy),
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
        textcoords='offset points',
        fontsize=16,
        fontweight='bold',
        color='black',
        ha=ha,
        va=va,
        zorder=20,
        bbox=dict(boxstyle='round,pad=0.1',  # or 'square,pad=0.3'
                  facecolor='white',
                  edgecolor='none',  # or 'black' for border
                  alpha=0.8)  # slight transparency
        )


def plot_station_metadata(statname,stat_phase_used,statlat,statlon,phi,dt,ax,version=None,annotate=True):
    # phi = np.radians(90 - phi)
    # half_dt = dt / 2 

    # #halfway points of each line
    # dx = half_dt * np.cos(phi)
    # dy = half_dt * np.sin(phi)

    # x0 = statlon - dx
    # x1 = statlon + dx
    # y0 = statlat - dy
    # y1 = statlat + dy

    # if version is 'New':
    #     ax.plot([x0, x1], [y0, y1], color='black', linewidth=3, transform=ccrs.PlateCarree(),zorder=10)
    # if version is 'Legend':
    #     ax.plot([x0, x1], [y0, y1], color='black', linewidth=3, transform=ccrs.PlateCarree())
    # if version is 'Old':
    #         ax.plot([x0, x1], [y0, y1], color="#FF4400", linewidth=3, transform=ccrs.PlateCarree())
# ax.scatter(
#     blob_df['longitude'],
#     blob_df['latitude'],
#     color='black',
#     s=10,
#     transform=ccrs.PlateCarree(),
#     zorder=10
# )
    if stat_phase_used == 'sks':
        ax.scatter(
            statlon,
            statlat,
            color = 'blue',
            s = 10, 
            transform=ccrs.PlateCarree(),
            zorder=8
        )

    if stat_phase_used == 'sks_deep':
        ax.scatter(
            statlon,
            statlat,
            color = 'orange',
            s = 10, 
            transform=ccrs.PlateCarree(),
            zorder=8
        )

    if stat_phase_used == 'all':
        ax.scatter(
            statlon,
            statlat,
            color = 'yellow',
            s = 10, 
            transform=ccrs.PlateCarree(),
            zorder=8
        )

    if stat_phase_used == 'complex':
        ax.scatter(
            statlon,
            statlat,
            color = 'grey',
            s = 10, 
            transform=ccrs.PlateCarree(),
            zorder=8
        )

    if annotate:
        ax.annotate(
        statname,
        xy=(statlon,statlat),
        xytext=(10, 10),
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
        textcoords='offset points',
        fontsize=16,
        fontweight='bold',
        color='black',
        zorder=8
        )


def build_cmap():
    COLOR_OCEAN = "white"
    COLOR_LAND  = "#f3f3f3"
    COLOR_COAST = "#a8a8a8"
    COLOR_LLVP  = "#d95757"
    THRESHOLD   = 0.4
    DATA_MAX    = 2.0
    GRAY_FLAT   = "#aaaaaa"
    warm_colors = [
        "#fffde0",
        "#fdd87a",
        "#f4931b",
        "#d94308",
        "#8b0000",
    ]
    warm_cmap = mcolors.LinearSegmentedColormap.from_list("warm", warm_colors, N=256)

    n_total = 512
    frac_gray = THRESHOLD / DATA_MAX
    n_gray = max(4, int(n_total * frac_gray))
    n_warm = n_total - n_gray

    gray_rgb = mcolors.to_rgba(GRAY_FLAT)
    gray_block = np.tile(gray_rgb, (n_gray, 1))
    warm_block = warm_cmap(np.linspace(0, 1, n_warm))

    combined = np.vstack([gray_block, warm_block])
    cmap = mcolors.ListedColormap(combined, name="aniso_warm")
    cmap.set_bad(color="white")
    cmap.set_under(color=GRAY_FLAT)

    norm = mcolors.Normalize(vmin=0.0, vmax=DATA_MAX)
    return cmap, norm

def subsample_anisotropy_grid(df, lat_stride=2, lon_stride=2):
    """
    Subsample by taking every Nth latitude and longitude
    This maintains a regular grid pattern
    """
    # Get unique sorted coordinates
    unique_lats = np.sort(df['lat'].unique())
    unique_lons = np.sort(df['lon'].unique())
    
    # Subsample coordinates
    selected_lats = unique_lats[::lat_stride]
    selected_lons = unique_lons[::lon_stride]
    
    print(f"Original grid: {len(unique_lats)} lats × {len(unique_lons)} lons")
    print(f"Subsampled grid: {len(selected_lats)} lats × {len(selected_lons)} lons")
    
    # Filter DataFrame to keep only selected coordinates
    df_sub = df[df['lat'].isin(selected_lats) & df['lon'].isin(selected_lons)].copy()
    
    return df_sub.reset_index(drop=True)