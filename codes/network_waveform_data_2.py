import pandas as pd
import numpy as np
import glob
import os
from obspy import geodetics
from obspy.taup import TauPyModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------- PARAMETERS -----------------
data_type   = 'complete'
data_tag    = 'swp'
station     = 'swp'
phase       = ['S', 'SKS']
model_name  = "iasp91"
date        = 'oct3'

print('Loading in filepaths for station and event data.')

# # ----------------- FILE PATHS -----------------
# waveforms = []
# for ph in phase:
#     dir_waveforms = f'../data/{data_type}/{station}/{data_tag}_{ph}_waveforms/'
#     waveforms.extend(glob.glob(dir_waveforms + '*'))

# /work/gcl3/BR/nicolasv03/senior_thesis/data/complete/australia/AU_event_info

dir_events      = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_event_info/'
station_file    = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_stations_info.txt'
output_path     = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_events_{date}.txt'

# ----------------- LOAD STATION DATA -----------------
stations_df = pd.read_csv(station_file, sep='|', header=0)
stations_df['evstation'] = stations_df['Station'].str.strip().str.upper()
stations_df = stations_df.rename(columns={'Latitude': 'evstationlat', 'Longitude': 'evstationlon'})

print('Starting to merge station data with event data.')

# ----------------- COMPILE EVENTS -----------------
events_df = pd.DataFrame()
for file in glob.glob(dir_events + '*'):
    network = os.path.basename(file).split('-')[0]
    sta = os.path.basename(file).split('-')[1]
    ev = pd.read_csv(file, sep=',', header=0)
    ev['evnetwork'] = network
    ev['evstation'] = sta.strip().upper()
    events_df = pd.concat([events_df, ev], ignore_index=True)

# ----------------- MERGE STATION COORDINATES -----------------
events_df['evstation'] = events_df['evstation'].str.strip().str.upper()
events_df = events_df.merge(stations_df[['evstation', 'evstationlat', 'evstationlon']], on='evstation', how='left')

print('Done merging station data with event data.')
print('Calculating geometric data for each event/station pair.')

# ----------------- CALCULATE GEODETIC METRICS -----------------
def compute_geodetics(row):
    try:
        dist, az, baz = geodetics.base.gps2dist_azimuth(row['evlat'], row['evlon'],
                                                        row['evstationlat'], row['evstationlon'])
        epic = geodetics.base.locations2degrees(row['evlat'], row['evlon'],
                                                row['evstationlat'], row['evstationlon'])
        return pd.Series([dist, az, baz, epic])
    except Exception:
        return pd.Series([None, None, None, None])

    
def compute_taupy_features(row,model_name=model_name):
    # evdp, epic, phase = input_tuple
    try:
        model_local = TauPyModel(model=model_name)
        arrivals = model_local.get_travel_times(
            source_depth_in_km=row['evdp'],
            distance_in_degree=row['evepic'],
            phase_list=[row['evphase']]
        )
        pierce = model_local.get_pierce_points(
            source_depth_in_km=row['evdp'],
            distance_in_degree=row['evepic'],
            phase_list=[row['evphase']]
            )
        if arrivals:
            arr = arrivals[0]
            # deepest piercing depth if available
            if pierce:
                pie = pierce[0]
                max_depth = max(row[-1] for row in pie.pierce)
               
            else:
                max_depth = None
            return arr.incident_angle, arr.takeoff_angle, arr.ray_param, max_depth
        else:
            return None, 0, None, None
    except Exception:
        return None, 1, None, None
    
def parallelize_dataframe(df, func, max_workers=32):
    input_data = df.to_dict('records')
    results = [None] * len(input_data)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, row): i for i, row in enumerate(input_data)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            results[i] = future.result()
    
    return pd.DataFrame(results, index=df.index)


# ----------------- LIST ALL WAVEFORMS -----------------
waveforms = []
for ph in phase:
    dir_waveforms = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_{ph}_waveforms/'
    waveforms.extend(glob.glob(dir_waveforms + '*.mseed'))

tick = 0 
# ----------------- BUILD WAVEFORM LOOKUP -----------------
# Assuming waveform filenames are like: NET-STA-YYYY-MM-DD-HHMMSS.mseed
waveform_dict = {}
for wf in waveforms:
    tick +=1
    base = os.path.basename(wf).replace('.mseed', '')
    parts = base.split('-')
    # print(parts)
    if len(parts) >= 5:
        # use combination of station + time as key for unique match
        key = f"{parts[1].strip().upper()}-{parts[2]}-{parts[3]}-{parts[4]}"
        waveform_dict[key] = wf



# ----------------- MAP WAVEFORMS TO EVENTS -----------------
# Create a key in events_df matching the waveform dict
events_df['waveform_key'] = (
    events_df['evstation'].str.strip().str.upper() + '-' + events_df['evtime']
)
# print(events_df['waveform_key'].head())
# Map waveform path
events_df['waveform_path'] = events_df['waveform_key'].map(waveform_dict)

# Remove events without a waveform
events_df = events_df[events_df['waveform_path'].notna()]

# Drop the temporary key if you want
events_df.drop(columns=['waveform_key'], inplace=True)


# Ensure all expected columns exist
required_cols = [
    'evnetwork','evstation','evstationlat','evstationlon',
    'evtime','evlat','evlon','evdp','evmg',
    'evphase','evdist','evaz','evbaz','evepic','evincident', 'evtakeoff', 'evrayparam', 'evpiercedepth',
    'windowstart','windowend','calcbaz','deltaphi',
    'sta_lta','snrbefore','snrafter','ellipticity'
]

for col in required_cols:
    if col not in events_df.columns:
        events_df[col] = None

# ----------------- RUN PARALLEL GEODETICS -----------------
print("Computing geodetic metrics...")
events_df[['evdist', 'evaz', 'evbaz', 'evepic']] = parallelize_dataframe(events_df, compute_geodetics)

# ----------------- RUN PARALLEL TAUPY FEATURES -----------------
print("Computing TauPy features...")
events_df[['evincident', 'evtakeoff', 'evrayparam', 'evpiercedepth']] = parallelize_dataframe(events_df, compute_taupy_features)


# ----------------- SAVE -----------------
events_df.to_csv(output_path, sep='|', index=False)
print(f"Saved {len(events_df)} events with waveforms to {output_path}")