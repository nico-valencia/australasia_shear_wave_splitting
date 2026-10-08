
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import filtering, analysis, plotting, diagnostics
from obspy import read
import traceback 
import time
import obspy
from station_keys import STATION_PHASE_MAP
from obspy.geodetics import base as geodetics

print(geodetics.HAS_GEOGRAPHICLIB)

# ==================== Load data ====================
# the next two datasets are 200 400 phase windowed datasets with newer updated sta lta values

dataType = 'complete'
networkName = 'swp'
stationName = 'swp'
freqType = 'normal'

raw_swp_eventdata = pd.read_csv(
    f'../data/{dataType}/{stationName}/{networkName}_measured_events_{freqType}_04032026.txt',
    sep='|'
)

dataType = 'complete'
networkName = 'AU'
stationName = 'australia'
freqType = 'normal'

raw_aus_eventdata = pd.read_csv(
    f'../data/{dataType}/{stationName}/{networkName}_measured_events_{freqType}_04032026.txt',
    sep='|'
)

raw_eventdata = pd.concat([raw_swp_eventdata, raw_aus_eventdata], ignore_index=True)
raw_eventdata = raw_eventdata.sort_values('evstation')

# ==================== Prepare data ====================
# Calculate SI errors
eventData = filtering.calculate_si_errors(raw_eventdata)

epicentral_max=360
epicentral_min=0
depth_min=0
depth_max=1e5
snrbefore_min=3.0
snrafter_min=0.0
sta_lta_min=0
mag_max=7.0
ellipticity_min=1.0
ellipticity_max=1e4
si_error_max=0.5
incident_max=90
pierce_depth_min=0
delta_travel_max = 25


# Apply quality filters
eventData = filtering.filter_events(
    eventData,
    epicentral_max=epicentral_max,
    epicentral_min=epicentral_min,
    depth_min=depth_min,
    depth_max=depth_max,
    snrbefore_min=snrbefore_min,
    snrafter_min=snrafter_min,
    sta_lta_min=sta_lta_min,
    mag_max=mag_max,
    ellipticity_min=ellipticity_min,
    ellipticity_max=ellipticity_max,
    si_error_max=si_error_max,
    incident_max=incident_max,
    pierce_depth_min=pierce_depth_min,
    delta_travel_max=delta_travel_max
)

eventData = eventData.drop_duplicates(subset=['evtime','evstation'])

# Remove shallow events with large incident angles (keeps only events with large epicentral distances and smaller incident angles)
eventData = eventData[~((eventData['evdp'] < 400) & (eventData['evincident'] > 25))]

# # Remove deep events with very large incident angles (sample)
eventData = eventData[~((eventData['evdp'] >= 400) & (eventData['evincident'] > 35))]

# ^^^^^^ these two filters are heavily used in the new analysis so keep them!!!


'''
april 5 notes
- added tilt to SI calculation and plotting (see analysis.py and spring_updates.ipynb)
- started looking at treating deep S phases and direct S phases separately (mainly dealing with incident angle and pierce depth filters)
- analysis of only deep S phases to identify trends and how this could be corraborated with sampling of anisotropy at tilted layer)
'''

print(f"Events after filtering: {len(eventData)}")

# Parameters
bin_width = 20
bin_minimum = 3
s_depth = 400

blob_stations = ['ARMA','BBOO','CAN','CNB','YNG']
# 'BCON'
# Get all unique stations
station_list = eventData['evstation'].unique()
print(f"Found {len(station_list)} unique stations")
print(f"Stations: {sorted(station_list)}")

# Filter to stations with minimum data
station_counts = eventData.groupby('evstation').size()
station_list = station_counts[station_counts >= 1].index.tolist()  # At least 10 measurements
print(f"\nProcessing {len(station_list)} stations with ≥10 measurements")

# Filter to only stations in dictionary and in blob list
station_list = [s for s in station_list if s in STATION_PHASE_MAP]
station_list = [s for s in station_list if s in blob_stations]

print(f"After dict filter: {len(station_list)} stations remaining")

model = obspy.taup.TauPyModel(model="iasp91")

from obspy.geodetics.base import locations2degrees,gps2dist_azimuth

from pyproj import Geod

g = Geod(ellps='WGS84')

blob_coords = pd.DataFrame(columns=['lat','lon','evlat','evlon','statlat','statlon','evdp','evphase','evepic','SI'])

blob_tick = 0 

# Loop through stations
for station_code in station_list:
    
    # if STATION_PHASE_MAP[station_code] == 'complex':
    #     continue
    
    try:
        station_events = eventData[eventData['evstation'] == station_code]
        # print(station_code)
        # print(station_events.head(3))
        results = analysis.analyze_station_comparison(station_events, station_code, bin_width=bin_width, bin_minimum=bin_minimum, s_depth=400, use_tilt=False, with_C=False)

        ####################

        # Use phase from station phase mapping
        data = results.get(STATION_PHASE_MAP[station_code])
        # print(data)
        # print(data.keys())
        # time.sleep(1)
        events = data['station_events']
        blob_events = events[(events['evphase'] == 'SKS') & 
                             (events['calcbaz'] > 130) &
                             (events['calcbaz'] < 160) 
        ]

        # print(len(blob_events))
        for i in range(0,len(blob_events)):
            event = blob_events.iloc[i]

            arrivals = model.get_ray_paths_geo(
                source_depth_in_km=event['evdp'],
                source_latitude_in_deg=event['evlat'],
                source_longitude_in_deg=event['evlon'],
                receiver_latitude_in_deg=event['evstationlat'],
                receiver_longitude_in_deg=event['evstationlon'],
                phase_list=['SKS']
            )


            arrival = arrivals[0]


            path = arrival.path

            cmb_inds = np.where(
                np.isclose(path['depth'], 2889, atol=1)
            )[0]

            receiver_cmb = path[cmb_inds[-1]]

            cmb_lat = receiver_cmb['lat']
            cmb_lon = receiver_cmb['lon']

            if cmb_lon < 150:
                continue
           

            blob_coords.loc[blob_tick,:] = [cmb_lat,cmb_lon,event['evlat'],event['evlon'],event['evstationlat'],event['evstationlon'],event['evdp'],event['evphase'],event['evepic'],event['wiggleSI_calc']]

            blob_tick += 1

            

        # plot the blob
        # output_path = f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/blob_plots/'
        # fig7, ax7 = analysis.publication_SI_curve(results, station_code,STATION_PHASE_MAP[station_code],s_depth=400,figsize=(12,8),with_C=False)
        # plt.savefig(output_path+f'{station_code}_{STATION_PHASE_MAP[station_code]}_plot.png', dpi=300, bbox_inches='tight')
        
        # station_events = station_events[
#     (station_events['evphase'] != 'SKS') |  # Keep if not SKS
#     (station_events['calcbaz'] < 120) |      # OR if calcbaz < 145
#     (station_events['calcbaz'] > 150)      # OR if calcbaz > 165
# ]


        results = []
    except Exception as e:
        print(f"  ✗ Failed on {station_code}: {e}")
        traceback.print_exc()

        plt.close('all')  # Clean up any open figures
        continue



print(blob_coords)
blob_coords.to_csv("/work/gcl3/BR/nicolasv03/senior_thesis/measurements/blob_plots/blob_coordinates_obspy_only.txt",sep='\t')