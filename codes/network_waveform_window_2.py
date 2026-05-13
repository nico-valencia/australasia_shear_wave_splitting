import pandas as pd 
import numpy as np 
import glob 
import sys
from obspy.signal.trigger import classic_sta_lta
import os
import copy
import traceback
from obspy import read
import splitwavepy as sw
import numpy as np
from support import covar_particle_motion, envelopes, envelopes_rms
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------- PARAMETERS -----------------
data_type   = 'complete'
data_tag    = 'swp'
station     = 'swp'
freqType    = "normal"
date        = '04032026'

events_file     = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_events_{date}.txt'
station_file    = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_stations_info.txt'
output_path     = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_windowed_events_{freqType}_{date}.txt'

# ----------------- LOAD STATION DATA -----------------
# stations_df = pd.read_csv(station_file, sep='|', header=0)
# stations_df['evstation'] = stations_df['Station'].str.strip().str.upper()
# stations_df = stations_df.rename(columns={'Latitude': 'evstationlat', 'Longitude': 'evstationlon'})

eventdata = pd.read_csv(events_file,sep='|')
eventdata['freq'] = freqType

eventdata = eventdata

def process_waveform(row,freqType=freqType):
    '''
    find waveform windows used for later procesing
    also find some other useful things, to be updated...
    '''

    try:
        wf = read(row['waveform_path'])
        pm_wf = copy.deepcopy(wf)

        # Frequency selection
        if freqType == 'low':
            freqmin, freqmax = 0.04, 0.125
        elif freqType == 'medium':
            freqmin, freqmax = 0.125, 0.2
        elif freqType =='high':
            freqmin, freqmax = 0.2, 1
        elif freqType == 'normal':
            freqmin, freqmax = 0.01, 0.125

        # Filtering
        wf.detrend('demean')
        wf.detrend('linear')
        wf.filter("bandpass", freqmin=freqmin, freqmax=freqmax)

        north = wf.select(component="N", channel='BH*')[0].data
        east  = wf.select(component="E", channel='BH*')[0].data
        sample_interval = wf[0].stats.delta

        if len(north) % 2 == 0: north = north[:-1]
        if len(east) % 2 == 0: east = east[:-1]

        n_sta_lta = classic_sta_lta(north,30,600)
        e_sta_lta = classic_sta_lta(east,30,600)

        '''
        Adjusting values here for STA LTA 3/16/2026
        before it was 100 and 400 (5 and 20 seconds)

        now it is gonna be 30 and 600 (1.5 and 30 seconds)
        '''

        time = list(range(len(north)))
        sum_sta_lta = n_sta_lta+e_sta_lta
        phase_ind = np.argmax(sum_sta_lta)
        sta_lta_max = sum_sta_lta[phase_ind]
        sta_lta_max_time = time[phase_ind]*sample_interval

        delta_travel = abs(sta_lta_max_time - 60)
        # if delta_travel >= 40:
        #     return None

        
        windowstart = phase_ind - 200 
        windowend = phase_ind + 400

        '''
        Adjusting window sizing 4/3/2026
        before it was 300 samples to the left and right (30 second window, 15s each side)
        now it is 200 samples to the left and 400 to the right (30 second window, 10s and 20s for the sides respectively)
        '''

        # Particle motion
        pm_wf.detrend('demean')
        pm_wf.detrend('linear')
        pm_wf.filter('bandpass', freqmin=0.02, freqmax=0.066)
        pm_wf_north = pm_wf.select(component="N")[0].data
        pm_wf_east  = pm_wf.select(component="E")[0].data
        if len(pm_wf_north) % 2 == 0: pm_wf_north = pm_wf_north[:-1]
        if len(pm_wf_east) % 2 == 0: pm_wf_east = pm_wf_east[:-1]

        calcbaz, xlam1, xlam2 = covar_particle_motion(pm_wf_north[windowstart:windowend],
                                                      pm_wf_east[windowstart:windowend])

        if abs(calcbaz - row['evbaz']) > 90:
            calcbaz = (calcbaz + 180) % 360


         # Envelope / SNR
        pair = sw.Pair(north, east, delta=sample_interval)
        # mean_snr, max_snr, ne_env, mean_snr2, max_snr2 = envelopes(pair, float(row['evbaz']), calcbaz, windowstart, windowend)

        # added rms to envelope measurement instead of median, gonna compare it (march 11,2026)
        mean_snr, mean_snr2, ne_env = envelopes_rms(pair, float(row['evbaz']), calcbaz, windowstart, windowend)


        return windowstart*sample_interval,windowend*sample_interval,calcbaz,sta_lta_max_time,mean_snr,mean_snr2,xlam1/xlam2,delta_travel



    except Exception:
        traceback.print_exc()
        # print('idk')
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        

def parallelize_dataframe(df, func, max_workers=16):
    input_data = df.to_dict('records')
    results = [None] * len(input_data)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, row): i for i, row in enumerate(input_data)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            results[i] = future.result()
    
    return pd.DataFrame(results, index=df.index)


# ----------------- RUN PARALLEL GEODETICS -----------------
print("Computing geodetic metrics...")
eventdata[['windowstart', 'windowend', 'calcbaz', 'sta_lta','snrbefore','snrafter','ellipticity','delta_travel']] = parallelize_dataframe(eventdata, process_waveform)

# ----------------- SAVE -----------------
eventdata.to_csv(output_path, sep='|', index=False)
print(f"Saved {len(eventdata)} events with waveforms to {output_path}")




