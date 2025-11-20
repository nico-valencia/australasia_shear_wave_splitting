import os
import glob
import numpy as np
import pandas as pd
from obspy import read
import splitwavepy as sw
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from support import SplittingIntensity
from tqdm import tqdm
import copy

# ----------------- PARAMETERS -----------------
data_type   = 'complete'
data_tag    = 'swp'
station     = 'swp'
freqType    = "normal"
date        = 'oct3'

events_file     = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_windowed_events_{freqType}_{date}.txt'
station_file    = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_stations_info.txt'
output_path     = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_measured_events_{freqType}_{date}.txt'

eventdata = pd.read_csv(events_file,sep='|')

def measure_waveform(row,freqType = freqType):
    try:
        wf = read(row['waveform_path'])

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

        evwindowstart = row['windowstart']
        evwindowend = row['windowend']
        pair = sw.Pair(north, east, delta=sample_interval)
        # print(f"length of pair before setting window:{len(pair.x)}")
        pair.set_window(float(evwindowstart), float(evwindowend))
        # print(f"length of pair after setting window:{len(pair.x)}")
        chopped_pair = pair.chop()
        # print(f"Length of pair after chopping: {len(chopped_pair.x)}")


        beta = ((((float(row['calcbaz']) - float(row['evbaz'])) % 180) + 90) % 180) - 90

        # Splitting intensity
        # print(f"this is me seeing what evbaz looks like {row['evbaz']}")
        # print(f"this is me seeing what calcbaz looks like {row['calcbaz']}")

        wiggleSIlow_act , wiggleSI_act , wiggleSIhigh_act = SplittingIntensity(chopped_pair,float(row['evbaz']))
        wiggleSIlow_calc, wiggleSI_calc, wiggleSIhigh_calc = SplittingIntensity(chopped_pair,float(row['calcbaz']))
        
        return beta,wiggleSIlow_act , wiggleSI_act , wiggleSIhigh_act,wiggleSIlow_calc, wiggleSI_calc, wiggleSIhigh_calc

    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

def parallelize_dataframe(df, func, max_workers=32):
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
eventdata[['deltaphi','wiggleSIlow_act','wiggleSI_act','wiggleSIhigh_act','wiggleSIlow_calc','wiggleSI_calc','wiggleSIhigh_calc']] = parallelize_dataframe(eventdata, measure_waveform)

# ----------------- SAVE -----------------
eventdata.to_csv(output_path, sep='|', index=False)
print(f"Saved {len(eventdata)} events with waveforms to {output_path}")











# data_type = 'complete'
# station = 'australia'
# data_tag = 'AU'
# freqType = 'high'

# # eventdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/data/'+data_type+'/'+station+'/'+data_tag+'_events_apr16.txt',sep='|')
# eventdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/data/'+data_type+'/'+station+'/'+data_tag+'_windowed_events_'+freqType+'_sep16.txt',sep='|')
# eventdata = eventdata.dropna(subset=['calcbaz'])

# output_dir      = '../data/'+data_type+'/'+station+'/'+data_tag
# dir_station     = '../data/'+data_type+'/'+station+'/'+data_tag+'_stations_info.txt'


# station_df  = pd.read_csv(dir_station,sep='|',header=0)
# station_lat = station_df['Latitude'].iloc[0] ; station_lon = station_df['Longitude'].iloc[0]

# eventdata['phi'] =0
# eventdata['dt']= 0
# eventdata['snr']= 0
# eventdata['lambda']= 0
# eventdata['dfast']=0
# eventdata['dlag']=0

# eventdata['wiggleSI_act'] = 0 
# eventdata['wiggleSIlow_act'] = 0 
# eventdata['wiggleSIhigh_act'] = 0 

# eventdata['wiggleSI_calc'] = 0 
# eventdata['wiggleSIlow_calc'] = 0 
# eventdata['wiggleSIhigh_calc'] = 0 



# # print(eventdata)

# phase = ['SKS','S']
# # phase = ['S']

# # station = 'STKA'
# waveforms = []
# for ph in phase:
#     # dir_waveforms   = '../data/sample/'+station+'/'+station+'_'+ph+'_waveforms/'
#     # dir = '/work/gcl3/BR/nicolasv03/senior_thesis/data/sample/'+station+'/'+station+'_'+ph+'_waveforms/'
#     dir = '/work/gcl3/BR/nicolasv03/senior_thesis/data/'+data_type+'/'+station+'/'+data_tag+'_'+ph+'_waveforms/'


#     waveforms.extend(glob.glob(dir+'*'))

# bruh = 0 
# result_ind = 0 

# waveforms = waveforms[:1000]

# def process_waveform(i, eventdata):
#     try:
#         x = read(i)
#         pieces  = i.split('-')
#         stat = pieces[1]
#         bit    = pieces[2] + "-" + pieces[3] + "-" + pieces[4]
#         time = bit.split('.')[0]+'.'+bit.split('.')[1]

#         # Match event
#         evrow = eventdata[(eventdata['evtime'] == time) & (eventdata['evstation'] == stat)]
#         if evrow.empty:
#             return None  # skip if not found
#         evind = evrow.index[0]

#         evbaz = evrow['evbaz'].iloc[0]
#         evcalcbaz = evrow['calcbaz'].iloc[0]
#         evwindowstart = evrow['windowstart'].iloc[0]
#         evwindowend = evrow['windowend'].iloc[0]

#         # Filter
#         x.detrend('demean')
#         x.detrend('linear')
#         x.filter("bandpass", freqmin=0.01, freqmax=0.125)

#         north = x.select(component="N")[0].data
#         east  = x.select(component="E")[0].data
#         sample_interval = x[0].stats.delta

#         if len(north) % 2 == 0: north = north[:-1]
#         if len(east) % 2 == 0:  east = east[:-1]

#         pair = sw.Pair(north, east, delta=sample_interval)
#         pair.set_window(float(evwindowstart), float(evwindowend))

#         beta = ((((float(evcalcbaz) - float(evbaz)) % 180) + 90) % 180) - 90

#         # Splitting intensity
#         wiggleSIlow_act , wiggleSI_act , wiggleSIhigh_act = SplittingIntensity(pair,float(evbaz))
#         wiggleSIlow_calc, wiggleSI_calc, wiggleSIhigh_calc = SplittingIntensity(pair,float(evcalcbaz))

#         print(wiggleSI_calc)

#         return {
#             'evtime': time,
#             'evstation': stat,
#             'beta': beta,
#             'wiggleSIlow_act': wiggleSIlow_act,
#             'wiggleSI_act': wiggleSI_act,
#             'wiggleSIhigh_act': wiggleSIhigh_act,
#             'wiggleSIlow_calc': wiggleSIlow_calc,
#             'wiggleSI_calc': wiggleSI_calc,
#             'wiggleSIhigh_calc': wiggleSIhigh_calc
#         }

#     except Exception as e:
#         print(f"Error processing {i}: {e}")
#         traceback.print_exc()
#         return None
    

# results = []
# with ProcessPoolExecutor(max_workers=64) as executor:
#     futures = [executor.submit(process_waveform, wf, eventdata) for wf in waveforms]
#     # for fut in as_completed(futures):
#     #     res = fut.result()
#     #     if res:
#     #         results.append(res)

#     for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing waveforms"):
#         res = fut.result()
#         if res:
#             results.append(res)

# # Merge results back into eventdata
# results_df = pd.DataFrame(results)
# # eventdata = eventdata.merge(results_df, on=['evtime','evstation'], how='left')

# # eventdata = eventdata.dropna(subset=['wiggleSI_act', 'wiggleSI_calc'])
# eventdata.to_csv(output_dir+'_events_measurements_'+freqType+'_sep16.txt', sep='|', index=False)