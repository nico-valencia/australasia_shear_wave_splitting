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
from scipy.stats import t as stats_t

from scipy.interpolate import interp1d
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



'''
attempting to see why SI values measure so low...

goals: 
plot SI corrected wiggles w/ radial and transverse, and see if derivative is working well 
identify stats on why the SI values are so low and then so high at different locations

results: THE WINDOW IS NOT EVEN A WINDOW WTFFFFFFF
'''

# ----------------- PARAMETERS -----------------
data_type   = 'complete'
data_tag    = 'AU'
station     = 'australia'
freqType    = "normal"
date        = 'sep17'

# Frequency selection
if freqType == 'low':
    freqmin, freqmax = 0.04, 0.125
elif freqType == 'medium':
    freqmin, freqmax = 0.125, 0.2
elif freqType =='high':
    freqmin, freqmax = 0.2, 1
elif freqType == 'normal':
    freqmin, freqmax = 0.01, 0.125

events_file     = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_windowed_events_{freqType}_{date}.txt'
station_file    = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_stations_info.txt'
output_path     = f'/work/gcl3/BR/nicolasv03/senior_thesis/data/{data_type}/{station}/{data_tag}_measured_events_{freqType}_{date}.txt'

eventdata = pd.read_csv(events_file,sep='|')


def SplittingIntensity(pair,baz):

    '''
    Calculate Splitting Intensity based on Chevrot (2000)
    Uses the clipped waveform based on input (pair)
    maximum amplitude of the transverse divided by 
    the maximum amplitude of the time derivative of the radial

    Written by Neala Creasy, altered May 2019 
    transcribed to Python by Nicolas Valencia
    '''

    data = copy.deepcopy(pair)
    data.rotateto(baz)          # make sure to define which back azimuth being used for this!!!

    R = data.x
    T = data.y
    sample_rate = data.delta 

    rd = np.diff(R) / sample_rate 

    T = T[:-1]
    Pop = np.dot(T,rd)
    NPop = np.linalg.norm(rd)**2

    SplitIntensity = -2 * Pop / NPop 

    m1est = SplitIntensity 

    S_expec = -0.5 * SplitIntensity * rd 

    # root mean square error 
    rdelta = np.sum((T + 0.5*SplitIntensity*rd)**2)
    ndf = GetNDF(T,len(T),len(T))
    # ndf = 0 # place holder for now, need to define what ndf is !!!! getndf(T,len(T),len(rd))
    variance = rdelta / (ndf -1)

    tin = stats_t.ppf(0.95,ndf-1)

    # sum of squared errors for variance 
    ssee = np.sum((rd-np.mean(rd))**2)

    sm2 = variance / ssee
    m1high = SplitIntensity + tin*np.sqrt(sm2)
    m1low = SplitIntensity - tin*np.sqrt(sm2)


    return m1low , m1est , m1high, S_expec

def GetNDF(S,n,norig):
    '''
    Compute the effective number of degrees of freedom in a time series. 

    taken from Wüstefeld et al. 2007
    altered by F. Link & M.Reiss 2019

    Borrowed from Jonathan Wolf. Transcribed to Python by Nicolas Valencia.
    '''

    S = np.asarray(S)
    if S.ndim == 1:
        S = S[:, np.newaxis]  # Ensure column vector

    # Apply a cosine taper at the ends (taper length 20% of the time window)
    tap = np.linspace(0, 1, int(n * 0.2))[::-1].reshape(-1, 1)  # Ensure it's a column vector
    A = S.copy()
    A[:len(tap)] *= (np.cos(tap * np.pi) + 1) / 2
    A[-len(tap):] *= (np.cos(tap[::-1] * np.pi) + 1) / 2

    # tap = np.linspace(0, 1, int(n * 0.2))[::-1]
    # A = S.copy()
    
    # Interpolate array
    n2 = 2 ** (round(np.log2(n - 0.1)) + 3)
    x = np.linspace(1, len(A), n2)
    interp_func = interp1d(np.arange(1, len(A) + 1), A.flatten(), kind='linear', fill_value='extrapolate')
    An = interp_func(x)
    
    # Compute FFT
    OUT = fft(An)

    # Compute the number of degrees of freedom
    nf = round((n2 + 2) / 2)  # Only take half of the spectrum to Nyquist frequency
    mag = np.abs(OUT[:nf])
    
    E2 = np.sum(mag**2)
    E4 = np.sum(mag**4)
    
    E2 -= 0.5 * mag[0]**2 + 0.5 * mag[-1]**2
    E4 = 4 * E4 / 3 - mag[0]**4 - mag[-1]**4
    
    # Compute the estimated NDF
    NDF = round(2 * (2 * E2 * E2 / E4 - 1))
    
    if NDF > norig:
        raise ValueError("get_ndf: NDF > NORIG")

    '''
    calculate the Number of Degrees-Of-Freedom of a summed and squared time
    series with spectrum array A. A is assumed to be a complex function of
    frequency with A[0] corresponding to zero frequency and A[N-1]
    corresponding to the Nyquist frequency.  If A[t] is gaussian distributed
    then ndf_spect should be the points in the original time series 2*(N-1).

    based on theory, the following expression should yield
    the correct number of degrees of freedom.(see appendix of silver
    and chan, 1990.  In practise, it gives a value that gives
    a df that is slightly too large.  eg 101 for an original time
    series of 100.  This is due to the fact that their is a slight
    positive bias to the estimate using only the linear term.

    modified after Walsh et al. 2013
    '''
    
    return NDF


stat = 'CMSA'

print(eventdata.columns)

for i in range(0,len(eventdata)):
    try:
        row = eventdata.iloc[i]
        wf = read(row['waveform_path'])
        # print(row['waveform_path'])

        if row['evstation'] != stat:
                continue
        
        # Filtering
        wf.detrend('demean')
        wf.detrend('linear')
        wf.filter("bandpass", freqmin=freqmin, freqmax=freqmax)

        north = wf.select(component="N", channel='BH*')[0].data
        east  = wf.select(component="E", channel='BH*')[0].data
        sample_interval = wf[0].stats.delta

        if len(north) % 2 == 0: north = north[:-1]
        if len(east) % 2 == 0: east = east[:-1]

        evwindowstart = int(row['windowstart'] / sample_interval)
        evwindowend = int(row['windowend'] / sample_interval)
        print(evwindowstart,evwindowend)
        pair = sw.Pair(north, east, delta=sample_interval)
        rt = copy.deepcopy(pair)
        rt.rotateto(row['evbaz'])
        pp = copy.deepcopy(pair)
        pp.rotateto(row['calcbaz'])

        si_1 = copy.deepcopy(pair)
        print(f"length of pair before setting window:{len(si_1.x)}")
        si_1.set_window(row['windowstart'],row['windowend'])
        print(f"length of pair after setting window:{len(si_1.x)}")
        # print(dir(si_1))
        # print(vars(pair.window))
        # print(si_1.window.start)
        new_chop = si_1.chop()
        print(f"length of pair after chopping window:{len(new_chop.x)}")


        si_2 = copy.deepcopy(pair)

        si_l, si, si_h, transexpec = SplittingIntensity(si_2,float(row['calcbaz']))

        # print('lengths of stuff')
        # print(len(si_2.y),len(transexpec))

        # print('length of si_1,si_2')
        # print(len(si_1.x),len(si_2.x))






        # si_1.chop()
        # si_2 = copy.deepcopy(pair)
        # si_2.set_window(row['windowstart'],row['windowend'])
        # window_time = si_1.t()[row['windowstart'],row['windowend']]

        # pair.set_window(float(evwindowstart), float(evwindowend))
        # beta = ((((float(row['calcbaz']) - float(row['evbaz'])) % 180) + 90) % 180) - 90

        fig = plt.figure(figsize=(20, 30))
        fig.supxlabel(f"stat:{row['evstation']} phase:{row['evphase']} depth:{row['evdp']} mag:{row['evmg']}")
        gs = gridspec.GridSpec(9, 6, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, :4])  # top-left
        ax2 = fig.add_subplot(gs[3:5, :4])  # bottom-left
        ax3 = fig.add_subplot(gs[6:8, :4])
        ax4 = fig.add_subplot(gs[:4, 4:])
        ax5 = fig.add_subplot(gs[5:, 4:])


        ax1.set_title('north / east')
        ax1.plot(pair.t(),pair.x,label='north')
        ax1.plot(pair.t(),pair.y,label='east')
        ax1.axvspan(pair.t()[evwindowstart], pair.t()[evwindowend], color='lightblue', alpha=0.3)
        ax1.legend()

        ax2.set_title(f"R/T:{round(row['evbaz'],0)}")
        ax2.plot(rt.t(),rt.x,label='radial')
        ax2.plot(rt.t(),rt.y,label='transverse')
        ax2.legend()

        ax3.set_title(f"P1/P2: {round(row['calcbaz'],0)}")
        ax3.plot(pp.t(),pp.x,label='p1')
        ax3.plot(pp.t(),pp.y,label='p2')
        ax3.legend()

        # ax4.set_title(f"SI: {round(row[''],0)}")
        ax4.plot(si_1.t()[evwindowstart:evwindowend],si_1.x[evwindowstart:evwindowend],label='p1')
        ax4.plot(si_1.t()[evwindowstart:evwindowend],si_1.y[evwindowstart:evwindowend],label='p2')
        ax4.legend()

        ax5.plot(si_1.t()[evwindowstart:evwindowend],si_1.x[evwindowstart:evwindowend],label='p1')
        ax5.plot(si_1.t()[evwindowstart:evwindowend],transexpec[evwindowstart:evwindowend],label='p1')




        # ax1.plot(pair.t(),pair.x)
        # ax1.plot(pair.t(),pair.y)
        # plt.figure(figsize=(12,16))

        # plt.plot(north)
        # plt.plot(east)
        plt.show()
        plt.close('all')

    except Exception:
        print('did not work')
        # print(f"Error: {exception}")
        traceback.print_exc()