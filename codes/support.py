from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.gridspec as gridspec
import os.path
try:
    import matplotlib
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# import splitwavepy 
from splitwavepy.core.data import Data, WindowPicker
import copy 
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.filter import envelope

from scipy.stats import t as stats_t

from scipy.interpolate import interp1d
from scipy.fftpack import fft

from obspy.taup import TauPyModel
model = TauPyModel(model="iasp91")

# from pyproj import Geod  # ← ADD THIS LINE
# import cartopy.crs as ccrs
import re





# import matplotlib.pyplot as plt

# turn these back on when not doing synthetic work, just turning it off for now to avoid dependency issues
# import cartopy.crs as ccrs
# from pyproj import Geod

#################### particle motion for plot_trace ####################

def trace_ppm(self, ax,windowstart,windowend, **kwargs):
    """Plot particle motion on *ax* matplotlib axis object, specifically for plot_trace code.
    Edited Nov 2024 to include windowstart and windowend for truncated traces
    """
    
    data = self.copy()
    # data.rotateto(0)
    x, y = data.x[windowstart:windowend] , data.y[windowstart:windowend]
    t = data.t()[windowstart:windowend]

    # # middle third of uncut trace 
    # x = x[int(len(x)*0.33):int(len(x)*0.66)]  
    # y = y[int(len(y)*0.33):int(len(y)*0.66)]
    # t = t[int(len(t)*0.33):int(len(t)*0.66)]

            
    # plot data
    # ax.plot(self.chop().y,self.chop().x)
    
    # multi-colored
    norm = plt.Normalize(t.min(), t.max())
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma', norm=norm, alpha=0.7)
    lc.set_array(t)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    # plt.colorbar(line)

    # set limit
    lim = np.abs(self.data()).max() * 1.1
    if 'lims' not in kwargs: kwargs['lims'] = [-lim, lim] 
    ax.set_aspect('equal')
    ax.set_xlim(kwargs['lims'])
    ax.set_ylim(kwargs['lims'])

    '''
    november 11 2024 
    adding option to put different labels
    '''

    # set labels
    if 'cmplabels' not in kwargs: kwargs['cmplabels'] = data.cmplabels

    if 'labels' in kwargs:
        # ax.set_xlabel('Radial')
        # ax.set_ylabel('Transverse')
        ax.set_xlabel(kwargs['labels'][0] + '   BAZ: ' + kwargs['baz'])
        ax.set_ylabel(kwargs['labels'][1])
    
    # turn off tick annotation
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    return

############################################################



#################### particle motion for eigenvalue method ####################

def _pppm(self, baz,calcbaz, ax, **kwargs):
    """Plot particle motion on *ax* matplotlib axis object.
    """
    
    data = self.copy()
    # data.rotateto(0)
    x, y = data.x , data.y
    t = data.t()
            
    # plot data
    # ax.plot(self.chop().y,self.chop().x)
    
    # multi-colored
    norm = plt.Normalize(t.min(), t.max())
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma', norm=norm, alpha=0.7)
    lc.set_array(t)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    # plt.colorbar(line)

    # set limit
    lim = np.abs(self.data()).max() * 1.1
    if 'lims' not in kwargs: kwargs['lims'] = [-lim, lim] 
    ax.set_aspect('equal')
    ax.set_xlim(kwargs['lims'])
    ax.set_ylim(kwargs['lims'])

    # set labels
    if 'cmplabels' not in kwargs: kwargs['cmplabels'] = data.cmplabels
    ax.set_xlabel('Radial'+ '\n' + 
                  '(Actual baz: ' + str(round(float(baz),3))+')' + '\n' + 
                  '(Calculated baz: ' + str(round(calcbaz,3)) + ' or ' + str(round((calcbaz+180),3)) + ')'  )
    ax.set_ylabel('Transverse')
    
    # turn off tick annotation
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    return

############################################################



#################### particle motion for transverse minimization method ####################

def trans_pppm(self, baz,calcbaz, ax, **kwargs):
    """Plot particle motion on *ax* matplotlib axis object.
    """
    
    data = self.copy()
    # data.rotateto(0)
    x, y = data.x , data.y
    t = data.t()
            
    # plot data
    # ax.plot(self.chop().y,self.chop().x)
    
    # multi-colored
    norm = plt.Normalize(t.min(), t.max())
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma', norm=norm, alpha=0.7)
    lc.set_array(t)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    # plt.colorbar(line)

    # set limit
    lim = np.abs(self.data()).max() * 1.1
    if 'lims' not in kwargs: kwargs['lims'] = [-lim, lim] 
    ax.set_aspect('equal')
    ax.set_xlim(kwargs['lims'])
    ax.set_ylim(kwargs['lims'])

    # set labels
    if 'cmplabels' not in kwargs: kwargs['cmplabels'] = data.cmplabels
    ax.set_xlabel('Radial'+ '\n' + 
                  '(Actual baz: ' + str(round(float(baz),3))+')' 
                  )
    ax.set_ylabel('Transverse')
    
    # turn off tick annotation
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    return

############################################################



#################### plots trace and particle motion only ####################

def plot_trace(data,baz, **kwargs):
    """
    Plot trace data and particle motion
    """

    fig = plt.figure(figsize=(12, 3))     
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    
    # trace
    ax0 = plt.subplot(gs[0])
    data._ptr(ax0, **kwargs)

    ####### oct 18th, 2024 ########
    # add a title based on passed kwargs 
    if 'title' in kwargs:
        ax0.set_title(f"{str(kwargs['title'])}")
    # only for trace portion of plot, more visible title
    #####################################


    
    # particle  motion
    ax1 = plt.subplot(gs[1])
    # data._ppm( ax1, **kwargs) 
    # trace_ppm(data,ax1,**kwargs) 
    trans_pppm(data, baz,0, ax1, **kwargs) 
    
                                
    # show
    plt.tight_layout()
    
    if not 'figname' in kwargs:
        kwargs['figname'] = "dataPlot.png"

    if not 'dpi' in kwargs:
        kwargs['dpi'] = 300

    # if 'title' in kwargs:
    #       plt.title(f"{str(kwargs['title'])}")

    plt.savefig(kwargs['figname'], bbox_inches='tight', dpi=kwargs['dpi'])

############################################################



#################### returns maximum lambda ratio ####################

def lambda_filter_2(self,**kwargs):             # used to filter only the most quality eigenvalue measurements (when including all data, not filtering during the measurement process)
                                                                    # uses all the data from eigenvalue method including errorbars and lambda ratio 
    if 'vals' not in kwargs: 
        kwargs['vals'] = self.lam1 / self.lam2
        
    return np.max(kwargs['vals']), self.dfast, self.dlag

############################################################



#################### returns maximum cross correlation (w/ preset constraints) ####################

def rotation_filter(self,minxc,maxfast,maxlag,**kwargs):             # used to filter only the most quality eigenvalue measurements 
                                                                    # uses all the data from eigenvalue method including errorbars and lambda ratio 
    if 'vals' not in kwargs: 
        kwargs['vals'] = self.xc
        
    if np.max(kwargs['vals']) >= minxc:
        if self.dlag <= maxlag:
            if self.dfast <= maxfast:
                        # return True
                        return np.max(kwargs['vals']), self.dfast, self.dlag
            else:
                        return False
        else:
                        return False 
    else:
                        return False
    
############################################################



#################### eigenvalue method splitting corrections and gridsearch ####################

def plot_eigen_measure(m,baz,title,**kwargs):
    # setup figure and subplots
    fig = plt.figure(figsize=(12,6)) 
    fig.suptitle("Eigenvalue Method for: " + title)
    gs = gridspec.GridSpec(2, 3,
                        width_ratios=[1,1,2]
                        )    
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[0,1])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[1,1])
    ax4 = plt.subplot(gs[:,2])
    
    # data to plot
    d1 = m.data.chop()
    d1f = m.srcpoldata().chop()
    d2 = m.data_corr().chop()
    d2s = m.srcpoldata_corr().chop()

    # display predicted back azimuth
    calcbaz = m.srcpol()  
    if calcbaz < 0:
           calcbaz = calcbaz + 180


    print(type(d1))
    print(calcbaz)
    
    # flip polarity of slow wave in panel one if opposite to fast
    # d1f.y = d1f.y * np.sign(np.tan(m.srcpol()-m.fast))
    
    # get axis scaling
    lim = np.abs(d2s.data()).max() * 1.1
    ylim = [-lim,lim]

    # original
    d1f._ptr(ax0,ylim=ylim,cmplabels=('Radial','Transverse'),**kwargs)
    _pppm(d1f,baz,calcbaz,ax1,lims=ylim,**kwargs)
    # corrected
    d2s._ptr(ax2,ylim=ylim,cmplabels=('Radial','Transverse'),**kwargs)
    _pppm(d2s,baz,calcbaz,ax3,lims=ylim,**kwargs)

    # error surface
    if 'vals' not in kwargs:
        # kwargs['vals'] = (m.lam1 - m.lam2) / m.lam2
        # kwargs['title'] = r'$(\lambda_1 - \lambda_2) / \lambda_2$'
        kwargs['vals'] = m.lam1 / m.lam2
        kwargs['title'] = r'$\lambda_1 / \lambda_2$'
    
    # add marker and info box by default
    if 'marker' not in kwargs: kwargs['marker'] = True
    if 'info' not in kwargs: kwargs['info'] = True
    if 'conf95' not in kwargs: kwargs['conf95'] = True
    m._psurf(ax4,**kwargs)
    
    # title
    if m.name != 'Untitled':
        plt.suptitle(m.name)
    
    # neaten
    plt.tight_layout()
    # plt.show()
    if not 'figname' in kwargs:
        kwargs['figname'] = "eigenM.png"

    if not 'dpi' in kwargs:
        kwargs['dpi'] = 300
    # plt.savefig(.png')

    plt.savefig(kwargs['figname'], bbox_inches='tight', dpi=kwargs['dpi'])

############################################################



#################### transverse minimization method splitting corrections and grid search ####################

def plot_trans_measure(m,baz,dist,depth,title,**kwargs):
    # setup figure and subplots
    fig = plt.figure(figsize=(14,7)) 
    # fig.suptitle("Transversal Min. Method for: " + title + "")
    fig.suptitle(f"Transversal Min. Method for {title}, Distance of {round(dist/1000,0)} km, Depth of {depth} km.")
    gs = gridspec.GridSpec(2, 3,
                        width_ratios=[1,1,2]
                        )    
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[0,1])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[1,1])
    ax4 = plt.subplot(gs[:,2])  
    


    # data to plot
    d1 = m.data.chop()
    d1f = m.srcpoldata().chop()
    d2 = m.data_corr().chop()
    d2s = m.srcpoldata_corr().chop()

    # display predicted back azimuth
    calcbaz = m.srcpol()  
    if calcbaz < 0:
            calcbaz = calcbaz + 180


    # print(type(d1))
    # print(calcbaz)

    # flip polarity of slow wave in panel one if opposite to fast
    # d1f.y = d1f.y * np.sign(np.tan(m.srcpol()-m.fast))

    # get axis scaling
    lim = np.abs(d2s.data()).max() * 1.1
    ylim = [-lim,lim]

    # original
    d1f._ptr(ax0,ylim=ylim,cmplabels=('Radial','Transverse'),**kwargs)
    trans_pppm(d1f,baz,calcbaz,ax1,lims=ylim,**kwargs)
    # corrected
    d2s._ptr(ax2,ylim=ylim,cmplabels=('Radial','Transverse'),**kwargs)
    trans_pppm(d2s,baz,calcbaz,ax3,lims=ylim,**kwargs)

    # error surface
    if 'vals' not in kwargs:
        # kwargs['vals'] = (m.lam1 - m.lam2) / m.lam2
        # kwargs['title'] = r'$(\lambda_1 - \lambda_2) / \lambda_2$'
        kwargs['vals'] = m.lam1 / m.lam2
        kwargs['title'] = r'$\lambda_1 / \lambda_2$'

    # add marker and info box by default
    if 'marker' not in kwargs: kwargs['marker'] = True
    if 'info' not in kwargs: kwargs['info'] = True
    if 'conf95' not in kwargs: kwargs['conf95'] = True
    m._psurf(ax4,**kwargs)

    # title
    if m.name != 'Untitled':
        plt.suptitle(m.name)

    # neaten
    plt.tight_layout()
    # plt.show()
    if not 'figname' in kwargs:
        kwargs['figname'] = "eigenM.png"

    if not 'dpi' in kwargs:
        kwargs['dpi'] = 300
    # plt.savefig(.png')


    plt.savefig(kwargs['figname'], bbox_inches='tight', dpi=kwargs['dpi'])

############################################################



#################### select smaller window for trace ####################

def plotdata(self,baz,calcbaz,**kwargs):
       
    """
    Plot trace data and particle motion
    """

    self.rotateto(baz)

    fig = plt.figure(figsize=(12, 3))     
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    
    # trace
    ax0 = plt.subplot(gs[0])
    self._ptr( ax0,cmplabels=('Radial','Transverse'), **kwargs)
    
        # d1f._ptr(ax0,ylim=ylim,cmplabels=('Radial','Transverse'),**kwargs)
    # trans_pppm(d1f,baz,calcbaz,ax1,lims=ylim,**kwargs)

    # particle  motion
    ax1 = plt.subplot(gs[1])
    # self._ppm( ax1, **kwargs)  
    trans_pppm(self,baz,calcbaz,ax1,**kwargs) #lims = ylim???? (normalizing data)
    
    # optional pick window
    if 'pick' in kwargs and kwargs['pick'] == True:
        windowpicker = WindowPicker(self,fig,ax0)
        windowpicker.connect()
                                
    # show
    plt.tight_layout()
    plt.show()

    self.rotateto(0)
    plt.close()

    # added oct 14 2024 to store values of window start and end 
    return windowpicker.x1 , windowpicker.x2

############################################################


### covariance method to determine particle motion ### 

def covar_particle_motion(north,east):

    '''
    Covariance analysis to calculate axes of particle motion and back azimuth
    Maximum and minimum variance correspond to axes directions

    Inputs:
    north: N/S component of trace
    east: W/E component of trace 
    make sure components are filtered between 15-50s!!!

    Outputs:
    baz: back azimuth of main particle motion (polarization direction)
    xlam1: first eigenvalue
    xlam2: second eigenvalue
    '''

    N = len(north)
    co11 = 0
    co12 = 0 
    co22 = 0

    # covariance matrix
    for k in range(N):
        co11 += east[k]*east[k]
        co12 += east[k]*north[k]
        co22 += north[k]*north[k]
    
    # calculate eigenvalues 
    xp = -co11 - co22
    xq = co11*co22 - co12**2
    rad = (xp/2)**2 - xq 

    xlam1 = -xp/2 + np.sqrt(rad)
    xlam2 = -xp/2 - np.sqrt(rad)

    d22 = co22 - xlam1 
    d21 = co12 
    yvec = -d21 / d22 

    # calculate back azimuth 
    phi = 90 - np.arctan(yvec) * 180 / np.pi

    return float(phi),float(xlam1),float(xlam2)

#########################################################


######## nov4th senior thesis post processing trace plotting ##########

def plot_trace_final(pair,evbaz,calcbaz,windowstart,windowend,env,**kwargs):

    data = copy.deepcopy(pair)

    dataevbaz = copy.deepcopy(pair)
    dataevbaz.rotateto(evbaz)

    datacalcbaz = copy.deepcopy(pair)
    datacalcbaz.rotateto(calcbaz)

    fig = plt.figure(figsize=(12, 9))     
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1]) 

    ### row 1 
    data = data
    # trace 1 
    ax0 = plt.subplot(gs[0])
    ax0.plot( data.t(), data.x,c='blue',linestyle='--')    
    ax0.plot( data.t(), data.y,c='red',linestyle='--')
    ax0.plot(data.t()[20:windowstart],env[20:windowstart],c='red')
    ax0.plot(data.t()[windowstart:windowend],env[windowstart:windowend],c='blue')
    ax0.plot(data.t()[windowend:-20],env[windowend:-20],c='red')
    ax0.plot( data.t()[windowstart:windowend], data.x[windowstart:windowend],c='blue',linestyle='--')    
    ax0.plot( data.t()[windowstart:windowend], data.y[windowstart:windowend],c='red',linestyle='--')
    # ax0.vlines(data.t()[windowstart:windowend],ymax=max(max(data.x),max(data.y)),ymin=max(max(data.x),max(data.y)))
    ax0.axvspan(data.t()[windowstart], data.t()[windowend], color='lightblue', alpha=0.3)
    ax0.set_title(kwargs['title'])


    # ax0.vlines(data.t()[windowend],ymax=max(max(data.x),max(data.y)),ymin=max(max(data.x),max(data.y)))

    kwargs = {'labels': ['N/S', 'E/W'], 'baz': 'n/a'}
    # labels = ['direction1,direction2']
    ax1 = plt.subplot(gs[1])
    trace_ppm(data,ax1,windowstart,windowend,**kwargs)

    ### row 2 
    data = dataevbaz
    # trace 2
    ax2= plt.subplot(gs[2])
    ax2.plot( data.t(), data.x,c='blue',linestyle='--')    
    ax2.plot( data.t(), data.y,c='red',linestyle='--')
    # ax2.plot(data.t()[20:windowstart-20],noiseenv,c='red')
    # ax2.plot(data.t()[windowstart+20:windowend-20],signalenv,c='blue')
    ax2.plot( data.t()[windowstart:windowend], data.x[windowstart:windowend],c='blue',linestyle='--')    
    ax2.plot( data.t()[windowstart:windowend], data.y[windowstart:windowend],c='red',linestyle='--')
    ax2.axvspan(data.t()[windowstart], data.t()[windowend], color='lightblue', alpha=0.3)

    ax3 = plt.subplot(gs[3])
    kwargs = {'labels': ['Radial', 'Transverse'],'baz': str(round(evbaz,0))}

    trace_ppm(data,ax3,windowstart,windowend,**kwargs)




    # ### row 3
    data = datacalcbaz
    # # trace 3
    ax4 = plt.subplot(gs[4])
    # stax = classic_sta_lta(dataevbaz.x,200,800)
    # stay = classic_sta_lta(dataevbaz.y,200,800)

    # ax4.plot(stax)
    # ax4.plot(stay)
    # ax4.plot(stax+stay)


    ax4.plot( data.t(), data.x,c='blue',linestyle='--')    
    ax4.plot( data.t(), data.y,c='red',linestyle='--')
    ax4.plot( data.t()[windowstart:windowend], data.x[windowstart:windowend],c='blue',linestyle='--')    
    ax4.plot( data.t()[windowstart:windowend], data.y[windowstart:windowend],c='red',linestyle='--')
    ax4.axvspan(data.t()[windowstart], data.t()[windowend], color='lightblue', alpha=0.3)


    ax5 = plt.subplot(gs[5])
    ppmkwargs = {'labels': ['Radial Prime', 'Transverse Prime'],'baz': str(round(calcbaz,0))}

    trace_ppm(data,ax5,windowstart,windowend,**ppmkwargs)

    ####### nov 18th, 2024 ########
    # add a title based on passed kwargs 
    # if 'title' in kwargs:
    #     ax0.set_title(kwargs['title'])
    # only for trace portion of plot, more visible title
    #####################################

    plt.tight_layout()

    return fig
    # plt.show()



    # # trace
    # ax0 = plt.subplot(gs[0])
    # data._ptr(ax0, **kwargs)

    # ####### oct 18th, 2024 ########
    # # add a title based on passed kwargs 
    # if 'title' in kwargs:
    #     ax0.set_title(f"{str(kwargs['title'])}")
    # # only for trace portion of plot, more visible title
    # #####################################


    
    # # particle  motion
    # ax1 = plt.subplot(gs[1])
    # # data._ppm( ax1, **kwargs) 
    # # trace_ppm(data,ax1,**kwargs) 
    # trans_pppm(data, baz,0, ax1, **kwargs) 
    
                                
    # # show
    # plt.tight_layout()
    
    # if not 'figname' in kwargs:
    #     kwargs['figname'] = "dataPlot.png"

    # if not 'dpi' in kwargs:
    #     kwargs['dpi'] = 300

    # # if 'title' in kwargs:
    # #       plt.title(f"{str(kwargs['title'])}")

    # plt.savefig(kwargs['figname'],dpi=800)

def envelopes(pair,evbaz,calcbaz,windowstart,windowend,**kwargs):

    '''
    calculate envelope of phase window and noise window
    looks at snr for orthogonal seismograph components
    november 11 2024 nico valencia

    1. get 1* data pair correlating with the 1* coordinate system
    2. add envelope in quadrature for 2d vector of energy 
    3. get snr ratio based on mean envelope and max envelope 

    *** envelope is rotationally invariant!!! 
    only need to measure one of these coordinate systems!!!

    '''
    
    data = copy.deepcopy(pair)
    windowstart = int(windowstart)
    windowend = int(windowend)

    try:

        # datacalcbaz.rotateto(calcbaz)

        # assumes 40 seconds of trace available before windowstart... or just measure from the beginning?
        # examines 30 seconds before windowstart to use as noise
        # or just measure from the beginning <----- doing this right now

        # NORTH AND EAST
        n_envelope = envelope(data.x)
        e_envelope = envelope(data.y)

        ne_env = np.sqrt(n_envelope**2 + e_envelope**2)

        ne_mean_snr = np.mean(ne_env[windowstart:windowend]) / np.mean(ne_env[20:windowstart])
        ne_max_snr = np.max(ne_env[windowstart:windowend]) / np.max(ne_env[20:windowstart])

        ### snr for end of trace 
        ne_mean_snr2 = np.mean(ne_env[windowstart:windowend]) / np.mean(ne_env[windowend:-20])
        ne_max_snr2 = np.max(ne_env[windowstart:windowend]) / np.max(ne_env[windowend:-20])




        return [ne_mean_snr,ne_max_snr,ne_env,ne_mean_snr2,ne_max_snr2] 
    
    except Exception as exception: 
        print(exception)
        return [0,0,0,0,0]


def envelopes_rms(pair, evbaz, calcbaz, windowstart, windowend, **kwargs):
    '''
    calculate envelope of phase window and noise window
    looks at snr for orthogonal seismograph components
    november 11 2024 nico valencia

    1. get 1* data pair correlating with the 1* coordinate system
    2. add envelope in quadrature for 2d vector of energy 
    3. get snr ratio based on RMS of envelope

    *** envelope is rotationally invariant!!! 
    only need to measure one of these coordinate systems!!!
    '''
    
    data = copy.deepcopy(pair)
    windowstart = int(windowstart)
    windowend = int(windowend)

    try:
        # NORTH AND EAST
        n_envelope = envelope(data.x)
        e_envelope = envelope(data.y)

        ne_env = np.sqrt(n_envelope**2 + e_envelope**2)

        # RMS SNR
        signal_rms = np.sqrt(np.mean(ne_env[windowstart:windowend]**2))
        noise_before_rms = np.sqrt(np.mean(ne_env[20:windowstart]**2))
        noise_after_rms = np.sqrt(np.mean(ne_env[windowend:-20]**2))
        
        ne_rms_snr = signal_rms / noise_before_rms if noise_before_rms > 0 else 0
        ne_rms_snr2 = signal_rms / noise_after_rms if noise_after_rms > 0 else 0

        return [ne_rms_snr, ne_rms_snr2, ne_env] 
    
    except Exception as exception: 
        print(exception)
        return [0, 0, 0]






    # r_signal_env = envelope(dataevbaz.x)
    # t_signal_env = envelope(dataevbaz.y)


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

    # S_expec = -0.5 * SplitIntensity * rd 

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


    return m1low , m1est , m1high


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
    nf = int(round((n2 + 2) / 2))  # Only take half of the spectrum to Nyquist frequency
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
     

def plot_great_circle_ray_path(eq_lat, eq_lon, st_lat, st_lon, phase_type = None,phase_name=None, ev_depth=None,s_depth=None, ax=None):
    """
    Plot a great circle path between an earthquake and a station on a world map.
    
    Parameters:
        eq_lat, eq_lon: Latitude and longitude of the earthquake
        st_lat, st_lon: Latitude and longitude of the station
        phase_name: Optional label (e.g., 'P', 'S')
        ax: Optional Cartopy axis to plot on. If None, a new figure will be created.
    """
    # Initialize geod object for WGS84 ellipsoid
    geod = Geod(ellps="WGS84")

    # Compute intermediate points along great circle
    npts = 100
    intermediate_points = geod.npts(eq_lon, eq_lat, st_lon, st_lat, npts)

    # Add endpoints manually
    lons = [eq_lon] + [pt[0] for pt in intermediate_points] + [st_lon]
    lats = [eq_lat] + [pt[1] for pt in intermediate_points] + [st_lat]

    existing_labels = [h.get_label() for h in ax.get_legend_handles_labels()[0]]

    # Plot earthquake and station
    if phase_type == 'SKS':
        label = 'SKS phase' if 'SKS phase' not in existing_labels else None

        ax.plot(eq_lon, eq_lat, 'b', marker = '*',markersize=6, transform=ccrs.PlateCarree(),label=label)
            
        # ax.plot(eq_lon, eq_lat, 'b^', marker = '*',markersize=6, transform=ccrs.Mollweide())

        # Plot great circle path
        ax.plot(lons, lats, 'k--', linewidth=0.5, alpha=0.5,transform=ccrs.Geodetic())

    if (phase_type == 'S') & (ev_depth < s_depth):
      
        label = 'S shallow' if 'S shallow' not in existing_labels else None
        ax.plot(eq_lon, eq_lat, 'r', marker ='*',markersize=6, transform=ccrs.PlateCarree(),label=label)
        # Plot great circle path
        ax.plot(lons, lats, 'k--', linewidth=0.5, alpha=0.5,transform=ccrs.Geodetic())

    if (phase_type == 'S') & (ev_depth >= s_depth):
      
        label = 'S deep' if 'S deep' not in existing_labels else None
        ax.plot(eq_lon, eq_lat, 'g', marker ='*',markersize=6, transform=ccrs.PlateCarree(),label=label)
        # Plot great circle path
        ax.plot(lons, lats, 'k--', linewidth=0.5, alpha=0.5,transform=ccrs.Geodetic())
      


    ax.plot(st_lon, st_lat, 'yellow', marker='v',markersize=6,  transform=ccrs.PlateCarree())
    # ax.plot(st_lon, st_lat, 'yellow', marker='v',markersize=6,  transform=ccrs.Mollweide())




    # Optional labeling
    if phase_name:
        ax.text(eq_lon + 2, eq_lat + 2, f"EQ ({phase_name})", color='red', transform=ccrs.PlateCarree())
        ax.text(st_lon + 2, st_lat + 2, "Station", color='blue', transform=ccrs.PlateCarree())

    return ax

def dms_to_dd(dms_str):
    # Remove extraneous quotes and whitespace
    dms_str = dms_str.strip().replace('"', '').replace('′', "'").replace('″', '"')

    # Match DMS pattern like: 57 51' 26.19" S
    match = re.match(r"(\d+)[^\d]+(\d+)[^\d]+([\d.]+)\s*([NSEW])", dms_str)
    if match:
        deg, minute, second, direction = match.groups()
        dd = float(deg) + float(minute) / 60 + float(second) / 3600
    else:
        # Try to match simpler patterns like "57 S"
        match_simple = re.match(r"(\d+)\s*([NSEW])", dms_str)
        if match_simple:
            dd = float(match_simple.group(1))
            direction = match_simple.group(2)
        else:
            return None  # Unable to parse

    if direction in ['S', 'W']:
        dd *= -1
    return dd














# ########## supplementary data plots ##########

# def extradata(m,baz,depth,angle,title,**kwargs):
#     # setup figure and subplots
#     fig = plt.figure(figsize=(16,8)) 
#     fig.suptitle("extra data for: " + title)
#     gs = gridspec.GridSpec(3, 3,
#                         width_ratios=[1,1,2]
#                         )    
#     ax0 = plt.subplot(gs[0,0])
#     ax1 = plt.subplot(gs[0,1])
#     ax2 = plt.subplot(gs[1,0])
#     ax3 = plt.subplot(gs[1,1])
#     ax4 = plt.subplot(gs[:,2]) 
#     ax5 = plt.subplot(gs[2,0:1]) 
#     ax6 = plt.subplot(gs[2,1:2])  

#     # data to plot
#     d1 = m.data.chop()
#     d1f = m.srcpoldata().chop()
#     d2 = m.data_corr().chop()
#     d2s = m.srcpoldata_corr().chop()

#     # display predicted back azimuth
#     calcbaz = m.srcpol()  
#     if calcbaz < 0:
#             calcbaz = calcbaz + 180


#     print(type(d1))
#     print(calcbaz)

#     # flip polarity of slow wave in panel one if opposite to fast
#     # d1f.y = d1f.y * np.sign(np.tan(m.srcpol()-m.fast))

#     # get axis scaling
#     lim = np.abs(d2s.data()).max() * 1.1
#     ylim = [-lim,lim]

#     # original
#     d1f._ptr(ax0,ylim=ylim,cmplabels=('Radial','Transverse'),**kwargs)
#     trans_pppm(d1f,baz,calcbaz,ax1,lims=ylim,**kwargs)
#     # corrected
#     d2s._ptr(ax2,ylim=ylim,cmplabels=('Radial','Transverse'),**kwargs)
#     trans_pppm(d2s,baz,calcbaz,ax3,lims=ylim,**kwargs)

#     # error surface
#     if 'vals' not in kwargs:
#         # kwargs['vals'] = (m.lam1 - m.lam2) / m.lam2
#         # kwargs['title'] = r'$(\lambda_1 - \lambda_2) / \lambda_2$'
#         kwargs['vals'] = m.lam1 / m.lam2
#         kwargs['title'] = r'$\lambda_1 / \lambda_2$'

#     # add marker and info box by default
#     if 'marker' not in kwargs: kwargs['marker'] = True
#     if 'info' not in kwargs: kwargs['info'] = True
#     if 'conf95' not in kwargs: kwargs['conf95'] = True
#     m._psurf(ax4,**kwargs)

#     # ray paths and arrival times
#     traveltimes = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=angle) #,phase_list=["P","S","SKS", "SKKS","ScS"])
#     arrivals = model.get_ray_paths(source_depth_in_km=depth, distance_in_degree=angle) #,phase_list=["P","S","SKS", "SKKS","ScS"])
#     arrivals.plot_rays(plot_type='cartesian',legend=False,show=False,fig=fig,ax=ax5)
#     arrivals.plot_times(legend=False,show=False,fig=fig,ax=ax6)
    




    

#     # ax6
#     # plt.table(traveltimes[0],traveltimes[1],loc=ax6)
#     # plt.axis('off')
#     print(traveltimes)
#     print(traveltimes[0])
#     print(len(traveltimes))
#     print(type(traveltimes))


#     # title
#     if m.name != 'Untitled':
#         plt.suptitle(m.name)

#     # neaten
#     plt.tight_layout()
#     # plt.show()
#     if not 'figname' in kwargs:
#         kwargs['figname'] = "eigenM.png"

#     if not 'dpi' in kwargs:
#         kwargs['dpi'] = 300
#     # plt.savefig(.png')


#     plt.savefig(kwargs['figname'], bbox_inches='tight', dpi=kwargs['dpi'])

# ##############################




#################### old code ####################
    
# def plot_measure(m,**kwargs):
#     # setup figure and subplots
#     fig = plt.figure(figsize=(12,6)) 
#     gs = gridspec.GridSpec(2, 3,
#                         width_ratios=[1,1,2]
#                         )    
#     ax0 = plt.subplot(gs[0,0])
#     ax1 = plt.subplot(gs[0,1])
#     ax2 = plt.subplot(gs[1,0])
#     ax3 = plt.subplot(gs[1,1])
#     ax4 = plt.subplot(gs[:,2])
    
#     # data to plot
#     d1 = m.data.chop()
#     d1f = m.srcpoldata().chop()
#     d2 = m.data_corr().chop()
#     d2s = m.srcpoldata_corr().chop()
    
#     # flip polarity of slow wave in panel one if opposite to fast
#     # d1f.y = d1f.y * np.sign(np.tan(m.srcpol()-m.fast))
    
#     # get axis scaling
#     lim = np.abs(d2s.data()).max() * 1.1
#     ylim = [-lim,lim]

#     # original
#     d1f._ptr(ax0,ylim=ylim,**kwargs)
#     d1._ppm(ax1,lims=ylim,**kwargs)
#     # corrected
#     d2s._ptr(ax2,ylim=ylim,**kwargs)
#     d2._ppm(ax3,lims=ylim,**kwargs)

#     # error surface
#     if 'vals' not in kwargs:
#         # kwargs['vals'] = (m.lam1 - m.lam2) / m.lam2
#         # kwargs['title'] = r'$(\lambda_1 - \lambda_2) / \lambda_2$'
#         kwargs['vals'] = m.lam1 / m.lam2
#         kwargs['title'] = r'$\lambda_1 / \lambda_2$'
    
#     # add marker and info box by default
#     if 'marker' not in kwargs: kwargs['marker'] = True
#     if 'info' not in kwargs: kwargs['info'] = True
#     if 'conf95' not in kwargs: kwargs['conf95'] = True
#     m._psurf(ax4,**kwargs)
    
#     # title
#     if m.name != 'Untitled':
#         plt.suptitle(m.name)
    
#     # neaten
#     plt.tight_layout()
#     # plt.show()
#     if not 'figname' in kwargs:
#         kwargs['figname'] = "eigenM.png"

#     if not 'dpi' in kwargs:
#         kwargs['dpi'] = 300

#     # plt.savefig(kwargs['figname'], bbox_inches='tight', dpi=kwargs['dpi'])
#     plt.show()


# def lambda_filter(self,minlam,maxfast,maxlag,**kwargs):             # used to filter only the most quality eigenvalue measurements 
#                                                                     # uses all the data from eigenvalue method including errorbars and lambda ratio 
#     if 'vals' not in kwargs: 
#         kwargs['vals'] = self.lam1 / self.lam2
        
#     if np.max(kwargs['vals']) >= minlam:
#         if self.dlag <= maxlag:
#             if self.dfast <= maxfast:
#                         # return True
#                         return np.max(kwargs['vals']), self.dfast, self.dlag
#             else:
#                         return False
#         else:
#                         return False 
#     else:
#                         return False

############################################################