#!/usr/bin/env python

########################################################################
# Notes:
# - General configuration appears below in the "config" section
# - Paths are generated and plotted in the main define_and_plot_xsecs()
#   routine (see that function for examples)
########################################################################

'''
Being used by Nico Valencia for southwest Pacific xsections of deep event earthquakes
Nov 21 2024
'''

import json
import os
import pickle
from obspy.taup import TauPyModel
import copy 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

#########
import sys
path2ucbpy = '/work/range/BR/utpalkumar/FWI/000_bsl_group/FWI_github/ucbpy/python3'
sys.path.insert(0, path2ucbpy)
############

import pyspl
from ModelA3d import ModelA3d
from Model1D import Model1D
from Sphere import delaz, gc_minor, shoot
from UCBColorMaps import cmapSved, cmapXved

from FigTools import plot_plates, plot_hotspots, plot_gc_clean, get_tectonics_path
from circle import circle_dist


########################################################################
# const

r_earth = 6371.0


########################################################################
# config

# parameter to plot (A3d descriptor name)
param = 'S'

# # model configuration
# model_config = {
#     # simply for naming of output files
#     'name': 'SEMUCB',
#     # A3d file
#     'modelfile': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/A3d.dat',
#     # 1D reference model (needed for plotting xi sections)
#     'refmodelfile': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/model1D.dat',
#     # spherical spline grid files
#     'gridfiles':  {
#         'S': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/hknots.dat',
#         'X': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/hknots2.dat'}}

# model configuration
model_config = {
    # simply for naming of output files
    'name': 'SEMUCB',
    # A3d file
    'modelfile': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/A3d.dat',
    # 1D reference model (needed for plotting xi sections)
    'refmodelfile': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/model1D.dat',
    # spherical spline grid files
    'gridfiles':  {
        'S': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/hknots.dat',
        'X': '/data/gcl2/BR/d.soergel/SEMUCB_A3d/hknots2.dat'}}

# min / max depths to plot
dmin = 50.0
dmax = 2900.0

# number of radial samples
nr = 200

# overall saturation level (%)
#vmax = 3.0
vmax = 2.5

# change in saturation above a specified depth (set to vmax to disable)
upper_sat = vmax
upper_sat_depth = 410.0

# whether to frame section closely
tight_axes = True
tight_axes_buffer_km = 150.0

# number of waypoints to mark on section / map
num_waypoints = 5

# saving sections
## whether to save
save = True
## where to save
# save_path = '/scr/01/barbara/,gung'
save_path = '/work/gcl3/BR/nicolasv03/senior_thesis/codes/Plotting'
## extensions and options: eps
#save_ext = 'eps'
#save_kwargs = {}
## extensions and options: png
save_ext = 'png'
save_kwargs = {'dpi': 300}

# include color bar?
colorbar = True

# figure size (inches: (w,h))
figure_config = {'figsize': (10,5)}

# whether to include lats / lons grid on map
map_draw_lats_lons = False


########################################################################
# globals

# stores hotspot locations
named_hotspots = None


########################################################################
# support routines

def recenter_cmap(cmap, new_center, old_center=0.5):
    '''Recenters the matplotlib colormap `cmap` around `new_center` (assuming
    it was previously centered at kwarg `old_center`, which defaults to 0.5).
    '''
    segmentdata = cmap._segmentdata
    new_segmentdata = {}
    for color in segmentdata:
        new_levels = []
        for level in segmentdata[color]:
            x, above, below = level
            if x <= old_center:
                new_x = x * new_center / old_center
            else:
                new_x = new_center + (1.0 - new_center) * (x - old_center) / (
                        1.0 - old_center)
            new_levels.append((new_x, above, below))
        new_segmentdata[color] = tuple(new_levels)
    return LinearSegmentedColormap(cmap.name + '-recentered', new_segmentdata)

def plot_deep_eqs(catalog, plons, plats, dist, width = 2.0):
    '''Plot deep earthquakes within distance `width` of the path specified by
    `plons` and `plats` (and along-section distance `dist`) from the eq catalog
    `catalog` (tuple: lons, lats, depths).
    '''
    # unpack eq catalog
    lons, lats, depths = catalog
    # to radians / colats
    thetas = np.pi / 2 - np.radians(lats)
    phis = np.radians(lons)
    p_thetas = np.pi / 2 - np.radians(plats).reshape((plats.size,1))
    p_phis = np.radians(plons).reshape((plons.size,1))
    # define distance-cosine threshold
    thresh = np.cos(np.radians(width))
    # compute distance cosines (path points x eqs)
    cos_dist = np.cos(thetas) * np.cos(p_thetas) + np.sin(thetas) * np.sin(p_thetas) * np.cos(phis - p_phis)
    # find and plot all eqs within the distance threshold
    i_eq, = np.nonzero(cos_dist.max(axis=0) >= thresh)
    for i in i_eq:
        i_p = np.argmax(cos_dist[:,i])
        x = np.sin(np.radians(dist[i_p])) * (r_earth - depths[i])
        y = np.cos(np.radians(dist[i_p])) * (r_earth - depths[i])
        plt.scatter(x, y, c='k', s=2, alpha=0.2)

def load_eq_catalog(fname, fields = None, delimiter=','):
    '''Load the earthquake catalog, potentially filtering only to the specified
    fields
    '''
    with open('%s/%s' % (get_tectonics_path(), fname)) as f:
        hfields = f.readline().split(delimiter)
        ls = [l.split(delimiter) for l in f.readlines()]
    if fields is None:
        return ls
    else:
        inds = [hfields.index(field) for field in fields]
        return [[l[i].strip() for i in inds] for l in ls]

def plot_xsec(p1, p2, model_config, xsec_name, fig_id_text, basemap_args,
        nx=100, show_eqs=True):
    # declare named_hotspots as global
    global named_hotspots

        # Original endpoints
    p1_orig = p1   # Left endpoint
    p2_orig = p2   # Right endpoint

    # Get azimuths along the great circle
    _, az_p1_to_p2 = delaz(p1_orig, p2_orig)
    _, az_p2_to_p1 = delaz(p2_orig, p1_orig)

    # Extend beyond p1
    overshoot_p1 = 10  # degrees
    az_beyond_p1 = (az_p1_to_p2 + 180) % 360
    p1_extended = shoot(p1_orig, overshoot_p1, az_beyond_p1)

    # Extend beyond p2
    overshoot_p2 = 20  # degrees
    az_beyond_p2 = (az_p2_to_p1 + 180) % 360
    p2_extended = shoot(p2_orig, overshoot_p2, az_beyond_p2)

    # Use extended points
    p1 = p1_extended
    p2 = p2_extended

    print(f"Extended p1: {p1}")
    print(f"Extended p2: {p2}")

    print(f"Azimuth from p1 to p2: {az_p1_to_p2:.1f}°")
    print(f"Azimuth from p2 to p1: {az_p2_to_p1:.1f}°")

    # p2_current = p2
    # # Calculate azimuth from ARMA to current source
    # delta_current, az_to_source = delaz(p1, p2_current)
    # print(f"Current distance: {delta_current:.1f}°, azimuth: {az_to_source:.1f}°")

    # # Extend p1 farther in the same direction
    # desired_distance = 120  # degrees (or whatever you want)
    # p2_extended = shoot(p1, desired_distance, az_to_source)

    # # also expand p1 so station is within nice view
    # delta_current, az_to_station = delaz(p2_extended, p1)
    # station_shoulder = desired_distance + 10  
    # p1_extended = shoot(p2_extended, station_shoulder, az_to_station)

    # print(f"Original p2: {p2_current}")
    # print(f"Extended p2: {p2_extended}")

    # # Use extended p1 for cross-section
    # p2 = p2_extended
    # p1 = p1_extended

    # # # truncate window to see only source of events ONLY FOR FITZ
    # p1 = shoot(p2,60,100)
    # p2new = shoot(p1,20,260)
    # # deltastat = 0 

    # deltastat,azstat=delaz(p2,p2new)
    # # print('deltastat:',deltastat)
    # p2 = p2new

    # # # # truncate window to see only source of events ONLY FOR STKA
    # p1 = shoot(p2,50,360)
    # p2new = shoot(p1,20,180)
    # # deltastat = 0 

    # deltastat,azstat=delaz(p2,p2new)
    # # print('deltastat:',deltastat)
    # p2 = p2new

    deltastat = 0 
    # setup sampling path
    # distance, azimuth
    print(p1,p2)
    delta, az = delaz(p1,p2)
    print(delta)
    # step size
    dx = delta / (nx - 1)
    # compute great-circle path
    lons, lats = gc_minor(p1, p2, dx)
    # lons, lats = shoot(p1,delta,100)
    print('delta = %f degrees' % (delta))
    print('dx = %f degrees' % (dx))

    # setup depth sampling
    dr = (dmax - dmin) / (nr - 1)
    r = r_earth - dmax + dr * np.arange(nr)
    print('dr = %f km' % (dr))

    # load the A3d model
    model = ModelA3d(model_config['modelfile'])
    model.load_from_file()
    coefs = model.get_parameter_by_name(param).get_values()

    #### interpolation phase 1 (model) ####

    # sample the model (polar coords in great-circle plane)
    # load the grid
    grid = np.loadtxt(model_config['gridfiles'][param], skiprows=1)
    # compute the sspl interpolant
    sspl = pyspl.SphericalSplines(grid[:,0], grid[:,1], grid[:,2])
    H = sspl.evaluate(lons.ravel(), lats.ravel())
    # compute the bspl interpolant
    bspl = pyspl.CubicBSplines(model.get_bspl_knots())
    V = bspl.evaluate(r)
    # sample
    x = (H * (V * coefs).T).T

    # plot xi as percent relative to isotropy
    if param == 'X':
        ref = Model1D(model_config['refmodelfile'])
        ref.load_from_file()
        vsv0 = ref.get_values(1000 * r, parameter='vsv').reshape((nr,1))
        vsh0 = ref.get_values(1000 * r, parameter='vsh').reshape((nr,1))
        x = (1.0 + x) * vsh0 ** 2 / vsv0 ** 2 - 1.0

    # to percent
    x *= 100

    # re-saturate in upper portion, if desired
    resat = False
    if upper_sat != vmax:
        w = np.ones(x.shape)
        w[r > r_earth - upper_sat_depth] *= vmax / upper_sat
        x *= w
        resat = True

    #### interpolation phase 2 (image) ####

    # define circular coord grid in the great-circle plane (polar coords) over
    # the interval [-delta / 2, delta / 2]
    t = dx * np.arange(nx) - delta / 2
    tg, rg = np.meshgrid(t,r)
    # to cartesian
    yg = np.cos(np.radians(tg)) * rg
    xg = np.sin(np.radians(tg)) * rg

    # define the image grid (cartesian) over [-r_earth, r_earth] x [0, r_earth]
    mx, my = 1000, 1000
    dmx = 2 * r_earth / (mx - 1)
    dmy = r_earth / (my - 1)
    xmg, ymg = np.meshgrid(dmx * np.arange(mx) - r_earth,
                           dmy * np.arange(my))

    # resample the interpolated model to the image grid
    x_res = griddata(
        np.vstack((xg.ravel(), yg.ravel())).T, x.ravel(),
        np.vstack((xmg.ravel(), ymg.ravel())).T)
    x_res = x_res.reshape(xmg.shape)

    #### plotting ####

    # setup plot (figure and axes)
    fig = plt.figure(frameon=False, **figure_config)
    ax = fig.add_axes([0.1,0,0.9,0.85])

    # do not shot axis frames / ticks
    plt.axis('off')

    # set up colormap (xi always recentered)
    if param == 'X':
        cmap, vmin = cmapXved(41, vmax)
        old_center = 1.0 - vmax / (vmax - vmin)
        cmap = recenter_cmap(cmap, 0.5, old_center=old_center)
    elif param == 'S':
        cmap = cmapSved(41)
    else:
        raise ValueError('Unrecognized parameter name: %s' % (param))

    # plot the resampled image
    imshow_config = {
            'cmap': cmap,
            'vmin': -vmax,
            'vmax': +vmax}
    im = plt.imshow(x_res,
            extent=[-r_earth,r_earth,0,r_earth],
            origin='lower', **imshow_config)

    # mask the core
    yc = np.cos(np.radians(t)) * (r_earth - dmax)
    xc = np.sin(np.radians(t)) * (r_earth - dmax)
    plt.fill_between(xc, np.zeros(yc.shape), yc, color=(1,1,1))

    # annotate depths of interest
    for d in [410, 650, 1000]:
        yd = np.cos(np.radians(t)) * (r_earth - d)
        xd = np.sin(np.radians(t)) * (r_earth - d)
        plt.plot(xd, yd, 'k--')

    # mark waypoints
    waypoint_dist = delta / (num_waypoints + 1) * np.arange(1,num_waypoints+1)
    waypoint_locs = []
    for i in range(num_waypoints):
        # distance
        d = waypoint_dist[i]
        # location (for map)
        waypoint_locs.append(shoot(p1, d, az))
        # to cartesian
        rw = np.asarray((r_earth - dmin,))
        yw = np.cos(np.radians(- delta / 2 + d)) * rw
        xw = np.sin(np.radians(- delta / 2 + d)) * rw
        # fill color
        facecolor = 'w'
        if i == num_waypoints - 1:
            facecolor = 'm'
        # plot
        plt.plot(xw, yw, ls='', marker='o', markerfacecolor=facecolor,
            markeredgecolor='k', markeredgewidth=2, ms=12, zorder=1000)

    # depth ticks (left side)
    tick_km_height = 100.0
    tick_width = 2
    for d in [500, 1000, 1500, 2000, 2500]:
        rd = r_earth - d
        td = - delta / 2
        y1 = np.cos(np.radians(td)) * rd
        x1 = np.sin(np.radians(td)) * rd
        y2 = y1 + tick_km_height * np.sin(np.radians(td))
        x2 = x1 - tick_km_height * np.cos(np.radians(td))
        plt.plot((x1,x2), (y1,y2), 'k-', lw=tick_width)

    # MARK STATION LOCATION ON CROSS-SECTION
    # Station is at p1_orig, which is overshoot_p1 from the left edge
    station_dist_from_left = overshoot_p1  # degrees from p1_extended
    station_angle = -delta/2 + station_dist_from_left  # Angle in cross-section coords
    
    # Plot at surface
    rs = r_earth - dmin
    ys = np.cos(np.radians(station_angle)) * rs
    xs = np.sin(np.radians(station_angle)) * rs

    rotation_angle = delta/2 - station_dist_from_left + 180  # 180 for upside-down

    plt.plot(xs, ys, marker=(3, 0, rotation_angle), markersize=15, 
            markerfacecolor='red', markeredgecolor='black', 
            markeredgewidth=2, ls='', zorder=1002)
    
    # plt.plot(xs, ys, marker='v', markersize=15, markerfacecolor='red',
    #         markeredgecolor='black', markeredgewidth=2, ls='', zorder=1002)

    # plot Sdiff raypath
    # eqdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/data/sample/FITZ/tonga_events.txt',sep='|')

    # eqdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/codes/Plotting/ARMA_chile_events.txt',sep=',')
    # eqdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/codes/Plotting/ARMA_weird_deep_S_events.txt',sep=',')
    # eqdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/codes/Plotting/ARMA_tonga_deep_S_events.txt',sep=',')
    # eqdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/codes/Plotting/ARMA_evbaz_120_to_150_events.txt',sep=',')

    eqdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/codes/Plotting/BBOO_evbaz_140_to_165_events.txt',sep=',')





    ### 4 events from Mariana trench area
    # eqdata = eqdata[eqdata['evlat']<0]
    # eqdata = eqdata[eqdata['evlon']>130]



    #### bkni phillipine subduction 
    # eqdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/data/sample/BKNI/BKNI_events_dec6.txt',sep='|')
    # eqdata = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/data/sample/STKA/STKA_events_nov18.txt',sep='|')

    # eqdata = eqdata[eqdata['evdp']>0]
    # eqdata = eqdata[eqdata['evlat']>0]
    # eqdata = eqdata[eqdata['evlat']<10]
    # eqdata = eqdata[eqdata['evlon']>120]
    
    # eqdata = eqdata[eqdata['evdp']>=300]
    taup_model = TauPyModel(model='iasp91')

    show_ray=True

    if show_ray:
        
        for i in range(0,len(eqdata)):
            eqdepth = float(eqdata['evdp'].iloc[i])
            eqdist = float(eqdata['evepic'].iloc[i])
            eqphase = str(eqdata['evphase'].iloc[i])
            print(eqdepth,eqdist)
        
            arrivals_tmp = taup_model.get_ray_paths(eqdepth,
                            eqdist, phase_list=[eqphase])
            arrivals = []
            for arrival in arrivals_tmp:
                # dist = arrival.purist_distance % 360.0
                # distance = arrival.distance
                # if distance < 0:
                #     distance = (distance % 360)
                # if abs(dist - distance) > 1E-5 * dist:
                #      # Mirror on axis.
                #     arrival = copy.deepcopy(arrival)
                #     arrival.path["dist"] *= -1.0
                # arrivals.append(arrival)

                # print(eqdepth,eqdist)
                # print(arrival.path["depth"][0])
                # print(np.rad2deg(arrival.path["dist"][0]))

                # eqdist-delta


                xr = np.sin(np.deg2rad(np.rad2deg(arrival.path["dist"])- delta / 2 - (eqdist-delta)+deltastat-overshoot_p1)) * (r_earth - arrival.path["depth"])
                yr = np.cos(np.deg2rad(np.rad2deg(arrival.path["dist"])- delta / 2 - (eqdist-delta)+deltastat-overshoot_p1)) * (r_earth - arrival.path["depth"])
            
                xr = -xr

                plt.plot(xr[0],yr[0],'o',c='gold',marker='*',markersize=10,markeredgecolor='black',# Outline color
    markeredgewidth=1)

                # if float(abs(eqdata['deltaphi'].iloc[i])) > 60:
                #     plt.plot(xr,yr, 'black',lw=1,linestyle='dotted',alpha=0.5)
                # else:
                #     plt.plot(xr,yr, 'black',lw=1,linestyle='dotted',alpha=0.5)
                #     # pass
                if eqphase == 'SKS':
                    plt.plot(xr,yr, 'black',lw=1,linestyle='dotted',alpha=0.5)
                if eqphase == 'S':
                    plt.plot(xr,yr, 'red',lw=1,linestyle='dotted',alpha=0.5)



        # fig.plot(x=np.rad2deg(arrival.path["dist"]+start_offset),y=np.array(r_earth-arrival.path["depth"]))

        

    #     fname = 'taup_path.gmt'
    #     dist,depr,latr,lonr = np.loadtxt(fname, usecols=(0,1,2,3), skiprows= 1, unpack = True)
    #     #print 'passed', depr[0], depr[1]
    #     #print  depr[0],latr[0],lonr[0]
    # # to radians / colats
    #     thetar = np.pi / 2 - np.radians(latr)
    #     phir = np.radians(lonr)
    #     distr = np.radians(dist - delta / 2)
    #     xr = np.sin(distr) * (depr)
    #     yr = np.cos(distr) * (depr)
    #     #print 'xr yr', xr, yr
    #     plt.plot(xr,yr, 'k-',lw=tick_width)

#         def plot_raypaths(fig, delta, phases, depth_event, start_offset,
# end_offset):
#      # add phases
#      if len(phases)>0:
#          arrivals_tmp = taup_model.get_ray_paths(depth_event,
# delta-start_offset-end_offset, phase_list=phases)

#          arrivals = []
#          for arrival in arrivals_tmp:
#              dist = arrival.purist_distance % 360.0
#              distance = arrival.distance
#              if distance < 0:
#                  distance = (distance % 360)
#              if abs(dist - distance) > 1E-5 * dist:
#                  # Mirror on axis.
#                  arrival = copy.deepcopy(arrival)
#                  arrival.path["dist"] *= -1.0
#              arrivals.append(arrival)

#          if not arrivals:
#              print("Phases not visible at these distances.")
#          else:
#              for arrival in arrivals:
# fig.plot(x=np.rad2deg(arrival.path["dist"]+start_offset),y=np.array(r_earth-arrival.path["depth"]))

#      return

    # frame edges of section
    edge_width = 2
    # - bottom
    ye = np.cos(np.radians(t)) * (r_earth - dmax)
    xe = np.sin(np.radians(t)) * (r_earth - dmax)
    plt.plot(xe, ye, 'k-', lw=edge_width)
    # - top
    ye = np.cos(np.radians(t)) * (r_earth - dmin)
    xe = np.sin(np.radians(t)) * (r_earth - dmin)
    plt.plot(xe, ye, 'k-', lw=edge_width)
    # - left 
    ye = np.cos(np.radians(- delta / 2)) * r
    xe = np.sin(np.radians(- delta / 2)) * r
    plt.plot(xe, ye, 'k-', lw=edge_width)
    # - right 
    ye = np.cos(np.radians(+ delta / 2)) * r
    xe = np.sin(np.radians(+ delta / 2)) * r
    plt.plot(xe, ye, 'k-', lw=edge_width)

    # add hotspots
    # how close a hotspot must be to be "in plane" (degrees)
    dist_thresh = 0
    hotspots_to_plot = []
    if named_hotspots is None:
        with open('%s/hotspots.json' % (get_tectonics_path())) as f:
            named_hotspots = json.load(f)
        print('Loaded hotspots')
    for hs in named_hotspots:
        # compute min distance to gc plane
        lat, lon = named_hotspots[hs]
        dists = np.asarray([delaz((lon,lat),p,delta_only=True)
            for p in zip(lons,lats)])
        imin = np.argmin(dists)
        # add to plot list if within thresh
        if dists[imin] <= dist_thresh:
            print('Have hotspot %s' % (hs))
            d = delaz(p1, (lons[imin],lats[imin]), delta_only=True)
            hotspots_to_plot.append((d,hs))
    # sort by distance, plotting with optional label dither (alternating up / down)
    hotspots_to_plot.sort()
    height_dither = 0.0 # 0.0 => disabled
    for d, hs in hotspots_to_plot:
        height = 50.0
        rw = height + np.asarray((r_earth - dmin,))
        yw = np.cos(np.radians(- delta / 2 + d)) * rw
        xw = np.sin(np.radians(- delta / 2 + d)) * rw
        marker = '^'
        marker = 3, 0, delta / 2 - d
        plt.plot(xw, yw, ls='', marker=marker, markerfacecolor=(0,1,0.1),
            markeredgecolor='k', markeredgewidth=2, ms=20,
            zorder=1001, clip_on=False)
        height = 350.0 + height_dither
        rw = height + np.asarray((r_earth - dmin,))
        yw = np.cos(np.radians(- delta / 2 + d)) * rw
        xw = np.sin(np.radians(- delta / 2 + d)) * rw
        plt.text(xw, yw, hs, rotation=delta/2 - d,
                ha='center', va='center', weight='bold',
                size=9)
        height_dither *= -1

    # # plot deep earthquakes
    # if show_eqs:
    #     # threshold for whether eq is "in plane"
    #     eq_search_width = 1.0
    #     catalog_str = load_eq_catalog('deep_100km.txt',
    #         fields=['Longitude', 'Latitude', 'Depth'])
    #     catalog = map(np.array,
    #         zip(*map(lambda xs: tuple(float(x) for x in xs), catalog_str)))  
    #     path_dists = t
    #     plot_deep_eqs(catalog, lons, lats, path_dists, width=eq_search_width)

    # tight axes?
    if tight_axes:
        # - bottom
        ye = np.cos(np.radians(t)) * (r_earth - dmax)
        y_bot_min = ye.min()
        # - top
        ye = np.cos(np.radians(t)) * (r_earth - dmin)
        y_top_max = ye.max()
        # - left 
        xe = np.sin(np.radians(- delta / 2)) * r
        x_left_min = xe.min()
        # - right 
        xe = np.sin(np.radians(+ delta / 2)) * r
        x_right_max = xe.max()
        plt.xlim((x_left_min - tight_axes_buffer_km,
                  x_right_max + tight_axes_buffer_km))
        plt.ylim((y_bot_min - tight_axes_buffer_km,
                  y_top_max + tight_axes_buffer_km))

    # parameter name
    if param == 'S':
        param_name = 'Vs'
    elif param == 'X':
        param_name = 'Xi'
    else:
        raise ValueError('Unrecognized parameter name: %s' % (param))

    # add color bar
    if colorbar:
        cax = fig.add_axes([0.02,0.2,0.015,0.3])
        ticks = np.arange(-vmax,vmax+0.001,1.0)
        cb = plt.colorbar(im, cax=cax, ticks=ticks, format='%3.0f')
        if resat:
            text = ['%2.0f [%.0f]' % (t,np.abs(t) / vmax * upper_sat) for t in ticks]
            cb.ax.set_yticklabels(text)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_weight('bold')
            l.set_size(8)
        cb.set_label('dln%s (%+.1f / %+.1f)' % (param_name, x.min(), x.max()),
                rotation=270.0, va='bottom', weight='bold', size=10)

    # add identifier text (i.e. section number)
    plt.figtext(0.9, 0.95, fig_id_text,
        weight='bold', ha='center', va='center', size=12,
        backgroundcolor=(1,1,1))

    # now, add a map
    if basemap_args:
        # define the map axes
        ax = fig.add_axes([0,0.65,0.35,0.35])

        # initialize the Basemap object
        m = Basemap(**basemap_args)

        # draw the edge of the map, and the continents
        m.drawmapboundary()
        m.fillcontinents(color=(0.8,0.8,0.8))

        # draw lat / lon grid
        if map_draw_lats_lons:
            m.drawparallels(np.arange(-90,90,10), dashes=[1,1])
            m.drawmeridians(np.arange(-180,180,10), dashes=[1,1])

        # draw plates, hotspots, and the cross-section path
        #plot_plates(m, color='r', linestyle='-', lw=0.5, lon360=True)
        #plot_hotspots(m, marker='o', ls='', markerfacecolor=(0,1,0.05),
        #   zorder=1000, ms=4)
        plot_gc_clean(m, p1, p2, 1.0, color='k', lw=1.0, linestyle='-')

        # Mark ORIGINAL station location on map
        m.scatter(*m(p1_orig[0], p1_orig[1]), color='red', marker='v',
                 s=100, edgecolor='black', linewidth=2, zorder=1002)

        # draw waypoints
        for i in range(num_waypoints):
            lon, lat = waypoint_locs[i]
            color = 1,1,1
            if i == num_waypoints - 1:
                color = 'm'
            m.scatter(*m(lon,lat), color=color, edgecolor='k', lw=1,
                    s=40, zorder=1001)

    # save section
    if save:
        if resat:
            xsec_name += '.resat_%.1f_at_%.1fkm' % (upper_sat, upper_sat_depth)
        # plt.savefig('%s/xsec_dln%s_%s.%s.vmax_%.1f.%s' % (
        #     save_path, param_name, xsec_name, model_config['name'],
        #     vmax, save_ext), **save_kwargs)
        plt.savefig(save_path+'/BBOO_chile_SKS_and_S_raypaths_20260504.png',dpi=1200)

    return fig

########################################################################
# main routine

def define_and_plot_xsecs():
    '''Define the section of interest and call plot_xsec()
    '''

    ####################################
    # example 1: single-section plotting
    # Fukao and Obayashi 2013 Fig. 10 H
    # section will be shown on-screen as well
    ##
    # file base name
    base_name = 'wp-SEMUCB'
    #B configuration for map
# if using "ortho" choose lon_0 and lat_0 to be the middle of the plotted
# great circle (br 05/08/19)
#     basemap_args = {
# #        'projection': 'merc',
#         'projection': 'ortho','lon_0':140.,'lat_0': -45.0,
#         'resolution': 'c'}
        #'llcrnrlon': 135.0, 'llcrnrlat': -45.0,
        #'urcrnrlon': 210.0, 'urcrnrlat':   0.0,
        #'lat_ts': -20.0}
    #    'urcrnrlon': 0.0, 'urcrnrlat': 50.0,
     #   'llcrnrlon': -90.0, 'llcrnrlat': -20.0,
      #  'lat_ts': -20.0}
    # endpoints from table 1 of FO13
#end points for Matt Jackson's hotspots
    #p1=0.0, -70.0
    #p2=0.0,-20.0
#
#     # p2 = 101.0396, 0.3262 #BKNI station
#     p2 = 141.5964, -31.8757 #stka station
# # -31.8757, 141.596405
#     p1 = 115.0, 10.0 #phillipine subdcution 
    # p2 = 125.64, -18.1 #FITZ station 

    # p1 = 151.63 , -30.412 #ARMA station
    # # p1 = -176.0, -25.0 #tonga subduction 
    # p2 = -71.71 , -32.75 #mean values of sources for chile subduction strange SKS deep mantle contamination
    # # p2 = -179 , -20.0 #tonga trench

    p1 = 136.06 , -32.81 #BBOO station
    p2 = -70.6 , -32.98 #mean values of sources for chile subduction strange SKS deep mantle contamination for BBOO

    mid_lon, mid_lat = shoot(p1, delaz(p1, p2)[0]/2, delaz(p1, p2)[1])
    basemap_args = {'projection': 'ortho', 'lon_0': mid_lon, 'lat_0': mid_lat+15, 'resolution': 'c'}
    
# label for section (typically used in rendering movie frames)
    section_label = ''
    plot_xsec(p1, p2, model_config, base_name, section_label, basemap_args,
        nx=300, show_eqs=True)
    # plot on screen
    plt.show()

    ####################################
    ## example 2: multi-section plotting
    ## Pacific APM sections (APM-normal)
    ## sections will not be shown on-screen (as there are in general too many)
    ###
    ## file base name
    #base_name = 'pac_apm'
    ## section width (degrees)
    #width = 120.0
    ## pacific plate euler pole
    #euler_pole = 96.2, -64.1 
    ## define sampling path (centers of APM-normal sections)
    #olons, olats = circle_dist(
    #    euler_pole[0], euler_pole[1], # euler pole
    #    90.0,                         # distance of path from pole (degrees)
    #    r_earth * np.radians(1.0))    # step size between samples (km)
    ## min / max lons for filtering sample path (circle_dist() returns full
    ## small circle path)
    #olon_min = 150.0
    #olon_max = 320.0
    ## generate paths
    #paths = []
    #for olon, olat in zip(olons, olats):
    #    # filter by lon range
    #    if olon >= olon_min and olon <= olon_max: 
    #        # configuration for the map
    #        basemap_args = {
    #            'projection': 'ortho',
    #            'lon_0': olon, 'lat_0': olat,
    #            'resolution': 'c'}
    #        # local azimuth to euler pole
    #        _, az = delaz((olon, olat), euler_pole)
    #        # compute endpoints of path
    #        if np.cos(np.radians(az)) > 0:
    #            p1 = shoot((olon,olat), width / 2, (az + 180) % 360.0)
    #            p2 = shoot((olon,olat), width / 2, az)
    #        else:
    #            p2 = shoot((olon,olat), width / 2, (az + 180) % 360.0)
    #            p1 = shoot((olon,olat), width / 2, az)
    #        paths.append((p1,p2,olon,olat,basemap_args))
    ## plot the paths
    #print('%i paths will be plotted' % (len(paths)))
    #for ix, path in enumerate(paths):
    #    ### TEST
    #    # dropping all but section 87 for testing purposes
    #    if ix != 87:
    #        continue
    #    ### TEST
    #    p1, p2, olon, olat, basemap_args = path
    #    lon_0 = olon
    #    if olon > 180:
    #        olon -= 360.0
    #    xsec_name = '%s_%3.3ioff_%.1fwidth' % (base_name, ix, width)
    #    fig_id_text = 'section %3.3i' % (ix)
    #    plot_xsec(p1, p2, model_config, xsec_name, fig_id_text, basemap_args,
    #        nx=300, show_eqs=False)
    #    # be sure to close the figures as we create them (when generating
    #    # many, this can eat up a lot of memory)
    #    plt.close()

def main():
    define_and_plot_xsecs()

if __name__ == '__main__':
    main()
