import pandas as pd
import os, sys, glob
import numpy as np
import enum # Enum for size units
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime as dt
# import pygmt
from math import * #added for error fix
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
import sys
from helper_functions.plotting import read_gmt_file, read_and_filter_gmt, plot_station_splitting, build_cmap
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
import matplotlib.colors as colors
import re
from support import dms_to_dd
from station_keys import STATION_PHASE_MAP
from scipy.interpolate import griddata

import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
})


'''
WILL BE SWITCHING FROM -180 TO 180 LON --> 0 TO 360 LON
MODULO 360 EVERY LONGITUDE DATA

'''

# #PacificSplittingMapCoordinates
# minlon, maxlon = 90.00, 180.00
# minlat, maxlat = -55.00, 5.00

#arma bounce points extent
minlon, maxlon = 156.00, 166.00
minlat, maxlat = -46.00, -36.00


# #blob region extent
# minlon, maxlon = 155.00, 166.00
# minlat, maxlat = -51.00, -36.00



slice_depth = 2800 #km

##### Data Paths #####

# measurements that I made
new_measurements_path       = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/swp_measurements_2.txt'

# potential CMB D" anisotropy contamination data
cmb_contamination_path      = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/blob_plots/blob_coordinates_ARMA_only.txt'

# Bernhard Steinberger lithospheric thickness model
lithosphere_thickness_path  = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/lithosphere_thickness/lith_ave_no_slabs.d'
                        
# Thorsten Becker SKS station average database (as of May 7 2026)
becker_sks_data_path        = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/sks_stat_avg/jsg_geodynamics.sks.avg.table.txt'

# Plate Boundary Data Bird (2003)
ridge_path      = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/ridge.gmt'       
transform_path  = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/transform.gmt'
trench_path     = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/plate_boundaries/trench.gmt'

# Bathymetry data from GEBCO
tif_files = glob.glob("/work/gcl3/BR/nicolasv03/GEBCO/*.tif")
src_files = [rasterio.open(f) for f in tif_files]

# SEMUCB slice at certain depth
tomo_data_path = f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/SEMUCB_slice_data/SEMUCB_depthslice_{slice_depth}km.npy'
# tomo_data_path = f"/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/SPiRaL_slice_data/SPiRaL_vsv_depthslice_{slice_depth}km.npy"

# WOlf 2026 TSR lower mantle diffSI
diffSI_path = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/wolf_lower_mantle/diff_splitting_grid_resampledforplotting.grd'

# bathymetry_path = '/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/bathymetry/gebco_2026_n6.807_s-50.49_w87.582_e193.256_geotiff.tif'


##### DataFrames #####
                                
new_data_df = pd.read_csv(new_measurements_path, sep='\t', encoding='cp1252') 
new_data_df[['Latitude','Longitude','phi','dt']] = new_data_df[['Latitude','Longitude','phi','dt']].apply(pd.to_numeric, errors='coerce')
new_data_df['Longitude'] = new_data_df['Longitude'] % 360

blob_df = pd.read_csv(cmb_contamination_path, delimiter=r'\s+') #, names=['latitude','longitude','evlat','evlon','statlat','statlon','evdp','evphase','evepic','SI'],header=None)
blob_df['lon'] = pd.to_numeric(blob_df['lon'], errors='coerce') % 360

lithos_df = pd.read_csv(lithosphere_thickness_path, header=None, delimiter=r'\s+', names=['longitude', 'latitude', 'thickness'], low_memory=False)
lithos_df['longitude'] = lithos_df['longitude'] % 360

# Thorsten Becker SKS station average database (as of May 7 2026)
stat_avg_df = pd.read_csv(becker_sks_data_path, header=None, delimiter=r'\s+', names=["station","lat","lon","azi","d_azi","dt","d_dt",
                                                                                      "type","elat","elon","edep","mlat","mlon","bazi"])
stat_avg_df['lon'] = stat_avg_df['lon'] % 360
stat_avg_df['elon'] = stat_avg_df['elon'] % 360

stat_avg_df['station_clean'] = stat_avg_df['station'].str.split('-', n=1).str[0]
stat_avg_df['num_of_splits'] = stat_avg_df['station'].str.extract(r'avg(\d+)').astype(int)
stat_avg_df = stat_avg_df[(minlon < stat_avg_df['lon']) & (stat_avg_df['lon'] < maxlon)]
stat_avg_df = stat_avg_df[(minlat < stat_avg_df['lat']) & (stat_avg_df['lat'] < maxlat)]

# print(stat_avg_df.head(5))
# base_station = (stat_avg_df.loc[5481, 'station'].split('-')[0])
# stat_avg_df = stat_avg_df[(stat_avg_df['station'].str.split('-').str[0] != base_station) | (stat_avg_df.index == 5481)]
stat_avg_df = (stat_avg_df.sort_values('num_of_splits', ascending=False).drop_duplicates(subset='station_clean').reset_index(drop=True))
stat_avg_df = stat_avg_df[stat_avg_df['station_clean'].isin(new_data_df['Station'])]
# stat_avg_df = stat_avg_df[stat_avg_df['station_clean'].isin(STATION_PHASE_MAP.keys())] # switch between plotting complex stations or not

# just for this one, plot only ARMA
new_data_df = new_data_df[new_data_df['Station'] == "ARMA"]

# ridge_gmt       = read_and_filter_gmt(ridge_path,minlon,maxlon,minlat,maxlat, max_gap=20)
# transform_gmt   = read_and_filter_gmt(transform_path,minlon,maxlon,minlat,maxlat, max_gap=20)
# trench_gmt      = read_and_filter_gmt(trench_path,minlon,maxlon,minlat,maxlat, max_gap=20)

# plate_df = pd.read_csv('/work/gcl3/BR/nicolasv03/senior_thesis/measurements/pm10242.csv',sep=',',header=None,names=['station','lat', 'lon', 'rate','angle','a','b','c'])
# plate_df['lon'] = pd.to_numeric(plate_df['lon'], errors='coerce') % 360
# print(plate_df.head(10))
# plate_df['lat_dd'] = plate_df['lat'].apply(dms_to_dd)
# plate_df['lon_dd'] = plate_df['lon'].apply(dms_to_dd)

# # Compute u (east) and v (north) components
# plate_df['angle_rad'] = np.deg2rad(plate_df['angle'])

# plate_df = plate_df[(plate_df['lat_dd'] >= minlat) & (plate_df['lat_dd'] <= maxlat) &
#                   (plate_df['lon_dd'] >= minlon) & (plate_df['lon_dd'] <= maxlon)] #filter within map bounds

# plate_df['u'] = plate_df['rate'] * np.sin(plate_df['angle_rad'])  # east component
# plate_df['v'] = plate_df['rate'] * np.cos(plate_df['angle_rad'])  # north component

# tomo_data = np.load(tomo_data_path, allow_pickle=True).item()
# lon2 = tomo_data["lon"]
# lat2 = tomo_data["lat"]
# slice_vals = tomo_data["values"]
# depth_km = tomo_data["depth"]


# # diffSI data from Wolf 2026 TSR
# diffSI_src = rasterio.open(diffSI_path)
# diffSI_data = diffSI_src.read(1).astype(float)
# height, width = diffSI_data.shape
# cols, rows = np.meshgrid(np.arange(width), np.arange(height))
# xs, ys = rasterio.transform.xy(diffSI_src.transform, rows, cols)
# lons = np.array(xs)[0]   # flatten row structure
# lats = np.array(ys)[:, 0]

# pad = 5.0

# window = from_bounds(
#     minlon - pad,
#     minlat - pad,
#     maxlon + pad,
#     maxlat + pad,
#     transform=diffSI_src.transform
# )

# diffSI_data = diffSI_src.read(1, window=window).astype(float)

# window_transform = diffSI_src.window_transform(window)

# h, w = diffSI_data.shape
# cols, rows = np.meshgrid(np.arange(w), np.arange(h))
# xs, ys = rasterio.transform.xy(window_transform, rows, cols)
# lon2d = np.array(xs)
# lat2d = np.array(ys)


# #########
# # --------------------------------------------------
# # Merge all GEBCO tiles into a single raster
# # --------------------------------------------------
# mosaic, out_transform = merge(src_files)

# # use merged raster metadata
# src = src_files[0]

# scale_factor = 32


# # --------------------------------------------------
# # Normalize longitudes to [-180, 180]
# # --------------------------------------------------
# minlon_wrap = ((minlon + 180) % 360) - 180
# maxlon_wrap = ((maxlon + 180) % 360) - 180

# crosses_dateline = minlon_wrap > maxlon_wrap


# # --------------------------------------------------
# # Helper function for reading/downsampling windows
# # --------------------------------------------------
# def read_window(bounds):
#     window = from_bounds(*bounds, transform=out_transform)

#     return mosaic[
#         0,
#         int(window.row_off):int(window.row_off + window.height):scale_factor,
#         int(window.col_off):int(window.col_off + window.width):scale_factor
#     ]


# # --------------------------------------------------
# # Crop raster
# # --------------------------------------------------
# if not crosses_dateline:

#     topo = read_window((
#         minlon_wrap,
#         minlat,
#         maxlon_wrap,
#         maxlat
#     ))

#     extent = [minlon, maxlon, minlat, maxlat]

# else:

#     topo_left = read_window((
#         minlon_wrap,
#         minlat,
#         180,
#         maxlat
#     ))

#     topo_right = read_window((
#         -180,
#         minlat,
#         maxlon_wrap,
#         maxlat
#     ))

#     topo = np.hstack([topo_left, topo_right])

#     extent1 = [minlon, 180, minlat, maxlat]
#     extent2 = [-180, maxlon, minlat, maxlat]


# # --------------------------------------------------
# # Replace nodata values with NaN
# # --------------------------------------------------
# nodata = src.nodata

# if nodata is not None:
#     topo = np.where(topo == nodata, np.nan, topo)
# ###########


# new_data_df = new_data_df[(minlon < new_data_df['Longitude']) & (new_data_df['Longitude'] < maxlon)]
# new_data_df = new_data_df[(minlat < new_data_df['Latitude']) & (new_data_df['Latitude'] < maxlat)]

# legend_df = pd.DataFrame([{
#     'Station': r'$\mathbf{\delta t}$ = 1.0',
#     'Latitude': -40,
#     'Longitude': 105,
#     'phi': 90,
#     'dt': 1
# }])

# # Create a figure and axis for the map
# fig = plt.figure(figsize=(20, 10))
# ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
# ax.set_extent([minlon,maxlon,minlat,maxlat], crs=ccrs.PlateCarree())
# # ax.gridlines(draw_labels=True,linewidth=0)

# import matplotlib.ticker as mticker

# gl = ax.gridlines(draw_labels=True, linewidth=0.25)

# # remove duplicate dateline label
# gl.xlocator = mticker.FixedLocator([154,156,158, 160, 162,164,166,168, 180])

# # optional latitude ticks
# gl.ylocator = mticker.FixedLocator([-46,-44,-42,-40,-38,-36,-34, 0, 10])

# # turn off top/right labels
# gl.top_labels = False
# gl.right_labels = False

# gl.xlabels_top = False
# gl.ylabels_right = False

# gl.xlabel_style = {'size': 14}
# gl.ylabel_style = {'size': 14}

# from matplotlib.ticker import FuncFormatter

# def deg_formatter(x, pos):
#     return f"{x:.0f}°"

# gl.xformatter = FuncFormatter(deg_formatter)
# gl.yformatter = FuncFormatter(deg_formatter)

# ax.coastlines()

vals = np.polyfit(blob_df['evepic'],blob_df['SI'],1)
print(vals)
xvals = np.linspace(100,120,100)
# sys.exit()

fig = plt.figure(figsize=(6,8))

plt.plot(blob_df['evepic'],blob_df['SI'],'o',c='black',alpha=0.8)
# plt.plot(xvals,vals[0]*xvals + vals[1],'-')
plt.xlim(93,127)
plt.ylim(-0.85,0.85)
plt.xlabel("Epicentral Distance [deg]",fontsize=20)
plt.ylabel("SI",fontsize=20)
plt.title("ARMA SKS Events 130 - 160 Back Azimuth",fontsize=20)

plt.xticks(
    [95, 100, 105, 110, 115, 120, 125],
    [f'{x}°' for x in [95,100, 105, 110, 115, 120, 125]],fontsize=14
)

plt.yticks([-0.75,-0.50,-0.25,0,0.25,0.5,0.75],fontsize=14)



# ##### choose basemap: lithospheric thickness or bathymetry data or diffSI data

# ### diffSI data
# cmap, norm = build_cmap()

# diffSI = ax.pcolormesh(
#     lon2d,
#     lat2d,
#     diffSI_data,
#     cmap=cmap,
#     norm=norm,
#     shading="auto",
#     transform=ccrs.PlateCarree(),
# )

# cbar = plt.colorbar(diffSI, ax=ax, orientation='horizontal', 
#                         pad=0.05, shrink=0.5, aspect=25)
# cbar.set_label(r"$\Delta$(SI)  —  Differential Splitting Intensity",fontsize=15)

# ### lithospheric thickness
# lon_grid = np.linspace(lithos_df['longitude'].min(), lithos_df['longitude'].max(), 200)
# lat_grid = np.linspace(lithos_df['latitude'].min(), lithos_df['latitude'].max(), 200)
# lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# thickness_grid = griddata((lithos_df['longitude'], lithos_df['latitude']),lithos_df['thickness'],
#                             (lon_mesh, lat_mesh),method='cubic')

# levels = np.linspace(lithos_df['thickness'].min(), lithos_df['thickness'].max(), 20)
# contour = ax.contourf(lon_mesh, lat_mesh, thickness_grid,
#                       levels=levels,
#                       cmap='RdYlBu_r',  # Or 'viridis', 'plasma', 'jet'
#                       alpha=0.6,
#                       transform=ccrs.PlateCarree(),
#                       zorder=1)

# cbar_thickness = plt.colorbar(contour, ax=ax, orientation='vertical', 
#                               pad=0.05, shrink=0.8)
# cbar_thickness.set_label('Lithosphere Thickness [km]', fontsize=14)


# ### bathymetry
# vmin = -4000   # ocean
# vmax = 2000    # land (australia peak elevation is ~2200m)

# if not crosses_dateline:
#     img = ax.imshow(topo,extent=extent,
#                     origin="upper",
#                     transform=ccrs.PlateCarree(),
#                     cmap="terrain",
#                     vmin=vmin,
#                     vmax=vmax,
#                     alpha=0.6)
#     cbar_bathymetry = plt.colorbar(img, ax=ax, shrink=0.6)
#     cbar_bathymetry.set_label("Bathymetry [m]")
# else:
#     img1 = ax.imshow(
#         topo_left,
#         extent=[minlon_wrap, 180, minlat, maxlat],
#         origin="upper",
#         transform=ccrs.PlateCarree(),
#         cmap="terrain",
#         vmin=vmin,
#         vmax=vmax,
#         alpha=0.6
#     )

#     img2 = ax.imshow(
#         topo_right,
#         extent=[-180, maxlon_wrap, minlat, maxlat],
#         origin="upper",
#         transform=ccrs.PlateCarree(),
#         cmap="terrain",
#         vmin=vmin,
#         vmax=vmax,
#         alpha=0.6
#     )

#     cbar_bathymetry = plt.colorbar(img1, ax=ax, shrink=0.6)
#     cbar_bathymetry.set_label("Bathymetry [m]")

### or we can plot tomo slice data as our background

# #only use this if using spiral! otherwise leave commented for semucb

# mean_vs = np.nanmean(slice_vals)
# slice_vals = 100 * np.log(slice_vals / mean_vs )



# im = ax.pcolormesh(
#     lon2,
#     lat2,
#     slice_vals,
#     cmap="RdBu",
#     shading="auto",
#     vmin=-2.5,
#     vmax=2.5,
#     transform=ccrs.PlateCarree()
# )

# # colorbar
# cbar = plt.colorbar(im, ax=ax,orientation='vertical', 
#                         pad=0.05, shrink=0.8)

# # plt.colorbar(contour, ax=ax, orientation='vertical', 
# #                               pad=0.05, shrink=0.8)

# cbar.set_label("dlnVs (%)",fontsize=15)

    
# ## plot the tectonic boundaries 
# for segment in ridge_gmt:
#     if len(segment) > 1:
#         lons, lats = zip(*segment)
#         ax.plot(lons, lats,color='darkred', linewidth=1.5,
#                 transform=ccrs.PlateCarree(), zorder=1)
        
# for segment in trench_gmt:
#     if len(segment) > 1:
#         lons, lats = zip(*segment)
#         ax.plot(lons, lats,color='darkred', linewidth=1.5,
#                 transform=ccrs.PlateCarree(), zorder=1)
        
# for segment in transform_gmt:
#     if len(segment) > 1:
#         lons, lats = zip(*segment)
#         ax.plot(lons, lats,color='darkred', linewidth=1.5,
#                 transform=ccrs.PlateCarree(), zorder=1)
        
# print(blob_df.head(20))


# define colormap + normalization
# cmap = plt.cm.vanimo   # or 'coolwarm', 'plasma', etc.

# from matplotlib.colors import LinearSegmentedColormap

# cmap = LinearSegmentedColormap.from_list(
#     'blue_grey_red',
#     ['blue', 'grey', 'red']
# )

# norm = colors.Normalize(
#     vmin= -0.5,
#     vmax= 0.5
# )

# # scatter plot
# sc = ax.scatter(
#     blob_df['lon'],
#     blob_df['lat'],
#     c=blob_df['SI'],          # <-- use c instead of color
#     cmap=cmap,
#     norm=norm,
#     edgecolor='black',
#     marker='o',
#     s=100,
#     alpha=0.5,
#     transform=ccrs.PlateCarree(),
#     zorder=10
# )

# # colorbar
# cbar = plt.colorbar(sc, ax=ax, orientation='vertical', 
#                         pad=0.025, shrink=0.9, aspect=25)
# cbar.set_label('Splitting Intensity')



# def plot_plate_motion(statlat,statlon,phi,dt,version=False):
#     # ax.plot(statlon,statlat, 'y', marker = 'v',markersize=6, transform=ccrs.PlateCarree())
#     phi = np.radians(90 - phi)
#     half_dt = dt / 2 

#     #halfway points of each line
#     dx = half_dt * np.cos(phi)
#     dy = half_dt * np.sin(phi)

#     x0 = statlon - dx
#     x1 = statlon + dx
#     y0 = statlat - dy
#     y1 = statlat + dy

#     if version:
#         ax.plot([x0, x1], [y0, y1], color='blue', linewidth=2, transform=ccrs.PlateCarree())
#     else:
#         ax.plot([x0, x1], [y0, y1], color='black', linewidth=1, transform=ccrs.PlateCarree())
# df_sorted = df.sort_values(by='column_name')

# platedf['lat_bin'] = (platedf['lat_dd'] / 2).round()
# platedf['lon_bin'] = (platedf['lon_dd'] / 2).round()

# platedf_sparse = platedf.drop_duplicates(
#     subset=['lat_bin', 'lon_bin']
# )

# #####
# spacing = 10  # degrees

# # Create bins
# plate_df['lat_bin'] = np.floor(plate_df['lat_dd'] / spacing) * spacing
# plate_df['lon_bin'] = np.floor(plate_df['lon_dd'] / spacing) * spacing

# # Average vectors within bins
# plate_df_binned = (
#     plate_df
#     .groupby(['lat_bin', 'lon_bin'], as_index=False)
#     .agg({
#         'u': 'mean',
#         'v': 'mean',
#         'lat_dd': 'mean',
#         'lon_dd': 'mean'
#     })
# )

# plt.quiver(
#     plate_df_binned['lon_dd'],
#     plate_df_binned['lat_dd'],
#     plate_df_binned['u'],
#     plate_df_binned['v'],
#     angles='xy',
#     scale_units='xy',
#     scale=20,
#     width=0.003,
#     color='grey',
#     alpha=0.5,
#     headwidth=4,
#     headlength=2.5,
#     headaxislength=2.0
# )

# plate_df = plate_df.sort_values(by=['lon_dd','lat_dd']).reset_index(drop=True)
# print(plate_df.head(20))

# plate_df['lat_bin'] = np.floor(plate_df['lat_dd'] / 3)#.round()
# plate_df['lon_bin'] = np.floor(plate_df['lon_dd'] / 3)#.round()
# plate_df_sparse = plate_df.drop_duplicates(
#     subset=['lat_bin', 'lon_bin']
# )
# # plate_df = plate_df.sort_values(by='lat_dd').reset_index(drop=True)

# print(plate_df_sparse.head(20))
# # plate_df.iloc[1::2]

# plate_df = plate_df_sparse
# plt.quiver(plate_df['lon_dd'], plate_df['lat_dd'], plate_df['u'], plate_df['v'],
#            angles='xy', scale_units='xy', scale=50, color='grey',alpha=0.9,
#            headwidth=3,headlength=3,headaxislength=3.5)

# # df = new_data_df
# for i in range(len(new_data_df)):
#     plot_station_splitting(new_data_df['Station'].iloc[i],
#                            new_data_df['Latitude'].iloc[i],
#                            new_data_df['Longitude'].iloc[i],
#                            new_data_df['phi'].iloc[i],
#                            4*new_data_df['dt'].iloc[i],
#                            ax=ax,
#                            version='New'
#                            )

# for i in range(len(legend_df)):
#     plot_station_splitting(legend_df['Station'].iloc[i],
#                            legend_df['Latitude'].iloc[i],
#                            legend_df['Longitude'].iloc[i],
#                            legend_df['phi'].iloc[i],
#                            4*legend_df['dt'].iloc[i],
#                            ax=ax,
#                            version='Legend')

# # print(stat_avg_df.head(5))
# for i in range(len(stat_avg_df)):
#     plot_station_splitting(stat_avg_df['station'].iloc[i],
#                            stat_avg_df['lat'].iloc[i],
#                            stat_avg_df['lon'].iloc[i],
#                            stat_avg_df['azi'].iloc[i],
#                            4*stat_avg_df['dt'].iloc[i],
#                            ax=ax,
#                            version='Old',
#                            annotate=False)
    
# ### plot station names only
# for i in range(len(new_data_df)):
#     plot_station_splitting(new_data_df['Station'].iloc[i],
#                            new_data_df['Latitude'].iloc[i],
#                            new_data_df['Longitude'].iloc[i],
#                            new_data_df['phi'].iloc[i],
#                            4*new_data_df['dt'].iloc[i],
#                            ax=ax,
#                            version='CMB_single_station',
#                            annotate=True
#                            )



# # Combine both datasets if you want to show all measurements
# combined_df = pd.concat([sks_meas_all_newdata])

# # Set colormap and normalization
# cmap = plt.cm.viridis # Or 'plasma', 'coolwarm', etc.
# norm = Normalize(vmin=0, vmax=1.5)

# # Plot scatter points colored by dt
# sc = ax.scatter(combined_df['Longitude'], combined_df['Latitude'],
#                 c=combined_df['dt'], cmap=cmap, norm=norm,
#                 s=40, edgecolor='k', linewidth=0.5,
#                 transform=ccrs.PlateCarree(), zorder=5)

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cax = inset_axes(ax,
#                  width="50%",  # width of the colorbar
#                  height="3%",  # height
#                  loc='lower center',
#                  bbox_to_anchor=(0.5, -0.1, 0.5, 1),  # (x0, y0, width, height)
#                  bbox_transform=ax.transAxes,
#                  borderpad=0)

# # Add a colorbar
# cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', shrink=0.3, pad=0.05)
# cbar.set_label('Delay Time (s)', fontsize=14)

# plt.title(f"CMB Pierce Points w/ SEMUCB-WM1 : {slice_depth} km",fontsize=25)
# plt.title(f"CMB Pierce Points w/ Diff SI",fontsize=25)

fig.savefig(f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/cmb_epicentral_plot_sep_23.pdf', dpi=600,bbox_inches="tight", pad_inches=0.2)
 #original dpi=300


# print(np.mean(blob_df['longitude']))
# print(np.mean(blob_df['latitude']))


# #PacificSplittingMapCoordinates
# minlon, maxlon = 90.00, 190.00
# minlat, maxlat = -50.00, 10.00

# #australia coordinates
# minlon, maxlon = 110.00, 160.00
# minlat, maxlat = -45.00, -10.00

# # indonesia coordinates
# minlon, maxlon = 90.00, 135.00
# minlat, maxlat = -15.00, 10.00

# #WestPacificSplittingMapCoordinates:
# minlon, maxlon = 70.00, 170.00
# minlat, maxlat = -60.00, 70.00

# #SEAsiaSplittingMapCoordinates:
# minlon, maxlon = 60.00, 160.00
# minlat, maxlat = -20.00, 20.00

# #USASplittingMapCoordinates:
# minlon, maxlon = -132.89, -66.09
# minlat, maxlat = 22.62, 51.64