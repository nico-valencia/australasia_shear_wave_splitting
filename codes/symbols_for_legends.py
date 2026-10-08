'''
Sep 8 2026 

Maps use markers that need to be show in a legend outside of the figure. 
This script generates a plot with stand alone markers so that they can be used within Illustrator to create a legend.


'''

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


legend_df = pd.DataFrame([{
    'Station': r'$\mathbf{\delta t}$ = 1.0',
    'Latitude': -45,
    'Longitude': 105,
    'phi': 90,
    'dt': 1
}])


fig, ax = plt.subplots(figsize=(20, 10))

ax.plot([0, 1], [0, 1], color="black", linewidth=5)
ax.plot([-2, -1], [-2, -1], color="red", linewidth=5)
ax.plot([-4, -3], [-4, -3], color="blue", linewidth=5)
ax.plot([2,3], [2,3], color="tomato", linewidth=5)

plt.quiver(
    0,1,
    1,1,
    angles='xy',
    scale_units='xy',
    scale=1,
    width=0.009,
    color='limegreen',
    alpha=0.9,
    headwidth=2,
    headlength=2,
    headaxislength=2.0,
    zorder=20
)

ax.plot(-2,2,
        marker='o',
        markersize=10,
        markerfacecolor='none',
        markeredgecolor='dimgrey',
        markeredgewidth=5.0,
        linestyle='None',
        alpha=0.8)  


# for i in range(len(legend_df)):
#     plot_station_splitting(legend_df['Station'].iloc[i],
#                            legend_df['Latitude'].iloc[i],
#                            legend_df['Longitude'].iloc[i],
#                            legend_df['phi'].iloc[i],
#                            4*legend_df['dt'].iloc[i],
#                            ax=ax,
#                            version='Legend')

ax.set_ylim(-5,5)
ax.set_xlim(-5,5)


fig.savefig(f'/work/gcl3/BR/nicolasv03/senior_thesis/measurements/publication_map/legend_symbols.pdf', bbox_inches="tight", pad_inches=0.2) #original dpi=300
