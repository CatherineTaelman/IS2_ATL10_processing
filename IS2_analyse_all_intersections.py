#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of IS-2 intersections:
    1. filter out duplicates in dataframe (e.g. idx (2-3) is same as (3-2))
    
    2. for each intersection, calculate:
        - time difference (delta_t)
        - corresponding difference in average freeboard and roughness (delta_mean_fb & delta_roughness)
        - IQR of all freeboards
        
    3. filter out "invalid" intersections, i.e. where delta_roughness > 0.1 between two beams of an intersection
    
    4. plot:
        1) scatter plot with delta_freeboard versus delta_time: does increasing time difference between two tracks result in increasingly different retrieved freeboard values?
        2) scatter plot for query_freeboard versus target_freeboard, with delta_time as color code. Also reports Pearson's correlation coeff in plot.

Note: roughness here means small-scale surface roughness and is approximated by the width of the Gaussian fit to the IS-2 photon height distribution

@author: Catherine Taelman
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import seaborn
import matplotlib.colors

# ---------------------------------------------------------------------------- #
# DEFINE DIRS

# path to data directory
DATA_DIR = Path('data')

# path to directory with intersections we want to analyze
INTERS_DIR = DATA_DIR / 'intersections' 

# ---------------------------------------------------------------------------- #
# grab all pickle file with intersection statistics
intersections_df_filepaths = glob.glob((INTERS_DIR / '*' / 'statistics_all_intersections.pkl').as_posix())

# collect dataframes per search radius
dfs = []

# loop over files containing intersection data for different search radii
for filepath in intersections_df_filepaths:
    filepath_parentdir = Path(filepath).parent

    # grab search radius that was used
    radius = int(filepath_parentdir.as_posix()[-4:-1])
    print(radius)
    
    FIG_DIR = filepath_parentdir / 'figures'
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

    # load intersection dictionary
    with open(filepath, 'rb') as f:
        intersections = pickle.load(f)   
    
    # collect indices of duplicate pairs
    duplicate_pairs = []
    
    # filter out duplicate pairs (target-query = query-target)
    for row in intersections.itertuples():
        q_fb = row.q_fb_avg
        idx = intersections.index[intersections['t_fb_avg'] == q_fb].to_list()
        
        if not idx==[]:
            idx_list = [row.Index, idx[0]]
            duplicate_pairs.append(tuple(sorted(idx_list)))
    
    # remove duplicates in the indices list (e.g. idx (2-3) is same as (3-2))
    duplicate_pairs = list(set(duplicate_pairs))
    
    # remove duplicates from dataframe
    intersections_subset = intersections.drop(index=[i[0] for i in duplicate_pairs])
    
    # ------------------------------------------------------------------------ #
    # CALCULATE TIME DIFF AND CORRESPONDING FB AND ROUGHNESS DIFF
    
    t1 = intersections_subset.t_timestamp.values
    t2 = intersections_subset.q_timestamp.values
    
    fb1 = intersections_subset.t_fb_avg.values
    fb2 = intersections_subset.q_fb_avg.values
    
    iqr1 = intersections_subset.t_fb_iqr.values
    iqr2 = intersections_subset.q_fb_iqr.values
    
    rough1 = intersections_subset.t_roughness_avg.values
    rough2 = intersections_subset.q_roughness_avg.values
    
    delta_t = np.float64(np.abs(t2-t1).astype('timedelta64[h]'))
    delta_iqr = np.abs(iqr2-iqr1)
    delta_mean_fb = np.abs(fb2-fb1)
    delta_roughness = np.abs(rough2-rough1)
    
    # ------------------------------------------------------------------------ #
    # FILTER OUT "INVALID" INTERSECTIONS (I.E. WHERE ROUGHNESS IS TOO DIFFERENT BETWEEN 2 TRACKS)
    
    # define threshold for filtering
    condition_roughness = delta_roughness<0.1
    
    delta_t = delta_t[condition_roughness]
    delta_iqr = delta_iqr[condition_roughness]
    delta_mean_fb = delta_mean_fb[condition_roughness]
    delta_roughness = delta_roughness[condition_roughness]
    
    fb1 = fb1[condition_roughness]
    fb2 = fb2[condition_roughness]

    # convert to cm
    fb1 = fb1*100
    fb2 = fb2*100
    delta_mean_fb = delta_mean_fb * 100
    
    delta_t = np.int16(delta_t)
    filtered_df = pd.DataFrame([delta_t, delta_mean_fb, delta_roughness, fb1, fb2]).T
    filtered_df.rename(columns={0: 'delta_t', 1: 'delta_mean_fb', 2: 'delta_roughness', 3: 'fb1', 4: 'fb2'}, inplace=True)
    filtered_df['search radius [m]'] = f'{radius}'
    dfs.append(filtered_df)

    # ------------------------------------------------------------------------- #
    # PLOT
    
    # plot fb_target vs fb_query #
    
    fb_scatterplot_outpath = FIG_DIR / f'scatter_fb1_vs_fb2_{radius}.png'

    cmap=plt.cm.get_cmap('viridis', len(np.unique(delta_t)))

    delta_t = np.int16(delta_t)
    
    # fitting a linear regression line per time difference + calculate pearson's correlation coeff
    pearsons_r = []
    for time_diff, time_color in zip(np.unique(delta_t), cmap.colors):
        x = fb1[delta_t==time_diff]
        y = fb2[delta_t==time_diff]
        m, b = np.polyfit(x, y, 1)
        r = np.corrcoef(x,y)
        pearsons_r.append(np.round(r[0][1],2))
    
    # --------------------------------------------------------------- #
    plt.figure()
    
    pal = seaborn.color_palette("viridis_r", as_cmap=True)
    seaborn.jointplot(data=filtered_df, x="fb1", y="fb2", hue="delta_t", palette=pal)
    plt.grid(alpha=0.5)
    plt.legend(title='$\Delta$-time', loc='upper left', fontsize=11, markerscale=1.3)
    plt.xlim([-15,160])
    plt.ylim([-15,160])
    
    # plot y=f(x) as reference
    x = np.linspace(-10, np.max([fb1,fb2]) + 10, 1000)
    plt.plot(x, x, linestyle='dotted', c='darkgray')
    
    plt.xlabel('Laser freeboard @time 1 [cm]', fontsize=12)
    plt.ylabel('Laser freeboard @time 2 [cm]', fontsize=12)
    plt.tick_params(axis='x', labelsize=11) 
    plt.tick_params(axis='y', labelsize=11) 
    
    try:
        str_r = [str(p_r) for p_r in pearsons_r]
        textstr = f"Correlation \nR = {str_r[0]} \nR = {str_r[1]} \nR = {str_r[2]} \nR = {str_r[3]} \nR = {str_r[4]}"
        plt.text(33, 104, textstr, bbox=dict(facecolor='white', edgecolor='lightgray',alpha=0.7), linespacing=1.7, fontsize=11)
    
    except:
        str_r
    
    plt.tight_layout()
    plt.savefig(fb_scatterplot_outpath)
    
# make global dataframe with ALL intersections
df = pd.concat(dfs)
df = df.reset_index(level=0).reset_index(drop=True)
  
df['mean_fb'] = np.mean([df['fb1'], df['fb2']], axis=0)

#%% PLOT 

# Define a custom color palette
custom_palette = ['peachpuff', 'chocolate', 'saddlebrown']
custom_palette = ['sandybrown', 'chocolate', 'saddlebrown']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["sandybrown","saddlebrown"])

# ------------------------------------------------------------------------- #

# plot nr intersections per delta_t, colour-coded for search radius

plot_outpath = FIG_DIR / 'nr_intersections_vs_time_plot.png'

# group df by time delta and search radius, and retrieve amount of samples inside each group
out = df.groupby(['delta_t', 'search radius [m]']).size()

plt.figure()
ax = plt.axes()

scatter = plt.scatter(out.index.get_level_values(level='delta_t').values, out.values, c=out.index.get_level_values(level='search radius [m]').values.astype(int), cmap=cmap)

# produce a legend with the unique colors from the scatter
legend = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Search radius [m]")
ax.add_artist(legend)

plt.xlabel('$\Delta$ time [hours]', fontsize=12)
plt.ylabel('number of intersections [-]', fontsize=12)
plt.tick_params(axis='x', labelsize=12) 
plt.tick_params(axis='y', labelsize=12) 
plt.grid(alpha=0.5)

plt.tight_layout()
plt.savefig(plot_outpath)

# ------------------------------------------------------------------------- #

# plot delta_fb vs delta_t BOXPLOTS

fb_vs_time_plot_outpath = FIG_DIR / 'delta_fb_vs_time_boxplots.png'

plt.figure(figsize=(7,5))

# Set the palette for all plots
seaborn.set_palette(custom_palette)
seaborn.boxplot(data=df, x='delta_t', y='delta_mean_fb', hue='search radius [m]', palette=custom_palette, showfliers=False, gap=.2, width=.5)

#plt.scatter(delta_t, delta_mean_fb, s=8)
plt.xlabel('$\Delta$ time [hours]', fontsize=12)
plt.ylabel('$\Delta$ freeboard [cm]', fontsize=12)

plt.tick_params(axis='x', labelsize=12) 
plt.tick_params(axis='y', labelsize=12) 
plt.grid(alpha=0.5)

plt.tight_layout()
plt.savefig(fb_vs_time_plot_outpath)

# ------------------------------------------------------------------------- #