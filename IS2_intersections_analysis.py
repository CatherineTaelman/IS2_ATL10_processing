#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inside 'geospatial' env

Analysis of IS-2 intersections:
    - filter out duplicates in dataframe (e.g. idx (2-3) is same as (3-2))
    - for each intersection, calculate:
        - time delta
        - corresponding delta in average freeboard and roughness values, as well as delta in freeboard histogram IQRs
    - plot:
        1) scatter plot with delta_freeboard versus delta_time: does increasing time difference between two tracks result in increasingly different retrieved freeboard values?
        2) scatter plot for query_freeboard versus target_freeboard, with delta_time as color code. Also reports Pearson's correlation coeff in plot.
            --> what is correlation between two tracks that intersect at different time stamps? Does increasing time difference influence the correlation?
            How is the relationship for thinner versus thicker retrieved freeboards?

@author: cat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import seaborn
import matplotlib.colors

#%% DEFINE DIRS

# define directories
PROJ_DIR = Path('/home/cat/onedrive/work/PhD/belgica_bank_study')
DATA_DIR = PROJ_DIR / 'data' 

RESULTS_DIR = PROJ_DIR / 'results' / 'altimetry' / 'IS2' / 'cross-over_study' / 'plot_intersections_for_paper'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

intersections_df_filepaths = [DATA_DIR / 'IS2' / 'ATL10' / 'intersections_across_beams_radius250' / 'all_intersections_stats_across_beams_radius250_dataframe.pkl',
                              DATA_DIR / 'IS2' / 'ATL10' / 'intersections_across_beams_radius500' / 'all_intersections_stats_across_beams_radius500_dataframe.pkl',
                              DATA_DIR / 'IS2' / 'ATL10' / 'intersections_across_beams_radius1000' / 'all_intersections_stats_across_beams_radius1000_dataframe.pkl']

radii = [250,500,1000]

# collect dataframes per search radius here
dfs = []


# loop over files containing intersection data for different search radii
for filepath, radius in zip(intersections_df_filepaths, radii):
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
    
    # FILTER OUT "INVALID" INTERSECTIONS
    
    # filter based on roughness conditions:
    # if delta_roughness < threshold --> keep sample
    
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
    
    # # define threshold for filtering
    # condition_IQR = delta_iqr<0.08
    condition_roughness = delta_roughness<0.1
    
    # # filter
    # delta_t = delta_t[condition_IQR & condition_roughness]
    # delta_iqr = delta_iqr[condition_IQR & condition_roughness]
    # delta_mean_fb = delta_mean_fb[condition_IQR & condition_roughness]
    # delta_roughness = delta_roughness[condition_IQR & condition_roughness]
    
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
    ##  plot fb_target vs fb_query ##
    
    fb_scatterplot_outpath = RESULTS_DIR / f'scatter_fb1_vs_fb2_{radius}.png'
    
    # plt.figure()
    # ax = plt.axes()
    
    cmap=plt.cm.get_cmap('viridis', len(np.unique(delta_t)))
    
    delta_t = np.int16(delta_t)
    
    # # # plot y=f(x) as reference
    # x = np.linspace(-5, np.max([fb1,fb2]), 1000)
    # plt.plot(x, x, linestyle='dotted', c='darkgray')
    
    # fitting a linear regression line per time difference + calculate pearson's correlation coeff
    pearsons_r = []
    for time_diff, time_color in zip(np.unique(delta_t), cmap.colors):
        x = fb1[delta_t==time_diff]
        y = fb2[delta_t==time_diff]
        m, b = np.polyfit(x, y, 1)
        #plt.plot(x, m*x + b, c=time_color, linestyle='dashed', linewidth=1)
        r = np.corrcoef(x,y)
        pearsons_r.append(np.round(r[0][1],2))
    
    # # scatter fb1 vs fb2
    # scatter = plt.scatter(fb1, fb2, s=5, c=delta_t)
    # #scatter = plt.scatter(fb1, fb2, s=5, c=delta_t, cmap=cmap)
    
    # # produce a legend with the unique colors from the scatter
    # legend = ax.legend(*scatter.legend_elements(),
    #                     loc="lower right", title="$\Delta$-time [hrs]")
    # ax.add_artist(legend)
    
    # str_r = [str(p_r) for p_r in pearsons_r]
    # textstr = f"Correlation \nR = {str_r[0]} \nR = {str_r[1]} \nR = {str_r[2]} \nR = {str_r[3]} \nR = {str_r[4]}"
    # ax.text(70, -7, textstr, bbox=dict(facecolor='white', alpha=0.5), linespacing=1.8)
    
    # # ticks = range(45,226, int(np.round(226/5))) # where to place ticks on cbar
    # # tick_labels = [str(int(item)) for item in sorted(np.unique(delta_t))] # labels of ticks
    # # cbar = plt.colorbar(ticks=ticks, label='$\Delta$-time [hours]')
    # # #plt.clim(ticks[0]-0.5, ticks[-1]+0.5)
    # # cbar.ax.set_yticklabels(tick_labels)  # vertically oriented colorbar
    
    # # plt.xlim(left=-0.1)
    # # plt.ylim(bottom=-0.1)
    # plt.xlabel('Laser freeboard @time 1 [cm]')
    # plt.ylabel('Laser freeboard @time 2 [cm]')
    # plt.grid(alpha=0.5)
    
    # plt.tight_layout()
    # plt.savefig(fb_scatterplot_outpath, dpi=300)

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
    
    str_r = [str(p_r) for p_r in pearsons_r]
    textstr = f"Correlation \nR = {str_r[0]} \nR = {str_r[1]} \nR = {str_r[2]} \nR = {str_r[3]} \nR = {str_r[4]}"
    plt.text(33, 104, textstr, bbox=dict(facecolor='white', edgecolor='lightgray',alpha=0.7), linespacing=1.7, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(fb_scatterplot_outpath, dpi=300)
    
# make global dataframe with ALL intersections included
df = pd.concat(dfs)
df = df.reset_index(level=0).reset_index(drop=True)
  
df['mean_fb'] = np.mean([df['fb1'], df['fb2']], axis=0)

#%% PLOT 

# Define a custom color palette
custom_palette = ['peachpuff', 'chocolate', 'saddlebrown']
custom_palette = ['sandybrown', 'chocolate', 'saddlebrown']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["sandybrown","saddlebrown"])

# ------------------------------------------------------------------------- #

# # plot delta_fb vs delta_t

# plot_outpath = RESULTS_DIR / 'fb_vs_time_plot.png'

# plt.figure(figsize=(7,5))
# plt.scatter(df['delta_t'], df['mean_fb'], s=8)
# plt.xlabel('$\Delta$ time [hours]')
# plt.ylabel('Mean laser freeboard [m]')
# plt.grid()

# plt.tight_layout()
# plt.savefig(plot_outpath, dpi=300)

# ------------------------------------------------------------------------- #
# # plot average fb1,fb2 vs delta_t

# plot_outpath = RESULTS_DIR / 'avg_fb1_2_vs_time_plot.png'

# plt.figure(figsize=(7,5))
# plt.scatter(delta_t, np.mean(fb1,fb2), s=8)
# plt.xlabel('$\Delta$ time [hours]')
# plt.ylabel('$\Delta$ roughness [m]')

# plt.tight_layout()
# plt.savefig(plot_outpath)

# ------------------------------------------------------------------------- #

# plot nr intersections per delta_t, colour-coded for search radius

plot_outpath = RESULTS_DIR / 'nr_intersections_vs_time_plot.png'

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
plt.savefig(plot_outpath, dpi=300)

# ------------------------------------------------------------------------- #

# plot delta_fb vs delta_t BOXPLOTS

fb_vs_time_plot_outpath = RESULTS_DIR / 'delta_fb_vs_time_boxplots.png'

plt.figure(figsize=(7,5))

# Set the palette for all plots
seaborn.set_palette(custom_palette)
seaborn.boxplot(data=df, x='delta_t', y='delta_mean_fb', hue='search radius [m]', palette=custom_palette, showfliers=False, gap=.2, width=.5)

#plt.scatter(delta_t, delta_mean_fb, s=8)
plt.xlabel('$\Delta$ time [hours]', fontsize=12)
plt.ylabel('$\Delta$ freeboard [cm]', fontsize=12)
#plt.title('IS-2 intersections: $\Delta$-freeboard versus $\Delta$-time')

plt.tick_params(axis='x', labelsize=12) 
plt.tick_params(axis='y', labelsize=12) 
plt.grid(alpha=0.5)

plt.tight_layout()
plt.savefig(fb_vs_time_plot_outpath, dpi=300)

#%%

# ------------------------------------------------------------------------- #