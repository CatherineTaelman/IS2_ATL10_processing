#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses pickle dictionaries with intersections per granule, created in 'IS2_find_intersections.py'.

Gather ALL IS-2 intersection stats into a global dataframe and save to pickle file for futher analysis.

Output dataframe has following data:
    intersections = {'t_timestamp':[],
                     't_fb_avg':[],
                     't_fb_std':[],
                     't_fb_iqr':[],
                     't_roughness_avg':[],
                     't_roughness_std':[],
                     'q_timestamp':[],
                     'q_fb_avg':[],
                     'q_fb_std':[],
                     'q_fb_iqr':[],
                     'q_roughness_avg':[],
                     'q_roughness_std':[],
                     'beamNum':[]
                     }
    
    with t=target granule and q=query granules intersecting with target granule

@author: Catherine Taelman
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pickle
from matplotlib_scalebar.scalebar import ScaleBar
import cartopy.crs as ccrs
import geopandas as gpd
from scipy.stats import iqr
import math

# ---------------------------------------------------------------------------- #
# DEFINE DIRS

# path to data directory
DATA_DIR = Path('data')

# path to directory with intersections we want to analyze
INTERS_DIR = DATA_DIR / 'intersections' / '2022-01-05_2022-01-09_deltaT_10days_radius_250m'

FIG_DIR = INTERS_DIR / 'figures'
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

outfile_path = INTERS_DIR / 'statistics_all_intersections.pkl'

# search radius that was used
radius = int(INTERS_DIR.stem[-4:-1])

plot_crossing = False

# ---------------------------------------------------------------------------- #
# PREPARE FOR ANALYSIS OF INTERSECTION DATA

# save intersection data (target+query granules) into global dataframe
intersections = {'t_timestamp':[],
                 't_fb_avg':[],
                 't_fb_std':[],
                 't_fb_iqr':[],
                 't_roughness_avg':[],
                 't_roughness_std':[],
                 't_beamNum' : [],
                 'q_timestamp':[],
                 'q_fb_avg':[],
                 'q_fb_std':[],
                 'q_fb_iqr':[],
                 'q_roughness_avg':[],
                 'q_roughness_std':[],
                 'q_beamNum':[]
                 }

intersections = pd.DataFrame(data = intersections)

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    
    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).
    
    values, weights -- np.ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    std = math.sqrt(variance)
    return average, std

# ---------------------------------------------------------------------------- #
# LOOP THROUGH ALL GRANULE INTERSECTION FILES TO COLLECT STATS INTO ONE GLOBAL DATAFRAME

# get list of all intersections dictionaries
all_dict_filepaths = glob.glob((INTERS_DIR/'processed_*.pkl').as_posix())

for filepath in all_dict_filepaths:
    # load intersection dictionary
    with open(filepath, 'rb') as f:
        intersection_dict = pickle.load(f)
    
    t_dfs_list = intersection_dict['target_samples'] # all samples from granule where some neighbouring samples from another granule were detected in prefedined radius
    query_dfs_list = intersection_dict['overlapping_query_samples'] # neighbouring samples corresponding to each target sample
    
    # loop over different target tracks and corresponding query tracks with intersections
    for t_df, query_dfs in zip(t_dfs_list, query_dfs_list):
    
        # grab timestamps, xy positions, freeboard values from the target samples 
        t_time = t_df.timestamps.values
        t_xy = [t_df.point_x.values, t_df.point_y.values]
        t_beam = t_df.beamNum.unique()

        # make sure there is only one beamnumber in target dataframe
        assert len(t_beam) == 1
        t_beam = t_beam[0]
                
        # concatenate all query samples (i.e. neighbours of targets) into one dataframe
        q_df = pd.concat(query_dfs)
        
        # remove duplicates in dataframe (i.e. same index)
        q_df = q_df[~q_df.index.duplicated(keep='first')]
        
        # grab timestamps, xy positions, freeboard values from the target samples 
        q_time = q_df.timestamps.values
        
        # if there are different timestamps (i.e. different granules intersecting) --> split up into multiple intersections
        q_df['date'] = pd.to_datetime(q_time).date
        q_unique_dates = np.unique(q_df['date'])
        
        # if there are multiple query tracks that intersect target track, loop through each one separately
        if len(q_unique_dates) > 1:
            print('more than 1 intersection')
            
            # remove column that was created in previous sjoin
            t_df = t_df.drop(['index_right'], axis=1)
            q_df = q_df.drop(['index_right'], axis=1)
            
            for unique_date in q_unique_dates:
                # take subset of querys on this date
                q_df_subset = q_df[q_df['date'] == unique_date]
                
                # list query beamNums
                q_beams = q_df_subset.beamNum.unique()
                
                # loop over query beams
                for q_beam in q_beams:
                
                    # only keep query samples of this beam
                    q_df_subset_beam = q_df_subset[q_df_subset['beamNum'] == q_beam]
                    
                    # drop target samples that do not intersect within radius 
                    t_points = gpd.GeoDataFrame(t_df, geometry=gpd.points_from_xy(t_df.lons, t_df.lats), crs=4326)
                    t_points_proj = t_points.to_crs({"init": "EPSG:3996"})
                    q_points = gpd.GeoDataFrame(q_df_subset_beam, geometry=gpd.points_from_xy(q_df_subset_beam.lons, q_df_subset_beam.lats), crs=4326)
                    q_points_proj = q_points.to_crs({"init": "EPSG:3996"})
                    
                    q_union = q_points_proj.buffer(radius).unary_union # buffer query samples with radius and take union
                    t_neighbours = t_points_proj["geometry"].intersection(q_union) # find target samples that intersect with this union
                    
                    # subset target samples to only keep neighbours of query samples
                    t_df_subset = t_df[~t_neighbours.is_empty]
                    t_xy = [t_df_subset.point_x.values, t_df_subset.point_y.values]
                    t_fb = t_df_subset.freeboard_new.values
                    
                    q_xy = [q_df_subset_beam.point_x.values, q_df_subset_beam.point_y.values]
                    q_fb = q_df_subset_beam.freeboard_new.values
                    
                    t_date_formatted = datetime.strftime(pd.to_datetime(t_time[0]), '%Y-%m-%d')
                    q_date_formatted = datetime.strftime(unique_date, '%Y-%m-%d')
                    
                    # if either target or query dataframes have too few samples, skip
                    if (len(t_df_subset) < 100) or (len(q_df_subset_beam) < 100):
                        continue
                    
                    # ---------------------------------------------------------------------------- #
                    # PLOT CROSSING TRACKS ON MAP FOR VISUAL INSPECTION
                    
                    if plot_crossing:
                        map_outdir = FIG_DIR / 'intersections_on_map'
                        Path(map_outdir).mkdir(parents=True, exist_ok=True)
                        crossing_map_outpath = map_outdir / f'intersection_{t_date_formatted}_{q_date_formatted}_beam_{t_beam}.png'
                        
                        # plot intersection
                        fig = plt.figure(figsize=(13,13))
                        
                        # Polar Stereographic projection corresponding to epsg 3996
                        projection = ccrs.Stereographic(
                                        central_latitude = 90,
                                        central_longitude = 0,
                                        false_easting = 0,
                                        false_northing = 0,
                                        true_scale_latitude = 75,
                                        globe = None
                                    )
                        
                            
                        # assign projection
                        ax = plt.axes(projection=projection)    
    
                        # add gridlines
                        gl = ax.gridlines(draw_labels = True)
                        gl.top_labels = False
                        gl.left_labels = False
                        
                        # add scalebar
                        ax.add_artist(ScaleBar(1))
                        
                        ax.scatter(t_xy[0], t_xy[1], s=3, label='target granule')
                        ax.scatter(q_xy[0], q_xy[1], s=3, label='query granule')
                        ax.legend(loc='lower right')
                        plt.title(f'Intersection {t_date_formatted} and {q_date_formatted}')
                        
                        plt.tight_layout()
                        plt.savefig(crossing_map_outpath)
                    
                    # ---------------------------------------------------------------------------- #
                    # CALCULATE STATISTICS FOR INTERSECTION
                    
                    # ------------------------------------------- #
                    ## FOR TARGET GRANULE ##
                    
                    t_fb = t_df_subset.freeboard_new.values
                    t_h_width_gaussian = t_df_subset.h_w_gauss.values
                    t_seg_lengths = t_df_subset.seg_length.values
                    
                    # weighted average and STD of freeboard values
                    t_fb_avg, t_fb_std = weighted_avg_and_std(t_fb, weights=t_seg_lengths)
                    
                    # interquartile range
                    t_iqr = iqr(t_fb)
                    
                    # weighted average of roughness (note: width of impulse-fitted Gaussian is proxy for roughness)
                    t_rough_avg = np.average(t_h_width_gaussian, weights=t_seg_lengths)
                    t_rough_std = np.std(t_h_width_gaussian)
                    
                    print(f'Target mean fb +- STD : {t_fb_avg} +- {t_fb_std}')
                    print(f'Target mean roughness +- STD : {t_rough_avg} +- {t_rough_std}')
                    print(f'Target IQR = {t_iqr}')
                    
                    # ------------------------------------------- #
                    ## FOR QUERY GRANULES ## 
                    
                    q_fb = q_df_subset_beam.freeboard_new.values
                    q_h_width_gaussian = q_df_subset_beam.h_w_gauss.values
                    q_seg_lengths = q_df_subset_beam.seg_length.values
                    
                    # weighted average and STD of freeboard values
                    q_fb_avg, q_fb_std = weighted_avg_and_std(q_fb, weights=q_seg_lengths)
                    
                    # interquartile range
                    q_iqr = iqr(q_fb)
                    
                    # weighted average of roughness (note: width of impulse-fitted Gaussian is proxy for roughness)
                    q_rough_avg = np.average(q_h_width_gaussian, weights=q_seg_lengths)
                    q_rough_std = np.std(q_h_width_gaussian)
                    
                    print(f'Query mean fb +- STD : {q_fb_avg} +- {q_fb_std}')
                    print(f'Query mean roughness +- STD : {q_rough_avg} +- {q_rough_std}')
                    print(f'Query IQR = {q_iqr}')
                    
                    # ------------------------------------------- #
                    # collect statistics of intersections into large dataframe for futher analysis
                    new_data_row = {'t_timestamp':[t_time[0]],
                                    't_fb_avg':[t_fb_avg],
                                    't_fb_std':[t_fb_std],
                                    't_fb_iqr':[t_iqr],
                                    't_roughness_avg':[t_rough_avg],
                                    't_roughness_std':[t_rough_std],
                                    't_beamNum':[t_beam],
                                    'q_timestamp':[q_df_subset_beam.timestamps.values[0]],
                                    'q_fb_avg':[q_fb_avg],
                                    'q_fb_std':[q_fb_std],
                                    'q_fb_iqr':[q_iqr],
                                    'q_roughness_avg':[q_rough_avg],
                                    'q_roughness_std':[q_rough_std],
                                    'q_beamNum':[q_beam]
                                     }
                    
                    new_data_row = pd.DataFrame(data = new_data_row)
                    intersections = pd.concat([intersections, new_data_row], ignore_index=True)      
        
        
        # if there is only a single intersecting track
        else:
            # list query beamNums
            q_beams = q_df.beamNum.unique()
            
            # loop over query beams
            for q_beam in q_beams:
            
                # only keep query samples of this beam
                q_df_subset = q_df[q_df['beamNum'] == q_beam]
            
                # drop target samples that do not intersect within radius 
                t_points = gpd.GeoDataFrame(t_df, geometry=gpd.points_from_xy(t_df.lons, t_df.lats), crs=4326)
                t_points_proj = t_points.to_crs({"init": "EPSG:3996"})
                q_points = gpd.GeoDataFrame(q_df_subset, geometry=gpd.points_from_xy(q_df_subset.lons, q_df_subset.lats), crs=4326)
                q_points_proj = q_points.to_crs({"init": "EPSG:3996"})
                
                q_union = q_points_proj.buffer(radius).unary_union # buffer query samples with 1km radius and form union
                t_neighbours = t_points_proj["geometry"].intersection(q_union) # find target samples that intersect with this union
                
                # subset target samples to only keep neighbours of query samples
                t_df_subset = t_df[~t_neighbours.is_empty]
                t_xy = [t_df_subset.point_x.values, t_df_subset.point_y.values]
                t_fb = t_df_subset.freeboard_new.values
                t_h_width_gaussian = t_df_subset.h_w_gauss.values
                t_seg_lengths = t_df_subset.seg_length.values
                
                # grab xy locations, freeboards, width Gaussian, segment lengts, values for this query beam
                q_xy = [q_df_subset.point_x.values, q_df_subset.point_y.values]
                q_fb = q_df_subset.freeboard_new.values 
                q_h_width_gaussian = q_df_subset.h_w_gauss.values # width of fitted gaussians per segment
                q_seg_lengths = q_df_subset.seg_length.values # segment lengths for normalizing
                
                t_date_formatted = datetime.strftime(pd.to_datetime(t_time[0]), '%Y-%m-%d')
                q_date_formatted = datetime.strftime(pd.to_datetime(q_time[0]), '%Y-%m-%d')
                
                # if either target or query dataframes have too few samples, skip
                if (len(t_df_subset) < 50) or (len(q_df_subset) < 50):
                    print('Less than 50 samples in either target or query dataframe, continuing to next intersection')
                    continue
            
                # PLOT CROSSING ON MAP # 
                
                if plot_crossing:
                    
                    map_outdir = FIG_DIR / 'crossovers_on_map'
                    Path(map_outdir).mkdir(parents=True, exist_ok=True)
                    crossing_map_outpath = map_outdir / f'crossing_{t_date_formatted}_{q_date_formatted}_beams_{t_beam}_{q_beam}.png'
                    
                    # plot intersection
                    fig = plt.figure(figsize=(13,13))
                    
                    # Polar Stereographic projection corresponding to epsg 3996
                    projection = ccrs.Stereographic(
                                    central_latitude = 90,
                                    central_longitude = 0,
                                    false_easting = 0,
                                    false_northing = 0,
                                    true_scale_latitude = 75,
                                    globe = None
                                )
                    
                        
                    # assign projection
                    ax = plt.axes(projection=projection)    

                    # add gridlines
                    gl = ax.gridlines(draw_labels = True)
                    gl.top_labels = False
                    gl.left_labels = False
                    
                    # add scalebar
                    ax.add_artist(ScaleBar(1))
                    
                    ax.scatter(t_xy[0], t_xy[1], s=3, label='target granule')
                    ax.scatter(q_xy[0], q_xy[1], s=3, label='query granule')
                    ax.legend(loc='lower right')
                    plt.title(f'Intersection {t_date_formatted} and {q_date_formatted}')
                    
                    plt.tight_layout()
                    plt.savefig(crossing_map_outpath)
                
                #%% CALCULATE STATISTICS FOR INTERSECTION
                
                # ------------------------------------------- #
                ## FOR TARGET GRANULE ##
                
                # weighted average and STD of freeboard values
                t_fb_avg, t_fb_std = weighted_avg_and_std(t_fb, weights=t_seg_lengths)
                
                # interquartile range
                t_iqr = iqr(t_fb)
                
                # weighted average of roughness (note: width of impulse-fitted Gaussian is proxy for roughness)
                t_rough_avg = np.average(t_h_width_gaussian, weights=t_seg_lengths)
                t_rough_std = np.std(t_h_width_gaussian)
                
                print(f'Target mean fb +- STD : {t_fb_avg} +- {t_fb_std}')
                print(f'Target mean roughness +- STD : {t_rough_avg} +- {t_rough_std}')
                print(f'Target IQR = {t_iqr}')
                
                # ------------------------------------------- #
                ## FOR QUERY GRANULES ## 
                
                # interquartile range
                q_iqr = iqr(q_fb) 
                
                # weighted average and STD of freeboard values
                q_fb_avg, q_fb_std = weighted_avg_and_std(q_fb, weights=q_seg_lengths)
                
                # weighted average of roughness (note: width of impulse-fitted Gaussian is proxy for roughness)
                q_rough_avg = np.average(q_h_width_gaussian, weights=q_seg_lengths)
                q_rough_std = np.std(q_h_width_gaussian)
                
                print(f'Query mean fb +- STD : {q_fb_avg} +- {q_fb_std}')
                print(f'Query mean roughness +- STD : {q_rough_avg} +- {q_rough_std}')
                print(f'Query IQR = {q_iqr}')
                
                # ------------------------------------------- #
                # collect statistics of intersections into large dataframe for futher analysis
                new_data_row = {'t_timestamp':[t_time[0]],
                                't_fb_avg':[t_fb_avg],
                                't_fb_std':[t_fb_std],
                                't_fb_iqr':[t_iqr],
                                't_roughness_avg':[t_rough_avg],
                                't_roughness_std':[t_rough_std],
                                't_beamNum':[t_beam],
                                'q_timestamp':[q_time[0]],
                                'q_fb_avg':[q_fb_avg],
                                'q_fb_std':[q_fb_std],
                                'q_fb_iqr':[q_iqr],
                                'q_roughness_avg':[q_rough_avg],
                                'q_roughness_std':[q_rough_std],
                                'q_beamNum':[q_beam]
                                 }
        
                new_data_row = pd.DataFrame(data = new_data_row)
                intersections = pd.concat([intersections, new_data_row], ignore_index=True)

# save intersection dictionary to pickle file for later use 
with open(outfile_path, 'wb') as f:
    pickle.dump(intersections, f)
