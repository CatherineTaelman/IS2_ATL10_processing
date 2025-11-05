#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find intersections (crossings) in IS-2 data
    - beam-to-beam (e.g. beam gt1l with another gt1l)
    - within 10 days time interval (t-10 < t < t+10)
    - filter out sample points with no freeboard_v4 values
    - filter out sample points that are not in Belgica Bank fast ice polygon (i.e. no samples over drifting ice!)
    - only in time period 15 jan - 15 may 2022, when fast-ice is stable --> looking at same ice in same location
   
Collect all intersections for this granule in dict
 - target_samples = samples from current granule with identified intersections
 - overlapping_query_samples = corresponding sample(s) from other granules that intersect with target_sample

Created on Sun Feb  9 17:53:56 2025

@author: cat
"""

import os
import glob
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from shapely.geometry import Point
import geopandas as gpd

#%% DEFINE DATA DIRS

radius = 250 # in meters 
overwrite=True

# define directories
PROJ_DIR = Path('/home/cat/onedrive/work/PhD/belgica_bank_study')
DATA_DIR = PROJ_DIR / 'data' 
IS2_DIR = DATA_DIR / 'IS2' / 'ATL10'

# dir where intersection dictionaries are saved as pickle files
OUT_DIR = IS2_DIR / f'intersections_across_beams_radius{radius}'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

#%% LOAD BELGICA BANK FAST ICE OUTLINE ROIs

# read in FYI and MYI ROIs
fastice_roi_path = DATA_DIR / 'shapefiles' / 'BB_fastice_ROI_4326.shp'
fastice_pol = gpd.read_file(fastice_roi_path)

#%% FIND GRANULES WITHIN SPECIFIED TIME WINDOW

# get list of all granules (filepaths) with re-processed freeboards in period jan-aug 2022
all_granules = glob.glob((IS2_DIR/'2022_*/re-processed_freeboards/*.h5').as_posix())

# define time window to search for intersections
delta_t = timedelta(days=10)
print(f'Time window for intersections: +/- {delta_t}')

# set time limits to search for overlaps (we want overlaps over stable fast-ice!)
start_time = datetime(2022,1,15)
end_time = datetime(2022,5,16)

# make list of corresponding timestamps (from filename)
valid_timestamps = []
valid_granules = []

for granule_path in all_granules:
    granule_basename = Path(granule_path).stem
    granule_timestamp = datetime.strptime(granule_basename[19:33], '%Y%m%d%H%M%S')
    if start_time < granule_timestamp < end_time:
        valid_granules.append(granule_path)
        valid_timestamps.append(granule_timestamp)

del granule_basename
del granule_timestamp
del all_granules

# dict with granule_basename : timestamp
granule_timestamp_dict = dict(zip(valid_granules, valid_timestamps))

# loop through granules and find other granules that overlap in time
for target_granule_path in sorted(valid_granules):
    # grab granule basename to name output file
    t_gran_basename = Path(target_granule_path).stem 
    print(f'Target granule is: {t_gran_basename}')
    t_gran_timestamp = granule_timestamp_dict[target_granule_path]
    
    # check if output file for this granule already exists
    out_dict_filename = f'{t_gran_basename}_intersections.pkl'
    out_dict_path =  OUT_DIR / out_dict_filename
    
    if out_dict_path.is_file() and not overwrite:
        print('Output file already exists, use `overwrite` to force')
        continue
    elif out_dict_path.is_file() and overwrite:
        print('Removing existing output file')
        out_dict_path.unlink()

    # read in data from target granule
    t_df = pd.read_hdf(target_granule_path)
    
    # drop data that is not over fast ice area
    points = gpd.GeoDataFrame(t_df, geometry=gpd.points_from_xy(t_df.lons, t_df.lats), crs=4326)
    t_df_fastice = gpd.sjoin(points, fastice_pol, how='inner', predicate = 'within')
    
    # drop data where there is no freeboard_V4 value
    t_df_fastice = t_df_fastice.dropna(subset=['freeboard_v4'])
    
    # grab target timestamp and set search window in time
    t_timestamp = t_df.timestamps.values[0]
    ts = pd.to_datetime(str(t_timestamp))
    t_timestamp_string = ts.strftime('%Y%m%d')
    
    # time search window (rough estimate based on granule filenames, exact filtering can happen further down if needed)
    min_t = ts - delta_t
    max_t = ts + delta_t
    
    # find other granules within specified time window --> will become tree to query from
    query_dict = {k:v for (k,v) in granule_timestamp_dict.items() if (v > min_t) and (v < max_t) and (v != t_gran_timestamp)}

    if query_dict == {}:
        continue
    
    #print(f'Query granules are: {query_dict.keys()}')
    
    # load data from query granules
    query_dfs = []
    for query_gran_path in query_dict.keys():
        q_df = pd.read_hdf(query_gran_path)
        
        # drop data that is not over fast ice area
        q_points = gpd.GeoDataFrame(q_df, geometry=gpd.points_from_xy(q_df.lons, q_df.lats), crs=4326)
        q_df_fastice = gpd.sjoin(q_points, fastice_pol, how='inner', predicate = 'within')
        
        # drop data where there is no freeboard_V4 value
        q_df_fastice = q_df_fastice.dropna(subset=['freeboard_v4'])
        
        query_dfs.append(q_df_fastice)
    
    if query_dfs == []:
        continue
    
    query_dataframe = pd.concat(query_dfs)
    
    # ------------------------------------------------------------------------------------------------- #
   
    intersections = {'target_samples': [], 
                     'overlapping_query_samples': []}
    
    # loop through different beams in target and query dataframes and find the k-nearest neighbours per freeboard sample in the target dataframe
    for beamNum in t_df_fastice.beamNum.unique():
        print(f'Looking for kNN in beam {beamNum} ... ')
        
        # keep track of indices for target and query samples
        t_idx = []
        neighs_idx = []
        # keep track of returned query dataframes (i.e. neighboring/overlapping samples)
        neighs_dfs = []
        
        # get target samples of only one beam
        t_df_subset = t_df_fastice[t_df_fastice['beamNum'] == beamNum]
        #q_df_subset = query_dataframe[query_dataframe['beamNum'] == beamNum]
        q_df_subset = query_dataframe # find intersections with ANY other beam
        
        # if not data in either target or query dataframes, continue to next beam
        if (t_df_subset.shape[0] == 0) or (q_df_subset.shape[0] == 0):
            print('Target or query dataframe empty, proceeding to next beam')
            continue

        # get target samples
        target_samples = t_df_subset[['point_x','point_y']].copy() # coord(x,y)
        target_samples = target_samples.values # array XY
        
        # get query samples to build tree
        query_samples = q_df_subset[['point_x','point_y']].copy() # coord(x,y)
        query_samples = query_samples.values # array XY
        
        # build KDtree from the query samples
        from scipy.spatial import KDTree
        tree = KDTree(query_samples)
        
        # loop through each fb point along track and find k-nearest neighbours within specified radius
        for target_idx, target_sample in enumerate(target_samples): 
            point_coord = [target_sample[0], target_sample[1]]
            query_idxs = tree.query_ball_point(point_coord, radius) # returns indices of neighbours in query tree within radius
            if query_idxs != []:
                t_idx.append(target_idx)
                neighs_idx.append(query_idxs)
                neighs = q_df_subset.iloc[query_idxs]
                neighs_dfs.append(neighs)

        # save targets and corresponding query returns (neighbours) to dict
        if not neighs_idx==[]:
            print('Found intersection!')
            targets = t_df_subset.iloc[t_idx]
            # save intersection data to dict
            intersections['target_samples'].append(targets)
            intersections['overlapping_query_samples'].append(neighs_dfs)

    # save intersection dictionary to pickle file for later use 
    if not intersections['target_samples'] == []:
        with open(out_dict_path, 'wb') as f:
            pickle.dump(intersections, f)