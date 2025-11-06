#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find beam-to-beam intersections in re-processed IS-2 data (i.e. output from 'reprocess_ATL10_freeboards.py')
    - within specified search radius (default=250m)
    - within specified time range (start_time, end_time) -> only granules acquired within this time range are used to query intersections
    - within specified time interval (default = 10 days, i.e. t-10 < t < t+10)
    - filter out data without valid 'freeboard_new' value (i.e. interpolated/original ATL10 freeboard)
    - optional: filter out invalid points using polygon as mask

Intersections are collected in a dictionary per granule, and saved to disk as pickle file ("original_filename"_intersections.pkl)
 - target_samples = samples from current granule with identified intersections
 - overlapping_query_samples = corresponding sample(s) from other granules that intersect with target_sample

@author: Catherine Taelman
"""

import glob
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import geopandas as gpd

# ---------------------------------------------------------------------------- #
# SET HYPERPARAMETERS #

# overwrite existing output?
overwrite=True

radius = 250 # in meters

# set time range for search -> query for all granules acquired within this time interval
start_time = datetime(2022,1,5)
end_time = datetime(2022,1,9)
print(f"Searching intersections from {start_time.strftime('%d-%m-%Y')} to {end_time.strftime('%d-%m-%Y')}")

# define the max. allowed time difference between two overpasses
delta_t = timedelta(days=10)

print(f'Using search radius {radius}m and time window +/- {delta_t.days} days')

# ---------------------------------------------------------------------------- #
# DEFINE DATA DIRS #

# path to data directory
DATA_DIR = Path('data')

# dir where intersection dictionaries are saved as pickle files
OUT_DIR = DATA_DIR / 'intersections' / f"{start_time.strftime('%Y-%m-%d')}_{end_time.strftime('%Y-%m-%d')}_deltaT_{delta_t.days}days_radius_{radius}m"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

masking_pol = None

# optional: read in masking polygon
# polygon_path = DATA_DIR / 'shapefiles' / 'BB_fastice_ROI_4326.shp'
# masking_pol = gpd.read_file(polygon_path)

# ---------------------------------------------------------------------------- #
# FIND GRANULES WITHIN SPECIFIED TIME WINDOW #

# frab all files to process
all_files = glob.glob((DATA_DIR / 'reprocessed_ATL10' / "*.h5").as_posix())

# make list of corresponding timestamps (from filename)
valid_timestamps = []
valid_granules = []

for granule_path in all_files:
    granule_basename = Path(granule_path).stem
    granule_timestamp = datetime.strptime(granule_basename[19:33], '%Y%m%d%H%M%S')
    if start_time < granule_timestamp < end_time:
        valid_granules.append(granule_path)
        valid_timestamps.append(granule_timestamp)

del granule_basename
del granule_timestamp
del all_files

# dict with granule_basename : timestamp
granule_timestamp_dict = dict(zip(valid_granules, valid_timestamps))

if len(valid_granules) == 0:
    print('No granules found within specified time interval!')
    print('------------------------------------------------')

# ---------------------------------------------------------------------------- #
# FIND OVERLAPPING GRANULES

# loop through granules and find other granules that overlap in time
for target_granule_path in sorted(valid_granules):
    # grab granule basename to name output file
    t_gran_basename = Path(target_granule_path).stem 
    print('-------------------------------------------------')
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
    
    if masking_pol != None:
        # drop data that is not within polygon
        points = gpd.GeoDataFrame(t_df, geometry=gpd.points_from_xy(t_df.lons, t_df.lats), crs=4326)
        t_df = gpd.sjoin(points, masking_pol, how='inner', predicate = 'within')
    
    # drop data where there is no freeboard_new value
    t_df = t_df.dropna(subset=['freeboard_new'])
    
    if t_df.empty:
        print('No intersections found')
        continue
    
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
        
        if masking_pol != None:
            # drop data that is not over fast ice area
            q_points = gpd.GeoDataFrame(q_df, geometry=gpd.points_from_xy(q_df.lons, q_df.lats), crs=4326)
            q_df = gpd.sjoin(q_points, masking_pol, how='inner', predicate = 'within')
        
        # drop data where there is no freeboard_new value
        q_df = q_df.dropna(subset=['freeboard_new'])
        query_dfs.append(q_df)
        
    if query_dfs == []:
        continue
    
    query_dataframe = pd.concat(query_dfs)
    
    # ------------------------------------------------------------------------------------------------- #
   
    intersections = {'target_samples': [], 
                     'overlapping_query_samples': []}
    
    # loop through different beams in target and query dataframes and find the k-nearest neighbours per freeboard sample in the target dataframe
    for beamNum in t_df.beamNum.unique():
        # print(f'Looking for kNN in beam {beamNum} ... ')
        
        # keep track of indices for target and query samples
        t_idx = []
        neighs_idx = []
        # keep track of returned query dataframes (i.e. neighboring/overlapping samples)
        neighs_dfs = []
        
        # get target samples of only one beam
        t_df_subset = t_df[t_df['beamNum'] == beamNum]
        # find intersections with ANY other beam
        q_df_subset = query_dataframe 
        
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
        
        # loop through each freeboard sample along track and find k-nearest neighbours within specified radius
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
    
    if intersections['target_samples'] == []:
        print('No intersections found')