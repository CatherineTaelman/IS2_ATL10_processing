#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-process ATL10 along-track freeboards of IceSat-2 to increase coverage in areas where lead tiepoints are sparse (e.g. over fast ice).

For each beam: SSHA is interpolated using AT MOST kNN (k=10) lead tiepoints (found along that beam). Max. distance to search for lead tie points is set to 100 km.

Adapted from Jack's code to reprocess ICESAT-2 freeboards independently of distance to lead.

How does it work?
Per beam, for each along-track sample that has an ATL07 height:
    1. find k closest lead tie points within 100 km distance along beam (using kNN search with k<=10).
    2. calculate an SSHA estimate (SSHA_interpolated) via a weighted interpolation scheme of the k-closest tie points (closest tie point has largest weight)
    3. calculate a freeboard estimate: freeboard_new = ATL07_height - SSHA_interpolated
    4. if original ATL10 freeboard > 10 (i.e. invalid), assign new freeboard value to this sample. Keep original ATL10 freeboards for all valid samples.
    5. save new dataframe to disk: "original_filename"_reprocessed.h5
    6. plot results and save figures to disk

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
            
from geospatial_helpers import lat_lon_to_projected
from read_ATL10 import load_data_IS2_beam

# ----------------------------------------------------------------------------------- #
# DEFINE DATA DIRS # 

# overwrite output file (if already exists)?
overwrite = True

# path to data directory
DATA_DIR = Path('data')

# where to save re-processed ATL10 files
OUT_DIR = DATA_DIR / 'reprocessed_ATL10'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# where to save figures
FIG_DIR = OUT_DIR / 'figures'
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

# frab all files to process
original_files = glob.glob((DATA_DIR / 'original_ATL10' / "*.h5").as_posix())

# # only process first file, as example 
# original_files = original_files[0]

# ----------------------------------------------------------------------------------- #
# READ IN ORIGINAL ATL10 DATA # 
    
# read ATL10 data into pandas dataframe
for i, filepath in enumerate(original_files):
    print('-----------------------------------------------')
    print('ATL10 file:', filepath)
    
    # dir to save plots of this specific granule
    sub_figdir = FIG_DIR / f'{Path(filepath).stem}'
    Path(sub_figdir).mkdir(parents=True, exist_ok=True)
    
    # path to outfile
    df_outpath = OUT_DIR / f'{Path(filepath).stem}_reprocessed.h5'

    # check if outfile already exists
    if df_outpath.is_file() and not overwrite:
        print('Output file already exists, use `overwrite` to force')
        continue
    elif df_outpath.is_file() and overwrite:
        print('Removing existing output file')
        df_outpath.unlink()
    
    # collect lead tiepoints of ALL beams into a global dataframe --> for kNN search!
    df_leads_collection = []
    
    # collect beam dataframes per track in list
    tracks = []
    
    # loop over all beams and extract lead tiepoints
    print('Loading data from all 6 beams ... ')
    for beam_num in range(1,7):
        try:
            df_fb, df_leads = load_data_IS2_beam(filepath, beam_num)
            df_leads_collection.append(df_leads)
            tracks.append([df_fb, df_leads])
            
        except KeyError:
            print(f"Key error for beam {beam_num}")
            tracks.append([pd.DataFrame(), pd.DataFrame()])
            continue

    # concatenate all lead dataframes into a global one
    df_leads_global = pd.concat(df_leads_collection)
   
    # ----------------------------------------------------------------------------------- #
    # RE-PROCESS FREEBOARD VALUES
    
    # collect all re-processed freeboard dataFrames into a list, to later concatenate it into a global re-processed dataFrame that is saved to disk
    re_processed_df_fb = []
    
    # calculate new freeboard estimates per beam
    for df_beam in tracks:
        
        df_fb = df_beam[0]
        df_leads = df_beam[1]
        
        # only proceed if dataFrames contain data
        if not df_fb.empty and not df_leads.empty:
        
            # ----------------------------------------------------------------------------------- #
            # STEP 1: kNN SEARCH k<=10 TO FIND CLOSEST LEAD TIE POINTS
            # note: using Euclidean distance on projected coordinates
            
            # project lat,lon to EPSG 3996
            df_fb[['point_x','point_y']] = df_fb.apply(lambda row: lat_lon_to_projected(row['lats'], row['lons']), axis=1, result_type='expand')
            df_leads_global[['point_x','point_y']] = df_leads_global.apply(lambda row: lat_lon_to_projected(row['lats'], row['lons']), axis=1, result_type='expand')
            df_leads[['point_x','point_y']] = df_leads.apply(lambda row: lat_lon_to_projected(row['lats'], row['lons']), axis=1, result_type='expand')
            
            # if at least 10 lead tiepoints available
            if len(df_leads) >= 10:
                nn1 = NearestNeighbors(n_neighbors=10, metric="euclidean")
                nn1.fit(df_leads[["point_x", "point_y"]])
                D, idx = nn1.kneighbors(df_fb[['point_x', 'point_y']])
            
            # if less than 10 lead tiepoints available
            elif len(df_leads)<10:
                nn1 = NearestNeighbors(n_neighbors=len(df_leads), metric="euclidean")
                nn1.fit(df_leads[["point_x", "point_y"]])
                D, idx = nn1.kneighbors(df_fb[['point_x', 'point_y']])
            
            # ----------------------------------------------------------------------------------- #
            # STEP 2: CALCULATE WEIGTHED INTERPOLATION OF LEAD SSHA FOR EACH SAMPLE
            
            lead_ssha = df_leads['lead_ssha'].to_numpy()
            # replace zeros by small value to avoid dividing by zero later
            D = np.where(D==0, 1e-16, D)
            
            # filter out lead tie points that are >100 km away
            D_masked = np.where(D>100000, np.nan, D) # filter out distances > 100 km
            idx_masked = np.where(D>100000, np.nan, idx)
            
            ssha_interp = []
            
            # iterate through all samples because interpolation requires to get rid of NaNs in the D and idx lists
            for row_idx, row_D in zip(idx_masked, D_masked):
                row_idx = row_idx.tolist()
                # remove nan values from list and convert index values to integers
                row_idx = [int(x) for x in row_idx if (np.isnan(x) == False)]
                
                # grab corresponding distances
                row_D = row_D.tolist()
                # filter out nans
                row_D = [x for x in row_D if (np.isnan(x) == False)]
                # turn into array again
                row_D = np.array(row_D)
                
                # grab lead ssha points at these indices and do interpolation for this sample
                ssha_interp_point = np.sum(lead_ssha[row_idx] * (1 / row_D**2)) / np.sum(1 / row_D**2)
                ssha_interp.append(ssha_interp_point)
            
            del D
            del idx
            del D_masked
            del idx_masked

            # ----------------------------------------------------------------------------------- #
            # SMOOTHEN INTERPOLATION RESULTS

            # window size for smoothing based on amount of available samples
            
            # calculate consecutive differences in along track distance
            diffs_x = df_fb['along_track_distance'].diff()[1:]
            # filter out data where along-track distance makes 'jump' (i.e. gaps in data) of > 0.1km
            diffs_x_filtered = diffs_x[diffs_x<0.1] * 1000 # convert to meters
            # calculate median distance between samples based on this selection
            median_dist_x = np.median(diffs_x_filtered)
            std_dist_x = np.std(diffs_x_filtered)
            
            # define number of samples to use in running average window (we want window of ~2.5 km)
            nr_points = np.round(2500/median_dist_x)

            # frac: Between 0 and 1. The fraction of the data used when estimating each y-value (i.e. freeboard)
            if len(ssha_interp) > nr_points:
                frac = nr_points/len(ssha_interp)
            else:
                # if there are less than 200 points to do the smoothing, use all those points
                frac = 1
            
            # LOWESS = Locally Weighted Scatterplot Smoothing
            lowess = sm.nonparametric.lowess(ssha_interp, np.arange(len(ssha_interp)), frac=frac, missing='none')
            ssha_interp_smooth = lowess[:, 1]
            
            # check if there is no data in ssh_interp (due to all lead tiepoints being too far away)
            if not ssha_interp_smooth.size==0:
                df_fb['ssha_interp'] = ssha_interp_smooth
            
            # ----------------------------------------------------------------------------------- #
            # STEP 3: FILL GAPS IN ORIGINAL ATL10 FREEBOARDS #
            
            # keep original ATL10 freeboards where freeboard values < 10 m
            condition = df_fb['freeboard'] < 10 

            if not ssha_interp_smooth.size==0:
                df_fb['freeboard_new'] = df_fb['freeboard'].where(condition, df_fb['height'] - df_fb['ssha_interp'], axis=0)
                
            # if no lead tiepoints were found along beam within 100 km distance, set freeboard_v4 to NaN
            if ssha_interp_smooth.size==0:
                df_fb['freeboard_new'] = np.nan
            
            # ----------------------------------------------------------------------------------- #
            ## PLOT (INTERPOLATED) SSHA & ORIGINAL VS. NEW FREEBOARDS ## 
            
            beam_nr = df_fb['beamNum'][0]
            
            # calculate rolling average of new freeboards for plotting
            rol_fb = df_fb['freeboard_new'].dropna().rolling(window=101, center=True, closed='both').mean().reindex(df_fb.index, method='pad')

            plt.figure(figsize=(10,5))
            plt.scatter(df_fb['along_track_distance'], df_fb['height'], label='Heights ATL07', c='darkred', s=1, alpha=0.3)
            plt.scatter(df_fb['along_track_distance'].where(df_fb['freeboard'] < 10), df_fb['freeboard'].where(df_fb['freeboard'] < 10), label='Freeboard ATL10', c='midnightblue', s=1, alpha=0.3)
            plt.scatter(df_fb['along_track_distance'].where(df_fb['freeboard'] > 10), df_fb['freeboard_new'].where(df_fb['freeboard'] > 10), label='Freeboard interpol.)', c='darkorange', s=1, alpha=0.5)
            plt.plot(df_fb['along_track_distance'], rol_fb, c='black', label='Re-processed freeboards', alpha=0.8)
            plt.plot(df_fb['along_track_distance'], df_fb['ssha_interp'], c='slategrey', label='SSHA interpolated', linewidth=2, linestyle='--')
            
            plt.xlabel('Along track distance')
            plt.ylabel('Height [m]')
            plt.title(f'Beam {beam_nr}: ATL07 heights & (interpolated) SSHA & freeboards')
            plt.legend(loc='upper right', markerscale=4, fontsize=8)
            
            plt.tight_layout()
            fig_path = sub_figdir / f'Reprocessing_beam_{beam_nr}.png'
            plt.savefig(fig_path, dpi=100)
            
            # save re-processed dataFrame in list
            re_processed_df_fb.append(df_fb)
    
    # ----------------------------------------------------------------------------------- #
    # SAVE NEW DATAFRAME TO DISK
    
    # concatenate re-processed freeboard dataframes into a global one
    if re_processed_df_fb:
        df_fb_global = pd.concat(re_processed_df_fb)
    
        # save new freeboard estimates to disk (HDF5 format)
        df_fb_global.to_hdf(df_outpath, key='df', mode='w')

        # ---------------------------------------------------------- #
        # PLOT RE-PROCESSED FREEBOARD FOR ALL BEAMS #
        
        colors = ['midnightblue', 'lightskyblue', 'indigo', 'mediumorchid','darkolivegreen', 'yellowgreen']
        
        fig,ax=plt.subplots(3,1,figsize=(10,5), sharex=True, sharey=True)
        
        axes = [0,0,1,1,2,2]
        
        for beam, ax_nr in zip(range(1,7), axes):
            
            rol_fb = df_fb_global['freeboard_new'].where(df_fb_global['beamNum']==beam).rolling(window=101, center=True, closed='both').mean()
            ax[ax_nr].plot(df_fb_global['along_track_distance'], rol_fb, color=colors[beam-1], label=f'beam {beam}')
            ax[ax_nr].set_ylabel('Along-track distance [km]')
            ax[ax_nr].set_ylabel('Freeboard [m]')
            ax[ax_nr].legend(loc='lower right')

        plt.suptitle('Re-processed freeboards per pair of beams')
        plt.tight_layout()
        fig_path = sub_figdir / 'Freeboards_all_beams.png'
        plt.savefig(fig_path, dpi=100)
        
        plt.close('all')
    # ----------------------------------------------------------------------------------- #
