#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-process freeboards of IceSat-2 

    VERSION 4: ssha is extrapolated using AT MOST kNN (k=10) lead tiepoints PER BEAM. Max distance to lead tie points is set to 100 km.

Adapted from Jack's code to reprocess ICESAT-2 freeboards independently of distance to lead

What does code does:
1. read in ATL10 data and return pandas Dataframe with relevant variables 
2. extrapolate ssha values to calculate freeboard values at distances up to 100 km --> 'freeboard_new'
3. plot ATL07 heights, original freeboard product from ATL10, and new freeboard product obtained from extrapolating ssh to all height points

"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import glob
import cartopy.crs as ccrs
from matplotlib_scalebar.scalebar import ScaleBar
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
            
from geospatial_helpers import lat_lon_to_projected
from read_ATL10 import load_data_IS2_beam


# define directories
PROJ_DIR = Path('/home/cat/onedrive/work/PhD/belgica_bank_study')
DATA_DIR = PROJ_DIR / 'data' / 'IS2' / 'ATL10'

# list of folders to process
folders = ['2022_01',
 '2022_02',
 '2022_03',
 '2022_04',
 '2022_05',
 '2022_06',
 '2022_07',
 '2022_08']

overwrite = False

for folder in folders:
    
    # dir to save re-processed freeboard values (dataframe to HDF5 file)
    DF_OUT_DIR = DATA_DIR / folder / 're-processed_freeboards'
    Path(DF_OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # frab all h5 files to process
    datfiles = glob.glob((DATA_DIR/folder/"*.h5").as_posix())

    #%% READ IN IS-2 DATA from original h5 files
    
    # Read IS-2 data into pandas dataframe
    for i, filepath in enumerate(datfiles):
        print('ATL10 file:', filepath)
        
        # dir to save plots
        OUT_DIR = PROJ_DIR / 'results' / 'altimetry' / 'IS2' / 're-processed_freeboards' / f'{Path(filepath).stem}'
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # file to save print statements of terminal during runtime
        output_terminal_filepath = OUT_DIR.parent / 'terminal_output.txt'
        
        with open(output_terminal_filepath, 'a') as file:
            print('\n ------------------------------------------- ', file=file)
            print(f'\n file: {Path(filepath).stem}', file=file)
        
        # path to outfile (dataframe saved as h5)
        df_outpath = DF_OUT_DIR / f'{Path(filepath).stem}_reprocessed.h5'
        
        # # check if outfile already exists
        # if df_outpath.is_file() and not overwrite:
        #     print('Output file already exists, use `overwrite` to force')
        #     continue
        # elif df_outpath.is_file() and overwrite:
        #     print('Removing existing output file')
        #     df_outpath.unlink()
        
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
        
        # # load data for all beams
        # tracks = []
        
        # print('Loading data from all 6 beams ... ')
        # for beam_num in beamNums:
        #     try:
        #         df_fb, df_leads = load_data_IS2_beam(filepath, beam_num)
        #         tracks.append([df_fb, df_leads])
        #     except KeyError:
        #         print(f"Key error for beam {beam_num}")
        #         tracks.append([pd.DataFrame(), pd.DataFrame()])
        #         continue
        
        #%% PLOT TRACK ON MAP
        
        # plt.figure(figsize=(10,10), dpi= 150)
        # # assign projection "NorthPolarStereo"
        # ax = plt.axes(projection=ccrs.NorthPolarStereo(true_scale_latitude=70))
        
        # cb2 = plt.scatter(df_leads_global['lons'], df_leads_global['lats'], c=df_leads_global['lead_ssh'], s=1, cmap='Blues', transform=ccrs.PlateCarree())
        
        # ax.coastlines()
        # #plt.colorbar(cb1, label=var+' (m)', shrink=0.3, extend='both')
        # plt.colorbar(cb2, label='lead ssh (m)', shrink=0.3, extend='both')
        # plt.title('Original ATL10 lead tiepoints for all 6 beams')
        
        # # Limit the map to 72 degrees latitude
        # ax.set_extent([-18, 9, 82, 76], ccrs.PlateCarree())
        # #ax.set_extent([-180, 180, 90, 75], ccrs.PlateCarree())
        # ax.add_artist(ScaleBar(1, box_color="silver", box_alpha=0.5, location='upper left'))
        # plt.tight_layout()
        # plt.savefig(OUT_DIR / 'original_lead_points_all_beams_on_map.png', dpi=150)
        
        # ---------------------------------------------------------------------------------- #
        
        # var='freeboard'
        
        # # for plotting, set max freeboard value to 0.3 m
        # df_plot = df_fb.copy()
        # df_plot.loc[df_plot['freeboard'] >= 0.3, 'freeboard'] = 0.3
        
        # plt.figure(figsize=(10,10), dpi= 150)
        # # assign projection "NorthPolarStereo"
        # ax = plt.axes(projection=ccrs.NorthPolarStereo(true_scale_latitude=70))
        
        # cb2 = plt.scatter(df_leads['lons'], df_leads['lats'], c=df_leads['lead_ssh'], s=8, cmap='Blues', transform=ccrs.PlateCarree())
        # cb1 = plt.scatter(df_plot['lons'], df_plot['lats'], c=df_plot['freeboard'], s=8, cmap='Oranges', transform=ccrs.PlateCarree())
        
        # ax.coastlines()
        # plt.colorbar(cb1, label=var+' (m)', shrink=0.3, extend='both')
        # plt.colorbar(cb2, label='lead ssh (m)', shrink=0.3, extend='both')
        # plt.title('Original ATL10 freeboard estimates and lead tiepoints')
        
        # # Limit the map to 72 degrees latitude
        # ax.set_extent([-180, 180, 90, 75], ccrs.PlateCarree())
        # ax.add_artist(ScaleBar(1, box_color="silver", box_alpha=0.5, location='upper left'))
        # plt.tight_layout()
        
        # plt.savefig(OUT_DIR / 'original_fb_and_lead_points_on_map.png', dpi=150)
        
        # ---------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
       
        #%%  SSH EXTRAPOLATION TO COMPUTE FREEBOARD INDEPENDENT OF DISTANCE TO NEAREST LEAD
        
        # collect all re-processed freeboard dataFrames into a list, to later concatenate it into a global re-processed dataFrame that is saved to disk
        re_processed_df_fb = []
        
        # calculate new freeboard estimates for ALL beams (strong and weak ones!)
        for df_beam in tracks:
            
            df_fb = df_beam[0]
            df_leads = df_beam[1]
            
            # only proceed if dataFrames contain data
            if not df_fb.empty and not df_leads.empty:
            
                # step 1: kNN to find closest ssh tiepoints for every sample in df_fb per beam
                ##  USE EUCLIDEAN DISTANCE ON PROJECTED COORDS ##
                # project lat,lon coordinates to EPSG 3996 (unit = meters!)
                df_fb[['point_x','point_y']] = df_fb.apply(lambda row: lat_lon_to_projected(row['lats'], row['lons']), axis=1, result_type='expand')
                df_leads_global[['point_x','point_y']] = df_leads_global.apply(lambda row: lat_lon_to_projected(row['lats'], row['lons']), axis=1, result_type='expand')
                
                ## VERSION 1
                df_leads[['point_x','point_y']] = df_leads.apply(lambda row: lat_lon_to_projected(row['lats'], row['lons']), axis=1, result_type='expand')
                
                if len(df_leads) >= 10:
                    nn1 = NearestNeighbors(n_neighbors=10, metric="euclidean")
                    nn1.fit(df_leads[["point_x", "point_y"]])
                    # idx = indices of 10-NN (i.e. indices of 10 closest lead points) in df_leads for every freeboard segment in df_fb
                    # D = distances to 10 closest lead points in df_leads
                    D, idx = nn1.kneighbors(df_fb[['point_x', 'point_y']])
                
                elif len(df_leads)<10:
                    print('Less than 10 lead tie points, so using kNN<10 !')
                    nn1 = NearestNeighbors(n_neighbors=len(df_leads), metric="euclidean")
                    nn1.fit(df_leads[["point_x", "point_y"]])
                    # idx = indices of k-NN (i.e. indices of k closest lead points) in df_leads for every freeboard segment in df_fb
                    # D = distances to k closest lead points in df_leads
                    D, idx = nn1.kneighbors(df_fb[['point_x', 'point_y']])
                
                # step 2: interpolate lead ssh (based on 10 NNs) to obtain a value for every fb point
                lead_ssh = df_leads['lead_ssh'].to_numpy()
                # replace zeros by small value to avoid dividing by zero later
                D = np.where(D==0, 1e-16, D)
                # perform interpolation
                ssh_interp_1 = np.sum(lead_ssh[idx] * (1 / D**2), axis=1) / np.sum(1 / D**2, axis=1)
                
                ## VERSION 4: kNN-search and ssha interpolation PER BEAM but with max 100 km distance criterion
                # use same kNN search as version 1, but filter out lead tiepoints > 100 km away
                # mask out indices and distances of lead points that are >100 km away
                D_masked = np.where(D>100000, np.nan, D) # filter out distances > 100 km
                idx_masked = np.where(D>100000, np.nan, idx)
                
                # perform interpolation
                ssh_interp_4 = []
                
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
                    
                    # grab lead ssh points at these indices and do interpolation for this sample
                    ssh_interp_point = np.sum(lead_ssh[row_idx] * (1 / row_D**2)) / np.sum(1 / row_D**2)
                    ssh_interp_4.append(ssh_interp_point)
                
                del D
                del idx
                del D_masked
                del idx_masked
                
                ## VERSION 2 & 3
                # step 1: kNN to find closest ssh tiepoints for every sample in df_fb
                nn2 = NearestNeighbors(n_neighbors=10, metric="euclidean")
                nn2.fit(df_leads_global[["point_x", "point_y"]])
                D, idx = nn2.kneighbors(df_fb[['point_x', 'point_y']])
                
                # step 2: filter out lead tiepoints that are more than 100 km away (only in VERSION 3!)
                # mask out indices and distances of lead points that are too far away (set distance criterion!)
                D_masked = np.where(D>100000, np.nan, D) # filter out distances > 100 km
                idx_masked = np.where(D>100000, np.nan, idx)
                
                # step 3: interpolate lead ssh (based on k-NNs) to obtain a value for every fb point
                ## interpolation where k of k-NN is dependent on distance sample to lead tie point
                lead_ssh = df_leads_global['lead_ssh'].to_numpy()
                # replace zeros by small value to avoid dividing by zero later
                D = np.where(D==0, 1e-16, D)
                
                # VERSION 2
                ssh_interp_2 = np.sum(lead_ssh[idx] * (1 / D**2), axis=1) / np.sum(1 / D**2, axis=1)
                
                # VERSION 3
                ssh_interp_3 = []
                
                # iterative because interpolation requires to get rid of NaNs in the D and idx lists
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
                    
                    # grab lead ssh points at these indices and do interpolation for this sample
                    ssh_interp_point = np.sum(lead_ssh[row_idx] * (1 / row_D**2)) / np.sum(1 / row_D**2)
                    ssh_interp_3.append(ssh_interp_point)
                    
                # step 4: # smoothing of interpolated points
                # calculate window size based on average distance between height samples
                
                # calculate consecutive differences in along track distance
                diffs_x = df_fb['along_track_distance'].diff()[1:]
                # filter out data where along-track distance makes 'jump' (i.e. gaps in data) of > 0.1km
                diffs_x_filtered = diffs_x[diffs_x<0.1] * 1000 # convert to meters
                # calculate median distance between samples based on this selection
                median_dist_x = np.median(diffs_x_filtered)
                std_dist_x = np.std(diffs_x_filtered)
                
                # step 5: define number of samples to use in running average window (we want window of ~2.5 km)
                nr_points = np.round(2500/median_dist_x)
                
                # with open(output_terminal_filepath, 'a') as file:
                #     print(f'\n Median distance between samples: {median_dist_x} m', file=file)
                #     print(f' STD of this distance: {std_dist_x} m', file=file)
                #     print(f' Averaging over {nr_points} points', file=file)
                
                # frac: Between 0 and 1. The fraction of the data used when estimating each y-value.
                if len(ssh_interp_1) > nr_points:
                    frac = nr_points/len(ssh_interp_1)
                else:
                    # if there are less than 200 points to do the smoothing, use all those points
                    frac = 1
                
                # LOWESS = Locally Weighted Scatterplot Smoothing
                lowess_1 = sm.nonparametric.lowess(ssh_interp_1, np.arange(len(ssh_interp_1)), frac=frac)
                ssh_interp_smooth_1 = lowess_1[:, 1]
                df_fb['ssh_interp_v1'] = ssh_interp_smooth_1
                
                lowess_2 = sm.nonparametric.lowess(ssh_interp_2, np.arange(len(ssh_interp_2)), frac=frac)
                ssh_interp_smooth_2 = lowess_2[:, 1]
                df_fb['ssh_interp_v2'] = ssh_interp_smooth_2
                
                lowess_3 = sm.nonparametric.lowess(ssh_interp_3, np.arange(len(ssh_interp_3)), frac=frac, missing='none')
                ssh_interp_smooth_3 = lowess_3[:, 1]
                
                # check if there is no data in the ssh_interp version 3 (due to all lead tiepoints being too far away)
                if not ssh_interp_smooth_3.size==0:
                    df_fb['ssh_interp_v3'] = ssh_interp_smooth_3
            
                lowess_4 = sm.nonparametric.lowess(ssh_interp_4, np.arange(len(ssh_interp_4)), frac=frac, missing='none')
                ssh_interp_smooth_4 = lowess_4[:, 1]
                
                # check if there is no data in the ssh_interp version 4 (due to all lead tiepoints being too far away)
                if not ssh_interp_smooth_4.size==0:
                    df_fb['ssh_interp_v4'] = ssh_interp_smooth_4
                    
                # plt.figure()
                # plt.plot(ssh_interp_3, linewidth=1, label='k<=10 across ALL beams')
                # plt.plot(ssh_interp_4, linewidth=1, label='k<=10 PER beam')
                # plt.plot(ssh_interp_smooth_3, linewidth=1, linestyle='--', color='midnightblue', label='v3 smoothed')
                # plt.plot(ssh_interp_smooth_4, linewidth=1,  linestyle='--', color='red',label='v4 smoothed')
                # plt.xlabel('height segment index')
                # plt.ylabel('ssha [m]')
                # plt.legend()
                # plt.title('Extrapolated ssha: version 3 vs 4')
                
                # plt.tight_layout()
                # fig_path = OUT_DIR / 'extrapolated_ssh_method_comparison.png'
                # plt.savefig(fig_path, dpi=150)
                
                # calculate new freeboard, using extrapolated ssh
                
                # use interpolated ssha to fill gaps in original ATL10 freeboards
                
                # keep original ATL10 freeboards where freeboard values < 10 m
                condition = df_fb['freeboard'] < 10 
                
                df_fb['freeboard_v1'] = df_fb['freeboard'].where(condition, df_fb['height'] - df_fb['ssh_interp_v1'], axis=0)
                df_fb['freeboard_v2'] = df_fb['freeboard'].where(condition, df_fb['height'] - df_fb['ssh_interp_v2'], axis=0)
                
                if not ssh_interp_smooth_3.size==0:
                    df_fb['freeboard_v3'] = df_fb['freeboard'].where(condition, df_fb['height'] - df_fb['ssh_interp_v3'], axis=0)
                
                # if no lead tiepoints were found across all 6 beams within 100 km distance, set freeboard_v3 to NaN
                if ssh_interp_smooth_3.size==0:
                    df_fb['freeboard_v3'] = np.nan
    
                if not ssh_interp_smooth_4.size==0:
                    df_fb['freeboard_v4'] = df_fb['freeboard'].where(condition, df_fb['height'] - df_fb['ssh_interp_v4'], axis=0)
                    
                # if no lead tiepoints were found along beam within 100 km distance, set freeboard_v4 to NaN
                if ssh_interp_smooth_4.size==0:
                    df_fb['freeboard_v4'] = np.nan
                
                ## PLOT FREEBOARD V0 vs V4 AND INTERPOLATED SSHA ## 
                beam_nr = df_fb['beamNum'][0]
                
                # calculate rolling average of freeboard V4 for plotting
                rol_fb = df_fb['freeboard_v4'].dropna().rolling(window=101, center=True, closed='both').mean().reindex(df_fb.index, method='pad')

                plt.figure(figsize=(10,5))
                plt.scatter(df_fb['along_track_distance'], df_fb['height'], label='Heights ATL07', c='darkred', s=1, alpha=0.3)
                plt.scatter(df_fb['along_track_distance'].where(df_fb['freeboard'] < 10), df_fb['freeboard'].where(df_fb['freeboard'] < 10), label='Freeboard ATL10', c='midnightblue', s=1, alpha=0.3)
                plt.scatter(df_fb['along_track_distance'].where(df_fb['freeboard'] > 10), df_fb['freeboard_v4'].where(df_fb['freeboard'] > 10), label='Freeboard extrapol.', c='darkorange', s=1, alpha=0.5)
                plt.plot(df_fb['along_track_distance'], rol_fb, c='black', label='Freeboard V4', alpha=0.8)
                plt.plot(df_fb['along_track_distance'], df_fb['ssh_interp_v4'], c='slategrey', label='ssha extrapolated', linewidth=2, linestyle='--')
                
                plt.xlabel('Along track distance')
                plt.ylabel('Height [m]')
                plt.title(f'Extrapolated ssha and freeboard V4 beam {beam_nr}')
                plt.legend(loc='upper right', markerscale=4, fontsize=8)
                
                plt.tight_layout()
                fig_path = OUT_DIR / f'Freeboard_V4_beam_{beam_nr}.png'
                plt.savefig(fig_path, dpi=150)
                
                # save re-processed dataFrame in list
                re_processed_df_fb.append(df_fb)
        
        # concatenate re-processed freeboard dataframes into a global one
        if re_processed_df_fb:
            df_fb_global = pd.concat(re_processed_df_fb)
        
            # save new freeboard estimates to disk (HDF5 format)
            df_fb_global.to_hdf(df_outpath, key='df_fb', mode='w')

            # ---------------------------------------------------------- #
             
            colors = ['midnightblue', 'lightskyblue', 'indigo', 'mediumorchid','darkolivegreen', 'yellowgreen']
            
            fig,ax=plt.subplots(3,1,figsize=(10,5), sharex=True, sharey=True)
            
            axes = [0,0,1,1,2,2]
            
            for beam, ax_nr in zip(range(1,7), axes):
                
                rol_fb = df_fb_global['freeboard_v4'].where(df_fb_global['beamNum']==beam).rolling(window=101, center=True, closed='both').mean()
                ax[ax_nr].plot(df_fb_global['along_track_distance'], rol_fb, color=colors[beam-1], label=f'beam {beam}')
                ax[ax_nr].set_ylabel('Along-track distance [km]')
                ax[ax_nr].set_ylabel('Freeboard [m]')
                ax[ax_nr].legend(loc='lower right')
                
            plt.suptitle('Re-processed freeboards along-track per beam')
            plt.tight_layout()
            fig_path = OUT_DIR / 'Freeboard_V4_all_beams.png'
            plt.savefig(fig_path, dpi=150)
            
#%%     PLOT 
        # # make scatter plot showing the values of the lead ssh points used in extrapolating ssh (for one point, taken at index 5)
        # y_ax1 = lead_ssh[idx[0,:]]
        # y_ax2 = D[0,:] / 1000
        
        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()

        # ax1.scatter(np.arange(1,11), y_ax1, color='royalblue', s=14)
        # ax2.scatter(np.arange(1,11), y_ax2, color='midnightblue', s=14, marker='_')
        
        # ax1.set_xlabel('k-NN')
        # ax1.set_ylabel('Lead height (ssh)', color='royalblue')
        # ax2.set_ylabel('Distance [km]', color='midnightblue')
        # plt.title('ssh tiepoints returned by kNN (k=10) search')
        
        # plt.tight_layout()
        # fig_path = OUT_DIR / 'ssh_tiepoints_returned_by_kNN_search.png'
        # plt.savefig(fig_path, dpi=150)
        
        # min_dist = np.min(D) / 1000
        # max_dist = np.max(D) / 1000
        
        # with open(output_terminal_filepath, 'a') as file:
        #     print(f'\n 10 closest SSH points are {min_dist} to {max_dist} km away', file=file)
            #print(f'10 closest SSH points are {min_dist} to {max_dist} km away')
        
        #%% PLOT ATL07 HEIGHT AND FREEBOARD (ORIGINAL + NEW VERSION) ALONG TRACK
        # fig,ax=plt.subplots(4,1,figsize=(12,12))
        
        # ax[0].scatter(df_fb['along_track_distance'], df_fb['height'],c=df_fb['height'], s=2, cmap='hot')
        # ax[0].set_ylim(0,2)
        # ax[0].set_ylabel('Heights ATL07 (m)',fontsize=12)
        # ax[0].set_xticklabels([])
        
        # # mask out invalid freeboard values for plotting (to make colorscale match with other plots)
        # df_fb_masked = np.where(df_fb['freeboard']>20, np.nan, df_fb['freeboard'])
        
        # ax[1].scatter(df_fb['along_track_distance'], df_fb_masked, c=df_fb['freeboard'], s=2, cmap='hot')
        # ax[1].set_ylabel('Freeboard (m)',fontsize=12)
        # ax[1].set_ylim(0,2)
        # ax[1].set_xticklabels([])
        # mean=str(np.round(np.mean(df_fb['freeboard']), 2))
        # ax[1].annotate('Mean: '+mean, xy=(0.01, 0.9),xycoords='axes fraction', fontsize=12)
        
        # ax[2].scatter(df_fb['along_track_distance'], df_fb['fb_v2'], c=df_fb['fb_v2'], s=2, cmap='hot')
        # ax[2].set_ylabel('Freeboard v2 (m)',fontsize=12)
        # ax[2].set_ylim(0,2)
        # ax[2].set_xticklabels([])
        # mean=str(np.round(np.mean(df_fb['fb_v2']), 2))
        # ax[2].annotate('Mean: '+mean, xy=(0.01, 0.9),xycoords='axes fraction', fontsize=12)
        
        # ax[3].scatter(df_fb['along_track_distance'], df_fb['fb_v3'], c=df_fb['fb_v3'], s=2, cmap='hot')
        # ax[3].set_ylabel('Freeboard v3 (m)',fontsize=12)
        # ax[3].set_ylim(0,2)
        # ax[3].set_xticklabels([])
        # mean=str(np.round(np.mean(df_fb['fb_v3']), 2))
        # ax[3].annotate('Mean: '+mean, xy=(0.01, 0.9),xycoords='axes fraction', fontsize=12)
        
        # # ax[3].scatter(df_fb['along_track_distance'], df_fb['ssh_flag'], c='k', s=2)
        # # ax[3].grid()
        # # ax[3].set_ylim(-1,3)
        # # ax[3].set_yticks(np.arange(0,3,1))
        # # ax[3].set_yticklabels(['sea ice', 'potential \n sea surface', 'sea surface'])
        # # ax[3].set_ylabel('ATL10 \nssh flag',fontsize=12)
        # # ax[3].set_xlabel('Along track distance (km)',fontsize=12)
        
        # for a in np.arange(0,4):
        #     ax[a].set_xlim(df_fb['along_track_distance'].iloc[0],df_fb['along_track_distance'].iloc[-1])
        #     width=1
        #     for index, row in df_fb.iterrows():
        #         x0 = row['along_track_distance'] 
        #         if row['ssh_flag'] > 0.5:
        #             ax[a].axvline(x0,c='y',alpha=.3, linewidth=width)
                    
        # plt.suptitle(f'IS-2 freeboard estimates beam {beamNum}')
        # plt.tight_layout()
        
        # fig_path = OUT_DIR / f'IS-2_freeboard_estimates_{Path(filepath).stem}.png'
        # plt.savefig(fig_path, dpi=300)
        
        #%% PLOT NEW TRACK ON MAP
        
        # plt.figure(figsize=(10,10), dpi= 150)
        # # assign projection "NorthPolarStereo"
        # ax = plt.axes(projection=ccrs.NorthPolarStereo(true_scale_latitude=70))
        
        # cb2 = plt.scatter(df_fb['lons'], df_fb['lats'], c=df_fb['ssh_interp'], s=8, cmap='Blues', transform=ccrs.PlateCarree())
        # cb1 = plt.scatter(df_fb['lons'], df_fb['lats'], c=df_fb['fb_v2'], s=8, cmap='Oranges', transform=ccrs.PlateCarree())
        
        # ax.coastlines()
        # plt.colorbar(cb1, label='Freeboard v2 (m)', shrink=0.4, extend='both')
        # plt.colorbar(cb2, label='Interpolated lead ssh (m)', shrink=0.4, extend='both')
        # plt.title('Freeboard estimates independent of distance to closest lead')
        
        # # Limit the map to 72 degrees latitude
        # ax.set_extent([-30, 20, 82, 77], ccrs.PlateCarree())
        # ax.add_artist(ScaleBar(1, box_color="silver", box_alpha=0.9, location='upper left'))
        # plt.tight_layout()
        
        # plt.savefig(OUT_DIR / 'fb_v2_on_map.png', dpi=150)
        
        
