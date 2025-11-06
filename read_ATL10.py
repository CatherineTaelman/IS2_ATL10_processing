#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Read data from IceSat-2 ATL10 h5 file into Pandas dataframes: 
    - one dataframe for freeboard data
    - one for lead data

@author: cat
"""

import h5py
from datetime import datetime, timedelta
import pandas as pd

def create_dataframes(f, beam_str, beam_num, return_fb, return_leads, spatial_extent):
    """

    Parameters
    ----------
    f : h5py object
        data from h5 file
    beam_str : string
        beam string ([choose from: '/gt1l', '/gt1r', '/gt2l', '/gt2r', '/gt3l', '/gt3r'])
    beam_num : int
        beam number (1,3,5 are strong beams, 2,4,6 are weak beams)
    return_fb: Boolean 
        if True, return the freeboard info as a DataFrame
    return_leads : Boolean 
        if True, returns lead info as a DataFrame
    spatial_extent : list
        limit dataframe to spatial extent defined as bounding box coordinates: [lon_min, lat_min, lon_max, lat_max]
        
    Returns
    -------
    df_fb : Pandas dataframe
        dataframe containing relevant freeboard variables
    df_leads : Pandas dataframe
        dataframe containing relevant lead variables
    """
    
    if return_fb:
        
        # time elapsed since 2018-01-01 (in seconds)
        delta_time = f[beam_str]['freeboard_segment']['delta_time'][:]
        
        # lat and lon
        lats = f[beam_str]['freeboard_segment']['latitude'][:]
        lons = f[beam_str]['freeboard_segment']['longitude'][:]
        
        # freeboard 
        fb = f[beam_str]['freeboard_segment']['beam_fb_height'][:]
        fb_std = f[beam_str]['freeboard_segment']['beam_fb_unc'][:]
        
        # freeboard confidence and freeboard quality flag
        fb_conf = f[beam_str]['freeboard_segment']['beam_fb_confidence'][:]
        fb_qf = f[beam_str]['freeboard_segment']['beam_fb_quality_flag'][:]
        
        # relative along track distance (remove the first point so it's distance relative to the start of the granule)
        alg_track_dist = (f[beam_str]['freeboard_segment/seg_dist_x'][:] - f[beam_str]['freeboard_segment/seg_dist_x'][0]) / 1000 # (in km)
        
        # along track distance from start of granule
        dist_x = f[beam_str]['freeboard_segment/seg_dist_x'][:] / 1000 # in km
       
        # ATL07 heights
        h = f[beam_str]['freeboard_segment']['heights']['height_segment_height'][:]
        
        # Width of Gaussian fit (from ATL07 heights)
        h_w_gauss = f[beam_str]['freeboard_segment']['heights']['height_segment_w_gaussian'][:]
        
        # segment length
        seg_len = f[beam_str]['freeboard_segment']['heights']['height_segment_length_seg'][:]
        
        # ID of each height segment (in 10 km swaths), from ATL07
        height_segment_id = f[beam_str]['freeboard_segment']['height_segment_id'][:]
        
        # ssh flag (based on sea surface type classification) --> ssh_flag==2 means it's a sea surface tiepoint
        # 0 to 2 'Identifies the height segments that are candidates for use as sea surface reference in freeboard calculations in ATL10. The flags are set as follows: 0 = sea ice; 1 = potential reference sea surface height; 2 = used in calculating reference sea surface height'
        h_ssh_flag = f[beam_str]['freeboard_segment']['heights']['height_segment_ssh_flag'][:]
        
        # surface type classification (# 0 = Cloud covered)
        surface_type = f[beam_str]['freeboard_segment']['heights']['height_segment_type'][:]
        
        # sea ice concentration (AMSR-2)
        sic = f[beam_str]['freeboard_segment']['heights']['ice_conc_amsr2'][:]
        
        # photon rate count
        ph_rate = f[beam_str]['freeboard_segment']['heights']['photon_rate'][:]
        
        # # Estimated background rate from sun angle, reflectance, surface slope
        # background_rate = f[beam_str]['sea_ice_segments/stats/backgr_calc'][:]
        
        # clouds flag, cloud probability p=(1-asr/t)*100)
        # 0 to 5 'clear_with_high_confidence clear_with_medium_confidence clear_with_low_confidence cloudy_with_low_confidence cloudy_with_medium_confidence cloudy_with_high_confidence'
        cloud_asr = f[beam_str]['freeboard_segment']['heights']['cloud_flag_asr'][:]

        # store into pandas dataframe
        df_fb = pd.DataFrame({
                           'freeboard':fb, 
                           'freeboard_std': fb_std,
                           'freeboard_conf': fb_conf,
                           'freeboard_qf': fb_qf,
                           'height': h,
                           'h_w_gauss': h_w_gauss,
                           'along_track_distance': alg_track_dist,
                           'dist_x': dist_x,
                           'seg_length': seg_len, 
                           'height_segment_id': height_segment_id,
                           'ssh_flag': h_ssh_flag, 
                           'surface_type': surface_type,
                           'sic': sic,
                           'photon_rate': ph_rate,
                           'cloud_flag_asr': cloud_asr,
                           'lons': lons, 
                           'lats': lats,
                           'timestamps': delta_time
                           })
        
        # limit to spatial extent of bounding box, if given
        if spatial_extent:
            df_fb = df_fb[(df_fb['lats'] > spatial_extent[1]) & (df_fb['lats'] < spatial_extent[3]) & (df_fb['lons'] > spatial_extent[0]) & (df_fb['lons'] < spatial_extent[2])].reset_index(drop=True)
        
        # convert elapsed time to actual timestamp
        df_fb['timestamps'] = df_fb['timestamps'].apply(lambda x: datetime(2018,1,1) + timedelta(seconds=x))
        
        # Could add in a filter based on the confidence and/or quality flag
        
        # Reset row indexing
        # df_fb=df_fb.reset_index(drop=True)
        
        df_fb['beamStr'] = beam_str
        df_fb['beamNum'] = int(beam_num)
        
    if return_leads:
        
        # lead tiepoints (note: these are ALL tiepoints along entire granule)
        lead_lats = f[beam_str]['leads']['latitude'][:]
        lead_lons = f[beam_str]['leads']['longitude'][:]
        lead_ssha = f[beam_str]['leads']['lead_height'][:]
        
        # along-track distance from the start of granule
        lead_dist_x = f[beam_str]['leads']['lead_dist_x'][:] / 1000 # in km
    
        df_leads = pd.DataFrame({'lats': lead_lats,
                                 'lons': lead_lons,
                                 'lead_ssha': lead_ssha,
                                 'lead_dist_x' : lead_dist_x
                                })
        
        # limit to spatial extent of bounding box, if given
        if spatial_extent:
            df_fb = df_fb[(df_fb['lats'] > spatial_extent[1]) & (df_fb['lats'] < spatial_extent[3]) & (df_fb['lons'] > spatial_extent[0]) & (df_fb['lons'] < spatial_extent[2])].reset_index(drop=True)
        
        df_leads['beamNum'] = int(beam_num)
        
    if not return_fb:
        df_fb = pd.DataFrame()
    if not return_leads:
        df_leads = pd.DataFrame()
        
    return df_fb, df_leads
    
def load_data_IS2_beam(filepath, beam_num, return_fb = True, return_leads = True, spatial_extent=None):
    """

    Parameters
    ----------
    filepath : string
        path to IS-2 data file (h5 format)
    beam_num : int
        beam numbers 1 3,5 are strong beams, the string changes based on IS2 orientation
    return_fb: Boolean (default=True)
        if True, return the freeboard info as a DataFrame
    return_leads : Boolean (default=True)
        if True, returns lead info as a DataFrame
    spatial_extent : list (default=None)
        limit dataframe to spatial extent defined as bounding box coordinates: [lon_min, lat_min, lon_max, lat_max]
        
    Returns
    -------
    df_fb : Pandas dataframe
        dataframe containing relevant freeboard variables
    df_leads : Pandas dataframe
        dataframe containing relevant lead variables

    """
    if not beam_num in [1,2,3,4,5,6]:
        print(f'Beam nr: {beam_num} not valid for IceSat-2. Choose from 1-6.')
        return
        
    #print(f'Beam nr: {beam_num}')
    
    # try reading in h5 file
    try:
        f = h5py.File(filepath, 'r')
        #print('got file')
    except FileNotFoundError:
        print(f'File not found: {filepath}')
        print('---------------------------------------------------------')
        return
    
    # grab spacecraft orientation to know which beam strings to use
    orientation_flag=f['orbit_info']['sc_orient'][0]

    if (orientation_flag==0):
        # backward orientation
        beamStrs=['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
                
    elif (orientation_flag==1):
        # forward orientation
        beamStrs=['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
        
    elif (orientation_flag==2):
        print('Transitioning, do not use for science!')


    beam_str=beamStrs[beam_num-1]

    df_freeboard, df_leads = create_dataframes(f, beam_str, beam_num, return_fb, return_leads, spatial_extent)
    
    return df_freeboard, df_leads


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    #beamNum (int): beam numbers 1 3,5 are always strong beams, the string changes based on IS2 orientation)
    strongBeams = [1, 3, 5]

    # path to ATL10 file
    filepath = "data/original_ATL10/processed_ATL10-01_20220101064950_01531401_006_01.h5"

    # collect data of different beams into list
    track_fb = []
    track_leads = []
    
    for beam_num in strongBeams:
        try:
            df_freeboard, df_leads = load_data_IS2_beam(filepath, beam_num)
            track_fb.append(df_freeboard)
            track_leads.append(df_leads)
            
        except KeyError:
            print(f"Key error for beam {beam_num}")
            continue

    # concatenate dataframes of seperate beams into global ones (one for freeboards, one for leads)
    track_fb_df = pd.concat(track_fb)
    track_leads_df = pd.concat(track_leads)
