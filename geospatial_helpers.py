#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions to work with geospatial data

@author: Catherine Taelman
"""

from osgeo import ogr
from osgeo import osr 
from datetime import datetime as dt 
from datetime import timedelta
from shapely.ops import transform
import pyproj
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import re

# --------------------------------------------------------------------------- # 
# --------------------------------------------------------------------------- # 

def get_bands(S1_path):
    S1_raster = rasterio.open(S1_path)
    
    sigma0_HH_dB = S1_raster.read()[0, :, :]
    sigma0_HV_dB = S1_raster.read()[1, :, :]
    
    return sigma0_HH_dB, sigma0_HV_dB, S1_raster

# --------------------------------------------------------------------------- # 

def get_mask(S1_mask_path):
    S1_mask_raster = rasterio.open(S1_mask_path)

    S1_mask = S1_mask_raster.read()[0, :, :]
    
    return S1_mask

# --------------------------------------------------------------------------- # 

def make_falseColour_RGB(sigma0_HH_dB, sigma0_HV_dB):
    # set 0 values (outside of data footprint) to nan
    sigma0_HH_dB[sigma0_HH_dB == 0] = np.nan
    sigma0_HV_dB[sigma0_HV_dB == 0] = np.nan

    # find min/max percentiles for good visualization --> dynamic range
    vmin_HH = np.nanpercentile(sigma0_HH_dB, 3)
    vmax_HH = np.nanpercentile(sigma0_HH_dB, 97)
    vmin_HV = np.nanpercentile(sigma0_HV_dB, 3)
    vmax_HV = np.nanpercentile(sigma0_HV_dB, 97)
    
    # or: use fixed range for HH and HV channels
    vmin_HH = -30
    vmax_HH = 0
    vmin_HV = -35
    vmax_HV = -5

    # create false_color RGB
    new_min = 0
    new_max = 1

    # linear map from sigma0 in dB to new_min and new_max
    HH_scaled = (sigma0_HH_dB - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
    HV_scaled = (sigma0_HV_dB - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min

    # clip values
    HH_scaled = np.clip(HH_scaled, new_min, new_max)
    HV_scaled = np.clip(HV_scaled, new_min, new_max)

    # stack scaled channels to fals-color RGB
    S1_RGB = np.stack((HV_scaled, HH_scaled, HH_scaled), 0)
    return S1_RGB
         
# --------------------------------------------------------------------------- # 

# project (lat,lon) coordinate to Polar Stereographic point (values expressed in meters)
def lat_lon_to_projected(lat, lon, inputEPSG=4326, outputEPSG=3996):

    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lat, lon)
    
    # create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)
    
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)
    
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    
    # transform point
    point.Transform(coordTransform)
    
    # return point in outputEPSG. Note: point_x matches with lon, and point_y matches with lat!
    return point.GetX(), point.GetY()

# --------------------------------------------------------------------------- # 
# project shapely Polygon given in lat/lon (EPSG 4326) to polar stereographic (EPSG 3996)

def polygon_from_4326_to_3996(shapely_polygon_4326):
    
    # define EPSG codes for projection
    wgs84 = pyproj.CRS('EPSG:4326')
    polarstereo = pyproj.CRS('EPSG:3996')
    
    # define transform function between both projections
    project = pyproj.Transformer.from_crs(wgs84, polarstereo, always_xy=True).transform
    
    # warp lat/lon polygon to EPSG 3996
    polygon_3996 = transform(project, shapely_polygon_4326)
    return polygon_3996

# --------------------------------------------------------------------------- # 
def convert_datenumber_to_date(matlab_datenum):
    """
Convert a Matlab datenumber to a Python datetime object. Note: Matlab datenumbers are 366 days behind Python ones!

    Parameters
    ----------
    matlab_datenum : str
        Matlab datenumber

    Returns
    -------
    date : datetime object
        python datetime
    """
    date_with_offset = dt.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1)
    # matlab datenumbers are 366 days behind the python ones --> correct for this offset!
    date = date_with_offset - timedelta(days=366)
    return date

# --------------------------------------------------------------------------- # 

def find_points_within_polygon(df, polygon_path):
    """
    Return the points of a dataframe that are spatially WITHIN a given polygon.
    Uses geopandas.sjoin()

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe where every row is a sample that has a lat/lon position
    polygon_path : path to geospatial file containing polygon geometry (e.g. shapefile)
        Polygon we want to use to filter the dataframe

    Returns
    -------
    points_within_polygon : pandas Dataframe
        Subset of the original Dataframe that contains only the points that lie WIHTIN the polygon

    """
    # load polygon into geodataframe (note: polygon defined with lat/lon!)
    polygon = gpd.read_file(polygon_path)
    
    # convert input df to geodataframe with crs 4326
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lons, df.lats), crs=4326)
    
    # find points that are WITHIN a polygon
    points_within_polygon = gpd.sjoin(points, polygon, how='inner', predicate = 'within')
    return points_within_polygon


def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    if direction == 'W' or direction == 'S':
        dd *= -1
    return dd;

def dd2dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]

# parse degree-minutes-seconds coordinate into decimal degrees
# Example usage: decimal_coord = parse_dms('''78°55'44.324"N''' )
def parse_dms(dms):
    '''

    Parameters
    ----------
    dms : TYPE
        DESCRIPTION.

    Returns
    -------
    decimal_coord : TYPE
        DESCRIPTION.

    '''
    deg, mins, secs, direction = re.split('[°\'"]', dms)
    decimal_coord = dms2dd(deg, mins, secs, direction)
    return (decimal_coord)

# parse degree-minutes coordinate into decimal degrees
# example usage: decimal_coord = parse_dm('''78°55.9'N''' )
def parse_dm(dm):
    deg, mins, direction = re.split('[°\']', dm)
    secs = '0'
    decimal_coord = dms2dd(deg, mins, secs, direction)
    return (decimal_coord)



## CALCULATE HAVERSINE DISTANCE ON LAT/LON COORDS ##
# # convert lat/lon coordinates to radians (needed as input for Haversine distance metric)
# df_leads['lats'] = np.deg2rad(df_leads['lats'])
# df_leads['lons'] = np.deg2rad(df_leads['lons'])
# df_fb['lats'] = np.deg2rad(df_fb['lats'])
# df_fb['lons'] = np.deg2rad(df_fb['lons'])

# # find 10 Nearest Neighbours lead points for every freeboard point
# nn = NearestNeighbors(n_neighbors=10, metric="haversine")
# nn.fit(df_leads[['lats', 'lons']])
# D, idx = nn.kneighbors(df_fb[['lats', 'lons']])

# # convert radial distances to meters
# earth_radius = 6371000
# D = D*earth_radius
