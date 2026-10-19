# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Calculate Toronto LTS from scratch
#
# This script calculates Level of Traffic Stress using city and government datasets. 
#
# Input data should be aligned to calculation timeframe if possible. Where it is possible to retroactively get the data at a specific time point, the "time point retroactivity" column includes a "yes". E.g. the cycling network file includes the dates that infrastructure was installed and/or updated.
#
# In general geojson format is preferred, since these are plain-text readable and tend to be smaller than shapefiles, but any geo file compatible with geopandas will work.
#
# | Data | Source | Time point retroactivity | Notes|
# | --- | --- | --- | --- |
# | Centreline network | [Toronto Open Data](https://open.toronto.ca/dataset/toronto-centreline-tcl/) | No | File is updated gradually over time, download at desired time point |
# | Cycling network file | Received from City | Yes | This script uses a version aligned with the centreline network |
# | Speed limit | [Government of Ontario](https://geohub.lio.gov.on.ca/datasets/mnrf::ontario-road-network-orn-road-net-element/about)  | No | Speed from GTAModel is used where available |
# | Number of lanes | Received from City | No | Provided privately by Charmalee |
# | GTAModel speed and volume | GTAModel contacts (via Bo) | Yes?| Optional, this info is in the 2021 LTS file shared by Bo until updated |
# | Street parking | Toronto Parking Authority (via Bo) | Unknown | This info is in the 2021 LTS file shared by Bo until updated |
# | Previous LTS version(s) | Bo | NA | Optional, used for testing and comparison |
#

# To do:
# - [x] Substitute Ontario lane info with City lane info
# - [ ] Bring in latest GTAModel speed data from James, verify it matches Bo's data (may not be identical, I'm not sure if it was generated in the same way)

# ## Source data

# downloaded from Open Data, Feb 24, 2025
centreline_filename = "data/Centreline_Version2-4326/Centreline - Version 2 - 4326.shp" 
# received from Katie Wittmann, City of Toronto
cycling_network_filename = "data/Existing_Cycling_Network_20250805/Existing_Cycling_Network_20250805.shp" 
speed_limit_filename = "data/ORNELEM/LIO-2025-10-03/ORN_SPEED_LIMIT.csv"
lanes_filename = "data/ORNELEM/LIO-2025-10-03/ORN_NUMBER_OF_LANES.csv" # wildly inaccurate!
city_lanes_filename = "data/LanesWidth_Dec112025/LanesWidth_Dec112025.shp"
road_class_filename = "data/ORNELEM/LIO-2025-10-03/ORN_ROAD_CLASS.csv"
lts_2021_filename = "data/july2021/LTS_July2021.shp" # created by Bo
#lts_2023_filename = 
lts_2024_filename = "data/LTS_network_Jan2024/Jan2024_lts.shp" # updated by Bo, by hand from previous versions

import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx
from matplotlib import pyplot as plt
from tqdm import tqdm
from pyogrio import read_dataframe # faster than geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ## Process centreline file

centreline = read_dataframe(centreline_filename)

# use UTM crs to get accurate distances between things
utm_crs = centreline.estimate_utm_crs()
centreline = centreline.to_crs(utm_crs)

# rename columns using field info - optional
centreline_fields = pd.read_csv("data/Centreline_Version2-4326/Centreline - Version 2 fields.csv")
centreline_cols = dict(zip(centreline_fields['field'], centreline_fields['name']))
centreline = centreline.rename(columns = centreline_cols)
centreline.columns = centreline.columns.str.lower() # lowercase column names

# get list of feature types
centreline[['feature_code','feature_code_desc']].value_counts()

# remove things that can't be biked on
centreline = centreline[~centreline['feature_code_desc'].isin(['Expressway', 'Expressway Ramp', 'River', 'Hydro Line', 'Major Railway',
                                  'Major Shoreline', 'Creek/Tributary', 'Minor Railway',  'Geostatistical line', 
                                                'Busway', 'Collector Ramp',  'Ferry Route', 'Minor Shoreline (Land locked)'])]

shape_orig = centreline.shape[0]

# ## Speed limit & number of lanes

# trim to just study area
try:
    orn_toronto = read_dataframe("data/ORNELEM/orn_toronto.geojson")
except: # select just Toronto area
    orn_geog = read_dataframe("data/ORNELEM/LIO-2025-10-03/ORN_ROAD_NET_ELEMENT.shp")
    toronto_boundary = centreline.union_all().convex_hull
    orn_toronto = orn_geog[orn_geog.to_crs(utm_crs).intersects(toronto_boundary)]
    orn_toronto.to_file("data/ORNELEM/orn_toronto.geojson")

speed_limits = read_dataframe(speed_limit_filename)
number_of_lanes_orn = read_dataframe(lanes_filename)
number_of_lanes = read_dataframe(city_lanes_filename)
road_class = read_dataframe(road_class_filename)

# +
# convert to UTM
orn_toronto = orn_toronto.to_crs(utm_crs)

# change to integer and filter to toronto
speed_limits['ORN_ROAD_NET_ELEMENT_ID'] = speed_limits['ORN_ROAD_NET_ELEMENT_ID'].astype('int32')
speed_limits = speed_limits[speed_limits['ORN_ROAD_NET_ELEMENT_ID'].isin(orn_toronto['OGF_ID'])]
number_of_lanes_orn['ORN_ROAD_NET_ELEMENT_ID'] = number_of_lanes_orn['ORN_ROAD_NET_ELEMENT_ID'].astype('int32')
number_of_lanes_orn = number_of_lanes_orn[number_of_lanes_orn['ORN_ROAD_NET_ELEMENT_ID'].isin(orn_toronto['OGF_ID'])]
road_class['ORN_ROAD_NET_ELEMENT_ID'] = road_class['ORN_ROAD_NET_ELEMENT_ID'].astype('int32')
road_class = road_class[road_class['ORN_ROAD_NET_ELEMENT_ID'].isin(orn_toronto['OGF_ID'])]
# -

road_class.columns

# +
# merge with geometry file
orn_toronto = orn_toronto.merge(number_of_lanes_orn[['ORN_ROAD_NET_ELEMENT_ID',
       'NUMBER_OF_LANES', 'EFFECTIVE_DATETIME',
       'SYSTEM_DATETIME']], left_on = 'OGF_ID', right_on = 'ORN_ROAD_NET_ELEMENT_ID', how = 'left')

orn_toronto = orn_toronto.merge(speed_limits[['ORN_ROAD_NET_ELEMENT_ID',
       'SPEED_LIMIT', 'EFFECTIVE_DATETIME',
       'SYSTEM_DATETIME']], on = 'ORN_ROAD_NET_ELEMENT_ID', how = 'left', suffixes=['_lanes', '_speed'])

orn_toronto = orn_toronto.merge(road_class[['ORN_ROAD_NET_ELEMENT_ID',
       'ROAD_CLASS', 'AGENCY_NAME']], on = 'ORN_ROAD_NET_ELEMENT_ID', how = 'left')

# remove edges with no speed limit info
orn_toronto = orn_toronto[~orn_toronto['SPEED_LIMIT'].isna()]

orn_toronto['SPEED_LIMIT'] = orn_toronto['SPEED_LIMIT'].astype('int')
orn_toronto['NUMBER_OF_LANES'] = orn_toronto['NUMBER_OF_LANES'].astype('int')
# -

orn_toronto['ROAD_CLASS'].value_counts()

orn_toronto[orn_toronto['ROAD_CLASS'].isin(['Freeway', 'Expressway / Highway'])].plot()

# look at which roads have speed limits over 80
orn_toronto[orn_toronto['SPEED_LIMIT'] >= 80].plot(column = 'SPEED_LIMIT')

# remove highways
orn_toronto = orn_toronto[~orn_toronto['ROAD_CLASS'].isin(['Freeway', 'Expressway / Highway'])]


# ### Calculate intersection between ORN and centreline file
#
# This is a little finicky. They overlap well, but not exactly.
#
# Some centreline segments are so short that they are difficult to match. 
#
# Procedure:
# - skip very short centreline segments (less than 2x buffer size) and assign later based on neighbours 
# - skip centreline segments that are 'Trail' or 'Walkway' (no speed limit)
# - for each centreline segment, buffer by ~10 metres
# - get intersecting ORN segments
# - if no intersecting segments are longer than ~2x buffer and the longest match is less than 80% of the segment length, reject match
# - otherwise, use the segment with the largest overlap and assign its lane and speed limit info
#

def map_match(centreline, match_network, value_columns, output_columns, overlap_handler, buffer = 15, exclude = ['Trail', 'Walkway']):
    """
    A crude map-matching between the city centreline network and another semi-aligned network 
    (e.g. other city maps, ORN map).

    Parameters:
    centreline (gdf): geodataframe of road network information
    match_network (gdf): network to map match and extract information from
    value_columns (list[str]): columns to match from non-centreline map. This overlap_handler functions are currently 
                                set up to require value_columns = [speed_col, lane_col] for the ORN network and 
                                value_columns = [width_col, lane_col] for the city network.
    output_columns (list[str]): columns written to centreline network. Must match value_columns in length and meaning.
    overlap_hander (function): function to deal with overlapping segments, logic is network-specific.
    buffer (float): distance in metres to buffer the centreline network for matches. Network must use a UTM CRS.
    exclude (list[str]): road segment types to exlude. Default to exclude is feature types that would not
                    have any vehicles (Trail, Walkway)
    
    Returns: 
    centreline (gdf): centreline network with new columns of matched data from match_network
    """
    assert centreline.crs == match_network.crs # networks must be in same reference system

    # slightly faster to calculate the buffer first and use it throughout
    buffered = centreline.buffer(buffer)

    # initialize output columns in centreline network
    for col in output_columns: 
        centreline[col] = np.nan

    for i, row in tqdm(centreline[~centreline['feature_code_desc'].isin(exclude)].iterrows()):  
        geom = buffered.loc[i]
        segment_length = row['geometry'].length
        # if the segment is too short to effectively match, set it aside and use connecting streets if possible. 
        if segment_length < 2*buffer:
            continue

        intersect = match_network[match_network.intersects(geom)]
        if len(intersect) == 0:
            #print("no intersection")
            continue

        intersect_lengths = intersect.intersection(geom).length
        # length of overlap is less than buffer and less than 90% overlap
        if np.all(intersect_lengths <= 2*buffer) and np.max(intersect_lengths) < 0.8*segment_length: 
            #print("no long intersection", np.max(intersect_lengths), segment_length)
            continue

        results = overlap_handler(intersect_lengths,
                                  intersect,
                                  value_columns[0],
                                  value_columns[1],
                                  segment_length,
                                  buffer)

        for col, value in results.items():
            centreline.loc[i, output_columns[value_columns.index(col)]] = value


# What to do with the overlapping items depends on the specifics of the dataset, so use separate helper functions for the two use cases so far.

# +
def overlap_handler_orn(intersect_lengths, intersect, speed_col, lane_col, segment_length, buffer):
    """
    What to do with intersecting segments for the ORN network. 
    If only one ORN segment intersects with the city segment, use its speed and lane info.
    Otherwise, if more than one segment matches it could be a bi-directional road represented
    as two one-way segments in the ORN. In this case, take the max speed and sum the number of lanes.
    If it's not two one-way segments, just take the max speed and lanes.

    Parameters:
    intersect_lengths (series): lengths of intersecting portions of segments that intersect with the centreline segment
    intersect (gdf): geodataframe of intersection with centreline segment
    speed_col (str): column from value_columns that refers to travel speed
    lane_col (str): column from value_columns that refers to number of lanes
    segment_length (float): not used in ORN version
    buffer (float): buffer in metres 

    Returns:
    dict: values from the ORN corresponding to each column
    """
    
    if len(intersect_lengths) == 1:
        max_intersection = intersect_lengths.index[intersect_lengths.argmax()]
        speed = intersect.loc[max_intersection, speed_col]
        lanes = intersect.loc[max_intersection, lane_col]
    else:
        # second largest and largest intersections to deal with bi-directional lanes
        max_ind2, max_ind  = np.argsort(intersect_lengths).values[-2:]
        # if overlaps are both larger than the buffer and segments are not bidirectional, keep both and sum lanes
        if np.all(intersect_lengths.iloc[[max_ind, max_ind2]] > 2*buffer) and np.all(intersect.iloc[[max_ind2, max_ind]]['DIRECTION'] != 'Both'):
            # use MAX speed limit of matching segments
            speed = intersect.loc[intersect_lengths.index[[max_ind, max_ind2]], speed_col].max()
            # SUM the number of lanes
            lanes = intersect.loc[intersect_lengths.index[[max_ind, max_ind2]], lane_col].sum()
        # feels like this could be written more efficiently with the if/else
        else: # otherwise just take the longest segment
            max_intersection = intersect_lengths.index[max_ind]
            speed = intersect.loc[max_intersection, speed_col]
            lanes = int(intersect.loc[max_intersection, lane_col])

    return {speed_col: speed, lane_col: lanes}

def overlap_handler_city(intersect_lengths, intersect, width_col, lane_col, segment_length, buffer):
    """
    What to do with intersecting segments for the city lanes network. 
    If only one ORN segment intersects with the city segment, use its speed and lane info.
    Otherwise, if more than one segment matches it could be a bi-directional road represented
    as two one-way segments in the ORN. In this case, take the max speed and sum the number of lanes.
    If it's not two one-way segments, just take the max speed and lanes.

    Parameters:
    intersect_lengths (series): lengths of intersecting portions of segments that intersect with the centreline segment
    intersect (gdf): geodataframe of intersection with centreline segment
    width_col (str): column from value_columns that refers to roadway width
    lane_col (str): column from value_columns that refers to number of lanes
    segment_length (float): length in metres of centreline segment
    buffer (float): buffer in metres 

    Returns:
    dict: values from the ORN corresponding to each column
    """
    if len(intersect_lengths) == 1:
        max_intersection = intersect_lengths.index[intersect_lengths.argmax()]
        width = intersect.loc[max_intersection, width_col]
        lanes = int(intersect.loc[max_intersection, lane_col])
    else:
        # second largest and largest intersections to deal with bi-directional lanes
        max_ind2, max_ind  = np.argsort(intersect_lengths).values[-2:]

        # if second-longest overlap is 80% of the segment or more, keep both and sum
        # this data has no indication of bidirectionality
        if intersect_lengths.iloc[max_ind2] > 0.8*segment_length:
            # sum the width
            width = intersect.loc[intersect_lengths.index[[max_ind, max_ind2]], width_col].sum()
            # SUM the number of lanes
            lanes = intersect.loc[intersect_lengths.index[[max_ind, max_ind2]], lane_col].sum()
        else: # otherwise just take the longest segment
            max_intersection = intersect_lengths.index[max_ind]
            width = intersect.loc[max_intersection, width_col]
            lanes = int(intersect.loc[max_intersection, lane_col])

    return {width_col: width, lane_col: lanes}


# -

match_network = orn_toronto
value_columns = ['SPEED_LIMIT', 'NUMBER_OF_LANES']
output_columns = ['speed_limit', 'number_of_lanes_orn']
overlap_handler = overlap_handler_orn

map_match(centreline, match_network, value_columns, output_columns, overlap_handler)

# ### Now fill in small gaps
#
# For unmatched centreline segments, look at neighbouring segments and assign their speed / lanes info.

# if this fails, it means that zero speeds were assigned in the above step and something has gone wrong
# most likely thing is a coordinate system mismatch, check that both datasets are using the same CRS
# can also plot small sections of each map to debug
assert centreline[(~centreline['feature_code_desc'].isin(['Trail', 'Walkway']))
                            & (centreline['speed_limit'].isna())].shape[0] < centreline.shape[0]

# how many segments are missing speed info?
centreline[(~centreline['feature_code_desc'].isin(['Trail', 'Walkway']))
                            & (centreline['speed_limit'].isna())].shape


def fill_info_gaps(centreline, fill_column, exclude = ['Trail', 'Walkway']):
    """
    Assign neighbouring speed limit or lanes to segments with missing data.

    Parameters:
    centreline (gdf): geodataframe of road network information
    fill_column (str): the column name to fill missing values in (e.g. 'speed_limit' or 'number_of_lanes_city')
    exclude (list[str]): road segment types to exlude. Default to exclude is feature types that would not
                    have any vehicles (Trail, Walkway)

    Returns:
    centreline (gdf): 
    """
    
    for i, row in tqdm(centreline[(~centreline['feature_code_desc'].isin(exclude))
                            & (centreline[fill_column].isna())].iterrows()):
        to_street = centreline.loc[centreline['from_intersection_id'] == row['to_intersection_id']]
        from_street = centreline.loc[centreline['to_intersection_id'] == row['from_intersection_id']]
    
        neighbouring_streets = pd.concat([to_street, from_street])
        
        # if it's a simple missing link and values on either side are the same:
        if len(to_street) == 1 and len(from_street) == 1 and np.all(to_street[[fill_column]].values 
                                                                    == from_street[[fill_column]].values):
            centreline.loc[i, fill_column] = to_street[fill_column].values[0]
    
        # else if only one value is present on all neighbouring streets, assign it
        elif len(neighbouring_streets[[fill_column]].dropna().value_counts()) == 1:
            centreline.loc[i, fill_column] = neighbouring_streets[fill_column].dropna().unique()[0]
    
        else:
            matching_class = neighbouring_streets[neighbouring_streets['feature_code_desc'] == row['feature_code_desc']]
    
            if len(matching_class[[fill_column]].dropna().value_counts()) == 1:
                centreline.loc[i, fill_column] = matching_class[fill_column].dropna().unique()[0]
    
            else: # assign max value of matching segments
                centreline.loc[i, fill_column] = matching_class[fill_column].dropna().max()

    return centreline


# +
nan_len_orig = centreline.shape[0] # start with the original size of the dataset (any number higher than the current number of NaNs would work)

nan_len = centreline[(~centreline['feature_code_desc'].isin(['Trail', 'Walkway']))
                            & (centreline['speed_limit'].isna())].shape[0]

while nan_len_orig - nan_len > 0:
    print(nan_len)
    nan_len_orig = nan_len
    centreline = fill_info_gaps(centreline, fill_column = 'speed_limit')
    nan_len = centreline[(~centreline['feature_code_desc'].isin(['Trail', 'Walkway']))
                            & (centreline['speed_limit'].isna())].shape[0]
# -

assert centreline.shape[0] == shape_orig

# Spot check plot to check that info looks correctly assigned. Speeds from the ORN should be correctly placed on the centreline network and these two maps should look the same. 

# +
# cartodb now requires a (free) api key: https://carto.com/basemaps/apikey/
with open("cartodb_api_key.txt","r") as carto_file:
    cartodb_key = carto_file.read()

# view the URLS for a provider by calling e.g. cx.providers.CartoDB.Voyager
url = 'https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=' + cartodb_key

# +
fig, axs = plt.subplots(1,2, figsize = (11,4.2))

xmin = 6.37e5
xmax = 6.4e5
ymin = 4.838e6
ymax = 4.8405e6

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.1)

#centreline[~centreline['feature_code_desc'].isin(['Trail', 'Walkway'])].plot(ax = ax, alpha = 0.5, color ='k')
orn_toronto.cx[xmin:xmax, ymin:ymax].plot(ax = axs[0], column = 'SPEED_LIMIT', legend = True, cax = cax, 
                                                   legend_kwds={"label": "Speed limit"
                                                               }, vmin = 30, vmax = 60)

centreline[~centreline['feature_code_desc'].isin(['Trail', 'Walkway'])].cx[xmin:xmax, ymin:ymax].plot(ax = axs[1], 
                                                                                         column = 'speed_limit', vmin = 30, vmax = 60 )

cx.add_basemap(ax = axs[0], crs = centreline.crs, source = url)
cx.add_basemap(ax = axs[1], crs = centreline.crs, source = url)

axs[0].set_title("Ontario Road Network")

axs[1].set_title("Centreline network")

for ax in axs:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])
    ax.set_xticks([])

plt.tight_layout()

plt.savefig("speed_limit_matching_comparison_east.png", dpi = 300)

# +
fig, axs = plt.subplots(1,2, figsize = (11,4.2))

xmin = 6.246e5
xmax = 6.264e5
ymin = 4.8340e6
ymax = 4.8355e6

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.1)

#centreline[~centreline['feature_code_desc'].isin(['Trail', 'Walkway'])].plot(ax = ax, alpha = 0.5, color ='k')
orn_toronto.cx[xmin:xmax, ymin:ymax].plot(ax = axs[0], column = 'SPEED_LIMIT', legend = True, cax = cax, 
                                                   legend_kwds={"label": "Speed limit"
                                                               }, vmin = 30, vmax = 60)

centreline[~centreline['feature_code_desc'].isin(['Trail', 'Walkway'])].cx[xmin:xmax, ymin:ymax].plot(ax = axs[1], 
                                                                                         column = 'speed_limit', vmin = 30, vmax = 60 )


cx.add_basemap(ax = axs[0], crs = centreline.crs, source = url, zoom = 15)
cx.add_basemap(ax = axs[1], crs = centreline.crs, source = url, zoom = 15)

axs[0].set_title("Ontario Road Network")

axs[1].set_title("Centreline network")

for ax in axs:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])
    ax.set_xticks([])

plt.tight_layout()

plt.savefig("speed_matching_comparison_southwest.png", dpi = 300)

# +
# check remaining missing segments
# mostly Laneway or Other, so minor streets

centreline[(~centreline['feature_code_desc'].isin(['Trail', 'Walkway']))
                            & (centreline['speed_limit'].isna())]['feature_code_desc'].value_counts()
# -

# ## Compare lane information received from the city
#
# We suspect there are serious accuracy issues with the number of lanes info from the Ontario Government and this is why we originally had trouble reproducing the LTS.

number_of_lanes = number_of_lanes.to_crs(utm_crs)

number_of_lanes['FUNC_CLA_1'].value_counts()

# remove highways
number_of_lanes = number_of_lanes[~number_of_lanes['FUNC_CLA_1'].isin(['Expressway'])]

match_network = number_of_lanes
value_columns = ['WIDTH', 'TRF_LANE']
output_columns = ['road_width', 'number_of_lanes_city']
overlap_handler = overlap_handler_city

map_match(centreline, match_network, value_columns, output_columns, overlap_handler)

# Fill in gaps

# +
nan_len_orig = centreline.shape[0] # just needs to be a number larger than the number of nans at the start

nan_len = centreline[(~centreline['feature_code'].isin([204001, 204002]))
                            & (centreline['number_of_lanes_city'].isna())].shape[0]

while nan_len_orig - nan_len > 0:
    print(nan_len)
    nan_len_orig = nan_len
    centreline = fill_info_gaps(centreline, fill_column = 'number_of_lanes_city', exclude = ['Trail', 'Walkway'])
    nan_len = centreline[(~centreline['feature_code'].isin([204001, 204002]))
                            & (centreline['number_of_lanes_city'].isna())].shape[0]

# +
# next: plots to compare number of lanes between ORN and city

# +
fig, axs = plt.subplots(1,2, figsize = (11,4.2))

xmin = 6.246e5
xmax = 6.265e5
ymin = 4.8340e6
ymax = 4.8355e6

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.1)

#centreline[~centreline['feature_code_desc'].isin(['Trail', 'Walkway'])].plot(ax = ax, alpha = 0.5, color ='k')
orn_toronto.cx[xmin:xmax, ymin:ymax].plot(ax = axs[0], column = 'NUMBER_OF_LANES', legend = True, cax = cax, 
                                                   legend_kwds={"label": "Number of lanes"
                                                               }, vmin = 1, vmax = 5)

centreline[~centreline['feature_code_desc'].isin(['Trail', 'Walkway'])].cx[xmin:xmax, ymin:ymax].plot(ax = axs[1], 
                                                                column = 'number_of_lanes_city', vmin = 1, vmax = 5 )


cx.add_basemap(ax = axs[0], crs = centreline.crs, source = url, zoom = 15)
cx.add_basemap(ax = axs[1], crs = centreline.crs, source = url, zoom = 15)

axs[0].set_title("Ontario Road Network")

axs[1].set_title("City lane width network")

for ax in axs:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])
    ax.set_xticks([])

plt.tight_layout()

plt.savefig("lanes_matching_comparison_southwest_city_data.png", dpi = 300)
# -

# Okay yes, this looks better - city lane data seems more accurate, not so many 2-lane arterial roads.

# ## Cycling network

cycling_network = read_dataframe(cycling_network_filename)
cycling_network = cycling_network.to_crs(utm_crs)

multiple_centrelines = cycling_network['CENTRELINE'].value_counts()[cycling_network['CENTRELINE'].value_counts() > 1].index

infra_types = cycling_network[cycling_network['CENTRELINE'].isin(multiple_centrelines)]['INFRA_HIGH'].unique()

infra_types

# There are a very small handful of places that have multiply defined centrelines in the cycling network. Keep the "best" infrastructure segment.

# rank infrastructure in the above list by which will result in a lower LTS score
# rails and cycle tracks: best
# bike lane: medium
# sharrows: worst
ranks = [3,2,1,1,1,1,2,3,3,1,3,1]

cycling_network['infra_rank'] = cycling_network['INFRA_HIGH'].map(dict(zip(infra_types, ranks)))

cycling_network = cycling_network.sort_values(by = 'infra_rank').drop_duplicates(subset = 'CENTRELINE', keep = 'first')

# merge cycling network file
centreline = centreline.merge(cycling_network.drop(columns = 'geometry'), left_on = 'centreline_id', right_on = 'CENTRELINE', how = 'left')

#e.g.
cycling_network[cycling_network['CENTRELINE'] == 14017010]

assert centreline.shape[0] == shape_orig

# Because the centreline network changes over time, some segments may not line up. Need to do spatial matching to get these aligned with the centreline file.

cycling_missing = cycling_network[~(cycling_network['CENTRELINE'].isin(centreline['centreline_id']))]

cycling_missing.shape

# +
# yellow on blue are missing segments
street = 'MacDonell Ave'
#street = 'Dufferin'
fig, ax = plt.subplots()
centreline[centreline['linear_name_full'].str.contains(street)].plot(ax = ax, linewidth = 4, label = 'centreline network segments')
centreline[(centreline['linear_name_full'].str.contains(street))
           & ~(centreline['INFRA_HIGH'].isna())].plot(ax = ax, color = 'red', linewidth = 3, alpha = 0.8, label = 'centreline segments with infra label')

cycling_network[cycling_network['LINEAR_NAM'].str.contains(street)].plot(ax = ax, color = 'yellow', linewidth = 2, alpha = 0.8,
                                                                                                             label = 'cycling network with label')

cx.add_basemap(ax = ax, crs = utm_crs, source=url)
ax.legend(loc = 'lower right')

# +
# now get specific overlaps
missing_segments_centreline = [] # store centreline network matches for missing segments
buffer = 5 # metres
locs = []

for i, row in cycling_missing.iterrows():
    # with a small buffer, this now works.
    intersect = centreline[(centreline['geometry'].intersects(row['geometry'].buffer(buffer)))]

    name_start = row['LINEAR_NAM'].lower().split(' ')[0]
                                
    # filter side streets, keep only if names match
    intersect2 =  intersect[(intersect['linear_name'].str.lower().str.contains(name_start))]

    # remove if they only overlap at the corners
    intersect3 = intersect2[~(intersect2.intersection(row['geometry'].buffer(buffer)).length <= buffer*1.25)]

    if len(intersect3) > 0:
        missing_segments_centreline.append(intersect3)
        locs.append(i)

# +
# orange sections are gaps filled in
street = 'MacDonell Ave'
#street = 'Dufferin'
fig, ax = plt.subplots()
#centreline[centreline['linear_name_full'].str.contains(street)].plot(ax = ax, linewidth = 4, label = 'centreline network segments')
centreline[~(centreline['INFRA_HIGH'].isna())].plot(ax = ax, color = 'blue', linewidth = 3, alpha = 0.8, 
                                                   label = 'centreline segments with infra label')

cycling_network.plot(ax = ax, color = 'yellow', linewidth = 2, alpha = 0.8, label = 'cycling network with label')

# these are the segments that have been matched
pd.concat(missing_segments_centreline).plot(ax = ax, color = 'orange', label = "matched sections")

ax.set_xlim(6.254e5, 6.261e5)
ax.set_ylim(4.8335e6, 4.8345e6)

cx.add_basemap(ax = ax, crs = utm_crs, source=url)
ax.legend(loc = 'lower right')

# +
# where orange is showing over blue here, we successfully matched the missing sections
fig, ax = plt.subplots(figsize = (12,12))

cycling_missing.plot(ax = ax, linewidth = 3)
pd.concat(missing_segments_centreline).plot(ax = ax, color = 'orange')
cx.add_basemap(ax = ax, crs = utm_crs, source = url)
plt.tight_layout()

plt.savefig("cycling_centreline_mismatch_matching.pdf")
# -

# Debug small missing sections still?

# add in missing info to centreline network
for i, ind in enumerate(locs):
    row = cycling_missing.loc[ind]
    centreline_ids = missing_segments_centreline[i]['centreline_id']
    centreline.loc[centreline['centreline_id'].isin(centreline_ids), ['CENTRELINE', 'LINEAR_NAM', 'FROM_STREE', 'TO_STREET', 'INFRA_HIGH',
       'INFRA_LOWO']] = row[['CENTRELINE', 'LINEAR_NAM', 'FROM_STREE', 'TO_STREET', 'INFRA_HIGH',
       'INFRA_LOWO']]

# ## Street parking
#
# In July 2021 file from Bo
#
# - Final speed: Average motor traffic speed during morning rush hour. The GTA model provides estimates only for major arterials; missing values are filled with the posted speed limit at the time. Some entries remain missing, mostly for walkways or multi-use pathways where LTS can be determined without vehicle speed. 
# - With_parking: Equals 1 if on-street parking exists on either side of the road segment, and 0 otherwise.
# - From Bo: GTASpeed_1–3 should be the speeds in morning peak hours, midday, and evening peak hours, respectively. Bo doesn't remember what BL_width and BL_speed represent. I think they are some intermediate values that I calculated to do the LTS assessment, not raw data.
#
#
# Bo suggests using open street map for lane information?
# Bo says he will check to see if he has GTAModel volume information in his records.
#

lts_2021_base = read_dataframe(lts_2021_filename)

lts_2021_base.columns

lts_2021_base.head()

# This one also has multiple rows for some centreline IDs, in particular the GTA model speed seems to vary.

lts_2021_base['GEO_ID'].value_counts().head(20)

lts_2021_base[lts_2021_base['GEO_ID'] == 1146742]

# keep the highest of matched GTAModel speeds for each geo_id
lts_2021_base = lts_2021_base.sort_values(by = 'Final_Spee').drop_duplicates(subset = 'GEO_ID', keep = 'last')

centreline = centreline.merge(lts_2021_base[['GEO_ID', 'with_parki', 'GTASpeed_1',
       'GTASpeed_2', 'GTASPeed_3', 'Final_Spee']], 
                                left_on = 'centreline_id', right_on = 'GEO_ID', how = 'left')

assert centreline.shape[0] == shape_orig

# ## Save file

date = "2025-02" # centreline network / cycling infra date
centreline.to_file("data/centreline_network_with_attributes_{}.geojson".format(date))
