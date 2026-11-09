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
# | GTAModel speed and volume | GTAModel contacts (via Bo, 2021), James Vaughan (2026) | Yes | We have Bo's 2021 matched version and the raw data from James. Both use 2016 modeled speed and volume, since this is the most accurately calibrated network |
# | Street parking | Toronto Parking Authority (via Bo) | Unknown | This info is in the 2021 LTS file shared by Bo until updated |
# | Previous LTS version(s) | Bo | NA | Optional, used for testing and comparison |
#

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
# Modeled speed and volume, 2016 GTAModel
gtamodel_speed_am = "data/TTS2016DemandV4.2Parameters/AMRoad/emme_links.shp"
gtamodel_speed_pm = "data/TTS2016DemandV4.2Parameters/PMRoad/emme_links.shp"
gtamodel_speed_md = "data/TTS2016DemandV4.2Parameters/MDRoad/emme_links.shp"

import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx
from matplotlib import pyplot as plt
from tqdm import tqdm
from pyogrio import read_dataframe # faster than geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ## Process centreline file

centreline_orig = read_dataframe(centreline_filename)
centreline = centreline_orig

# use UTM crs to get accurate distances between things
utm_crs = centreline.estimate_utm_crs()
centreline = centreline.to_crs(utm_crs)

# rename columns using field info - optional
centreline_fields = pd.read_csv("data/Centreline_Version2-4326/Centreline - Version 2 fields.csv")
centreline_cols = dict(zip(centreline_fields['field'], centreline_fields['name']))
centreline = centreline.rename(columns = centreline_cols)
centreline.columns = centreline.columns.str.lower() # lowercase column names

assert np.all(centreline['centreline_id'].value_counts() == 1)

# get list of feature types
centreline[['feature_code','feature_code_desc']].value_counts()

# remove things that can't be biked on
centreline = centreline[~centreline['feature_code_desc'].isin(['Expressway', 'Expressway Ramp', 'River', 'Hydro Line', 'Major Railway',
                                  'Major Shoreline', 'Creek/Tributary', 'Minor Railway',  'Geostatistical line', 
                                                'Busway', 'Collector Ramp',  'Ferry Route', 'Minor Shoreline (Land locked)'])]

shape_orig = centreline.shape[0]

# ## GTAModel speed
#
# - [x] segment to City of Toronto
# - [x] merge GTAModel data into single dataframe
# - [x] match to centreline network
# - [x] compare with speed data from Bo
#
# NOTE: the @lkspd column is the **free-flow** speed, not the congested speed. To calculate the congested speed, use the travel time column ('TIMAU'), the number of minutes for a vehicle to traverse the link, and the 'LENGTH' column. LENGTH is ''the length of the link in KMs. This does not have to be the same as the shape length.''
#
# Info from [documentation](https://tmg.utoronto.ca/doc/1.6/gtamodel/user_guide/Outputs/shapefile.html) and consultation with James Vaughan.
#
# From Bo, on the GTAModel_1 _2 _3 columns in his 2021 network: "my gut feeling is that I need to map the GTA speed from one shapefile to the toronto road network shapefile. I tried different complementary strategies targeting different road segments. I think I used the AM speed when it’s available and speed limit when it's not. The discrepancy may be because the matching was not done very well. I think I tried something called “spatial matching” (an ArcGIS function) and the three gta speeds are basically three complementary tries. And then I checked many road segments manually and correct things if they didn't look right."

# @lkspd is the vehicle speed column
gtamodel_am = read_dataframe(gtamodel_speed_am)
gtamodel_pm = read_dataframe(gtamodel_speed_pm)
gtamodel_md = read_dataframe(gtamodel_speed_md)

gtamodel = gtamodel_am[['ID', 'INODE', 'JNODE', 'LENGTH', 
                        'LANES', 'TIMAU', '@lkspd', 'MODES', 'geometry']].merge(gtamodel_pm[['ID', 'INODE', 'JNODE', 
                                                                                            'LENGTH', 'TIMAU', '@lkspd']], 
                                                                                               on = ['ID', 'INODE', 'JNODE'],
                                                                             how = 'outer', suffixes = ['_am', '_pm'])
gtamodel = gtamodel.merge(gtamodel_md[['ID', 'INODE', 'JNODE', 
                                       'LENGTH', 'TIMAU', '@lkspd']], on = ['ID', 'INODE', 'JNODE'], how = 'outer')

# calculate congested speed
gtamodel['speed_am'] = gtamodel['LENGTH_am'] / (gtamodel['TIMAU_am'] /60)
gtamodel['speed_md'] = gtamodel['LENGTH'] / (gtamodel['TIMAU'] /60)
gtamodel['speed_pm'] = gtamodel['LENGTH_pm'] / (gtamodel['TIMAU_pm'] /60)

# all links match up between the times of day
assert np.all(gtamodel['ID'].value_counts() == 1)

gtamodel = gtamodel.to_crs(utm_crs)

# ### Filter GTAModel modes and links
#
# Modes 'm' and 'r' are transit lines - can remove these.
#
# Also remove 'LANES' == 0: these seem like transit station entrances?

fig, ax = plt.subplots(figsize = (10,10))
gtamodel[gtamodel['LANES'] == 0].plot(ax = ax)
centreline.plot(ax = ax, color = 'k', linewidth = 0.5)
ax.set_xlim(6.20e5, 6.3e5)
ax.set_ylim(4.832e6, 4.844e6)

# remove segments with 0 lanes - these look like transit station entrances
gtamodel = gtamodel[~(gtamodel['LANES'] == 0)]

modes = list(gtamodel['MODES'].unique())

gtamodel['MODES'].value_counts()

# visualize mode labels to check which ones we can remove (e.g. highways, transit lines)
for i in range(1,5):
    fig, ax = plt.subplots(figsize = (10,10))
    centreline.plot(ax = ax, color = 'k', linewidth = 0.5)
    gtamodel[gtamodel['MODES'].isin(modes[(i-1)*10:10*i])].plot(ax = ax, column = 'MODES', linewidth = 1, legend = True)
    ax.set_xlim(6.15e5, 6.5e5)
    ax.set_ylim(4.83e6, 4.85e6)

modes_to_remove = ["chijfedHIJKbpg", "m", "r", "chijfedHIJKv"] # highways, transit lines, and abstracted zone connectors

gtamodel = gtamodel[~gtamodel['MODES'].isin(modes_to_remove)]

fig, ax = plt.subplots(figsize = (10,10))
centreline.plot(ax = ax, color = 'k', linewidth = 0.5)
gtamodel.plot(ax = ax, column = 'MODES', linewidth = 1, legend = True)
ax.set_xlim(6.20e5, 6.3e5)
ax.set_ylim(4.835e6, 4.847e6)

gtamodel_am['LANES'].value_counts()

# ### Match GTAModel links to centreline network
#
# - [x] subset GTAModel to bounds of centreline (might be able to skip this if checking for actual intersection
# - [x] The GTAmodel network is bi-directional: links are present in both directions. Drop bidirectional links with the same attributes.
# - [x] for each centreline link, find the best-matching GTAModel link

boundary = centreline.union_all().convex_hull

gtamodel_to = gtamodel[gtamodel.intersects(boundary.buffer(1000))]
gtamodel_to.to_file("data/gtamodel_to.geojson")

# exclude these road classes from the centreline network - not present in GTAModel network
exclude = ['Trail', 'Walkway', 'Local', 'Laneway', 'Pending', 'Access Road', 'Other']

centreline[~centreline['feature_code_desc'].isin(exclude)]['feature_code_desc'].value_counts()

# Some collector and minor arterial segments are in the GTAModel network, but not all. Can't just remove them. 

# +
fig, ax = plt.subplots(figsize = (10,10))

gtamodel_to.plot(ax = ax, color = 'grey', linewidth = 1.5)
centreline[~centreline['feature_code_desc'].isin(exclude)].plot(ax = ax, linewidth = 1, column = 'feature_code_desc', legend = True)

ax.set_xlim(6.2e5, 6.4e5)
ax.set_ylim(4.831e6, 4.845e6)
# -

# Drop duplicate reverse-direction links
#
# Speeds may be different in different directions, but we don't store directional information. (We could, but we don't.)
# Here we just keep the highest speed in either direction.

# define an ID with to drop duplicates on - ID is 'INODE-JNODE' and can order the nodes by size to drop duplicates
# I got that idea from Kyla
gtamodel_to['ID_sort'] = gtamodel_to[['INODE', 'JNODE']].min(axis = 1).astype('str') + '-' + gtamodel_to[['INODE', 'JNODE']].max(axis = 1).astype('str')

# Because we want to keep the highest speed from any time of day, do this with groupby instead of drop_duplicates

gtamodel_to_speed = gtamodel_to.groupby('ID_sort')[['speed_am', 'speed_pm', 'speed_md']].max().reset_index()

gtamodel_to = gtamodel_to[['ID', 'ID_sort', 'INODE', 'JNODE', 'LANES', 'MODES', 'geometry']].drop_duplicates(subset = 'ID_sort').merge(gtamodel_to_speed, on = 'ID_sort')

assert np.all(gtamodel_to['ID_sort'].value_counts() == 1)


def map_match(centreline, match_network, value_columns, output_columns, overlap_handler, length_threshold,
              buffer = 15, 
              exclude = ['Trail', 'Walkway']):
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
        if segment_length < length_threshold:
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
def overlap_handler_gtamodel(intersect_lengths, intersect, speed_col, lane_col, segment_length, buffer):
    """
    What to do with intersecting segments for the GTAModel network 
    If only one segment intersects with the city segment, use its speed and lane info.
    Otherwise, if more than one segment matches, take the longest match and its speed and lanes.

    Parameters:
    intersect_lengths (series): lengths of intersecting portions of segments that intersect with the centreline segment
    intersect (gdf): geodataframe of intersection with centreline segment
    speed_col (str): column from value_columns that refers to travel speed
    lane_col (str): column from value_columns that refers to number of lanes
    segment_length (float): not used in ORN version
    buffer (float): buffer in metres 

    Returns:
    dict: values from the GTAModel network corresponding to each column
    """
    
    # for this network, keep only convincing matches that have good overlap with the segment
    if np.max(intersect_lengths) < 0.5*segment_length: 
        return {speed_col: np.nan, lane_col: np.nan}

    # use the highest overlap
    max_intersection = intersect_lengths.index[intersect_lengths.argmax()]
    speed = intersect.loc[max_intersection, speed_col]
    lanes = intersect.loc[max_intersection, lane_col]

    return {speed_col: speed, lane_col: lanes}


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

match_network = gtamodel_to
overlap_handler = overlap_handler_gtamodel
buffer = 15
length_threshold = 2*buffer

# AM speed

# +
value_columns = ['speed_am', 'LANES']
output_columns = ['speed_am', 'number_of_lanes_gtamodel']

map_match(centreline, match_network, value_columns, output_columns, overlap_handler, 
          length_threshold, buffer, exclude = exclude)
# -

# MD speed

# +
value_columns = ['speed_md', 'LANES']
output_columns = ['speed_md', 'number_of_lanes_gtamodel']

map_match(centreline, match_network, value_columns, output_columns, overlap_handler, 
          length_threshold, buffer, exclude = exclude)
# -

# PM speed

# +
value_columns = ['speed_pm', 'LANES']
output_columns = ['speed_pm', 'number_of_lanes_gtamodel']

map_match(centreline, match_network, value_columns, output_columns, overlap_handler, 
          length_threshold, buffer, exclude = exclude)
# -

assert np.all(centreline['centreline_id'].value_counts() == 1)

# how many segments are missing speed info?
centreline[(~centreline['feature_code_desc'].isin(exclude))
                            & (centreline['speed_am'].isna())].shape

# +
fig, axs = plt.subplots(1,2, figsize = (12,8))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.5)
centreline.plot(ax = axs[0], column = 'speed_am', vmin = 20, vmax = 70, legend = True, cax = cax)

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.5)
gtamodel_to.plot(ax = axs[1], column = 'speed_am', vmin = 20, vmax = 70, legend = True, cax = cax)

for ax in axs:
    ax.set_xlim(6.28e5, 6.35e5)
    ax.set_ylim(4.833e6, 4.84e6)
    ax.set_xticks([])
    ax.set_yticks([])

axs[0].set_title("Centreline network matched speeds")
axs[1].set_title("GTAModel network speeds")
plt.tight_layout()
plt.savefig("gtamodel_matched_speed_pre_gap_fill.png", dpi = 250)

# +
fig, ax = plt.subplots(figsize = (8,8))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.5)
centreline.dropna(subset = 'speed_am').plot(ax = ax, color = 'limegreen')
gtamodel_to.plot(ax = ax, color = 'k', linewidth = 0.5)
#cx.add_basemap(ax = ax, zoom = 13)

ax.set_xlim(6.25e5, 6.4e5)
ax.set_ylim(4.832e6, 4.842e6)
plt.tight_layout()
plt.savefig("gta_speed_matching_downtown_pre_gap_fill.png", dpi = 200)


# -

def fill_info_gaps(centreline, fill_column, exclude = ['Trail', 'Walkway'], exclude_roads = [], 
                   exclude_segments = [], name_match = False,
                  check_distance = False):
    """
    Assign neighbouring speed limit or lanes to segments with missing data.

    Parameters:
    centreline (gdf): geodataframe of road network information
    fill_column (str): the column name to fill missing values in (e.g. 'speed_limit' or 'number_of_lanes_city')
    exclude (list[str]): road segment types to exlude. Default to exclude is feature types that would not
                    have any vehicles (Trail, Walkway)
    exclude_roads (list[str]): road segment names to exclude (matching 'linear_name_full' column)
    exclude_segments (list[str]): centreline segment IDs to exclude (matching 'centreline_id' column)
    name_match: if True, one of the neighbouring streets must have the same name

    Returns:
    centreline (gdf): 
    """
    
    for i, row in tqdm(centreline[(~centreline['feature_code_desc'].isin(exclude)) 
                                  & (~centreline['linear_name_full'].isin(exclude_roads))
                                  & (~centreline['centreline_id'].isin(exclude_segments))
                            & (centreline[fill_column].isna())].iterrows()):

        to_street = centreline.loc[centreline['from_intersection_id'] == row['to_intersection_id']]
        from_street = centreline.loc[centreline['to_intersection_id'] == row['from_intersection_id']]
    
        neighbouring_streets = pd.concat([to_street, from_street])

        if name_match == True:
            # filter to same-name streets
            neighbouring_streets = neighbouring_streets[neighbouring_streets['linear_name_full'] == row['linear_name_full']]

        # if no neighbours, skip
        if len(neighbouring_streets) == 0:
            continue

        # if none of the neighbours have values, skip
        if len(neighbouring_streets[[fill_column]].dropna()) == 0:
            continue
        
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


# #### Identify roads to exclude from gap filling
#
# This crude map-matching sometimes grabs perpendicular segments at intersections. To avoid propagating these, follow these checks for gap filling:
# - If less than some fraction of the named road was matched to GTAModel, don't fill other gaps
# - If only one named segment was matched, don't fill other gaps

centreline['length'] = centreline.length

road_lengths = centreline[(~centreline['feature_code_desc'].isin(exclude))].groupby('linear_name_full')['length'].sum().reset_index()

matched_road_lengths = centreline[(~centreline['feature_code_desc'].isin(exclude))
            & (~centreline['speed_am'].isna())].groupby('linear_name_full')['length'].agg([sum, lambda x: x.nunique()]).reset_index()

matched_road_lengths = matched_road_lengths.rename(columns= {'<lambda_0>':'segments'})

road_lengths = road_lengths.merge(matched_road_lengths, on = 'linear_name_full', how = 'left').fillna(0)

road_lengths.shape

road_lengths['fraction_matched'] = road_lengths['sum'] / road_lengths['length']

exclude_roads = list(road_lengths[(road_lengths['segments'] == 1)
              | (road_lengths['fraction_matched'] < 0.3)]['linear_name_full'])

# Exclude centreline segments from gap filling that are beyond some cutoff distance from gtamodel network

outer_distance = 100 # if more than 100 m from gtamodel segments, don't attempt to match
exclude_segments = []
for i, row in tqdm(centreline.iterrows()):
    if gtamodel_to.distance(row['geometry']).min() > outer_distance: 
        exclude_segments.append(row['centreline_id'])

len(exclude_segments)

# +
fig, ax = plt.subplots(figsize = (8,8))

centreline[~centreline['centreline_id'].isin(exclude_segments)].plot(ax = ax, color = 'limegreen')
gtamodel_to.plot(ax = ax, color = 'k', linewidth = 0.5)
#cx.add_basemap(ax = ax, zoom = 13)

ax.set_xlim(6.25e5, 6.4e5)
ax.set_ylim(4.832e6, 4.842e6)
plt.tight_layout()

# +
fill_column = 'speed_am'
nan_len_orig = centreline.shape[0] # start with the original size of the dataset (any number higher than the current number of NaNs would work)

nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude)) 
                                  & (~centreline['linear_name_full'].isin(exclude_roads))
                                  & (~centreline['centreline_id'].isin(exclude_segments))
                            & (centreline[fill_column].isna())].shape[0]

while nan_len_orig - nan_len > 0:
    print(nan_len)
    nan_len_orig = nan_len
    centreline = fill_info_gaps(centreline, fill_column = fill_column, exclude = exclude, 
                                exclude_roads = exclude_roads, exclude_segments = exclude_segments, name_match = True)
    nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude)) 
                                  & (~centreline['linear_name_full'].isin(exclude_roads))
                            & (~centreline['centreline_id'].isin(exclude_segments))
                            & (centreline[fill_column].isna())].shape[0]

# +
fill_column = 'speed_md'
nan_len_orig = centreline.shape[0] # start with the original size of the dataset (any number higher than the current number of NaNs would work)

nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude)) 
                                  & (~centreline['linear_name_full'].isin(exclude_roads))
                                    & (~centreline['centreline_id'].isin(exclude_segments))
                            & (centreline[fill_column].isna())].shape[0]

while nan_len_orig - nan_len > 0:
    print(nan_len)
    nan_len_orig = nan_len
    centreline = fill_info_gaps(centreline, fill_column = fill_column, exclude = exclude, 
                                exclude_roads = exclude_roads, exclude_segments = exclude_segments, name_match = True)
    nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude)) 
                                  & (~centreline['linear_name_full'].isin(exclude_roads))
                                    & (~centreline['centreline_id'].isin(exclude_segments))
                            & (centreline[fill_column].isna())].shape[0]

# +
fill_column = 'speed_pm'
nan_len_orig = centreline.shape[0] # start with the original size of the dataset (any number higher than the current number of NaNs would work)

nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude)) 
                                  & (~centreline['linear_name_full'].isin(exclude_roads))
                            & (~centreline['centreline_id'].isin(exclude_segments))
                            & (centreline[fill_column].isna())].shape[0]

while nan_len_orig - nan_len > 0:
    print(nan_len)
    nan_len_orig = nan_len
    centreline = fill_info_gaps(centreline, fill_column = fill_column, exclude = exclude, 
                                exclude_roads = exclude_roads, exclude_segments = exclude_segments, name_match = True)
    nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude)) 
                                  & (~centreline['linear_name_full'].isin(exclude_roads))
                            & (~centreline['centreline_id'].isin(exclude_segments))
                            & (centreline[fill_column].isna())].shape[0]
# -

# ### Plots to spot-check GTAModel matching

# +
fig, axs = plt.subplots(1,2, figsize = (12,8))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.5)
centreline.plot(ax = axs[0], column = 'speed_am', vmin = 20, vmax = 70, legend = True, cax = cax)

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.5)
gtamodel_to.plot(ax = axs[1], column = 'speed_am', vmin = 20, vmax = 70, legend = True, cax = cax)

for ax in axs:
    ax.set_xlim(6.28e5, 6.35e5)
    ax.set_ylim(4.833e6, 4.84e6)
    ax.set_xticks([])
    ax.set_yticks([])

axs[0].set_title("Centreline network matched speeds")
axs[1].set_title("GTAModel network speeds")
plt.tight_layout()
plt.savefig("gtamodel_matched_speed_post_gap_fill.png", dpi = 250)

# +
fig, axs = plt.subplots(1,2, figsize = (13,8))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.5)
centreline.plot(ax = axs[0], column = 'speed_am', vmin = 20, vmax = 70, legend = True, cax = cax)

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.5)
gtamodel_to.plot(ax = axs[1], column = 'speed_am', vmin = 20, vmax = 70, legend = True, cax = cax)

for ax in axs:
    ax.set_xlim(6.15e5, 6.3e5)
    ax.set_ylim(4.8315e6, 4.843e6)
    ax.set_xticks([])
    ax.set_yticks([])

axs[0].set_title("Centreline network matched speeds")
axs[1].set_title("GTAModel network speeds")
plt.tight_layout()
plt.savefig("gta_speed_matching_panel_west.png", dpi = 200)

# +
fig, ax = plt.subplots(figsize = (8,8))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.5)
centreline.dropna(subset = 'speed_am').plot(ax = ax, color = 'limegreen')
gtamodel_to.plot(ax = ax, color = 'k', linewidth = 0.5)
#cx.add_basemap(ax = ax, zoom = 13)

ax.set_xlim(6.15e5, 6.3e5)
ax.set_ylim(4.8315e6, 4.843e6)
plt.tight_layout()
plt.savefig("gta_speed_matching_west.png", dpi = 200)

# +
fig, ax = plt.subplots(figsize = (8,8))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.5)
centreline.dropna(subset = 'speed_am').plot(ax = ax, color = 'limegreen')
gtamodel_to.plot(ax = ax, color = 'k', linewidth = 0.5)
#cx.add_basemap(ax = ax, zoom = 13)

ax.set_xlim(6.15e5, 6.3e5)
ax.set_ylim(4.839e6, 4.849e6)
plt.tight_layout()
plt.savefig("gta_speed_matching_northwest.png", dpi = 200)

# +
fig, ax = plt.subplots(figsize = (8,8))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.5)
centreline.dropna(subset = 'speed_am').plot(ax = ax, color = 'limegreen')
gtamodel_to.plot(ax = ax, color = 'k', linewidth = 0.5)
#cx.add_basemap(ax = ax, zoom = 13)

ax.set_xlim(6.25e5, 6.4e5)
ax.set_ylim(4.832e6, 4.842e6)
plt.tight_layout()
plt.savefig("gta_speed_matching_downtown.png", dpi = 200)

# +
fig, ax = plt.subplots(figsize = (8,8))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.5)
centreline.dropna(subset = 'speed_am').plot(ax = ax, color = 'limegreen')
gtamodel_to.plot(ax = ax, color = 'k', linewidth = 0.5)
#cx.add_basemap(ax = ax, zoom = 13)

ax.set_xlim(6.25e5, 6.4e5)
ax.set_ylim(4.839e6, 4.849e6)
plt.tight_layout()
plt.savefig("gta_speed_matching_northeast.png", dpi = 200)
# -

# Still some stubby segments getting matched that shouldn't be, but I think this is okay for now.

# ## Speed limit & number of lanes (ORN & city)

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

match_network = orn_toronto
value_columns = ['SPEED_LIMIT', 'NUMBER_OF_LANES']
output_columns = ['speed_limit', 'number_of_lanes_orn']
overlap_handler = overlap_handler_orn
buffer = 15
length_threshold = 2*buffer

map_match(centreline, match_network, value_columns, output_columns, overlap_handler, length_threshold = length_threshold)

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

# +
exclude = ['Trail', 'Walkway']
fill_column = 'speed_limit'

nan_len_orig = centreline.shape[0] # start with the original size of the dataset (any number higher than the current number of NaNs would work)

nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude))
                            & (centreline[fill_column].isna())].shape[0]

while nan_len_orig - nan_len > 0:
    print(nan_len)
    nan_len_orig = nan_len
    centreline = fill_info_gaps(centreline, fill_column = fill_column, exclude = exclude)
    nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude))
                            & (centreline[fill_column].isna())].shape[0]
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

# eglinton between Laird and Bayview is coming up as 2 lanes in the city data - this seems to be an error in the data
number_of_lanes[number_of_lanes['LOC_STREET'].str.contains('EGLINTON')].cx[6.3e5:6.32e5,4.841e6:4.844e6]

number_of_lanes['FUNC_CLA_1'].value_counts()

# remove highways
number_of_lanes = number_of_lanes[~number_of_lanes['FUNC_CLA_1'].isin(['Expressway'])]

match_network = number_of_lanes
value_columns = ['WIDTH', 'TRF_LANE']
output_columns = ['road_width', 'number_of_lanes_city']
overlap_handler = overlap_handler_city
buffer = 15
length_threshold = 2*buffer

map_match(centreline, match_network, value_columns, output_columns, overlap_handler, length_threshold = length_threshold)

# Fill in gaps

# +
exclude = ['Trail', 'Walkway']
fill_column = 'number_of_lanes_city'
nan_len_orig = centreline.shape[0] # just needs to be a number larger than the number of nans at the start

nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude))
                            & (centreline[fill_column].isna())].shape[0]

while nan_len_orig - nan_len > 0:
    print(nan_len)
    nan_len_orig = nan_len
    centreline = fill_info_gaps(centreline, fill_column = fill_column, exclude = exclude)
    nan_len = centreline[(~centreline['feature_code_desc'].isin(exclude))
                            & (centreline[fill_column].isna())].shape[0]

# +
fig, axs = plt.subplots(1,2, figsize = (11,4.2))

xmin = 6.246e5
xmax = 6.265e5
ymin = 4.8340e6
ymax = 4.8355e6

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.1)

#centreline[~centreline['feature_code_desc'].isin(['Trail', 'Walkway'])].plot(ax = ax, alpha = 0.5, color ='k')
number_of_lanes.cx[xmin:xmax, ymin:ymax].plot(ax = axs[0], column = 'TRF_LANE', legend = True, cax = cax, 
                                                   legend_kwds={"label": "Number of lanes"
                                                               }, vmin = 1, vmax = 5)

centreline[~centreline['feature_code_desc'].isin(['Trail', 'Walkway'])].cx[xmin:xmax, ymin:ymax].plot(ax = axs[1], 
                                                                column = 'number_of_lanes_city', vmin = 1, vmax = 5 )


cx.add_basemap(ax = axs[0], crs = centreline.crs, source = url, zoom = 15)
cx.add_basemap(ax = axs[1], crs = centreline.crs, source = url, zoom = 15)

axs[0].set_title("City data")

axs[1].set_title("Matched to centreline")

for ax in axs:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])
    ax.set_xticks([])

plt.tight_layout()

plt.savefig("lanes_matching_comparison_southwest.png", dpi = 300)
# -

# ### Plots to compare number of lanes between ORN and city

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

assert centreline.shape[0] == shape_orig

# ## Street parking
#
# In July 2021 file from Bo:
# - Final speed: Average motor traffic speed during morning rush hour. The GTA model provides estimates only for major arterials; missing values are filled with the posted speed limit at the time. Some entries remain missing, mostly for walkways or multi-use pathways where LTS can be determined without vehicle speed. 
# - With_parking: Equals 1 if on-street parking exists on either side of the road segment, and 0 otherwise.
# - Bo doesn't remember what BL_width and BL_speed represent. I think they are some intermediate values that I calculated to do the LTS assessment, not raw data.

lts_2021_base = read_dataframe(lts_2021_filename)

lts_2021_base = lts_2021_base.to_crs(utm_crs)

lts_2021_base.columns

lts_2021_base[np.any(lts_2021_base[['GTASpeed_1', 'GTASpeed_2', 'GTASPeed_3']] > 0, axis = 1)][['LF_NAME', 'NRN_SpeedL', 'NBRLANES_2', 'Left_Parki', 'Right_Park', 'GTASpeed_1',
       'GTASpeed_2', 'GTASPeed_3', 'Shape_Leng', 'Final_Spee', 'BL_width',
        'with_parki', 'LTS_BL', 'LTS_mixed', 'LTS']].drop_duplicates(subset = 'LF_NAME').head(20)

# This one also has multiple rows for some centreline IDs, in particular the GTA model speed seems to vary.
# This could be because both directions were getting matched, same as I had.

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




