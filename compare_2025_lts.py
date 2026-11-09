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

# # Compare 2025 LTS map recalculated from scratch and as a result of sequential updates

# lts 2025 map sent to city: 2025 centreline file, all LTS values from Bo's 2021 network unless cycling infra or speed changed, and any new or changed centreline segments also had their lts recalculated. (script is update_lts.ipynb.)
#
# vs. lts 2025 calculated from scratch

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from pyogrio import read_dataframe # faster than geopandas
import contextily as cx
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from shapely.geometry import LineString, Point, MultiLineString
from tqdm import tqdm
import shapely, json, matplotlib

from utils.functions import lts_colour

use_maxspeed = False
#use_maxspeed = "AM"
lts_recalc = read_dataframe("data/lts_2025-02_maxspeed_{}.geojson".format(use_maxspeed))
lts_2025 = read_dataframe("data/jan2025_lts.geojson")

lts_recalc['lts'] = lts_recalc['lts'].astype('int')
lts_2025['lts'] = lts_2025['lts'].astype('int')

# +
# SET ST. GEORGE LTS TO max 2 - THIS IS A MANUAL MODIFICATION
#lts_recalc.loc[(lts_recalc['linear_name_full'].str.contains('St George St'))
#        & (lts_recalc['lts'] == 3), 'lts'] = 2
# -

# ## Plot both networks

labels = ["new", "updated"] # new = calculated from scratch, updated = updated from Bo's lts
for i, lts in enumerate([lts_recalc, lts_2025]):
    lts = lts_colour(lts)
    lts_arterial = lts[lts['feature_code_desc'].isin(['Major Arterial', 'Minor Arterial'])]

    fig, ax = plt.subplots(figsize = (12,10))

    lts.plot(ax = ax, color = lts['colour'], linewidth = 0.5)
    lts_arterial.plot(ax = ax, color= lts_arterial['colour'], linewidth = 1)
    
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("LTS network 2025 - {}".format(labels[i]))
    
    plt.tight_layout()
    plt.savefig("2025_lts_{}_maxspeed_{}.pdf".format(labels[i], use_maxspeed))
    plt.savefig("2025_lts_{}_maxspeed_{}.png".format(labels[i], use_maxspeed), dpi = 250)

# ## Segment-level comparison

# both networks are constructed with the same centreline network and are directly comparable
np.all(lts_recalc[['centreline_id', 'linear_name_full']] == lts_2025[['centreline_id', 'linear_name_full']])

lts_2025.length.sum() / 1000

lts_2025[~(lts_2025['lts'] == lts_recalc['lts'])].length.sum() / 1000

# About 13% of the labels are not identical in the recalculated version (by length). How many of these affect high/low stress?

lts_2025[~(lts_2025['lts'] == lts_recalc['lts'])].length.sum() / lts_2025.length.sum() 

lts_recalc.columns

lts_2025 = lts_2025.merge(lts_recalc[['centreline_id', 'lts', 'speed_limit', 'number_of_lanes_city', 'speed_assumed',
       'lanes_assumed']], on = 'centreline_id', suffixes = ("_updated", "_new"))

lts_2025[~(lts_2025['lts_updated'] == lts_2025['lts_new'])][['linear_name_full', 'feature_code_desc', 'lts_updated', 'lts_new']]

# segments classified as low stress in the update version labeled as high stress in the new
low_to_high = lts_2025[(lts_2025['lts_updated'].isin([1,2])) 
          & (lts_2025['lts_new'].isin([3,4]))]

high_to_low = lts_2025[(lts_2025['lts_updated'].isin([3,4])) 
          & (lts_2025['lts_new'].isin([1,2]))]

# 234 km previously labeled low stress are now high stress.

low_to_high.length.sum() / 1000

# 92 km previously labeled high stress are now low stress.

high_to_low.length.sum() / 1000

# ~5\% of segments have changed labels from a stress perspective. 

((low_to_high.length.sum() / 1000) + (high_to_low.length.sum() / 1000)) / (lts_2025.length.sum() / 1000 )

# +
#low_to_high.plot(color = low_to_high['colour'])
# -

high_to_low.plot(color = high_to_low['colour'])

# # Calculate accessibility and compare

# - [x] convert network to graph
# - [x] get signals and stop signs
# - [x] get origin and destination nodes
# - [x] get job data

network = lts_2025

# ## Update network to remove bridge nodes
#
# Code from route_choice/network_data.ipynb

# bridge nodes identified from route choice work - remove these for routing purposes
bridge_nodes = pd.read_csv("../route_choice/bridge_nodes_2023.csv")

network['length'] = network.length


# +
def delete_non_intersecting_node(node_id, lts_network, precision = 0.01):

    """
    For a given node in lts_network, attempts to merge pairs of segments with the same name that are connected to that node and delete the node.

    Note: if desired, we could remove only one of the nodes and not merge the other two segments, since that would still effectively remove the intersection. 
    Could do that if the attributes didn't align, for instance.
    Right now there is NO checking for attribute alignment. Possibly implement this once I decide where in the pipeline it goes.
    """
    
    segments = lts_network[((lts_network['from_intersection_id'] == node_id)
                | (lts_network['to_intersection_id'] == node_id))]

    merged_centrelines = []
    to_merge_list = []

    # option 1: segment names
    # check if we're working with 2 pairs of segments with the same name
    names = segments['linear_name_full'].unique()
    try:
        assert np.all(segments['linear_name_full'].value_counts() == 2)            
    except AssertionError: # if there are two segments with the same name, merge those two, then merge the remaining
        try:
            assert np.any(segments['linear_name_full'].value_counts() == 2) 
        except: # if we can't resolve how they should be merged, then leave them alone
            print("Can't delete node {}".format(node_id))
            print(segments['linear_name_full'].unique())
            
            return lts_network, merged_centrelines # unchanged

    if len(names) == 2:
        for name in names:
            to_merge_list.append(segments[segments['linear_name_full'] == name])

    elif len(names) == 3:
        name1 = segments['linear_name_full'].value_counts()[segments['linear_name_full'].value_counts() == 2].index[0]
        to_merge_list.append(segments[segments['linear_name_full'] == name1])
        to_merge_list.append(segments[segments['linear_name_full'] != name1])
    
    shape_orig = lts_network.shape[0]

    for to_merge in to_merge_list:
        # new linestring geometry - need to convert to multilinestring first
        geom = shapely.line_merge(MultiLineString([to_merge.iloc[0]['geometry'], to_merge.iloc[1]['geometry']]))
    
        # now delete old rows from dataframe and add new one with new attributes
        lts_network = lts_network[~lts_network['centreline_id'].isin(to_merge['centreline_id'])] # delete
    
        # keep a list of which centrelines were merged
        merged_centrelines.append(to_merge['centreline_id'].values)
        
        # keep the longer of the two centreline_ids by default
        new_row = to_merge.sort_values(by = 'length', ascending = False).iloc[[0]]

        # drop the middle node that is being deleted
        new_nodes = to_merge[['from_intersection_id', 'to_intersection_id']][~(to_merge[['from_intersection_id', 'to_intersection_id']] == node_id)]
        
        # sometimes the node to delete is not in the centre and we need to flip a segment
        # if there are two nans or no nans in one of the columns, need to flip a segment
        if np.any(new_nodes.isna().sum(axis = 0) != 1):
            fnode_old = to_merge.iloc[0]['from_intersection_id']
            tnode_old = to_merge.iloc[0]['to_intersection_id']
            ind = to_merge.index[0]
            to_merge.loc[ind, 'geometry'] = to_merge.iloc[0]['geometry'].reverse() # reverse geometry
            to_merge.loc[ind, 'from_intersection_id'] = tnode_old
            to_merge.loc[ind, 'to_intersection_id'] = fnode_old
            new_nodes = to_merge[['from_intersection_id', 'to_intersection_id']][~(to_merge[['from_intersection_id', 'to_intersection_id']] == node_id)]

        # rename from and to nodes
        new_row['from_intersection_id'] = new_nodes[~new_nodes['from_intersection_id'].isna()]['from_intersection_id'].values[0]
        new_row['to_intersection_id'] = new_nodes[~new_nodes['to_intersection_id'].isna()]['to_intersection_id'].values[0]
        
        # check that node is deleted
        assert np.all(new_row[['from_intersection_id', 'to_intersection_id']] != node_id)
    
        # update geometry
        new_row['geometry'] = shapely.line_merge(MultiLineString([shapely.set_precision(to_merge.iloc[0]['geometry'], precision), 
                                                                 shapely.set_precision(to_merge.iloc[1]['geometry'], precision)]))
        new_row['length'] = new_row['geometry'].length
        
        # check that new length is sum of old length
        assert np.isclose(new_row['length'],to_merge['length'].sum(), atol=precision*2)[0] 
    
        lts_network = pd.concat([lts_network, new_row])
    
    assert lts_network.shape[0] == shape_orig - 2
    
    # option 2: check osm matches - could do this as a validation step
    # FIXME not implemented

    return lts_network, merged_centrelines

    
# -

network_new = network.copy()

merged_centrelines_all = []
for node_id in tqdm(bridge_nodes['node_id']):
    network_new, merged_centrelines = delete_non_intersecting_node(node_id, network_new)
    merged_centrelines_all += merged_centrelines

assert np.all(network_new.columns == network.columns)

network = network_new


# ### Create graph for routing

def create_simplified_graph_from_network(network, weights):
    """modified from Daniel's code
    This creates a graph with only the shortest edge between nodes if parallel edges exist

    Parameters:
    network (gdf): GeoDataFrame to turn into a graph
    weights (list[str]): list of columns in `network` to supply as edge weights for the graph

    Returns:
    graph (nx.Graph()): networkx undirected graph corresponding to network
    """
    graph = nx.Graph() #undirected graph, needs to be multigraph if we want to keep parallel edges
    for _, row in network.iterrows():
        geometry = row.geometry
        distance = geometry.length
        
        if isinstance(geometry, LineString):
            #start_node = tuple(geometry.coords[0])
            #end_node = tuple(geometry.coords[-1])
            start_node = int(row['from_intersection_id'])
            end_node = int(row['to_intersection_id'])
            
            edge = graph.get_edge_data(start_node, end_node)
            if edge != None: # another parallel edge exists, keep the shortest one
                length_prev = edge['length']
                
                if length_prev < distance:
                    continue
                else: # update edge
                    graph[start_node][end_node]['name'] = row['centreline_id']
                    # add additional weights
                    for weight in weights:
                        graph[start_node][end_node][weight] = row[weight]

            else:       
                graph.add_edge(start_node, end_node,
                           name=row['centreline_id'])
                # add additional weights
                for weight in weights:
                    graph[start_node][end_node][weight] = row[weight]
                

    return graph


# Define a travel time attribute in minutes

speed = 15 # km/h
speed_ms = speed/3.6
network['time'] = (network['length'] / speed_ms) / 60

weights = ['length', 'time', 'lts_new', 'lts_updated']

G = create_simplified_graph_from_network(network, weights)

# Add node LTS for accessibility calculation

# - need signals and stop signs dataset
# - then calculate node LTS following route choice code

# ## Traffic signals
#
# - These aren't aligned to the centreline network, so just do spatial matching to node
# - Remove any installed 2024 or later
# - Check if any are upgraded / changed
# - Dataset includes LPI activation date separate from activation date. Could also include this in network?
#
# Steps:
# - get nodes from centreline network
# - spatially match signals to nodes
#
#
# Mode of control: from glossary of https://www.toronto.ca/wp-content/uploads/2021/01/964c-TSSignal-OptimizationSynchro-11-Guidelines.pdf
#
# - SA1: Semi-actuated 1. A signal operation in which one or more phases are actuated, however
# the main street, side street and associated pedestrian phases are on
# recall. Vehicle detection is provided for actuated phases (typically a
# turning movement).
# - SA2: Semi-actuated 2. A signal operation in which the signal display will not change to the side street unless a vehicle or pedestrian has been detected. [In these cases no walk sign is given unless the pedestrian button is pushed.](https://www.toronto.ca/home/311-toronto-at-your-service/find-service-information/article/?kb=kA06g000001cwPTCAY)
# - SAP: Semi-actuated Pedestrian. A signal operation in which signals will not change to the side street unless a vehicle or pedestrian has been detected, and in which the side street will serve the pedestrian "walk" phase regardless of whether or
# not a pedestrian call has been received.
# - FT: Fixed Time. A signal operation in which the vehicle signal indication changes
#  automatically from the main street to the side street, and back, even if
# there are no vehicles/pedestrians wishing to cross the main street.
# - PA: Pedestrian Actuated. A signal operation at a mid-block pedestrian crossing that is actuated by
#   a pedestrian pushbutton. It provides pedestrians with a protected
#  crossing opportunity by requiring motorists to stop at the signal.
# - MPS: Midblock Pedestrian Signal. A control device dedicated primarily to providing pedestrian right-of-way
#   at midblock locations. Main road traffic is fully signalised. The pedestrian
# phase is called by pedestrian pushbuttons.
# - SAV: Semi-actuated vehicle. A signal operation in which there is no pedestrian crossing across the
# of main street and side street phase is actuated and extendible by vehicle
# call only.
# - IPS: Intersection pedestrian signal. A control device dedicated primarily to providing pedestrian right-of-way
#  at an intersection. Main road traffic is fully signalized while the side
#  road(s) are controlled with stop signs. They are also known as "Half
# Signals".

utm_crs = network.crs

signals = read_dataframe("../Cycling_data/route_choice/traffic_signals/Traffic Signal - 4326.geojson")
signals_2025 = signals[signals['ACTIVATIONDATE'].dt.year <= 2025].to_crs(utm_crs)

signals['CONTROL_MODE'].value_counts()

# Get node coordinates

# round points to 10 cm to not get precision error duplicates of nodes
# node1 is at the start of segment, node 2 is at the end of segment
network["node1"] = shapely.wkt.loads(shapely.wkt.dumps(shapely.get_point(network['geometry'], 0), rounding_precision = 1))
network["node2"] = shapely.wkt.loads(shapely.wkt.dumps(shapely.get_point(network['geometry'], -1), rounding_precision = 1))

# +
# all overlaps should be the same if digitization direction is respected
fnodes = network[['from_intersection_id', 'node1']].drop_duplicates()
tnodes = network[['to_intersection_id', 'node2']].drop_duplicates()

fnodes.columns = ['node_id', 'geometry']
tnodes.columns = ['node_id', 'geometry']
# -

# Nodes appear multiple times in the network as "from" and "to" nodes of various segments. We want to construct a dataframe of unique nodes. After dropping duplicates on ID and geometry, we still have some duplicated nodes by node ID.

(fnodes['node_id'].value_counts() == 2).sum()

(tnodes['node_id'].value_counts() == 2).sum()

# +
nodes_df = pd.concat([fnodes,tnodes]).drop_duplicates()
print(nodes_df.shape)

# round to 20 cm precision
nodes_df['geometry'] = shapely.set_precision(nodes_df['geometry'],0.2)
nodes_df = nodes_df.drop_duplicates()
print(nodes_df.shape)
# -

node_count = nodes_df['node_id'].value_counts().reset_index()
node_count.head(3)


# By inspecting these cases, they are all clearly the same node, just with slight differences in coordinate precision. 
# - Drop duplicates on node ID above and keep first. 

nodes_df = pd.concat([fnodes,tnodes])
nodes_df['geometry'] = shapely.set_precision(nodes_df['geometry'],0.2)
nodes_df = nodes_df.drop_duplicates(subset = 'node_id', keep = 'first')

# #### Join signals to nodes
#
# Want to strike a balance between merging signal info wherever possible (i.e. only midblock crossings left out) and not merging the wrong intersections together.
#
# Better to err on the side of a larger buffer: in practice, I could detour to a pedestrian crossover when crossing the street, if these are available.

nodes_df = gpd.GeoDataFrame(nodes_df, geometry = nodes_df['geometry'])
nodes_df = nodes_df.set_crs(utm_crs)

# Remove 93 midblock crossings - not going to be part of network

signals_2025 = signals_2025[~(signals_2025['CONTROL_MODE'] == 'PA - MPS')]


# Need to also check for things like street names matching, where possible

def match_signals(nodes_df, network, signals, buffer = 30):
    """
    Match signals to centreline network nodes.

    Parameters:
    nodes_df (gdf): GeoDataFrame of node IDs and point geometry
    network (gdf): corresponding centreline network with the same named nodes and corresponding edges
    signals (gdf): GeoDataFrame of signal type and point geometry
    buffer (float): buffer in metres to search around centreline nodes. Typical 4-lane road is 15 metres wide or so.

    Returns:
    nodes_df (gdf): GeoDataFrame of node IDs with matched signals
    """
    node_ids = []
    no_node = []
    no_name_node = []
    multiples = []
    distance = []

    for i, row in tqdm(nodes_df.iterrows()):
        # switch the order to potentially assign multiple nodes to a signal
        intersect = signals[signals['geometry'].intersects(row['geometry'].buffer(buffer))]
    
        # this may break if the buffer size changes
        # but for now, there is only one signal in the buffer for each node
        assert len(intersect) <= 1
        
        if len(intersect) == 0:
            no_node.append(row.name)
            node_ids.append(np.nan)
            distance.append(np.nan)
            continue
        
        # check street names for matches
        centreline_select = network[(network['from_intersection_id'] == row['node_id']) 
                | (network['to_intersection_id'] == row['node_id'])]
    
        # check that at least one street name matches between centreline and signal street names
        try:
            assert np.any(centreline_select['linear_name'].str.lower().str.split(' ', expand = True)[0].isin(
                                                                    [intersect.iloc[0]['MAIN_STREET'].lower().split(' ')[0],
                                                                    intersect.iloc[0]['SIDE1_STREET'].lower().split(' ')[0],
                                                                    intersect.iloc[0]['SIDE2_STREET'].lower().split(' ')[0]]))
        except AssertionError: # don't include that match
            no_name_node.append(row.name)
            node_ids.append(np.nan)
            distance.append(np.nan)
            continue
    
        node_ids.append(intersect['PX'].values[0])
        distance.append(intersect.distance(row['geometry']).values[0])

    nodes_df['signal_PX'] = node_ids
    nodes_df['signal_distance'] = distance
    
    return nodes_df


buffer = 30
nodes_df = match_signals(nodes_df, network, signals_2025, buffer = buffer)

# ensure at least some things got matched
assert nodes_df['signal_PX'].isna().sum() < nodes_df.shape[0]

print("{}% of signals were not assigned to a centreline network node (no node within {} metres).".format(
                        round((signals_2025[~signals_2025['PX'].isin(nodes_df['signal_PX'])].shape[0])
                                                                        / len(signals_2025),3)*100,
                                                                        buffer))

# 86% of these are midblock signals for private access.

missing = signals_2025[~signals_2025['PX'].isin(nodes_df['signal_PX'])]

missing[(missing['PRIVATE_ACCESS'] != "None")
    | (missing['SIDE2_STREET'] == "PRIVATE ACCESS")].shape[0] / len(missing)

# Only 6\% aren't a midblock crossing

(missing['MIDBLOCK_ROUTE'] == "None").sum() / len(missing)

# ## Stop signs
#
# - Get from OSM (osm_stop_signs.ipynb)
# - Using osmnx, download specific time period of data (check code used for Tessa's paper)
# - Can we tell which direction the stop signs apply to?
#

stop_signs = read_dataframe("../Cycling_data/route_choice/stop_signs/osm_stop_signs_and_signals.geojson")
stop_signs = stop_signs.to_crs(utm_crs)

# create a column combining highway and stop columns
stop_signs['stop'] = stop_signs['stop'].fillna("")
stop_signs['control'] = stop_signs['highway'] + stop_signs['stop']


# In every case I have inspected (the first 10 or so), using the closest intersecting point works. 
# With a 30 m buffer, we were catching some that don't actually apply, so changed the buffer to 20 m. The overlap seems quite close in most cases, and we are planning to use just all-way stops. 

# Note: this function is very similar to match_stop_signs but the logic of handling overlaps is different.

def match_stop_signs(nodes_df, network, stop_signs, buffer = 20):
    """
    Match signals to centreline network nodes.

    Parameters:
    nodes_df (gdf): GeoDataFrame of node IDs and point geometry
    network (gdf): corresponding centreline network with the same named nodes and corresponding edges
    stop_signs (gdf): GeoDataFrame of stop sign type and point geometry (from OSM)
    buffer (float): buffer in metres to search around centreline nodes. Typical 4-lane road is 15 metres wide or so.

    Returns:
    nodes_df (gdf): GeoDataFrame of node IDs with matched stop signs and signals
    """

    controls = []
    street_counts = []
    no_node = []
    no_name_node = []
    multiples = []
    distance = []
    

    for i, row in tqdm(nodes_df.iterrows()):
        # switch the order to potentially assign multiple nodes to a signal
        intersect = stop_signs[stop_signs['geometry'].intersects(row['geometry'].buffer(buffer))]
    
        if len(intersect) == 0:
            no_node.append(row.name)
            controls.append(np.nan)
            street_counts.append(np.nan)
            distance.append(np.nan)
            continue
    
        # get centreline component
        centreline_select = network[(network['from_intersection_id'] == row['node_id']) 
                | (network['to_intersection_id'] == row['node_id'])]
    
        # if more than one osm node is grabbed
        if len(intersect) > 1:
            # use the closest one
            #assert intersect['control'].unique().shape[0] == 1 # if only one treatment type, assign it
            intersect = intersect.loc[[intersect.distance(row['geometry']).idxmin()]]
            #if intersect['control'].unique().shape[0] > 1:
            
        dist = intersect.distance(row['geometry']).values[0]
    
        controls.append(intersect['control'].values[0])
        street_counts.append(intersect['street_count'].values[0])
        distance.append(dist)

    # assign to dataframe
    nodes_df['osm_control'] = controls
    nodes_df['street_count'] = street_counts
    nodes_df['osm_control_distance'] = distance

    return nodes_df

nodes_df = match_stop_signs(nodes_df, network, stop_signs, buffer = 20)

# Compare OSM and city traffic signal info

print("There are {} nodes matched to city traffic signals".format(nodes_df[~nodes_df['signal_PX'].isna()].shape[0]),
      "and {} nodes matched to OSM traffic signals".format(nodes_df[nodes_df['osm_control'] == 'traffic_signals'].shape[0]),
      "with {} nodes matched to both types of signals.".format(nodes_df[ ~(nodes_df['signal_PX'].isna()) 
    & (nodes_df['osm_control'] == 'traffic_signals')].shape[0]))

# This is a pretty high overlap rate. We're not going to use the OSM traffic signal data, but it's encouraging.

# makes sure 0 isn't a used signal ID
assert len(nodes_df[nodes_df['signal_PX'] == 0]) == 0

nodes_df['signal_PX'] = nodes_df['signal_PX'].fillna(0)
nodes_df['signal_PX'] = nodes_df['signal_PX'].astype('int')

# ## Calculate intersection LTS

# Now create sub-network with only low-stress edges
#
# Use Bo's routing to account for intersection stress

nodes_df['osm_control'] = nodes_df['osm_control'].fillna("")

# drop nodes that are no longer in the network
nodes_df = nodes_df[nodes_df['node_id'].isin(np.unique(network[['from_intersection_id', 'to_intersection_id']].values.flatten()))]

# some indices were duplicated, causing issues
nodes_df = nodes_df.reset_index()


def assign_intersection_lts(nodes_df, network, lts_column = 'lts'):
    """
    Assign intersection LTS

    Parameters:
    nodes_df (gdf): GeoDataFrame of node IDs and point geometry
    network (gdf): corresponding centreline network with the same named nodes and corresponding edges
    lts_column (str): column name for the column that contains edge lts

    Returns:
    nodes_df (gdf): GeoDataFrame of node IDs with an LTS value for each node
    """

    nodes_df['intersection_lts'] = np.nan
    nodes_df['intersection_lts_message'] = np.nan
    nodes_df['intersection_lts_message'] = nodes_df['intersection_lts_message'].astype('object')
    
    for i, row in tqdm(nodes_df.iterrows()):
        node = row['node_id']
        # any edge to or from this node
        edges = network[(network['from_intersection_id'] == node)
                | (network['to_intersection_id'] == node)]
        signal = row['signal_PX']
        control = row['osm_control']
        max_lts = edges[lts_column].max()
        node_lts = int(max_lts) # set to max of intersecting roads as default
        message = "Node LTS is max intersecting LTS"
        if node_lts > 2:
            if signal != 0: # then is traffic signal
                node_lts = 2
                message = "LTS 3-4 with traffic signals"
        elif node_lts <= 2:
            if signal != 0 or control == 'stopall': # use only all-way stops
                node_lts = 1
                message = "LTS 1-2 with traffic signals or stop"
    
        nodes_df.loc[i, 'intersection_lts'] = node_lts
        nodes_df.loc[i, 'intersection_lts_message'] = message

    return nodes_df


nodes_df = assign_intersection_lts(nodes_df, network, lts_column = 'lts_updated')

nodes_df2 = assign_intersection_lts(nodes_df, network, lts_column = 'lts_new')

nodes_df = nodes_df.merge(nodes_df2[['node_id', 'intersection_lts']], on = 'node_id', 
                          suffixes = ('_updated', '_new')).drop(columns = 'intersection_lts_message')

# this gives an identical result for both networks evidently
(nodes_df['intersection_lts_updated'] == nodes_df['intersection_lts_new']).sum() / nodes_df.shape[0]

# +
# add node lts to G
lts_columns = ['lts_new', 'lts_updated']

for col in lts_columns:
    attrs = {}
    for node in tqdm(G.nodes):
        attrs[node] = nodes_df[nodes_df['node_id'] == node][f'intersection_{col}'].values[0]
    nx.set_node_attributes(G, attrs, col)


# -

# ## Create low stress graph for routing

def create_lowstress_graph(G, exclude_nodes, lts_column, speed = 15, lts_threshold = 2):
    """
    Based on Bo's _build_current_nx function.

    Create a graph G_curr that includes only low-stress links 
    (by setting travel time very high on high-stress links)
    This graph incorporates intersection LTS by setting the cost high 
    on any link that touches a high-stress intersection because that 
    link will not be routable at the endpoint.

    Parameters:
    G (networkx Graph):
    exclude_nodes (list[int]): list of node IDs to exclude, e.g. nodes that are actually bridges
    lts_column (str): label for the LTS edge attribute
    speed (float): assumed cycling speed in km/h
    lts_threshold (int): low-stress cycling threshold, one of [1, 2, 3, 4]
    
    Returns:
    G_curr (networkx Graph): graph with travel time cost in minutes
    """

    speed_ms = speed/3.6 # cycling speed in metres per second

    edges = []
    
    for fnode, tnode in G.edges():
        if (G[fnode][tnode][lts_column] > lts_threshold) or (G.nodes[fnode][lts_column] > lts_threshold): # if node is high-stress or unroutable
            cost = 100 # this just has to be a value that exceeds the routing limit in later steps
        else:
            cost = (G[fnode][tnode]['length'] / speed_ms) / 60 # travel time in minutes
        edges.append((fnode, tnode, {'time': cost}))
    G_curr = nx.Graph() # undirected graph
    G_curr.add_edges_from(edges)
    return G_curr


G_curr_new = create_lowstress_graph(G, [], 
                                    'lts_new', speed = 15, lts_threshold = 2)
G_curr_updated = create_lowstress_graph(G, [], 
                                    'lts_updated', speed = 15, lts_threshold = 2)

num_highstress_edges = 0
lts_column = 'lts_new'
for fnode, tnode in G.edges:
    if G[fnode][tnode][lts_column] > 2:
        num_highstress_edges += 1

len(G.edges)

len(network)

(network[lts_column] > 2).sum()

num_highstress_edges = 0
for fnode, tnode in G_curr_new.edges:
    if G_curr_new[fnode][tnode]['time'] == 100:
        num_highstress_edges += 1

num_highstress_edges

# These are reasonably close, so that's good.

# ## Get jobs per DA

da_job = pd.read_csv("../Cycling_data/accessibility/Employment_Toronto.csv")

# Bo's data (I think 2016 TTS)
da_pop_job = pd.read_csv("../ML4Bilevel/data/trt_da/pop_job_census.csv")

# ## Map DAs to a representative node
#
# Bo has a DA to node mapping from 2021. I don't have the code for how this is generated,
# but my understanding is he identified nodes close to DA centroids.
#
# Procedure:
# - get DA centroids
# - for each DA centroid, find the closest node that is not a bridge node, use this one for the representative node
#

da_boundaries = read_dataframe("../Cycling_data/accessibility/lda_000a21a_e/lda_000a21a_e.shp")

# subset DAs to Toronto
census_geog = pd.read_csv("../Cycling_data/accessibility/2021_98260004.csv")

# Toronto census geography labels: https://www150.statcan.gc.ca/n1/pub/92-195-x/2021001/other-autre/geograph/code-eng.htm
#
# Toronto's CSD ID is 3520005.

toronto_das = census_geog[census_geog['CSDDGUID_SDRIDUGD'].str.contains('3520005')]['DADGUID_ADIDUGD'].drop_duplicates()

assert np.all(toronto_das.str[-8:].isin(da_boundaries['DAUID']))

toronto_das = toronto_das.str[-8:]

toronto_das = da_boundaries[da_boundaries['DAUID'].isin(toronto_das)].to_crs(utm_crs)

# this is about the right number of DAs
toronto_das.shape

# don't use bridge nodes as the DA centroid nodes
exclude_nodes = list(bridge_nodes['node_id'])
nodes_df_da = nodes_df[~nodes_df['node_id'].isin(exclude_nodes)]

node_id = []
for i, da in tqdm(toronto_das.iterrows()):
    # argmin returns iloc index, not loc index
    node_id.append(nodes_df_da.iloc[nodes_df_da.distance(da['geometry'].centroid).argmin()]['node_id'])

toronto_das['node_id'] = node_id
toronto_das['node_id'] = toronto_das['node_id'].astype('int')

toronto_das.head()

# These are mostly the same as Bo's DAs, but some are different - could be because of changing DA boundaries / labels / changing node labels in updated centreline and/or updated DA boundaries for 2021.

# +
# plot to check
fig, ax = plt.subplots(figsize = (15,15))

toronto_das.boundary.plot(ax = ax, color = 'k', linewidth = 0.5)

nodes_df_da[nodes_df_da['node_id'].isin(node_id)].plot(ax = ax, color = 'r', markersize = 2)

ax.set_xticks([])
ax.set_yticks([])

# uses 2021 census data
plt.savefig("da_to_node_2021.pdf")
plt.savefig("da_to_node_2021.png", dpi = 200)
# -

# This seems reasonable

da_job['da_emp'] = da_job['da_emp'].astype('str')

# add job data
toronto_das = toronto_das.merge(da_job, left_on = 'DAUID', right_on = 'da_emp', how = 'left')

from networkx import single_source_dijkstra_path_length


# ### calculate travel time matrix for all OD pairs

def orig2des_length(orig, destinations, T, G, impedance = 'time'):
    """
    From Bo's code, get destination nodes reachable in time <= T from origin node `orig` using graph G. 

    Parameters:
    orig (int): origin node
    destinations (list[int]): list of destination nodes to check
    T (float): impedance cutoff (e.g. travel time limit in minutes)
    G: network graph
    impedance: edge weight to use for calculating impedance (must be in same units as T)

    Returns: 
    dict: destination nodes (keys) and impedance value to reach them, e.g. travel time (values)
    """
    if orig not in G.nodes():
        return {}
    reachable = single_source_dijkstra_path_length(G=G, source=orig, cutoff=T, weight=impedance)
    return {des: val for des, val in reachable.items() if des in destinations and des != orig}


def get_reachable_pairs(destinations, G, T=30):
    """
    Get OD pairs from the list of nodes in 'destinations' that are connected in the network G 
    under travel time limit T.

    Parameters:
    destinations (list[int]): list of node IDs to check for connectedness
    G (nx.Graph): network graph, nodes in `destinations` must be in G
    T (float): impedance cutoff for routing 
    """
    od_pairs_all = {}
    for node in tqdm(destinations):
        reachable = orig2des_length(node, destinations, T, G)
        if len(reachable) > 0:
            od_pairs_all[node] = reachable

    return od_pairs_all


destinations = list(toronto_das['node_id'])


def nested_keys_to_int(d):
    res = {}
    for orig, des in d.items():
        res[int(orig)] = {}
        for node, time in des.items():
            res[int(orig)][int(node)] = time
    return res


# connected OD pairs within T minutes of biking (regardless of stress level)
try:
    with open("od_pairs_all.json", "r") as od_file:
       od_pairs_all_str = json.load(od_file) # careful: reloads nodes as str
    od_pairs_all = nested_keys_to_int(od_pairs_all_str) # convert back to int
    print("loading connected OD pairs from file")
except:
    destinations = list(toronto_das['node_id'])
    od_pairs_all = get_reachable_pairs(destinations, G)
    with open("od_pairs_all.json", "w") as od_file:
        json.dump(od_pairs_all, od_file) 

    # travel time matrix style
    od_pairs_df = pd.DataFrame.from_dict(od_pairs_all, orient = 'index')
    od_pairs_df.to_csv("od_pairs_2021_DAs_all.csv", index = None)

# #### Diagnostics for OD pairs

od_pairs_df = pd.read_csv("od_pairs_2021_DAs_all.csv")

total_pairs = len(toronto_das)**2

np.sum(od_pairs_df.isna()) / total_pairs

destinations = list(toronto_das['node_id'])

np.sum(~od_pairs_df.isna())

# There are 2.4 million OD pairs connected within 30 minutes travel time. Bo's args_adj_ratio has 1.1 million pairs

from utils.functions import load_file

args, _ = load_file("prob/trt/args_adj_ratio.pkl")

od_pairs_Bo = args['destinations']

od_pairs_Bo_df = pd.DataFrame.from_dict(od_pairs_Bo, orient = 'index')

np.sum(~od_pairs_Bo_df.isna())

args_orig, _ = load_file("prob/trt/args.pkl")

len(od_pairs_Bo.keys())

od_pairs_Bo_df_orig = pd.DataFrame.from_dict(args_orig['destinations'], orient = 'index')

np.sum(~od_pairs_Bo_df_orig.isna())


# 1.3 million connected pairs, but Bo only has 3054 DAs instead of the full 3700. There was some removal process, can't remember what exactly. 

# ### Calculate accessibility

def cal_acc(G, od_pairs, potential, impedence='time', time_limit = 30, by_orig=False):
    """
    calculate the accessibility of the network

    Parameters:

    Returns:
    """
    # retrieve information

    acc = 0
    acc_by_orig = {}

    for orig in tqdm(od_pairs):
        orig_acc = 0
        lengths = single_source_dijkstra_path_length(G, source=orig, cutoff=time_limit, weight=impedence)
        reachable_des = [des for des in lengths if des in od_pairs[orig]]
        for des in reachable_des:
            acc += potential[des]
            orig_acc += potential[des]
        acc_by_orig[orig] = orig_acc
    if by_orig:
        return acc, acc_by_orig
    return acc


da2job = dict(zip(toronto_das['node_id'].astype('int'), toronto_das['employment'].fillna(0)))

acc = cal_acc(G, od_pairs_all, da2job)

acc

acc_new, acc_by_orig_new = cal_acc(G_curr_new, od_pairs_all, da2job, by_orig = True)

acc_new

acc_updated, acc_by_orig_updated = cal_acc(G_curr_updated, od_pairs_all, da2job, by_orig = True)

acc_updated

acc_new_df = pd.DataFrame.from_dict(acc_by_orig_new, orient = 'index').reset_index() 
acc_updated_df = pd.DataFrame.from_dict(acc_by_orig_updated, orient = 'index').reset_index() 

acc_new_df.columns = ['node_id', 'acc']
acc_updated_df.columns = ['node_id', 'acc']

acc_df = acc_new_df.merge(acc_updated_df, on = 'node_id', suffixes = ("_new", "_updated"))

np.corrcoef(acc_df['acc_updated'], acc_df['acc_new'])

from scipy.stats import pearsonr

pearsonr(acc_df['acc_updated'], acc_df['acc_new'])

# +
fig, ax = plt.subplots()

ax.scatter(acc_df['acc_updated'], acc_df['acc_new'], alpha = 0.3, s = 10)

ax.plot([0,500000], [0,500000], color = 'k', linestyle = '--')

ax.set_ylabel("Accessibility with new LTS")
ax.set_xlabel("Accessibility with updated LTS")
plt.tight_layout()
plt.savefig("new_vs_updated_lts_2025_maxspeed_{}.pdf".format(use_maxspeed))
plt.savefig("new_vs_updated_lts_2025_maxspeed_{}.png".format(use_maxspeed), dpi = 250)
# -

toronto_das = toronto_das.merge(acc_df, on = 'node_id')

toronto_das['acc_diff'] = toronto_das['acc_new'] - toronto_das['acc_updated']

toronto_das['acc_diff_frac'] = toronto_das['acc_diff'] / (toronto_das['acc_updated'] + 1)

toronto_das['acc_diff_norm'] = toronto_das['acc_diff'] / toronto_das['acc_updated'].max()

# +
fig, ax = plt.subplots(figsize = (12, 12))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.5)
toronto_das.plot(ax = ax, column = 'acc_diff_norm', cmap = 'BrBG', vmin = -0.5, vmax = 0.5, legend = True, cax = cax, 
                legend_kwds = {"label": "Accessibility difference: (new - updated) / max(updated)"})

ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig("accessibility_difference_2025_maxspeed_{}.pdf".format(use_maxspeed))
plt.savefig("accessibility_difference_2025_maxspeed_{}.png".format(use_maxspeed), dpi = 250)

# +
fig, ax = plt.subplots(figsize = (12, 12))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.5)
toronto_das.plot(ax = ax, column = 'acc_diff_frac', cmap = 'BrBG', vmin = -1, vmax = 1, legend = True, cax = cax, 
                legend_kwds = {"label": "Fractional accessibility difference: (new - updated) / updated"})

ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig("fractional_accessibility_difference_2025_maxspeed_{}.pdf".format(use_maxspeed))
plt.savefig("fractional_accessibility_difference_2025_maxspeed_{}.png".format(use_maxspeed), dpi = 250)
# -

lts_2025['lts_diff'] = lts_2025['lts_new'] - lts_2025['lts_updated']

stress_dict = {1:0, 2:0, 3:1, 4:1}

lts_2025['stress_updated'] = lts_2025['lts_updated'].map(stress_dict)
lts_2025['stress_new'] = lts_2025['lts_new'].map(stress_dict)

lts_2025['stress_diff'] = lts_2025['stress_new'] - lts_2025['stress_updated']

lts_2025_arterial = lts_2025[lts_2025['feature_code_desc'].isin(['Major Arterial', 'Minor Arterial'])]

# +
fig, ax = plt.subplots(figsize = (15, 15))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.5)


toronto_das.plot(ax = ax, column = 'acc_diff_frac', alpha = 0.2, cmap = 'RdBu', vmin = -1, vmax = 1)


lts_2025.plot(ax = ax, linewidth = 0.5, column = 'stress_diff', cmap = 'RdBu_r', vmin = -1.5, vmax = 1.5, legend = True, cax=cax,
    legend_kwds={"label": "Stress difference (new - updated)"})

lts_2025_arterial.plot(ax = ax, linewidth = 1, column = 'stress_diff', cmap = 'RdBu_r', vmin = -1.5, vmax = 1.5)

ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

plt.savefig("lts_2025_stress_diff_maxspeed_{}.pdf".format(use_maxspeed))
plt.savefig("lts_2025_stress_diff_maxspeed_{}.png".format(use_maxspeed), dpi = 250)
# -

# Red segments means new is high stress and updated is low-stress; blue means new is low-stress and updated is high stress
#
# Red shading means new accessibility is *lower* than the updated version accessibility.

toronto_das['acc_new_plot'] = toronto_das['acc_new'] + 1
toronto_das['acc_updated_plot'] = toronto_das['acc_updated'] + 1

# +
fig, axs = plt.subplots(1,2, figsize = (16,8))

divider = make_axes_locatable(axs[0])
cax1 = divider.append_axes("right", size="5%", pad=0.5)

divider = make_axes_locatable(axs[1])
cax2 = divider.append_axes("right", size="5%", pad=0.5)

toronto_das.plot(ax = axs[0], column = 'acc_new_plot', norm=matplotlib.colors.LogNorm(vmin=1, vmax=6*10**5), legend = True, cax = cax1,
                legend_kwds = {"label":"Accessibility to jobs"})
toronto_das.plot(ax = axs[1], column = 'acc_updated_plot', norm=matplotlib.colors.LogNorm(vmin=1, vmax=6*10**5), legend = True, cax = cax2,
                legend_kwds = {"label":"Accessibility to jobs"})

axs[0].set_title("2025 LTS new (automated)")
axs[1].set_title("2025 LTS updated from Bo's 2021 network")

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("2025_accessibility_comparison_new_vs_updated_maxspeed_{}.pdf".format(use_maxspeed))
plt.savefig("2025_accessibility_comparison_new_vs_updated_maxspeed_{}.png".format(use_maxspeed), dpi = 250)
# -

# ## Checking individual segments

# The section of Bayview that is low-stress in the new one and seems to be bumping up accessibility there - that section has a painted bike lane (though it's 4 lanes and looks scary),
# maybe the gtamodel speed is low there?
#
# I think it's okay - posted speed limit is 50, GTAModel speed is also close to 50. Leave this one alone. Bike lane with speed < 56 and 4 lanes is LTS 2. 

lts_2025[lts_2025['linear_name_full'].str.contains('Bayview Ave')].cx[6.3e5:6.32e5,4.8423e6:4.844e6].plot(column = 'lts_new')

lts_2025[lts_2025['linear_name_full'].str.contains('Bayview Ave')].cx[6.3e5:6.32e5,4.8423e6:4.844e6][[
      'linear_name_full', 'infra_high', 'infra_lowo',
       'lts_updated', 'lts_new', 'speed_limit', 'speed_assumed', 'lanes_assumed']]

lts_recalc.columns

lts_recalc[lts_recalc['linear_name_full'].str.contains('Bayview Ave')].cx[6.3e5:6.32e5,4.8423e6:4.844e6][['speed_limit',
       'number_of_lanes_city', 'infra_high', 'infra_lowo', 'with_parki',
       'gtaspeed_1', 'speed_am', 'speed_md', 'speed_pm', 'lts',
       'speed_assumed', 'lanes_assumed', 'rule', 'message']]

# St. George and Davenport - both have a painted bike lane and some sections are high-stress in the new method. Should these be low stress?

# Eglinton between Bayview and Laird - should probably be high-stress? This is coming up as low stress because it is recorded (wrongly) as having 2 lanes.

lts_2025[lts_2025['linear_name_full'].str.contains('Eglinton Ave E')].cx[6.3e5:6.32e5,4.841e6:4.844e6].plot(column = 'lts_new')

lts_2025.columns

lts_2025[lts_2025['linear_name_full'].str.contains('Eglinton Ave E')].cx[6.3e5:6.32e5,4.841e6:4.844e6][['infra_high', 'infra_lowo',
       'lts_updated', 'lts_new', 'speed_limit', 'speed_assumed', 'lanes_assumed']]


