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

# # Assign LTS
#
# - Use decision tree chart, functions modified from https://github.com/mbonsma/LTS-OSM/blob/main/lts_functions.py
# - Include rule statements to help with interpretation and debugging

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pyogrio import read_dataframe # faster than geopandas

# decision rule glossary based on network attributes
# these are modified from Bike Ottawa's stressmodel code (https://github.com/BikeOttawa/stressmodel)
# and from https://github.com/mbonsma/LTS-OSM
rule_message_dict = {'s1':'This way is a separated path because it has the centreline feature code Trail.',
                     's2':'This way is a separated path because it has the centreline feature code Walkway.',
                     's3':'This way is a separated path because it has a cycle track or contraflow bike lane (cycling network file).',
                     's4':'This way is a separated path because it is a multi-use trail or park road (cycling network file).',

                     'b1':'LTS is 1 because there is parking present, the speed is less than or equal to 40 and there are 2 lanes or less.',
                     'b2':'Increasing LTS to 2 because the speed is between 41-50 km/h, there are 2 lanes or less, and parking present.',  
                     'b3':'Increasing LTS to 3 because the speed is between 51-56 km/h and parking present.',  
                     'b4':'Increasing LTS to 4 because speed is > 56 km/h.',
                     'b5':'Not assigning LTS yet because speed or lanes info is missing.',
                  
                     'c1':'LTS is 1 because there is no parking, the speed is less than or equal to 50 and there are 2 lanes or less.',
                     'c2':'Increasing LTS to 2 because there 3 or 4 lanes, speed <= 56, and no parking.',
                     'c3':'Increasing LTS to 3 because the speed is <= 56 km/h, > 4 lanes, and no parking.',
                     'c4':'Increasing LTS to 4 because speed is > 56 km/h.',
                     'c5':'Not assigning LTS yet because speed or lanes info is missing.',

                     'm1':'LTS is 1 because there are 3 or fewer lanes, speed is <= 40, and road class is either Local, Pending, Other, or Access Road.',
                     'm2':'Increasing LTS to 2 because there are 3 or fewer lanes, speed is <= 40, and road class is not Local, Pending, Other, or Access Road (higher vehicle volume).',
                     'm3':'Increasing LTS to 2 because there are 3 or fewer lanes, speed is <= 50, and road class is either Local, Pending, Other, or Access Road.',
                     'm4':'Increasing LTS to 3 because there are 3 or fewer lanes, speed is <= 50, and road class is not Local, Pending, Other, or Access Road (higher vehicle volume).',
                     'm5':'Increasing LTS to 3 because there are 4-5 lanes and speed is <= 40.',
                     'm6':'Increasing LTS to 4 because speed is higher than 50 km/h or there are more than 5 lanes.',
                     'm7':'Not assigning LTS yet because speed or lanes info is missing.',
                    
                     'o1':'LTS 1 assigned to remaining NaNs that are Other or Laneway.',
                     'r1':'Highest LTS of same-named streets assigned to remaining NaNs.'}

simplified_message_dict = {'s3':r'highway $=$ "cycleway"',
                     's1':r'highway $=$" path"', 
                     's2':r'separated, highway $=$" footway", not a crossing', 
                     's7':r'cycleway* $=$ "track"', 
                     's8':r'cycleway* $=$ "opposite_track"', 
                     'b1':r'bike lane w/ parking, speed $\leq$ 40 km/h, $\leq$ 2 lanes',
                     'b2':r'bike lane w/ parking, speed 41-50 km/h, $\leq 2 lanes', 
                     'b3':r'bike lane w/ parking, speed 51-56 km/h', 
                     'b4':r'bike lane w/ parking, speed $>$ 56 km/h',  
                     'b5':r'Speed or lane info is missing',
                     'c1':r'bike lane no parking, $\leq$ 50 km/h, $\leq$ 2 lanes',
                     'c2':r'bike lane no parking, $\leq$ 56 km/h, 3-4 lanes',
                     'c3':r'bike lane no parking, $\leq$ 56 km/h, $>$ 4 lanes',
                     'c4':r'bike lane no parking, $>$ 56 km/h',
                     'c5':r'Speed or lane info is missing',
                     'm1':r'mixed traffic, speed $\leq$ 40 km/h, $\leq$ 3 lanes, local road', 
                     'm2':r'mixed traffic, speed $\leq$ 40 km/h, $\leq$ 3 lanes, not local road', 
                     'm3':r'mixed traffic, speed 41-50 km/h, $\leq$ 3 lanes, local road', 
                     'm4':r'mixed traffic, speed 41-50 km/h, $\leq$ 3 lanes, not local road', 
                     'm5':r'mixed traffic, speed $\leq$ 40 km/h, 4-5 lanes',
                     'm6':r'mixed traffic, speed $>$ 50 km/h or $>$ 5 lanes',
                     'm7':r'Speed or lane info is missing',
                     'o1':'LTS 1 for "Other" or "Laneway" with missing info.',
                     'r1':'Highest LTS of same-named streets with missing info.'} 

# Road class / speed / number of lanes:
# - local: low speeds and under 2500 vehicles daily volume and are typically found in residential areas.
# - Collector roads, which feature medium travel speeds and support traffic movement for between 2500 and 8000 vehicles daily between local roads and arterial roads.
# - Arterial roads, which feature traffic speeds of at least 40 km/h, daily traffic volumes of over 8000 vehicles, and exist primarily to move traffic across longer distances.
#
# According to [this road classification document](https://www.toronto.ca/wp-content/uploads/2018/01/950a-Road-Classification_Summary-Document.pdf), "major arterial" roads have at minimum 4 peak hour lanes. 

# ## Functions

from utils.lts_functions import is_separated, bike_lane_with_parking, bike_lane_without_parking, mixed_traffic

# ## Set options

use_max_speed = False # if true, use the max of speed limit and GTAModel speeds. If False, use GTAModel speeds where available.

# ## Load data

centreline = read_dataframe("data/centreline_network_with_attributes_2025-02.geojson")

centreline['feature_code_desc'].value_counts()

centreline[centreline['centreline_id'] == 30023164].plot(column = 'Final_Spee', alpha = 0.5)

centreline['centreline_id'].value_counts()

# ## Generate network

lts = centreline.copy()
lts['lts'] = np.nan

# ### Calculate a final speed based on modeled speed and posted speed limits
#
# Bo used GTA Model speeds where available. I first tried using the max of GTAModel speed and posted speed, but this gave a very great quantity of LTS 4 (due to 50 km/h speed limits on residential roads in the suburbs and the cutoff being 48 km/h at the time). 

# only one of these speeds is ever > 0
np.max(np.sum(lts[['GTASpeed_1', 'GTASpeed_2', 'GTASPeed_3']] > 0, axis = 1))

(lts[['GTASPeed_3']] > 0).sum()

(lts[['GTASpeed_2']] > 0).sum()

# +
lts['gta_speed'] = np.nanmax(lts[['GTASpeed_1', 'GTASpeed_2', 'GTASPeed_3']], axis = 1)
lts['gta_speed'] = lts['gta_speed'].replace(0, np.nan) # replace 0 gta speed with NaN
lts['final_speed'] = lts['gta_speed']
lts['final_speed'] = lts['final_speed'].fillna(lts['speed_limit'])
lts['final_speed_max'] = np.nanmax(lts[['speed_limit', 'gta_speed']], axis = 1)

if use_max_speed == True:
    lts['speed_assumed'] = lts['final_speed_max']
else:
    lts['speed_assumed'] = lts['final_speed']
# -

# there are some cases where the speed limit has been lowered since 2021
lts[(lts['Final_Spee'] > lts['speed_limit'])
     & (lts['gta_speed'].isna())][['linear_name_full', 'speed_limit', 'gta_speed', 'Final_Spee', 'final_speed']]

# use city lane info
lts['lanes_assumed'] = lts['number_of_lanes_city']

# ## Apply LTS calculations

lts = is_separated(lts, 'feature_code_desc', 'INFRA_HIGH')

# +
# get subset with no LTS, bike lanes, and parking
bl_with_parking = lts.loc[(lts['lts'].isna())
                & (lts['INFRA_HIGH'].isin(['Bike Lane', 'Bike Lane - Contraflow', 'Bike Lane - Buffered']))
                & (lts['with_parki'] == 1)]

bl_with_parking = bike_lane_with_parking(bl_with_parking, 'lanes_assumed', 'speed_assumed')

# update lts
lts.loc[bl_with_parking.index, 'lts'] = bl_with_parking['lts']
lts.loc[bl_with_parking.index, 'rule'] = bl_with_parking['rule']

# +
#subset with no LTS, bike lanes, and no parking
bl_without_parking = lts.loc[(lts['lts'].isna())
                & (lts['INFRA_HIGH'].isin(['Bike Lane', 'Bike Lane - Contraflow', 'Bike Lane - Buffered']))
                & (lts['with_parki'] == 0)]

bl_without_parking = bike_lane_without_parking(bl_without_parking, 'lanes_assumed', 'speed_assumed')

# update lts
lts.loc[bl_without_parking.index, 'lts'] = bl_without_parking['lts']
lts.loc[bl_without_parking.index, 'rule'] = bl_without_parking['rule']

# +
# for any segments since 2021 that don't have parking specified, assume they have parking (higher standard)
bl_with_parking = lts.loc[(lts['lts'].isna())
                & (lts['INFRA_HIGH'].isin(['Bike Lane', 'Bike Lane - Contraflow', 'Bike Lane - Buffered']))]

print(bl_with_parking.shape[0])

bl_with_parking = bike_lane_with_parking(bl_with_parking, 'lanes_assumed', 'speed_assumed')

# update lts
lts.loc[bl_with_parking.index, 'lts'] = bl_with_parking['lts']
lts.loc[bl_with_parking.index, 'rule'] = bl_with_parking['rule']
# -

lts[lts['lts'].isna()]['feature_code_desc'].value_counts()

lts['lts'].value_counts()

# +
mixed = lts[(lts['lts'].isna())]

mixed = mixed_traffic(mixed, 'lanes_assumed', 'speed_assumed', 'feature_code_desc')

# update lts
lts.loc[mixed.index, 'lts'] = mixed['lts']
lts.loc[mixed.index, 'rule'] = mixed['rule']
# -

lts['lts'].value_counts()

# ### Assign any missing by road class

lts[lts['lts'].isna()][['linear_name_full', 'number_of_lanes_city', 'speed_limit', 'final_speed', 'gta_speed', 'Final_Spee', 'feature_code_desc']]

# +
# assign LTS 1 to 'other' by inspection

lts.loc[(lts['lts'].isna())
    & (lts['feature_code_desc'].isin(['Other', 'Laneway'])), 'rule'] = 'o1'

lts.loc[(lts['lts'].isna())
    & (lts['feature_code_desc'].isin(['Other', 'Laneway'])), 'lts'] = 1


# -

# Attempt to assign remainder based on neighbouring segments

for street in lts[lts['lts'].isna()].drop_duplicates(subset = 'linear_name_full')['linear_name_full']:
    # look at the same street's already-labeled sections
    possible_lts_vals = lts[(lts['linear_name_full'] == street)]['lts'].unique()
    possible_lts_vals = possible_lts_vals[~np.isnan(possible_lts_vals)]

    if len(possible_lts_vals) >= 1: 
        print(street, possible_lts_vals)
        # assign the max
        lts.loc[(lts['lts'].isna())
                & (lts['linear_name_full'] == street), 'rule'] = 'r1'
        lts.loc[(lts['lts'].isna())
                & (lts['linear_name_full'] == street), 'lts'] = max(possible_lts_vals)


lts.loc[lts['lts'].isna()]

# assign LTS 4 to remaining
lts.loc[lts['lts'].isna(), 'lts'] = 4

assert lts['lts'].isna().sum() == 0

lts['rule'].value_counts()

lts['message'] = lts['rule'].map(rule_message_dict)

keep_columns = ['centreline_id', 'linear_name_full',
       'linear_name_full_legal',
       'linear_name_desc', 'linear_name_label', 'from_intersection_id',
       'to_intersection_id', 'oneway_dir_code', 'oneway_dir_code_desc',
       'feature_code', 'feature_code_desc',  'speed_limit',
       'number_of_lanes_city', 'INFRA_HIGH', 'INFRA_LOWO',
       'with_parki', 'GTASpeed_1', 'GTASpeed_2', 'GTASPeed_3', 'Final_Spee',
       'geometry', 'lts', 'gta_speed', 'final_speed', 'final_speed_max', 'speed_assumed', 'lanes_assumed', 'rule', 'message']

date="2025-02"
lts[keep_columns].drop_duplicates().to_file("data/lts_{}_maxspeed_{}.geojson".format(date, 
                                                                use_max_speed))

# ## Tests

assert np.all(lts.loc[lts['feature_code_desc'].isin(['Trail', 'Walkway']), 'lts'] == 1)

assert np.all(lts.loc[lts['INFRA_HIGH'].isin(['Cycle Track', 'Bi-Directional Cycle Track',
                                                   'Cycle Track - Contraflow', 'Multi-Use Trail', 'Multi-Use Trail - Boulevard', 'Multi-Use Trail - Entrance',
                                                   'Park Road']), 'lts'] == 1)

lts.loc[lts['final_speed'] <= 56, 'lts'].value_counts()

lts.loc[lts['final_speed'] > 56, 'lts'].value_counts()

lts['final_speed'].isna().sum()

# only trails etc. should have missing speeds
lts[lts['final_speed'].isna()]['feature_code_desc'].value_counts()


