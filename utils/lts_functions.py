import numpy as np
import pandas as pd

# these cutoffs match what is in the LTS_tables.docx file

def is_separated(gdf, feature_col, infra_col):
    """
    Assign LTS 1 to feature codes 'Trail' and 'Walkway'
    Assign LTS 1 to separated cycling infrastructure
    """

    conditions = [(gdf[feature_col] == 'Trail'), #s1
                  (gdf[feature_col] == 'Walkway'), #s2
                  (gdf[infra_col].isin(['Cycle Track', 'Bi-Directional Cycle Track',
                                                   'Cycle Track - Contraflow'])), #s3
                  (gdf[infra_col].isin(['Multi-Use Trail', 'Multi-Use Trail - Boulevard', 'Multi-Use Trail - Entrance',
                                                   'Park Road']))] #s4

    values = ['s1', 's2', 's3', 's4']

    gdf['rule'] = np.select(conditions, values, default = "")

    rule_dict = {'s1':1, 's2':1, 's3':1, 's4':1}

    gdf['lts'] = gdf['rule'].map(rule_dict)

    return gdf


def bike_lane_with_parking(gdf, lane_column, speed_column):
    """
    If segments have bike lanes and parking
    Caution: this select procedure will overwrite whatever dataframe you put in,
    Only run this on the subset with parking and bike lanes.
    """

    conditions = [
        (gdf[lane_column] <= 2) & (gdf[speed_column] <= 40),
        (gdf[lane_column] <= 2) & (gdf[speed_column] <= 50),
        (gdf[speed_column] <= 56), # any number of lanes
        (gdf[lane_column].isna()) | (gdf[speed_column].isna())]

    # lts values for each condition

    values = ['b1', 'b2', 'b3', 'b5']

    rule_dict = {'b1':1, 'b2':2, 'b3':3, 'b4':4, 'b5':np.nan}
    
    gdf['rule'] = np.select(conditions, values, default="b4")

    gdf['lts'] = gdf['rule'].map(rule_dict)

    return gdf


def bike_lane_without_parking(gdf, lane_column, speed_column):
    """
    If segments have bike lanes and no parking
    Caution: this select procedure will overwrite whatever dataframe you put in,
    Only run this on the subset with no parking and bike lanes.
    """

    conditions = [
        (gdf[lane_column] <= 2) & (gdf[speed_column] <= 50),
        (gdf[lane_column] <= 4) & (gdf[speed_column] <= 56),
        (gdf[speed_column] <= 56),
        (gdf[lane_column].isna()) | (gdf[speed_column].isna())]

    values = ['c1', 'c2', 'c3', 'c5']

    rule_dict = {'c1':1, 'c2':2, 'c3':3, 'c4':4, 'c5':np.nan}
    
    gdf['rule'] = np.select(conditions, values, default="c4")

    gdf['lts'] = gdf['rule'].map(rule_dict)

    return gdf


def mixed_traffic(gdf, lane_column, speed_column, road_class_col):
    """
    If segments have no bike infrastructure
    Caution: this select procedure will overwrite whatever dataframe you put in,
    Only run this on the subset of interest.
    """
    conditions = [
        (gdf[lane_column] <= 3) & (gdf[speed_column] <= 40) & (gdf[road_class_col].isin(['Local', 'Pending', 'Other', 'Access Road', 'Laneway'])),
        (gdf[lane_column] <= 3) & (gdf[speed_column] <= 40), # not in those road classes                                                       
        (gdf[lane_column] <= 3) & (gdf[speed_column] <= 50) & (gdf[road_class_col].isin(['Local', 'Pending', 'Other', 'Access Road', 'Laneway'])),
        (gdf[lane_column] <= 3) & (gdf[speed_column] <= 50), 
        (gdf[lane_column] <= 5) & (gdf[speed_column] <= 40),
        (gdf[lane_column].isna()) | (gdf[speed_column].isna())]

    # lts values for each condition

    values = ['m1', 'm2', 'm3', 'm4', 'm5', 'm7']

    rule_dict = {'m1':1, 'm2':2, 'm3':2, 'm4':3, 'm5':3, 'm6':4, 'm7':np.nan}

    gdf['rule'] = np.select(conditions, values, default="m6")

    gdf['lts'] = gdf['rule'].map(rule_dict)
    
    return gdf

def lts_colour(df):
    """
    Creates a column called 'colour' which maps lts values to a colour.
    It also makes all columns lowercase, maybe undesirable, will see
    """
    
    # lowercase all columns
    df.columns = df.columns.str.lower()
    
    # define lts colours for plotting
    conditions = [
        (df['lts'] == 1),
        (df['lts'] == 2),
        (df['lts'] == 3),
        (df['lts'] == 4),
        ]

    # create a list of the values we want to assign for each condition
    values = ['g', 'b', 'y', 'r']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['colour'] = np.select(conditions, values, default = 'k')
    
    return df

def stress_colour(df):
    """
    Creates a column called 'colour' which maps lts values to a high/low stress colour.
    It also makes all columns lowercase, maybe undesirable, will see
    """
    
    # lowercase all columns
    df.columns = df.columns.str.lower()
    
    # define lts colours for plotting
    conditions = [
        (df['lts'] == 1),
        (df['lts'] == 2),
        (df['lts'] == 3),
        (df['lts'] == 4),
        ]

    # create a list of the values we want to assign for each condition
    values = ['g', 'g', 'orangered', 'orangered']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['stress_colour'] = np.select(conditions, values)
    
    return df