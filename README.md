# Calculate LTS from city data

This repo contains code and notes for calculating Level of Traffic Stress with city data.
So far, this has only been implemented for the City of Toronto, and there are hard-coded
variables (in particular, file, dataset, and column names) that will need to be adjusted
for using with other cities.

## Usage

To calculate LTS for the City of Toronto, run these scripts:
1. `create_network.py` - aligns input data to the centreline network to create a harmonized geospatial file.
2. `assign_lts.py` - use the [LTS decision tree](#lts-tables) to calculate LTS for the network created in step 1.
3. `compare_2025_lts.py` - plot LTS values on a map, calculate accessibility, compare results of updating network by hand from Bo's 2021 network and regenerating LTS from scratch programmatically.

## Data sources

Most data used in this calculation is publicly available government datasets; some was provided privately. Data sources and links are described in the following table.

| Data | Source | Time point retroactivity | Notes|
| --- | --- | --- | --- |
| Centreline network | [Toronto Open Data](https://open.toronto.ca/dataset/toronto-centreline-tcl/) | No | File is updated gradually over time, download at desired time point |
| Cycling network file | Received from City | Yes | This script uses a version aligned with the centreline network, but the cycling network is also publicly available on [Toronto Open Data](https://open.toronto.ca/dataset/cycling-network/) |
| Speed limit | [Government of Ontario](https://geohub.lio.gov.on.ca/datasets/mnrf::ontario-road-network-orn-road-net-element/about)  | No | Speed from GTAModel is used where available |
| Number of lanes | Received from City | No | Provided privately |
| GTAModel speed and volume | GTAModel contacts (via Bo, 2021), James Vaughan (2026) | Yes | We have Bo's 2021 matched version and the raw data from James. Both use 2016 modeled speed and volume, since this is the most accurately calibrated network |
| Street parking | Toronto Parking Authority (via Bo) | Unknown | This info is in the 2021 LTS file shared by Bo until updated |
| Previous LTS version(s) | Bo | NA | Optional, used for testing and comparison |

## LTS tables

These tables summarize the LTS criteria we use for calculating LTS based on city data.
They are based on [Furth et al. 2016](https://doi.org/10.3141/2587-06) with some modifications to make thresholds 
more clear. These changes are described in footnotes. 

#### Separated – cycle track, contraflow cycle track, trail, path, or park road: LTS 1

#### Bike lane with on-street parking (more stressful):

| Number of lanes (both directions) | Speed limit or modelled speed (km/h) | LTS |
| ---- | ---- | ---- |
|<= 2 |	speed <= 40 | 	1 |
|<= 2 |	40 < speed <= 50[^1] |	2 |
|Any  | 50 < speed <= 56[^2] |	 3 |
|Any  | speed > 56[^2] | 4  |

#### Bike lane without on-street parking:

| Number of lanes (both directions) | Speed limit or modelled speed (km/h) | LTS |
| ---- | ---- | ---- |
| <= 2 |	speed <= 50[^1] |	1
| <= 4 | 50 < speed <= 56 |	2
| > 4 | 50 < speed <= 56[^2] |	 3
|Any | speed > 56[^2] |	4


[^1]: in Furth et al. 2016, this value is 30 mph (48 km/h), which we have rounded to 50 km/h to match typical Canadian speed limits.
[^2]: in Furth et al. 2016, LTS 3 is 35 mph (56 km/h), but LTS 4 is listed as “40 mph or more” (64 km/h). It is ambiguous whether the line between LTS 3 and LTS 4 should be 35 or 40 mph.

#### Mixed traffic:

| Number of lanes (both directions) | Speed limit or modelled speed (km/h) | 	ADT |	LTS |
| ---- | ---- | ---- | ---- |
| <= 3[^3] |	speed <= 40  |	<= 3000[^4] |	1 |
| <= 3	 |speed <= 40 |	>3000 |	2 |
| <= 3 |	40 < speed <= 50 |	<= 3000[^4] |	2 |
| <= 3	| 40 < speed <= 50 |	>3000 |	3 |
| 3 < lanes <= 5	| speed <= 40 |	Any |	3 |
|Any	| speed > 50 |	Any |	4 |
| > 5|	Any |	Any |	4 |

[^3]: in Furth the lane values are 2, 4, 6, etc. but here we choose (3, 5) to be the midpoint since the cutoffs are ambiguous.
[^4]: Based on City of Toronto road classifications, I consider the road classes “Local”, “Pending”, “Other”, and “Access Road” to meet this ADT threshold. Local roads should have an ADT <= 2500. 



