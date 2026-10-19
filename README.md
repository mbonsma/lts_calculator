\# Calculate LTS from city data



This repo contains code and notes for calculating Level of Traffic Stress with city data.

So far, this has only been implemented for the City of Toronto, and there are hard-coded

variables (in particular, file, dataset, and column names) that will need to be adjusted

for using with other cities.







\## LTS tables – Toronto analysis



These tables summarize the LTS criteria we use for calculating LTS based on city data.

They are based on \[Furth et al. 2016]() with some modifications to make thresholds 

more clear. These changes are described in footnotes. 



\#### Separated – cycle track, contraflow cycle track, trail, path, or park road: LTS 1





\#### Bike lane with on-street parking (more stressful):



| Number of lanes (both directions) | Speed limit or modelled speed (km/h) | LTS |

| ---- | --- | -- |

|<= 2 |	speed <= 40 | 	1 |

|<= 2 |	40 < speed <= 50\[^1] |	2 |

|Any  | 50 < speed <= 56\[^2] |	 3 |

|Any  | speed > 56\[^2] | 4  |



\#### Bike lane without on-street parking:



| Number of lanes (both directions) | Speed limit or modelled speed (km/h) | LTS |

| ---- | --- | -- |

|<= 2 |	speed <= 50\[^1] |	1

|<= 4 | 50 < speed <= 56 |	2

|> 4 | 50 < speed <= 56\[^2] |	 3

|Any | speed > 56\[^2] |	4



\[^1] Note: in Furth et al. 2016, this value is 30 mph (48 km/h), which we have rounded to 50 km/h to match typical Canadian speed limits.

\[^2] Note: in Furth et al. 2016, LTS 3 is 35 mph (56 km/h), but LTS 4 is listed as “40 mph or more” (64 km/h). It is ambiguous whether the line between LTS 3 and LTS 4 should be 35 or 40 mph.



\#### Mixed traffic:



| Number of lanes (both directions) | Speed limit or modelled speed (km/h) | 	ADT |	LTS |

| ---- | --- | ---- | ---- |

|<= 3\[^3] |	speed <= 40  |	<= 3000\[^4] |	1 |

|<= 3	 |speed <= 40 |	>3000 |	2 |

|<= 3 |	40 < speed <= 50 |	<= 3000\[^4] |	2 |

|<= 3	| 40 < speed <= 50 |	>3000 |	3 |

|3 < lanes <= 5	| speed <= 40 |	Any |	3 |

|Any	| speed > 50 |	Any |	4 |

|> 5|	Any |	Any |	4 |



\[^3]Note: in Furth the lane values are 2, 4, 6, etc. but here we choose (3, 5) to be the midpoint since the cutoffs are ambiguous.

\[^4]Based on City of Toronto road classifications, I consider the road classes “Local”, “Pending”, “Other”, and “Access Road” to meet this ADT threshold. Local roads should have an ADT <= 2500. 



