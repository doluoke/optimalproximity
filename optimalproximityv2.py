# Enter your script here.


#%%capture
#!pip install contextily==0.99.0

#--------------------------------------------------------------
import civis
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import civis.io
import zipfile
import geopandas as gpd
from shapely.geometry import Point,multipoint
from shapely.ops import nearest_points
import itertools
from scipy import stats
#import contextily as ctx
from fiona.crs import from_epsg
import warnings
warnings.filterwarnings('ignore')

import matplotlib
#%matplotlib inline

#--------------------------------------------------------------
print('This code started')
#table = os.environ["table_01"]
#table2 = os.environ["table_02"]
#print('This is table 01:' , table)
#print('This is table 02:' ,table2)
#print('This type table 01:' ,type(table))
#print('This type table 02:' ,type(table2))
#--------------------------------------------------------------

# record level data 

unmatched_query= '''

    
    SELECT
     userdefinedfltyid,
      latitude as latitude_ref,
      longitude as longitude_ref,serious_damage_bucket__hazus,flood_zone_type__hazus,
      neighborhood__hazus
    FROM ''' + '''fed.task6_civis_estimate_serious_damage_leftjoin_hud__demographics''' + ''' as ia
    WHERE data_boolean__hud=0
    
    ORDER BY userdefinedfltyid

'''

unmatched_query_table = civis.io.read_civis_sql(
    unmatched_query,"City of Houston",use_pandas=True
)


print(unmatched_query)


#--------------------------------------------------------------
# record level data 

highwater_query= '''
SELECT
     
        objectid_1,
        lat as latitude,
        long as longitude


FROM ''' + '''dev.high_water_rescue''' + ''' as h

   
ORDER BY h.objectid_1

'''
print('This code completed')
highwater_table = civis.io.read_civis_sql(
    highwater_query,"City of Houston",use_pandas=True
)


print(highwater_query)
#You may uncomment this
#highwater_table.head()
