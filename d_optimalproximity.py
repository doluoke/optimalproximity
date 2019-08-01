# Enter your script here.


#%%capture
#!pip install contextily==0.99.0

#--------------------------------------------------------------
import civis
import os
import sys
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
import time
import random


import matplotlib
#%matplotlib inline
# Function Declaration

def convert_wgs_to_utm(lon, lat):
    utm_band = str(int((np.floor((lon + 180) / 6 )  + 1)))
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return int(epsg_code)

#--------------------------------------------------------------
print('This code started')
# Record Start Time
then2 = time.time() #Time before the operations start

#conn = create_engine('redshift+psycopg2://dokeowo@host.amazonaws.com:5439/database')
#print('This is 0 after python', sys.argv[0], 'and type is:', type(sys.argv[0]))
#print('This is 1 after python', sys.argv[1], 'and type is:', type(int(sys.argv[1])))
#print('This is 2 after python', sys.argv[2], 'and type is:', type(float(sys.argv[2])))
#print('This is 3 after python', sys.argv[3], 'and type is:', type(sys.argv[3]))
#print('This is 3 after python', sys.argv[4], 'and type is:', type(sys.argv[4]))

#table_01 = '''fed.task6_civis_estimate_serious_damage_leftjoin_hud__demographics'''
table_01     = sys.argv[1]
longitude_01 = sys.argv[2]
latitude_01  = sys.argv[3]
#ref_01       = int(sys.argv[4])
indkey_01    = sys.argv[4]

table_02     = sys.argv[5]
longitude_02 = sys.argv[6]
latitude_02  = sys.argv[7]
#ref_02       = int(sys.argv[9])
indkey_02    = sys.argv[8]

#ref_prj = int(sys.argv[11])
table_out = sys.argv[9]
database = sys.argv[10]

# This assumes that the coordinates in table 01 and table_02 are referenced to WGS84 geographic coordinate system (Lat & Long)
ref_01 = 4326
ref_02 = ref_01

print(table_01)
print(longitude_01)
print(latitude_01)
print(ref_01)
print(indkey_01)
print(table_02)
print(longitude_02)
print(latitude_02)
print(ref_02)
print(indkey_02)
#print(ref_prj)
print(table_out)
print(database)


#table_02 = '''dev.high_water_rescue'''
# ref_01 = 4326
# ref_02 = 4326
#ref_prj = 32615
#table = os.environ["table_01"]
#table2 = os.environ["table_02"]
#print('This is table 01:' , table)
#print('This is table 02:' ,table2)
#print('This type table 01:' ,type(table))
#print('This type table 02:' ,type(table2))
#--------------------------------------------------------------

# record level data 

table01_query= ''' SELECT ''' +  indkey_01 + ''', ''' + latitude_01 + ''' as latitude_ref,''' + longitude_01 + ''' as longitude_ref FROM ''' + table_01 + ''' as ia ORDER BY ''' + indkey_01


table01_query_table = civis.io.read_civis_sql(
    table01_query,database,use_pandas=True
)


#print(table01_query)

# 1b) Convert the imported table01 table to GeoDataFrame
table01_query_geo = gpd.GeoDataFrame(table01_query_table)
table01_query_geo['point'] = table01_query_geo.apply(lambda x: Point(x['longitude_ref'],x['latitude_ref']), axis = 1)
table01_query_geo['geometry'] = table01_query_geo.point

# Compute the mean position to determine the EPSG 
mean_long_ref = np.mean(table01_query_geo['longitude_ref'])
mean_lat_ref = np.mean(table01_query_geo['latitude_ref'])

ref_prj = convert_wgs_to_utm(mean_long_ref,mean_lat_ref)

# 1c) Convert umatched point -table01_query_geo from Geographic to UTM Zone 15N
table01_query_geo_crs = table01_query_geo
# Assign WGS 84 Coordinate System
table01_query_geo_crs.crs = from_epsg(ref_01)
# Project from WGS to Projected Coordinate System
table01_query_geo_prj = table01_query_geo_crs.to_crs(epsg=ref_prj)



#--------------------------------------------------------------
# record level data 


table02_query= ''' SELECT ''' +  indkey_02 + ''', ''' + latitude_02 + ''' as latitude,''' + longitude_02 + ''' as longitude FROM ''' + table_02 + ''' as h ORDER BY ''' + indkey_02

print('This code completed')
table02_table = civis.io.read_civis_sql(
    table02_query,database,use_pandas=True
)


print(table02_query)
#You may uncomment this
#table02_table.head()


# 2b) Convert the imported matched point to GeoDataFrame

# Convert coordinates to numeric. This part of the code may slow it down
table02_table['latitude'] = table02_table.latitude.convert_objects(convert_numeric=True)
table02_table['longitude'] = table02_table.longitude.convert_objects(convert_numeric=True)

# Convert to Geographic Points
table02_geo = gpd.GeoDataFrame(table02_table)
table02_geo['point'] = table02_geo.apply(lambda x: Point(x['longitude'],x['latitude']), axis = 1)
table02_geo['geometry'] = table02_geo.point

# Compute the mean position to determine the EPSG 
mean_long_02 = np.mean(table02_geo['longitude'])
mean_lat_02 = np.mean(table02_geo['latitude'])

ref_prj_02 = convert_wgs_to_utm(mean_long_02,mean_lat_02)

# 4c) Convert table02 point -table02_geo from Geographic to UTM Zone 15N
#print(debris_geo.crs)
table02_geo_crs = table02_geo
# Assign WGS 84 Coordinate System
table02_geo_crs.crs = from_epsg(ref_02)
print(table02_geo.crs)
#Project from WGS to Projected Coordinate System
table02_geo_prj = table02_geo.to_crs(epsg=ref_prj_02)
table02_geo_prj.head()

# Step II: Spatial indexing in GeoPandas (kdTree) to find the nearest neighbor
from scipy.spatial import cKDTree 

# 1b) Define function for kdTree

def ckdnearest(gdA,acol, gdB, bcol):   
    nA = np.array(list(zip(gdA.geometry.x, gdA.geometry.y)) )
    nB = np.array(list(zip(gdB.geometry.x, gdB.geometry.y)) )
    btree = cKDTree(nB)
    # Find the 2 closest points
    
    dist, idx = btree.query(nA, 2)

    # Choose the closest point other than itself
    
    indx_zero = dist[:,0] == 0
    dist[indx_zero,0] = dist[indx_zero,1]
    idx[indx_zero,0] = idx[indx_zero,1]
    
    df = pd.DataFrame.from_dict({
                             'ID_of_Table_A' : gdA[acol].values, 
                             'ID_of_NearestPoint_B' : gdB.loc[idx[:,0], bcol].values,
                             'distance_to_nearest_B_meters': dist[:,0].astype(float)
                             })
    #df.set_index(['ID_of_Table_A','ID_of_NearestPoint_B','distance_to_nearest_B'],inplace=True)

    return df


# 2) Estimate Nearest Distance of table01 points to events:

# 2c) table02 rescues

# Record Start Time
then = time.time() #Time before the operations start
nearest_table02 =ckdnearest(table01_query_geo_prj,indkey_01, table02_geo_prj, indkey_02)

# Record Stop Time
now = time.time() #Time after it finished
print("It took: ", now-then, " seconds to compute the nearest point")
#..................................................................
#create table dr.optimal authorization dokeowo;

optimalproimity = civis.io.dataframe_to_civis(nearest_table02, database,table_out,existing_table_rows = 'drop')

# Record Stop Time
now2 = time.time() #Time after it finished
print("Time taken to run the entire code : ", now2-then2, " seconds")
#optimalproimity.result()
print('I got to the last line')
