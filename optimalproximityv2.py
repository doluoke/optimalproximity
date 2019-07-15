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
from sqlalchemy import create_engine


import matplotlib
#%matplotlib inline

#--------------------------------------------------------------
print('This code started')
#conn = create_engine('redshift+psycopg2://dokeowo@host.amazonaws.com:5439/database')
#print('This is 0 after python', sys.argv[0], 'and type is:', type(sys.argv[0]))
#print('This is 1 after python', sys.argv[1], 'and type is:', type(int(sys.argv[1])))
#print('This is 2 after python', sys.argv[2], 'and type is:', type(float(sys.argv[2])))
#print('This is 3 after python', sys.argv[3], 'and type is:', type(sys.argv[3]))
#print('This is 3 after python', sys.argv[4], 'and type is:', type(sys.argv[4]))

#table_01 = '''fed.task6_civis_estimate_serious_damage_leftjoin_hud__demographics'''
table_01 = sys.argv[1]
ref_01   = int(sys.argv[2])
table_02 = sys.argv[3]
ref_02   = int(sys.argv[4])
ref_prj = int(sys.argv[5])
table_out = sys.argv[6]

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

table01_query= '''

    
    SELECT
      userdefinedfltyid,
      latitude as latitude_ref,
      longitude as longitude_ref
    FROM ''' + table_01 + ''' as ia
    
    
    ORDER BY userdefinedfltyid

'''

table01_query_table = civis.io.read_civis_sql(
    table01_query,"City of Houston",use_pandas=True
)


#print(table01_query)

# 1b) Convert the imported table01 table to GeoDataFrame
table01_query_geo = gpd.GeoDataFrame(table01_query_table)
table01_query_geo['point'] = table01_query_geo.apply(lambda x: Point(x['longitude_ref'],x['latitude_ref']), axis = 1)
table01_query_geo['geometry'] = table01_query_geo.point

# 1c) Convert umatched point -table01_query_geo from Geographic to UTM Zone 15N
table01_query_geo_crs = table01_query_geo
# Assign WGS 84 Coordinate System
table01_query_geo_crs.crs = from_epsg(ref_01)
# Project from WGS to Projected Coordinate System
table01_query_geo_prj = table01_query_geo_crs.to_crs(epsg=ref_prj)



#--------------------------------------------------------------
# record level data 

table02_query= '''
SELECT
     
        objectid_1,
        lat as latitude,
        long as longitude


FROM ''' + table_02 + ''' as h

   
ORDER BY h.objectid_1

'''
print('This code completed')
table02_table = civis.io.read_civis_sql(
    table02_query,"City of Houston",use_pandas=True
)


print(table02_query)
#You may uncomment this
#table02_table.head()


# 2b) Convert the imported matched point to GeoDataFrame
table02_geo = gpd.GeoDataFrame(table02_table)
table02_geo['point'] = table02_geo.apply(lambda x: Point(x['longitude'],x['latitude']), axis = 1)
table02_geo['geometry'] = table02_geo.point

# 4c) Convert table02 point -table02_geo from Geographic to UTM Zone 15N
#print(debris_geo.crs)
table02_geo_crs = table02_geo
# Assign WGS 84 Coordinate System
table02_geo_crs.crs = from_epsg(ref_02)
print(table02_geo.crs)
#Project from WGS to Projected Coordinate System
table02_geo_prj = table02_geo.to_crs(epsg=ref_prj)
table02_geo_prj.head()

# Step II: Spatial indexing in GeoPandas (kdTree) to find the nearest neighbor
from scipy.spatial import cKDTree 

# 1b) Define function for kdTree

def ckdnearest(gdA,acol, gdB, bcol):   
    nA = np.array(list(zip(gdA.geometry.x, gdA.geometry.y)) )
    nB = np.array(list(zip(gdB.geometry.x, gdB.geometry.y)) )
    btree = cKDTree(nB)
    dist, idx = btree.query(nA,k=1)
    df = pd.DataFrame.from_dict({
                             'ID_of_TableA' : gdA[acol].values, 
                             'ID_of_NearestPoint' : gdB.loc[idx, bcol].values,
                             'distance': dist.astype(float)
                             })
    return df


# 2) Estimate Nearest Distance of table01 points to events:

# 2c) table02 rescues
nearest_table02 =ckdnearest(table01_query_geo_prj,'userdefinedfltyid', table02_geo_prj,'objectid_1')

#..................................................................
#create table dr.optimal authorization dokeowo;

optimalproimity = civis.io.dataframe_to_civis(nearest_table02, "City of Houston",table_out)
#optimalproimity.result()
print('I got to the last line')
