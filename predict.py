import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt 
import mapclassify
from pathlib import Path
import osmnx as ox
from sklearn.cluster import DBSCAN
import os
import osmnx as ox
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--place_mame", help="name of the place",type=str)
parser.add_argument("-b","--bbox",help="upper-left and lower-right points: lat1 lon1 lat2 lon2",type=str)
parser.add_argument("-m","--pred_model_file", help="prediction model file",type=str)

args = parser.parse_args()

place_mame = args.place_mame
pred_model_file = args.pred_model_file
bbox = [float(i) for i in args.bbox.split()]
p1 = bbox[0:2]
p2 = bbox[2:]
places=[
{'place_mame' : place_mame,
'p1' : p1,
'p2' : p2},
]

# Load road network
dfimgID = gpd.read_file(f'output/mapillary/{place_mame}-road.geojson')

outputkey='w_crash' ## make sure you have this key

dfimgID = dfimgID.loc[:,['u','v','osmid','imgID',outputkey,'geometry']]

classes = ['animal--bird',
 'animal--ground-animal',
 'construction--barrier--ambiguous',
 'construction--barrier--concrete-block',
 'construction--barrier--curb',
 'construction--barrier--fence',
 'construction--barrier--guard-rail',
 'construction--barrier--other-barrier',
 'construction--barrier--road-median',
 'construction--barrier--road-side',
 'construction--barrier--separator',
 'construction--barrier--temporary',
 'construction--barrier--wall',
 'construction--flat--bike-lane',
 'construction--flat--crosswalk-plain',
 'construction--flat--curb-cut',
 'construction--flat--driveway',
 'construction--flat--parking',
 'construction--flat--parking-aisle',
 'construction--flat--pedestrian-area',
 'construction--flat--rail-track',
 'construction--flat--road',
 'construction--flat--road-shoulder',
 'construction--flat--service-lane',
 'construction--flat--sidewalk',
 'construction--flat--traffic-island',
 'construction--structure--bridge',
 'construction--structure--building',
 'construction--structure--garage',
 'construction--structure--tunnel',
 'human--person--individual',
 'human--person--person-group',
 'human--rider--bicyclist',
 'human--rider--motorcyclist',
 'human--rider--other-rider',
 'marking--continuous--dashed',
 'marking--continuous--solid',
 'marking--continuous--zigzag',
 'marking--discrete--ambiguous',
 'marking--discrete--arrow--left',
 'marking--discrete--arrow--other',
 'marking--discrete--arrow--right',
 'marking--discrete--arrow--split-left-or-straight',
 'marking--discrete--arrow--split-right-or-straight',
 'marking--discrete--arrow--straight',
 'marking--discrete--crosswalk-zebra',
 'marking--discrete--give-way-row',
 'marking--discrete--give-way-single',
 'marking--discrete--hatched--chevron',
 'marking--discrete--hatched--diagonal',
 'marking--discrete--other-marking',
 'marking--discrete--stop-line',
 'marking--discrete--symbol--bicycle',
 'marking--discrete--symbol--other',
 'marking--discrete--text',
 'marking-only--continuous--dashed',
 'marking-only--discrete--crosswalk-zebra',
 'marking-only--discrete--other-marking',
 'marking-only--discrete--text',
 'nature--mountain',
 'nature--sand',
 'nature--sky',
 'nature--snow',
 'nature--terrain',
 'nature--vegetation',
 'nature--water',
 'object--banner',
 'object--bench',
 'object--bike-rack',
 'object--catch-basin',
 'object--cctv-camera',
 'object--fire-hydrant',
 'object--junction-box',
 'object--mailbox',
 'object--manhole',
 'object--parking-meter',
 'object--phone-booth',
 'object--pothole',
 'object--sign--advertisement',
 'object--sign--ambiguous',
 'object--sign--back',
 'object--sign--information',
 'object--sign--other',
 'object--sign--store',
 'object--street-light',
 'object--support--pole',
 'object--support--pole-group',
 'object--support--traffic-sign-frame',
 'object--support--utility-pole',
 'object--traffic-cone',
 'object--traffic-light--general-single',
 'object--traffic-light--pedestrians',
 'object--traffic-light--general-upright',
 'object--traffic-light--general-horizontal',
 'object--traffic-light--cyclists',
 'object--traffic-light--other',
 'object--traffic-sign--ambiguous',
 'object--traffic-sign--back',
 'object--traffic-sign--direction-back',
 'object--traffic-sign--direction-front',
 'object--traffic-sign--front',
 'object--traffic-sign--information-parking',
 'object--traffic-sign--temporary-back',
 'object--traffic-sign--temporary-front',
 'object--trash-can',
 'object--vehicle--bicycle',
 'object--vehicle--boat',
 'object--vehicle--bus',
 'object--vehicle--car',
 'object--vehicle--caravan',
 'object--vehicle--motorcycle',
 'object--vehicle--on-rails',
 'object--vehicle--other-vehicle',
 'object--vehicle--trailer',
 'object--vehicle--truck',
 'object--vehicle--vehicle-group',
 'object--vehicle--wheeled-slow',
 'object--water-valve',
 'void--car-mount',
 'void--dynamic',
 'void--ego-vehicle',
 'void--ground',
 'void--static',
 'void--unlabeled']

classcode={}
for i in range(0,len(classes)):
    classcode[classes[i]] = i

def encodePreds(data):
    imgList = list(data['imgID'].unique())
    codes = []
    for imgID in imgList:
        thiscode = [0] * len(classes)
        tmp=data[data['imgID']==imgID]
        for i, row in tmp.iterrows():
            item = row['item']
            thiscode[classcode[item]]+=1
        codes.append(thiscode)
        
    codedf = pd.DataFrame(list(zip(imgList, codes)),columns =['imgID', 'code'])
    
    return codedf

# Encode road characteristics

codedpreds=[]
for place in places:
    place_mame = place['place_mame']
    north, west = place['p1']
    south, east = place['p2']   
    pred = pd.read_csv(f'output/mapillary/{place_mame}-predictions.csv')
    codedpreds.append(encodePreds(pred))
    
codedpreds = pd.concat(codedpreds)  

codedpredsdict = {}
for i, row in codedpreds.iterrows():
    codedpredsdict[row['imgID']] = row['code']

pred = pd.read_csv(f'output/mapillary/{place_mame}-predictions.csv')

v = []
ind = []

for i, row in dfimgID.iterrows():
    imgID = row['imgID']
    try:
        v.append(codedpredsdict[imgID])
        ind.append(i)
    except:
        pass

#newroad = road.loc[ind, ['g_crash_m']]
newroad = dfimgID.loc[ind, :]

for i in range(0,len(classcode)):
    newroad[f'x{str(i)}']=None

for i in range(0,len(classcode)):
    c=[]
    for j in range(0,len(v)):
        c.append(v[j][i])
    newroad[f'x{str(i)}']=c

#newroad.to_file(f'data/{place_mame}/data-reload.geojson', index=False, driver='GeoJSON')



# Predict

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


all_data = newroad.copy()

# Trim input

subcls=[
 'construction--barrier--ambiguous',
 'construction--barrier--concrete-block',
 'construction--barrier--curb',
 'construction--barrier--fence',
 'construction--barrier--guard-rail',
 'construction--barrier--other-barrier',
 'construction--barrier--road-median',
 'construction--barrier--road-side',
 'construction--barrier--separator',
 'construction--barrier--temporary',
 'construction--barrier--wall',
 'construction--flat--bike-lane',
 'construction--flat--crosswalk-plain',
 'construction--flat--curb-cut',
 'construction--flat--driveway',
 'construction--flat--parking',
 'construction--flat--parking-aisle',
 'construction--flat--pedestrian-area',
 'construction--flat--rail-track',
 'construction--flat--road',
 'construction--flat--road-shoulder',
 'construction--flat--service-lane',
 'construction--flat--sidewalk',
 'construction--flat--traffic-island',
 'construction--structure--bridge',
 'construction--structure--building',
 'construction--structure--garage',
 'construction--structure--tunnel',
 'human--person--individual',
 'human--person--person-group',
 'human--rider--bicyclist',
 'human--rider--motorcyclist',
 'human--rider--other-rider',
 'marking--continuous--dashed',
 'marking--continuous--solid',
 'marking--continuous--zigzag',
 'marking--discrete--ambiguous',
 'marking--discrete--arrow--left',
 'marking--discrete--arrow--other',
 'marking--discrete--arrow--right',
 'marking--discrete--arrow--split-left-or-straight',
 'marking--discrete--arrow--split-right-or-straight',
 'marking--discrete--arrow--straight',
 'marking--discrete--crosswalk-zebra',
 'marking--discrete--give-way-row',
 'marking--discrete--give-way-single',
 'marking--discrete--hatched--chevron',
 'marking--discrete--hatched--diagonal',
 'marking--discrete--other-marking',
 'marking--discrete--stop-line',
 'marking--discrete--symbol--bicycle',
 'marking--discrete--symbol--other',
 'marking--discrete--text',
 'marking-only--continuous--dashed',
 'marking-only--discrete--crosswalk-zebra',
 'marking-only--discrete--other-marking',
 'marking-only--discrete--text',
 'nature--vegetation',
 'object--banner',
 'object--bike-rack',
 'object--catch-basin',
 'object--cctv-camera',
 'object--fire-hydrant',
 'object--junction-box',
 'object--manhole',
 'object--parking-meter',
 'object--phone-booth',
 'object--pothole',
 'object--sign--advertisement',
 'object--sign--ambiguous',
 'object--sign--back',
 'object--sign--information',
 'object--sign--other',
 'object--sign--store',
 'object--street-light',
 'object--support--pole',
 'object--support--pole-group',
 'object--support--traffic-sign-frame',
 'object--support--utility-pole',
 'object--traffic-cone',
 'object--traffic-light--general-single',
 'object--traffic-light--pedestrians',
 'object--traffic-light--general-upright',
 'object--traffic-light--general-horizontal',
 'object--traffic-light--cyclists',
 'object--traffic-light--other',
 'object--traffic-sign--ambiguous',
 'object--traffic-sign--back',
 'object--traffic-sign--direction-back',
 'object--traffic-sign--direction-front',
 'object--traffic-sign--front',
 'object--traffic-sign--information-parking',
 'object--traffic-sign--temporary-back',
 'object--traffic-sign--temporary-front',
 'object--trash-can',
 'object--vehicle--bicycle',
 'object--vehicle--bus',
 'object--vehicle--car',
 'object--vehicle--caravan',
 'object--vehicle--motorcycle',
 'object--vehicle--on-rails',
 'object--vehicle--other-vehicle',
 'object--vehicle--trailer',
 'object--vehicle--truck',
 'object--vehicle--vehicle-group',
 'object--vehicle--wheeled-slow',
]

svIND = [f"x{i}" for i,cls in enumerate(classes) if any(x in cls for x in subcls)]

data = all_data.copy()

X = data[svIND].values
Y = data[outputkey]

import matplotlib.pyplot as plt
import itertools
import numpy as np
import pickle

filename = pred_model_file

loaded_model = pickle.load(open(filename, 'rb'))

Y_pred = loaded_model.predict(X)

data['RiskLevel']=['High' if i >0 else 'Low' for i in Y_pred ]

data.to_file(f'data/{place_mame}/Predictions.geojson', index=False, driver='GeoJSON')

# Filter based on distance

points = pd.read_csv(f'output/mapillary/{place_mame}-points.csv')
points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.longitude, points.latitude))

new = data.merge(points, left_on='imgID', right_on='imgID')

new1 = gpd.GeoDataFrame(new, geometry=new.geometry_x)
new2 = gpd.GeoDataFrame(new, geometry=new.geometry_y)
new['distance'] = new1['geometry'].distance(new2['geometry'])

save = new[new['distance']<3*10e-5]
save = gpd.GeoDataFrame(save, geometry=save.geometry_x)
del save['geometry_x']
del save['geometry_y']

save.to_file(f'data/{place_mame}/Predictions_final.geojson', index=False, driver='GeoJSON')

