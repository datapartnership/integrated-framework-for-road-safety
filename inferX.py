import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t","--access_token", help="Mapillary token",type=str)
parser.add_argument("-p","--place_mame", help="name of the place",type=str)
parser.add_argument("-b","--bbox",help="upper-left and lower-right points: lat1 lon1 lat2 lon2",type=str)
parser.add_argument("-c","--config_file", help="model config file",type=str)
parser.add_argument("-ckp","--checkpoint_file", help="checkpoint file",type=str)
parser.add_argument("-f","--force_infer", help="force to re-infer",type=str, default='yes')


args = parser.parse_args()
access_token = args.access_token
place_mame = args.place_mame
config_file = args.config_file
checkpoint_file = args.checkpoint_file
bbox = [float(i) for i in args.bbox.split()]
p1 = bbox[0:2]
p2 = bbox[2:]

force_infer = True if args.force_infer=='yes' else False
if force_infer:
    print('Will force to re-infer.')

#force_infer=True

import torch
if torch.cuda.is_available():
    device = f'cuda:{torch.cuda.current_device()}'
else: 
    device = 'cpu'
print(f'using device: {device}')


# ## Prepare the model

from mmdet.apis import init_detector, inference_detector
import mmcv

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device=device)# cpu


# ## Get images information for ROI

import mercantile, mapbox_vector_tile, requests, json, os
from vt2geojson.tools import vt_bytes_to_geojson
from pathlib import Path
import pandas as pd

# define an empty geojson as output
output= { "type": "FeatureCollection", "features": [] }


east, south, west, north = [p2[1], p2[0], p1[1], p1[0]]

#filter_values = ['object--support--utility-pole','object--street-light']
# get the tiles with x and y coors intersecting bbox at zoom 14 only
tiles = list(mercantile.tiles(west, south, east, north, 14))
print("Number of tiles: ",len(tiles))

mapillary_out_dir = 'output/mapillary'
Path(mapillary_out_dir).mkdir(parents=True, exist_ok=True)

# loop through all tiles to get IDs of Mapillary data
Path(f"{mapillary_out_dir}/tile-geojson/").mkdir(parents=True, exist_ok=True)
imgIDs = []
longitude = []
latitude = []
tileIDs = []
for i, tile in enumerate(tiles):
    print('Downloading images for Tile ', i)

    tile_file = f'{mapillary_out_dir}/tile-geojson/{tile.x}-{tile.y}-{tile.z}.geojson'
    if not os.path.isfile(tile_file):
        tile_url = 'https://tiles.mapillary.com/maps/vtp/mly1_public/2/{}/{}/{}?access_token={}'.format(tile.z,tile.x,tile.y,access_token)
        
        response = requests.get(tile_url)
        data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z)

        # write tile
        with open(tile_file, 'w') as f:
            json.dump(data, f)
    else:
        with open(tile_file) as f:
            data = json.load(f)

    
    tile_id = f"{tile.x}-{tile.y}-{tile.z}"
    
    
    tile_dir = f"{mapillary_out_dir}/tiles/{tile_id}"
    img_dir_tile = f"{tile_dir}/image"
    json_dir_tile = f"{tile_dir}/json"
    Path(tile_dir).mkdir(parents=True, exist_ok=True)
    Path(img_dir_tile).mkdir(parents=True, exist_ok=True)
    Path(json_dir_tile).mkdir(parents=True, exist_ok=True)
    
    points = [feature for feature in data['features'] if feature['geometry']['type']=='Point']
    #print(len(points))
    
    for point in points:
        
        '''
        point['geometry']['coordinates']
        point['properties']['captured_at']
        point['properties']['compass_angle']
        point['properties']['id']
        point['properties']['is_pano']
        point['properties']['sequence_id']
        '''
        
        imgIDs.append(point['properties']['id'])
        lon, lat = point['geometry']['coordinates']
        longitude.append(lon)
        latitude.append(lat)
        
        tileIDs.append(tile_id)
        
    '''
        # get img info
        imgId = point['properties']['id']
        graph_img_url = graph_img_url_base.format(imgId=imgId, access_token=access_token)
        json_img_file = f"{json_dir_tile}/{imgId}.json"
        if not os.path.isfile(json_img_file):
            json_img = requests.get(graph_img_url).json()
            with open(json_img_file, 'w') as f:
                json.dump(json_img, f, indent=2)
        else:
            with open(json_img_file) as f:
                json_img = json.load(f)
        
        # download image
        img_url = json_img['thumb_1024_url']
        #print(img_url)
    '''

    '''
    try:
        ## apply filter
        #filtered_data = [feature for feature in data['features'] if feature['properties']['value'] in filter_values]
        # no filter
        filtered_data = [feature for feature in data['features'] if feature['properties']['value']]
        for feature in filtered_data:

            if (feature['geometry']['coordinates'][0] > west and feature['geometry']['coordinates'][0] < east)\
              and (feature['geometry']['coordinates'][1] > south and feature['geometry']['coordinates'][1] < north):
                
                output['features'].append(feature)
                
    except: pass
    '''

images = pd.DataFrame(list(zip(imgIDs, longitude, latitude, tileIDs)), columns =['imgID', 'longitude', 'latitude', 'tileID'])
# make sure points are limited to the boundary
images = images[images['longitude'] > west]
images = images[images['longitude'] < east]
images = images[images['latitude'] > south]
images = images[images['latitude'] < north]
images.to_csv(f'{mapillary_out_dir}/{place_mame}-points.csv', index=False)

'''with open(f'{mapillary_out_dir}/mapillary.geojson', 'w') as f:
    json.dump(output, f)'''


# ## Load road network

import geopandas as gpd
import osmnx as ox
import geopandas as gpd

#road = gpd.read_file('data/Bogota/bogota_roadway_trafficlight.geojson', bbox=[west,south,east,north])
if True:#place_mame == 'Padang':
    networkfile = f'data/{place_mame}/network_simplified.gpkg'
    if not Path(networkfile).is_file():
        G = ox.graph_from_address('Padang, Indonesia', dist=12*1000, network_type='drive', simplify = True) 
        ox.save_graph_geopackage(G, filepath=networkfile)
        ox.plot_graph(G)
        
    road = gpd.read_file(networkfile, layer='edges', bbox=[west,south,east,north])

# ## Find the nearest image to the centroid of a segment

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
def findNearestPoint(gdA, gdB):

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.centroid.x, x.centroid.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

images = gpd.GeoDataFrame(images, geometry=gpd.points_from_xy(images.longitude, images.latitude))
road = findNearestPoint(road, images)
road.to_file(f'output/mapillary/{place_mame}-road.geojson', driver='GeoJSON')    
#road.head()


# ## Download images 

import os
graph_img_url_base = 'https://graph.mapillary.com/{imgId}?fields=id,computed_geometry,altitude,atomic_scale,camera_parameters,camera_type,captured_at,compass_angle,computed_altitude,computed_compass_angle,computed_rotation,exif_orientation,geometry,height,thumb_256_url,thumb_1024_url,thumb_2048_url,merge_cc,mesh,quality_score,sequence,sfm_cluster,width&access_token={access_token}'

for i, row in road.iterrows():
    tile_id = row['tileID']
    
    tile_dir = f"{mapillary_out_dir}/tiles/{tile_id}"
    img_dir_tile = f"{tile_dir}/image"
    json_dir_tile = f"{tile_dir}/json"

    # get img info
    imgId = row['imgID']
    graph_img_url = graph_img_url_base.format(imgId=imgId, access_token=access_token)
    json_img_file = f"{json_dir_tile}/{imgId}.json"
    if not os.path.isfile(json_img_file):
        json_img = requests.get(graph_img_url).json()
        with open(json_img_file, 'w') as f:
            json.dump(json_img, f, indent=2)
    else:
        with open(json_img_file) as f:
            json_img = json.load(f)
            
    try:
    
        # download image
        img_url = json_img['thumb_1024_url']
    
        picPath = f'{img_dir_tile}/{imgId}.jpeg'
        exist = os.path.exists(picPath)
        if not exist:
            r = requests.get(img_url)
            f = open(picPath, 'wb')
            f.write(r.content)
            f.close()
            
    except: print('empty? :', json_img_file)
    
    
    
    

    
    


# ## Infer

import mmcv
classes = mmcv.list_from_file('data/classes.txt')
def postprocess(data, threshold):
        # Format output following the example ObjectDetectionHandler format
        output = []

        for image_index, image_result in enumerate(data):
            output.append([])
            if isinstance(image_result, tuple):
                bbox_result, segm_result = image_result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = image_result, None
                

            for class_index, class_result in enumerate(bbox_result):

                class_name = classes[class_index]
                for bbox in class_result:
                    bbox_coords = bbox[:-1].tolist()

                    score = float(bbox[-1])
                    if score >= threshold:
                        output[image_index].append({
                            class_name: bbox_coords,
                            'score': score
                        })
        return output


from glob import glob
for tile_id in list(road['tileID'].unique()):
    # loop over tiles, each is a folder
    
    tile_dir = f"{mapillary_out_dir}/tiles/{tile_id}"
    img_dir_tile = f"{tile_dir}/image"
    json_dir_tile = f"{tile_dir}/json"
    pred_dir_tile = f"{tile_dir}/pred"
    predchunk_dir_tile = f"{tile_dir}/predchunk"
    
    Path(tile_dir).mkdir(parents=True, exist_ok=True)
    Path(img_dir_tile).mkdir(parents=True, exist_ok=True)
    Path(json_dir_tile).mkdir(parents=True, exist_ok=True)
    Path(pred_dir_tile).mkdir(parents=True, exist_ok=True)
    Path(predchunk_dir_tile).mkdir(parents=True, exist_ok=True)
    
    # get img list
    imgList = glob(f"{img_dir_tile}/*.jpeg")
    if len(imgList) < 1: break

    # Remove empty images
    imgList = [img for img in imgList if os.path.getsize(img)/1024 > 1]
        
    #ncpu = 2
    step = 200 #int(len(imgList)/ncpu)+1
    chunks = [imgList[x:x+step] for x in range(0, len(imgList), step)]
    for chunkid, imgList in enumerate(chunks):
        pred_chunk_file = f"{predchunk_dir_tile}/{chunkid}.json"
        
        # if chunk file exist, skip to next chunk
        if os.path.isfile(pred_chunk_file) and not force_infer: continue
            
        results = inference_detector(model, imgList)
        preds = postprocess(results, 0.8)
        # write files
        for i, pred in enumerate(preds):
            
            imgID = imgList[i].split('/')[-1].split('.')[0]
            predFile = f"{pred_dir_tile}/{imgID}.json"

            with open(predFile, 'w') as outfile:
                json.dump(pred, outfile, indent=2)
        
        # save chunk file
        with open(pred_chunk_file, 'w') as fp:
            pass
    


# ## Merge inference to road network


item, x1, y1, x2, y2, score, imgID = ([] for i in range(7))
imgList = []
for i, row in road.iterrows():
    tile_id = row['tileID']
    
    tile_dir = f"{mapillary_out_dir}/tiles/{tile_id}"
    img_dir_tile = f"{tile_dir}/image"
    json_dir_tile = f"{tile_dir}/json"
    pred_dir_tile = f"{tile_dir}/pred"

    # get img info
    imgId = row['imgID']
    if imgId in imgList: continue
    imgList.append(imgId)
    graph_img_url = graph_img_url_base.format(imgId=imgId, access_token=access_token)
    pred_file = f"{pred_dir_tile}/{imgId}.json"
    
    if os.path.isfile(pred_file):
        with open(pred_file) as f:
            preds = json.load(f)
        for pred in preds:
            itemName = list(pred.keys())[0]
            item.append(itemName)
            x1.append(pred[itemName][0])
            y1.append(pred[itemName][1])
            x2.append(pred[itemName][2])
            y2.append(pred[itemName][3])
            
            score.append(pred['score'])
            imgID.append(imgId)
        
    else:
        pass

dfpred = pd.DataFrame(list(zip(item,x1,y1,x2,y2,score,imgID)), columns=['item','x1','y1','x2','y2','score','imgID'])
dfpred.to_csv(f'{mapillary_out_dir}/{place_mame}-predictions.csv', index=False)


