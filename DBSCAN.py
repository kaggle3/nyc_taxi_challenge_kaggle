#author:sushma

import matplotlib.pyplot as plt
import numpy as np,pandas as pd
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
data=pd.read_csv('train_with_weather.csv')
sl=pd.read_csv('speed_limit.csv')
#print(data.info())
#print(sl.info())

#dropoff_long=sl['dropoff_longitude'].values
#dropoff_lat=sl['dropoff_latitude'].values
coordinates=sl[['longitude','latitude']].as_matrix()
#print(coordinates.shape)
db = DBSCAN(eps=0.0005/6371., min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coordinates))
cluster_labels=db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coordinates[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))
print(cluster_labels,num_clusters)
def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)
centermost_points = clusters.map(get_centermost_point)
lats, lons = zip(*centermost_points)
rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
#rs = rep_points.apply(lambda row: data[(data['lat']==row['lat']) &amp;&amp; (data['lon']== row['lon'])].iloc[0], axis=1)
plt.figure()
plt.scatter(coordinates[:,0],coordinates[:,1])
plt.show()
