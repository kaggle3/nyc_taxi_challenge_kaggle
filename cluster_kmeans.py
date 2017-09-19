#auther Sushma and Krishna
#This program is for K-mean clustering of speed data on basis of latitude and longitude .
#The cluster of speed is mapped to cluster in the location database to longitude and latitude.
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN,k_means
data=pd.read_csv('train_with_weather.csv')
sl=pd.read_csv('speed_limit.csv')
test=pd.read_csv('test_with_weather.csv')
coordinates_1=(data[['pickup_longitude','pickup_latitude']])
coordinates_2=(sl[['longitude','latitude']])
coordinates_22 = coordinates_2.rename(index=str, columns={"longitude":"pickup_longitude","latitude":"pickup_latitude"})
print(coordinates_1.info())
print(coordinates_22.info())
frames = [coordinates_22,coordinates_1]

coordinates_111 = pd.concat(frames)

coordinates = coordinates_111


#number of cluster 53942
print(coordinates.info())
coordinates = coordinates.reindex()
coordinates_array = np.array(coordinates)

sample_ind = np.random.permutation(len(coordinates_array))[:500000]

kmeans = MiniBatchKMeans(n_clusters=21310,init=np.array(coordinates_22),random_state=42).fit(coordinates_array[sample_ind])

sl.loc[:, 'speed_cluster'] = kmeans.predict(sl[['latitude', 'longitude']])
#sl.loc[:, 'dropoff_cluster'] = kmeans.predict(sl[['latitude', 'dropoff_longitude']])

centroid_1 =kmeans.cluster_centers_
labels_1 = kmeans.labels_
data.to_csv('train_cluster.csv')
test.to_csv('test_cluster.csv')
sl.to_csv('sl_cluster.csv')
centroid=pd.DataFrame(centroid_1)
centroid.to_csv('centroid1.csv')
labels=pd.DataFrame(labels_1)
labels.to_csv('labels1.csv')
print(centroid)
print(labels)



'''sample_ind = np.random.permutation(len(np.array(coordinates)))[:50000]
kmeans = MiniBatchKMeans(n_clusters=53942, batch_size=10000).fit(coordinates[sample_ind])


sl.loc[:, 'speed_cluster'] = kmeans.predict(sl[['latitude', 'longitude']])
#sl.loc[:, 'dropoff_cluster'] = kmeans.predict(sl[['latitude', 'dropoff_longitude']])'''