"""Author: Sagar Shelke
This is a script for feature engineering on train and test data.
Some features are taken from other kaggle kernels.

We have extensively used Pandas DataFrame"""

import pandas as pd
import seaborn as sns
import numpy as np
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geopy.distance import vincenty
import xgboost as xgb
import sys

print("Timer is set! and execution is started")
start = time.time()
warnings.filterwarnings("ignore")


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)

N= 100000 # number of samles in a row to plot

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

# Read dataset
print("--> Reading data-set")
train = pd.read_csv("./nyc/train.csv")

print("This is description of training set")
print(train.info())
print("--------------------------------------")
test = pd.read_csv("./nyc/test.csv")
print("This is description of training set")
print(test.info())
print("---------------------------------------")

# Check dataset
print("--> checking data-set for balancing")
if train.id.nunique() == train.shape[0]:
    print("ID is unique")
else:
    print("Oops! ID is not unique")

if len(np.intersect1d(train.id.values, test.id.values)) == 0:
    print("no ID's are matching")
else:
    print("ID's are matching")

if train.count().min() == train.shape[0]:
    print("We do not need to worry about missing values")
else:
    print("Some values are missing")

# convert datetime object into string
print("--> Separating date and time")

train["pickup_datetime"] = pd.to_datetime(train.pickup_datetime)
test["pickup_datetime"] = pd.to_datetime(test.pickup_datetime)

train.loc[:, "pickup_date"] = train["pickup_datetime"].dt.date
test.loc[:, "pickup_date"] = test["pickup_datetime"].dt.date

train["store_and_fwd_flag"] = 1* (train.store_and_fwd_flag.values == "Y")
test["store_and_fwd_flag"] = 1 * (train.store_and_fwd_flag.values == "Y")
print("-----------------------------------")

# pickup date and number of pickups plotting
"""Plotting Train and Test instances vs month"""

plt.plot(train.groupby("pickup_date").count()[["id"]], "o-", label= "train")
plt.plot(test.groupby("pickup_date").count()[["id"]], "o-", label= "train")
plt.title("Train and Test instances versus month")
plt.legend(loc=0)
plt.ylabel("number of instances")
plt.savefig("train_test.png")

# calculate distances and add into dataset
print("--> Calculating distance")

train.loc[:, "distance_h"] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, "direction"] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, "distance"] = train.apply(lambda x: vincenty((x['pickup_latitude'],x['pickup_longitude']), (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)

test.loc[:, "distance_h"] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, "direction"] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, "distance"] = test.apply(lambda x: vincenty((x['pickup_latitude'],x['pickup_longitude']), (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)

train.loc[:, "center_latitude"] = (train["pickup_latitude"].values + train["dropoff_latitude"].values) / 2
train.loc[:, "center_longitude"] = (train["pickup_longitude"].values + train["dropoff_longitude"].values) / 2

test.loc[:, "center_latitude"] = (test["pickup_latitude"].values + test["dropoff_latitude"].values) / 2
test.loc[:, "center_longitude"] = (test["pickup_longitude"].values + test["dropoff_longitude"].values) / 2
print("----------------------------------")

# add date-time feature
print("--> Adding datetime features")

train.loc[:, "pickup_weekday"] = train["pickup_datetime"].dt.dayofweek
train.loc[:, "pickup_weekofyear"] = train["pickup_datetime"].dt.weekofyear
train.loc[:, "pickup_month"] = train["pickup_datetime"].dt.month
train.loc[:, "pickup_hour"] = train["pickup_datetime"].dt.hour
train.loc[:, "pickup_minute"] = train["pickup_datetime"].dt.minute
train.loc[:, "pickup_week_hour"] = train["pickup_weekday"] * 24 + train["pickup_hour"]

test.loc[:, "pickup_weekday"] = test["pickup_datetime"].dt.dayofweek
test.loc[:, "pickup_weekofyear"] = test["pickup_datetime"].dt.weekofyear
test.loc[:, "pickup_month"] = test["pickup_datetime"].dt.month
test.loc[:, "pickup_hour"] = test["pickup_datetime"].dt.hour
test.loc[:, "pickup_minute"] = test["pickup_datetime"].dt.minute
test.loc[:, "pickup_week_hour"] = test["pickup_weekday"] * 24 + test["pickup_hour"]
print("-----------------------------------")

# Adding speech features
print("--> Adding speed features")

train.loc[:, "avg_speed_h"] = 1000 * train["distance_h"] / train["trip_duration"]
train.loc[:, "avg_speed_m"] = 1000 * train["distance_dummy_manhattan"] / train["trip_duration"]
train.loc[:, "avg_speed"] = 1000 * train["distance"] / train["trip_duration"]

print("-------------------------------------")

"""Plot some information"""

fig, ax = plt.subplot(ncols = 2, sharey = True)
ax[0].plot(train.groupby("pickup_hour").mean()["avg_speed"], "bo-", lw = 2, alpha = 0.7) #here we are taking mean of average speed for each pickup_hour
ax[1].plot(train.groupby("pickup_weekday").mean()["avg_speed"], "ro-", lw= 2, alpha = 0.7)
ax[0].set_xlabel("pickup_hour")
ax[1].set_xlabel("pickup_weekday")
ax[0].set_ylabel("average_speed")
fig.suptitle("Speed Analysis")
plt.show()

fig_2, ax_2 = plt.subplot(ncols = 2, sharey = True)
ax_2[0].plot(train.groupby("pickup_hour").mean()["trip_duration"], "bo-", lw = 2, alpha = 0.7) #here we are taking mean of average speed for each pickup_hour
ax_2[1].plot(train.groupby("pickup_weekday").mean()["trip_duration"], "ro-", lw= 2, alpha = 0.7)
ax_2[0].set_xlabel("pickup_hour")
ax_2[1].set_xlabel("pickup_weekday")
ax_2[0].set_ylabel("trip_duration_average")
fig_2.suptitle("Trip Duration Analysis")
plt.show()


# PCA (Principal component analysis)

print("---> Performing PCA")
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA()
X_1 = pca.fit_transform(train[["pickup_latitude", "pickup_longitude"]])
train.loc[:, "pickup_pca0"] = X_1[:, 0]
train.loc[:, "pickup_pca1"] = X_1[:, 1]

X_2 = pca.fit_transform(train[["dropoff_latitude", "dropoff_longitude"]])
train.loc[:, "dropoff_pca0"] = X_2[:, 0]
train.loc[:, "dropoff_pca1"] = X_2[:, 1]

X_3 = pca.fit_transform(test[["pickup_latitude", "pickup_longitude"]])
test.loc[:, "pickup_pca0"] = X_3[:, 0]
test.loc[:, "pickup_pca1"] = X_3[:, 1]

X_4 = pca.fit_transform(test[["dropoff_latitude", "dropoff_longitude"]])
test.loc[:, "dropoff_pca0"] = X_4[:, 0]
test.loc[:, "dropoff_pca1"] = X_4[:, 1]

# PCA distance feature

train.loc[:, "pca_manhattan"] = np.abs(train["dropoff_pca1"] - train["pickup_pca1"]) + np.abs(train["dropoff_pca0"] - train["pickup_pca0"])
test.loc[:, "pca_manhattan"] = np.abs(test["dropoff_pca1"] - test["pickup_pca1"]) + np.abs(test["dropoff_pca0"] - test["pickup_pca0"])
# Clustering (k-mean)

print("---> Performing clustering")

longitude = list(train.pickup_longitude) + list(train.dropoff_longitude)
latitude = list(train.pickup_latitude) + list(train.dropoff_latitude)


plt.figure(figsize=(10,10))
plt.plot(longitude, latitude, ".", alpha = 0.4, markersize= 0.05)
plt.show()


sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])


# plotting longitude-latitude clusters

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], s=10, lw=0,
           c=train.pickup_cluster[:N].values, cmap='tab20', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# Adding another dataset

print("---> Adding new dataset")
train_1 = pd.read_csv("/Users/IQ-PC/Desktop/nyc/fastest_routes_train_part_1.csv", usecols=["id", "total_distance", "total_travel_time", "number_of_steps"])

train_2 = pd.read_csv("/Users/IQ-PC/Desktop/nyc/fastest_routes_train_part_2.csv", usecols=["id", "total_distance", "total_travel_time", "number_of_steps"])

test_street = pd.read_csv("/Users/IQ-PC/Desktop/nyc/fastest_routes_test.csv", usecols=["id", "total_distance", "total_travel_time", "number_of_steps"])

train_street_info = pd.concat((train_1, train_2))

train = train.merge(train_street_info, how="left", on="id")

test = test.merge(test_street, how="left", on="id")

# checking features before training

print("---> Checking feature before training")

do_not_use_for_train =["id", "pickup_datetime", "trip_duration", "pickup_date", "dropoff_datetime", "avg_speed_h", "avg_speed_m", "avg_speed"]

feature_names = [f for f in train.columns if f not in do_not_use_for_train]

print("we will be training on {} features.".format(len(feature_names)))

feature_stats = pd.DataFrame({"feature": feature_names})

feature_stats.loc[:, "train_mean"] = np.nanmean(train[feature_names].values, axis=0).round(4)
feature_stats.loc[:, "test_mean"] = np.nanmean(test[feature_names].values, axis=0).round(4)

feature_stats.loc[:, "train_std"] = np.nanstd(train[feature_names].values, axis=0).round(4)
feature_stats.loc[:, "test_std"] = np.nanstd(test[feature_names].values, axis=0).round(4)

feature_stats.loc[:, "train_nan"] = np.mean(np.isnan(train[feature_names].values), axis=0).round(3)
feature_stats.loc[:, "test_nan"] = np.mean(np.isnan(test[feature_names].values), axis=0).round(3)

feature_stats.loc[:, "train_test_mean_diff"] = np.abs(feature_stats["train_mean"]- feature_stats["test_mean"])/np.abs(feature_stats["train_std"] +feature_stats["test_std"]) *2

feature_stats.loc[:, "train_test_nan_diff"] = np.abs(feature_stats["train_nan"] - feature_stats["test_nan"])

feature_stats = feature_stats.sort_values(by="train_test_mean_diff")

print(feature_stats[["feature", "train_test_mean_diff"]].tail())

# checking for missing values and filling if there are any

print("--> checking NaN values in dataframe")

print("Checking for train dataframe")
print(train.isnull().any())
print("filling empty values, if there are any!")
train = train.fillna(method="ffill")

print("Checking for test data-frame")
print(test.isnull().any())
print("filling empty values, if there are any!")
test = test.fillna(method="ffill")

# Merge with augmented dataset

test_augmented = pd.read_csv("./nyc/test_augmented.csv")
test = pd.merge(test, test_augmented, on="id")

train_augmented = pd.read_csv("./nyc/train_augmented.csv")
train = pd.merge(train, train_augmented, on="id")

# Add snow data

snow = pd.read_csv("/Users/IQ-PC/Desktop/nyc/weather_nyc.csv")
print(snow.info())
snow = snow.replace("T", np.nan)
print(snow.head(20))
snow.rename(columns={"date": "pickup_date"}, inplace=True)
snow["pickup_date"] = snow["pickup_date"].astype("str")

snow = snow.drop(["maximum temerature", "minimum temperature", "average temperature"], axis=1)

train = pd.merge(train, snow, on="pickup_date")

train = train.fillna(method="ffill")

test = pd.merge(test, snow, on="pickup_date")

test = test.fillna(method="ffill")

print(train.info())
# Saving data-frame to CSV files
# we are doing this to save time for different model trials

print("---> Saving dataframes to CSV files")

train.to_csv("./nyc/train_final_feat.csv", sep=",")

test.to_csv("./nyc/test_final_feat.csv", sep=",")

print("Successfully saved DataFrames! Import them in training program")