"""Author: Sagar Shelke

This is custom regressor gradient boosting is used on the output of AdaBoost
where each instant is given weight

With creative combination and parameter tuing this program can give better output"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

def three_r (train, test):

    start = time.time()
    print("Training Custom Regressor")

    y = np.log(train["trip_duration"].values + 1)

    Xtr, Xv, Ytr, Yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

    print("Three Regression is Running")
    rand_for1 = AdaBoostRegressor(n_estimators=5, base_estimator=DecisionTreeRegressor(max_depth=15, min_samples_leaf=25, min_samples_split=10))
    rand_for1.fit(Xtr, Ytr)

    y2 =Ytr - rand_for1.predict(Xtr)
    rand_for2 = AdaBoostRegressor(n_estimators=5, base_estimator=DecisionTreeRegressor(max_depth=15, min_samples_leaf=25, min_samples_split=10))
    rand_for2.fit(Xtr, y2)

    y3 = y2 - rand_for2.predict(Xtr)
    rand_for3 = AdaBoostRegressor(n_estimators=5, base_estimator=DecisionTreeRegressor(max_depth=15, min_samples_leaf=25, min_samples_split=10))
    rand_for3.fit(Xtr, y3)

    Y_pred = sum(rand_for.predict(Xv) for rand_for in (rand_for1, rand_for2, rand_for3))

    rmse = np.sqrt(mean_squared_error(Yv, Y_pred))

    print("New Three Reg RMSLE ={}".format(rmse))

    end = time.time()
    print("Execution Time of Custom Regressor ={}".format(end - start))
    print("-----------------------------------------")

if __name__ == "__main__":

    # import engineered features

    print("Reading Data")
    train_weather = pd.read_csv("./nyc/train_with_weather.csv")
    test_weather = pd.read_csv("./nyc/test_with_weather.csv")

    do_not_use_for_train = ["id", "pickup_datetime", "trip_duration", "pickup_date", "dropoff_datetime", "avg_speed_h",
                            "avg_speed_m", "avg_speed"]

    feature_names = [f for f in train_weather.columns if f not in do_not_use_for_train]
    print("Features with weather: Xg boost")
    three_r(train=train_weather, test=test_weather)
