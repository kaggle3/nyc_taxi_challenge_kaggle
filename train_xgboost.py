"""Author: Sagar Shelke

This program trains XGboost regressor on a single set of parameters."""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def xgb_r(train, test):

    start = time.time()
    print("Training XG boost Regressor")

    y = np.log(train["trip_duration"].values + 1)
    Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)
    dtest = xgb.DMatrix(test[feature_names].values)

    watchlist = [(dtrain, "train"), (dvalid, "valid")]

    xgb_pars = {"min_child_weight": float(2.0), "learning_rate": float(0.025), "cosample_bytree": float(0.45),
                "max_depth": 20,
                "subsample": float(1.0), "gamma": 0.8,
                }

    model = xgb.train(xgb_pars, dtrain, 1000, watchlist, early_stopping_rounds=150, maximize=False, verbose_eval=10)
    print("Modeling RSMLE %5f" % model.best_score)

    end = time.time()
    print("XGBR execution time ={}".format(end - start))
    print("------------------------------------------")

if __name__ == "__main__":

    # import engineered features

    print("Reading Data")
    train_weather = pd.read_csv("./nyc/train_with_weather.csv")
    test_weather = pd.read_csv("./nyc/test_with_weather.csv")


    do_not_use_for_train = ["id", "pickup_datetime", "trip_duration", "pickup_date", "dropoff_datetime", "avg_speed_h",
                            "avg_speed_m", "avg_speed"]

    feature_names = [f for f in train_weather.columns if f not in do_not_use_for_train]
    print("Features with weather: Xg boost")
    xgb_r(train=train_weather, test=test_weather)
