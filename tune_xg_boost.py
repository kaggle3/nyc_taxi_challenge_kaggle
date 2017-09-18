"""Author: Sagar Shelke
This is a program to tune Xg boost using Hyperopt tuning library and Tree of Parzen Estimators (TPE) algorithm
This program can be easily generalized to tune any machine/deep learning algorithm

We are saving results on test data if error on validation data goes below particular threshold """

import pandas as pd
import numpy as np
import xgboost as xgb
import sys
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

def objective(space):

    space["colsample_bytree"] = float(space["colsample_bytree"])
    space['max_depth'] = int(space['max_depth'])
    space['learning_rate'] = float(space['learning_rate'])
    space['subsample'] = float(space['subsample'])
    space['min_child_weight'] = float(space['min_child_weight'])

    print("Reading Data")

    train = pd.read_csv("./nyc/train_final_feat.csv")
    test = pd.read_csv("./nyc/test_final_feat.csv")

    do_not_use_for_train = ["id", "pickup_datetime", "trip_duration", "pickup_date", "dropoff_datetime", "avg_speed_h",
                            "avg_speed_m", "avg_speed"]

    feature_names = [f for f in train.columns if f not in do_not_use_for_train]

    y = np.log(train["trip_duration"].values + 1)
    Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

    d_train = xgb.DMatrix(train[feature_names].values)
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)
    dtest = xgb.DMatrix(test[feature_names].values)

    watchlist = [(dtrain, "train"), (dvalid, "valid")]

    print("Building Model")

    model = xgb.train(space, d_train, 1000, early_stopping_rounds=150, maximize=False, verbose_eval=10)
    rmsle = round(model.best_score, 5)
    print("Modeling RSMLE {}".format(rmsle))
    print(space)

    if rmsle < 0.36500:

        ytest = model.predict(dtest)
        if test.shape[0] == ytest.shape[0]:
            print("Test is successful")
        else:
            print("Oops! There is some problem with dimention")
        test["trip_duration"] = np.exp(ytest) - 1
        test[["id", "trip_duration"]].to_csv("./results/"+str(rmsle)+"_submission.csv.gz", index=False, compression="gzip")

    print("-------------------------------------")

    return {'loss': rmsle, 'status': STATUS_OK}


if __name__ == "__main__":

    trial = 0

    space = {
        'learning_rate': hp.quniform('eta', 0.005, 0.05, 0.005),
        'max_depth': hp.quniform('max_depth', 3, 20, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 1, 0.05),
    }

    trials = Trials()
    print("optimizing function")
    sys.stdout = open("tune_xg_boost_without_snow.txt", "wt")
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=300,
                trials=trials)
    print(best)
