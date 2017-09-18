"""Author: Sagar Shelke

This program takes engineered features as input and apply several machine learning regression algorithms
on problem.

Results are logged to a text file"""

import pandas as pd
import seaborn as sns
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xgboost as xgb
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

print("Model Training has been Started")

sys.stdout = open("gboot_newalgo_with_snow_1.txt", "wt")


def svm_r(train, test):
    start = time.time()

    print("Training SVM Regressor")
    scalar = StandardScaler().fit(train[feature_names].values)
    X_train = scalar.fit_transform(train[feature_names].values)

    X_test = scalar.fit_transform(test[feature_names].values)

    # when you convert labels into log before training. Root Mean Square Error becomes Root Mean Square Log Error
    y = np.log(train["trip_duration"].values + 1)

    Xtr, Xv, Ytr, Yv = train_test_split(X_train, y, test_size=0.2, random_state=1987)

    model = SVR(kernel="rbf", C=1e3, gamma=0.1)
    model.fit(Xtr, Ytr)
    Y_pred = model.predict(Xv)

    error = np.sqrt(mean_squared_error(Yv, Y_pred))

    print("SVM regression error ={}".format(error))

    end = time.time()
    print("Execution time for SVM ={}".format(end - start))
    print("---------------------------------------")


def dec_tree_r(train, test):

    start = time.time()

    # uncomment this code if you want to perform grid-search hyper-parameter tuning.
    # we have used GridSearch from sklearn. You could also use other standard
    # libraries such as hyper-opt

    """"
    cv_param ={"max_depth": [5, 10, 15, 20, 25], "min_samples_leaf": [5, 10, 15, 20, 25], "min_samples_split": [5, 10, 15, 20, 25]}

    y = np.log(train["trip_duration"].values + 1)
    x = train[feature_names].values

    print("Tuning Decision Tree")

    model = DecisionTreeRegressor()
    grid_search = GridSearchCV(model, cv_param, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(x, y)

    print("Best Estimator={}".format(grid_search.best_params_))

    print(grid_search.best_estimator_)
    cv_res = grid_search.cv_results_

    print("Cross Validation Results")
    for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
        print(np.sqrt(-mean_score), params)
    """
    y = np.log(train["trip_duration"].values + 1)

    Xtr, Xv, Ytr, Yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

    print("Training a Random Forest Regressor")

    model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=25, min_samples_split=10)
    model.fit(Xtr, Ytr)
    Y_pred = model.predict(Xv)

    error_ = np.sqrt(mean_squared_error(Yv, Y_pred))
    end = time.time()

    print("RMSLE with Decision Tree ={}".format(error_))
    print("Execution time for tuning ={}".format(end - start))
    print("---------------------------------")


def ran_for_r(train, test):

    start = time.time()
    y = np.log(train["trip_duration"].values + 1)

    Xtr, Xv, Ytr, Yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

    print("Training a Random Forest Regressor")

    model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=10, n_jobs=-1)
    model.fit(Xtr, Ytr)
    Y_pred = model.predict(Xv)

    error_ = np.sqrt(mean_squared_error(Yv, Y_pred))

    print("RMSLE with Random forest = {}".format(error_))

    end = time.time()
    print("Random Forest Execution Time={}".format(end - start))
    print("-------------------------------------------")


def xgb_r(train, test):
    start = time.time()
    print("Training XG boost Regressor")

    y = np.log(train["trip_duration"].values + 1)
    Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)
    dtest = xgb.DMatrix(test[feature_names].values)

    watchlist =[(dtrain, "train"),(dvalid, "valid")]

    # This is inefficient grid search. We have used hyperopt library with Tree of Parzen Estimators (TPE)
    # for actual tuning

    """
    print("---> Starting grid search")

    i = 0
    for MCW in [10, 20, 50, 75, 100]:
        for ETA in [0.05, 0.01, 0.1, 0.15]:
            for CS in [0.3, 0.4, 0.5]:
                for MD in [4, 6, 8, 10, 12, 15]:
                    for SS in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        for LAMBDA in [0.5, 0.8, 1, 1.5, 2, 3]:
                            xgb_pars = {"min_child_weight": MCW, "eta": ETA, "cosample_bytree": CS, "max_depth": MD,
                                             "subsample": SS, "lambda": LAMBDA,
                                             "nthread": -1, "booster": "gbtree", "silent": 1, "eval_metric": "rmse",
                                             "objective": "reg:linear"}

                            print("{}th combination of hyper-parameters".format(i))
                            print(xgb_pars)
                            model = xgb.train(xgb_pars, dtrain, 500, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=10)
                            print("Modeling RSMLE %5f" % model.best_score)
                            i = i + 1
                            print("------------------------------------------")
    end = time.time()

    print("Training time {} seconds".format(end - start))

    print("-------------------------------")
    """
    xgb_pars = {"min_child_weight": 2.0, "eta": 0.025, "cosample_bytree": 0.45, "max_depth": 20,
                                             "subsample": 1.0,
                                             "nthread": -1, "booster": "gbtree", "silent": 1, "eval_metric": "rmse",
                                             "objective": "reg:linear"}

    model = xgb.train(xgb_pars, dtrain, 1000, watchlist, early_stopping_rounds=150, maximize=False, verbose_eval=10)
    print("Modeling RSMLE %5f" % model.best_score)

    #print(model.get_parameters())

    end = time.time()
    print("XGBR execution time ={}".format(end - start))
    print("------------------------------------------")

    # Random CV search with 5 fold cross validation
    """
    cv_params = {"max_depth": [4, 6, 8, 10, 12, 15], "min_child_weight": [10, 20, 50, 75, 100],
                 "eta": [0.05, 0.01, 0.1, 0.15], "cosample_bytree": [0.3, 0.4, 0.5], "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
                 "lambda": [0.5, 0.8, 1, 1.5, 2, 3]}
    
    ind_params = {'n_estimators': 1000, 'seed': 0, 'objective': 'reg:linear'}
    
    optimize_model = RandomizedSearchCV(xgb.XGBRegressor(**ind_params),cv_params,scoring="accuracy", cv=5, n_jobs=-1,
                                        n_iter=5)
    
    optimize_model.fit(d_train, y)
    
    print("These are cross-validation results")
    print(optimize_model.cv_results_)
    print("-----------------------")
    print("Best estimator")
    print(optimize_model.best_estimator_)
    print("-----------------------")
    print("Best Score")
    print(optimize_model.best_score_)
    print("------------------------")
    print("Best Parameters")
    print(optimize_model.best_params_)
    print("----------------------")
    """

    # calculating feature importance
    
    print("Feature importance column")
    feature_importance_dict = model.get_fscore()
    
    f_1 = pd.DataFrame({"f": list(feature_importance_dict.keys()), "importance": list(feature_importance_dict.values())})
    
    print(feature_importance_dict)
    f_1.sort_values(by="importance", inplace= True)
    
    # Plot feature vs importance curve
    
    print("plotting feature importance")
    
    #f_1.plot(kind = 'barh', x = 'f', figsize = (8,8), color = 'orange')
    
    fig, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(model, height=0.8, ax=ax)
    plt.savefig("feature_importance.png")
    print("---------------------------")
    
    # Predict on test set
    
    ypred = model.predict(dtest)
    
    fig, ax = plt.subplots(ncols=2)
    ax[0].scatter(ypred, yv, s=0.1,alpha=0.1)
    ax[0].set_xlabel("log(prediction)")
    ax[0].set_ylabel("log(ground_truth)")
    ax[1].scatter(np.exp(ypred), np.exp(yv), s=0.1, alpha=0.1)
    ax[1].set_xlabel("prediction")
    ax[1].set_ylabel("ground truth")
    plt.title("Predictions")
    plt.savefig("groundtruth_vs_validation.png")
    
    # Model submission
    
    ytest = model.predict(dtest)
    
    if test.shape[0] == ytest.shape[0]:
        print("Test is successful")
    else:
        print("Oops! There is some problem with dimention")
        
    test["trip_duration"] = np.exp(ytest) - 1
    
    test[["id", "trip_duration"]].to_csv("groupthree_submission.csv.gz", index= False, compression = "gzip")
    
    print("validtaion data prediction mean = {}".format(ypred.mean()))
    print("test data prediction mean = {}".format(ytest.mean()))
    
    fig, ax = plt.subplots(nrows=2, sharex= True, sharey= True)
    sns.distplot(ypred, ax=ax[0], color="blue", label="validation prediction")
    sns.distplot(ytest, ax=ax[1], color="green", label="test prediction")
    ax[0].legend(loc=0)
    ax[1].legend(loc=0)
    plt.savefig("validation_test_predict.png")


def gbt_r(train, test):

    start = time.time()
    print("sklearn GBRT is running")
    y = np.log(train["trip_duration"].values + 1)

    Xtr, Xv, Ytr, Yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

    model = GradientBoostingRegressor(max_depth=15, learning_rate=1.0, n_estimators=1000, min_samples_leaf=25,
                                      min_samples_split=10)
    model.fit(Xtr, Ytr)
    print("Predicting Results")
    Y_pred = model.predict(Xv)
    mse = mean_squared_error(Yv, Y_pred)
    rmse = np.sqrt(mse)

    print("Accuracy with Gradient Boosted Tree={}".format(rmse))
    end = time.time()

    print("sklearn GBRT execution time ={}".format(end - start))
    print("-----------------------------------------")


def three_r(train, test):

    # This was custom implementation. Random forest is bagging with decision tree as base estimator. We tried to combine
    # it with boosting .i.e. we trained next Random Forest on the error of previous .
    # This approach gives better accuracy than Decision Trees and with correct hyper-parameter
    # tuning accuracy can increase
    start = time.time()
    print("Training Custom Regressor")

    y = np.log(train["trip_duration"].values + 1)

    Xtr, Xv, Ytr, Yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

    print("Three Regression is Running")
    rand_for1 = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=25, n_jobs=-1, min_samples_split=10)
    rand_for1.fit(Xtr, Ytr)

    y2 =Ytr - rand_for1.predict(Xtr)
    rand_for2 = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=25, n_jobs=-1, min_samples_split=10)
    rand_for2.fit(Xtr, y2)

    y3 = y2 - rand_for2.predict(Xtr)
    rand_for3 = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=25, n_jobs=-1, min_samples_split=10)
    rand_for3.fit(Xtr, y3)

    Y_pred = sum(rand_for.predict(Xv) for rand_for in (rand_for1, rand_for2, rand_for3))

    rmse = np.sqrt(mean_squared_error(Yv, Y_pred))

    print("New Three Reg RMSLE ={}".format(rmse))

    end = time.time()
    print("Execution Time of Custom Regressor ={}".format(end - start))
    print("-----------------------------------------")

if __name__ == "__main__":

    # import engineered features
    # Performance metrics is root mean square log error

    print("Reading Data")
    train_weather = pd.read_csv("./nyc/train_with_weather.csv")
    test_weather = pd.read_csv("./nyc/test_with_weather.csv")

    do_not_use_for_train = ["id", "pickup_datetime", "trip_duration", "pickup_date", "dropoff_datetime", "avg_speed_h",
                            "avg_speed_m", "avg_speed"]

    feature_names = [f for f in train_weather.columns if f not in do_not_use_for_train]

    print("Starting Training")
    print("Features with weather: Decision Tree")
    dec_tree_r(train=train_weather, test=train_weather)
    print("Features with weather: Random Forest Tree")
    ran_for_r(train=train_weather, test=test_weather)
    print("Features with weather: Xg boost")
    xgb_r(train=train_weather, test=test_weather)
    print("Feature with weather: sklearn GBRT")
    gbt_r(train=train_weather, test=test_weather)
    print("Three Regression Algorithm")
    three_r(train=train_weather, test=test_weather)


