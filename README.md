# nyc_taxi_challenge

This is a repository of our participation in New york City Taxi Trip Duration challenge on Kaggle (07/20/2017 - 09/15/2017). This was the first time we participated on any Kaggle competition.
Our standing : 227 / 1257 (Top 23%)

You can get data to start with from
https://www.kaggle.com/c/nyc-taxi-trip-duration/data

Programmming Language : Python
Important Libraries : Numpy, Pandas, sklearn, XGboost, Hyperopt

feature_engineering.py - This program takes data given on Kaggle website and perform feature engineering. We were able to cretate 47 features. A new CSV file is created at the end with new features added.

train_algorithms.py - This program tests different algorithm on new dataset created after feature engineering. (Remember there is no free lunch!)

After this program we found out XGboost regressor works better than any other algorithm. Next stage was hyperparameter tuning

tune_xg_boost.py - This program used Hyperopt tuning library with Tree of Parzen Estimators (TPE) algorithm (note* This library also implements Random Search). We tuned six paremeters after reading XGboost documentation. http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html
This program tests models on test data (along with saving prediction results) whose error on validation set goes below threshold set by us.

train_xgboost.py - This program train XGboost regressor on a single set of parameters. (You would like to do this after parameter tuning)

three_reg.py - This was out attempt to use both bagging and boostig in a single model. We did not get time to tune and work more on this but with near optimal values, this beat decision tree.

speedlimit.csv-Created by taking data from neywork govt website of traffic data in the city of manhattan.Contains the location(long and lat) of speed limit points and speed limit value.

adaboost.py-DTR has been used as the base estimator and number of estimators to be taken to be 2.Obsevation-no improvement from Xgboost has been seen.

weather_nyc.csv-contains weather data which has been collected from the officail website maintained by newyork state govt.


DBSCAN.py-to cluster the spatial points (long,lat) and later use these clusters to assign avg. speed limit to each of the cluster memebers.





