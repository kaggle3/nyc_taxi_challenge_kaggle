import sklearn,pandas as pd,numpy as np
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split as data_split
from sklearn.metrics import mean_squared_error

train = pd.read_csv('train_with_weather.csv')

train = train.drop(["_merge"], axis=1)

do_not_use =["id", "pickup_datetime", "trip_duration", "pickup_date", "dropoff_datetime", "avg_speed_h", "avg_speed_m",
             "avg_speed"]

feature_names =[f for f in train.columns if f not in do_not_use]

y = np.log(train["trip_duration"].values +1)

Xtr, Xv, Ytr, Yv = data_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

#print("Training first model")



print("Training second model")

ada_reg_2=AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=15, min_samples_leaf=25, min_samples_split=10), n_estimators=2)
ada_reg_2.fit(Xtr,Ytr)

y_pred_2=ada_reg_2.predict(Xv) 

mse_2 = mean_squared_error(Yv, y_pred_2)

print("PMSLE for Reg_2={}".format(np.sqrt(mse_2)))
"""
plt.figure()
plt.scatter(X_train, y, c="k", label="training samples")
plt.plot(X_train, y_pred_1, c="g", label='linear error', linewidth=2)
plt.plot(X, y_pred_2, c="r", label="square error", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("AdaBoosted Decision Tree Regression")
plt.legend()
plt.show()
"""