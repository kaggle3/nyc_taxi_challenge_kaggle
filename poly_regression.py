#Creator:- Krishna Dogney
#This program is for applying Polunomial regression
# on the featrued engineeried data
#sklearn library and pandas is used for this program.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


df =pd.read_csv('train_final_feat.csv')
df.info()

corre_mat = df.corr()

print(corre_mat["trip_duration"].sort_values(ascending=False))

#picking up the data coloumn for polynomial regression


p = df[['passenger_count','distance_h','pickup_weekday','pickup_hour','pickup_month','avg_speed_h','pickup_cluster','total_travel_time',
        'distance_dummy_manhattan',]]
s = df[['trip_duration']]

#print(p)
q = np.array(p)
X = q.astype(np.float)

Y = np.array(s)
Y = Y.astype(np.float)

#scaling the data. It is very important step.
Y[:,0]= Y[:,0]/1000
X[:,3]= q[:,3]/10
X[:,5]= q[:,5]/10
X[:,6]= q[:,6]/100
m =100

#Fitting data polnomil regression regression with degree used is 3
poly_fetures = PolynomialFeatures(degree=3,include_bias=False)
X_poly = poly_fetures.fit_transform(X)
print('poly feture')

#Training the model
print('size of X_poly is ')
print(np.shape(X_poly))
lin_reg = LinearRegression()
lin_reg.fit((X_poly),np.log(Y))

#predicting
predict_time = lin_reg.predict(X_poly)
    #print(trip_durat_transpose)

#Root mean square error calcluation.
rmse = np.sqrt(mean_squared_error(np.log(Y), predict_time))
print("square root or is")
print(rmse)