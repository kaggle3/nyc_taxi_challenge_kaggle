#Creator:- Krishna Dogney
#This program is for applying linear regression
# on the featrued engineeried data
#sklearn library and pandas is used for this program.

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.metrics import mean_squared_error
df1 = pd.read_csv('train_data_10000.csv')
#df2 = df1["trip_duration"].values

#print(df1.info())
y =[]
trip_dur = []
dist_tot = []
number_of_step =[]


#creating list of  featured data to be given as input yo the linear regression

distance = df1.distance_h.tolist()
pickup_date = df1.pickup_date.tolist()
trip_duration =df1.trip_duration.tolist()
pickup_hour =df1.pickup_hour.tolist()
pickup_month=df1.pickup_month.tolist()
avg_speed_h=df1.avg_speed_h.tolist()
total_travel_time=df1.total_travel_time.tolist()
pickup_weekday = df1.pickup_weekday.tolist()
Steps = df1.number_of_steps.tolist()

#Scaling the data. A very important step

scaling_factor = 100
scaling_factor1 = 10
trip_duration_scaling = [x / scaling_factor for x in trip_duration]
trip_distance_scaling = [x / scaling_factor1 for x in distance]
Steps_scaling = [x / scaling_factor1 for x in Steps]

#Selecting data from specific time range for allpying linear regression
#Example all the taxi ride booked between 7am to 9 am
#and is it a week day or weekend

for k in range(len(distance)):
    if(pickup_weekday[k]<=1<=4):
        if (pickup_hour[k]==10 or pickup_hour[k]==11 ):
            #if (pickup_hour[k]>12):
                 if(pickup_month[k]== 2):

                        trip_dur.append(trip_duration_scaling[k])
                        dist_tot.append(trip_distance_scaling[k])
                        number_of_step.append(Steps_scaling[k])

'''plt.scatter(trip_dur,dist_tot)
plt.show()'''

distance_tot = np.array([dist_tot])
trip_durat = np.array([trip_dur])
Steps_array = np.array([number_of_step])

distance_tot_transpose = distance_tot.T
trip_durat_transpose = trip_durat.T
number_of_step_transpose = Steps_array.T

#Fitting model on linear regression
lin_reg = LinearRegression()
lin_reg.fit((distance_tot_transpose),np.log(trip_durat_transpose))

#Predicting the output

predict_time = lin_reg.predict(distance_tot_transpose)
print(trip_durat_transpose)

# Calculating Error in prediction i.e. Root Mean Square Error
rmse = np.sqrt(mean_squared_error(np.log(trip_durat_transpose), predict_time))
print("square root or is")
print(rmse)
#print(np.array(dist_tot).reshape(1,-1))

print(lin_reg.coef_)
print(lin_reg.intercept_)
#print(lin_reg.intercept_)
#y = mx+c

#Plotting the data
plt.scatter(trip_dur,dist_tot)
plt.plot(y)
plt.show()
