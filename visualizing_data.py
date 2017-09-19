#Creator:- Krishna Dogney
#This program is for visualizing data in specific time frame
#For example taxi ride between 8am to 10am and data and time extraction form the
#library numpy and striptime is used

import csv
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#reading csv file
f = open('first_train.csv')

csv_f = csv.reader(f)
row_distance=[]
row_trip_duration=[]
row_pickup_time = []
row_trip_duration_int_scaling = []
row_pickup_time_datetime = []
row_trip_duration_int_scaling_79 = []
row_distance_79 = []
row_pickup_time_79 = []
x_mat_ones_append =[]
row_pickup_month = []


#extracting coloumn of specific name for csv file
for x in csv_f:
   row_distance.append(x[5])
   row_trip_duration.append(x[6])
   row_pickup_time.append(x[2])
   row_pickup_month.append(x[1])

row_distance.remove('distance')
row_trip_duration.remove('trip_duration')
row_pickup_time.remove('pickup_time')
row_pickup_month.remove('pickup_date')

for j in range(len(row_pickup_time)):
    pickup_time = row_pickup_time[j]
    datetime_object = datetime.strptime(pickup_time, '%H:%M:%S').time()
    row_pickup_time_datetime.append(datetime_object.hour)

    pickup_month = row_pickup_month[j]
    datemonth_object = datetime.strptime(pickup_time, '%H:%M:%S').time()
    row_pickup_month.append(datemonth_object.month)
#print(row_pickup_time_datetime)

print("month")
print(row_pickup_month)


row_distance_float = list(map(float,row_distance))
row_trip_duration_int = list(map(int,row_trip_duration))
#N = len(row_trip_duration_int)-1
#for i in range(0,N):
 #  row_trip_duration_int_scaling.append()

scaling_factor = 1000

row_trip_duration_int_scaling = [x / scaling_factor for x in row_trip_duration_int]


#now = datetime.now()
#am_7 = now.replace(hour=7, minute=0, second=0, microsecond=0)
#am_9 = now.replace(hour=9, minute=0, second=0, microsecond=0)

for k in range(len(row_pickup_time)):
    if (row_pickup_time_datetime[k]>8):
        if (row_pickup_time_datetime[k]<10):

            row_distance_79.append(row_distance_float[k])
            row_trip_duration_int_scaling_79.append(row_trip_duration_int_scaling[k])




plt.scatter(row_distance_79,row_trip_duration_int_scaling_79)
plt.show()
