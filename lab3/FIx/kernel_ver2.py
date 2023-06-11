from __future__ import division
from math import radians, cos, sin, asin, sqrt, exp
from datetime import datetime
from pyspark import SparkContext
from pyspark.broadcast import Broadcast
import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km

def day_difference(day1, day2):
    diff = abs(datetime.strptime(str(day1), "%Y-%m-%d") - datetime.strptime(str(day2), "%Y-%m-%d"))
    no_days = diff.days
    return no_days

def hour_diff(time1, time2):
    diff = abs(datetime.strptime(time1, "%H:%M:%S") - datetime.strptime(time2, "%H:%M:%S"))
    diff = (diff.total_seconds()) / 3600
    return diff

def gaussian_kernel(u, h):
    return np.exp(-u**2 / (2 * h**2))

def sum_kernel(distance_kernel, day_kernel, time_kernel):
    res = distance_kernel + day_kernel + time_kernel
    return res

def product_kernel(distance_kernel, day_kernel, time_kernel):
    res = distance_kernel * day_kernel * time_kernel
    return res

# Value to predict
target_latitude = 58.68681
target_longitude = 15.92183
target_date = '2014-05-17'
hour_list= ["00:00:00", "22:00:00", "20:00:00", "18:00:00", "16:00:00", "14:00:00", "12:00:00", "10:00:00", "08:00:00", "06:00:00", "04:00:00"]


# Kernel width
h_dist = 100
h_day = 15
h_time = 5

# Create SparkContext
sc = SparkContext(appName="Lab 3 ML")

temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
station_file = sc.textFile("BDA/input/stations.csv")

temperature_data = temperature_file.map(lambda x: x.split(';'))
stations_data = station_file.map(lambda x: x.split(';'))

#Filter the previous data until target_date, this can save some computation
target_date_strip = datetime.strptime(target_date, '%Y-%m-%d')
prev_temp = temperature_data.filter(lambda x: datetime.strptime(x[1], '%Y-%m-%d') <= target_date_strip)

# pre-calculate station_distkernel and broadcast it (to speed up process)
station_distkernel = stations_data.map(lambda x: (x[0], gaussian_kernel(haversine(target_longitude, target_latitude, float(x[4]), float(x[3])), h_dist))).collectAsMap()
broadcast_station_distkernel = sc.broadcast(station_distkernel)

#station id, hour, temp, day_kernel,date, cache it to save time
temperature_data_datekernel = prev_temp.map(lambda x: (x[0], x[2], x[3], gaussian_kernel(day_difference(target_date, x[1]), h_day),x[1])).cache()


predictions = {}
for hour in hour_list:
    target_hour_strip = datetime.strptime(hour, '%H:%M:%S')
    #filter; here, it will keep data from previous date, and current date before current hour. Using "or" is crucial.
    temp_filtered = temperature_data_datekernel.filter(lambda x: datetime.strptime(x[1], '%H:%M:%S') <= target_hour_strip 
                                                       or datetime.strptime(x[4], '%Y-%m-%d') < target_date_strip)

    #temp, dist_kernel, day_kernel, hour_kernel
    temp_kernels = temp_filtered.map(lambda x: (float(x[2]),
                                                broadcast_station_distkernel.value[x[0]],
                                                x[3],
                                                gaussian_kernel(hour_diff(hour, x[1]), h_time)))
    #temp,sumkernel, prodkernel                                
    temp_both_kernels = temp_kernels.map(lambda x: (x[0], sum_kernel(x[1], x[2], x[3]), product_kernel(x[1], x[2], x[3])))

    #sum_kernel, sum_kernel*temp, prod_kernel, prod_kernel*temp 
    cal_kerneltemp = temp_both_kernels.map(lambda x :(x[1],x[0]*x[1],x[2],x[0]*x[2]))
    
    #sum all the elements
    sum_kerneltemp = cal_kerneltemp.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))
    
    #sum_prediction,prod_prediction
    predictions[hour]=(sum_kerneltemp[1]/sum_kerneltemp[0],sum_kerneltemp[3]/sum_kerneltemp[2])

# Convert predictions dictionary to RDD
predictions_rdd = sc.parallelize(list(predictions.items()))
predictions_rdd = predictions_rdd.coalesce(1)
predictions_rdd = predictions_rdd.sortByKey()
# Save the RDD as text files
predictions_rdd.saveAsTextFile("BDA/output/predictions")                                                   
