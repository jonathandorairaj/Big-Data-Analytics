from __future__ import division
from math import radians, cos, sin, asin, sqrt, exp
from datetime import datetime
from pyspark import SparkContext
from pyspark.broadcast import Broadcast
import numpy as np

##########################
#check comment at the end#
##########################

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
target_date = '1980-05-17'

# Hyperparameters: widths for day, distance, and time
h_dist = 100
h_day = 30
h_time = 12

# Create SparkContext
sc = SparkContext(appName="Lab 3 ML")

temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
station_file = sc.textFile("BDA/input/stations.csv")

temperature_data = temperature_file.map(lambda x: x.split(';'))
stations_data = station_file.map(lambda x: x.split(';'))

target_date_strip = datetime.strptime(target_date, '%Y-%m-%d')
prev_temp = temperature_data.filter(lambda x: datetime.strptime(x[1], '%Y-%m-%d') < target_date_strip)

# Calculate station_distkernel and broadcast it
station_distkernel = stations_data.map(lambda x: (x[0], gaussian_kernel(haversine(target_longitude, target_latitude, float(x[4]), float(x[3])), h_dist))).collectAsMap()
broadcast_station_distkernel = sc.broadcast(station_distkernel)

temperature_data_datekernel = prev_temp.map(lambda x: (x[0], x[1], x[2], x[3], gaussian_kernel(day_difference(target_date, x[1]), h_day))).cache()

def forecast(use_sum_kernel, target_latitude, target_longitude, target_date, temperature_data, stations_data):

    h_time = 12

    predictions = {}

    for hour in ["00:00:00", "22:00:00", "20:00:00", "18:00:00", "16:00:00", "14:00:00", "12:00:00", "10:00:00", "08:00:00", "06:00:00", "04:00:00"]:
        temp_kernels = temperature_data.map(lambda x: (float(x[3]),
                                                broadcast_station_distkernel.value[x[0]],
                                                x[4],
                                                gaussian_kernel(hour_diff(hour, x[2]), h_time))
                                    )

        if use_sum_kernel:
            temp_kernel = temp_kernels.map(lambda x: (x[0], sum_kernel(x[1], x[2], x[3])))
        else:
            temp_kernel = temp_kernels.map(lambda x: (x[0], product_kernel(x[1], x[2], x[3])))

        kernel_kerneltemp = temp_kernel.map(lambda x: (x[1], x[0]*x[1]))
        sum_kernel_value = kernel_kerneltemp.map(lambda x: x[0]).reduce(lambda x, y: x + y)
        sum_kerneltemp = kernel_kerneltemp.map(lambda x: x[1]).reduce(lambda x, y: x + y)

        predictions[hour] = sum_kerneltemp / sum_kernel_value

    return predictions

# Use broadcast variables in the forecast function
temp_forecast_sum = forecast(use_sum_kernel=True,
                             target_latitude=target_latitude,
                             target_longitude=target_longitude,
                             target_date=target_date,
                             temperature_data=temperature_data_datekernel,
                             stations_data=broadcast_station_distkernel
                             )

temp_forecast_sum_rdd = sc.parallelize(list(temp_forecast_sum.items()))
temp_forecast_sum_rdd = temp_forecast_sum_rdd.coalesce(1)
temp_forecast_sum_rdd = temp_forecast_sum_rdd.sortByKey()
temp_forecast_sum_rdd.saveAsTextFile("BDA/output/sum")

temp_forecast_prod = forecast(use_sum_kernel=False,
                              target_latitude=target_latitude,
                              target_longitude=target_longitude,
                              target_date=target_date,
                              temperature_data=temperature_data_datekernel,
                              stations_data=broadcast_station_distkernel
                              )

temp_forecast_prod_rdd = sc.parallelize(list(temp_forecast_prod.items()))
temp_forecast_prod_rdd = temp_forecast_prod_rdd.coalesce(1)
temp_forecast_prod_rdd = temp_forecast_prod_rdd.sortByKey()
temp_forecast_prod_rdd.saveAsTextFile("BDA/output/prod")

#The code is slow, So I kind of cheat it to predict date on '1980-05-17', it will took 35 minutes, 
#The current out put are
#SUM kernel:
#('00:00:00', 4.7135832609073214)
#('04:00:00', 4.8050296922608036)
#('06:00:00', 4.8601404936976671)
#('08:00:00', 4.9179656286143931)
#('10:00:00', 4.9764221253256427)
#('12:00:00', 5.0338658572589612)
#('14:00:00', 5.0890764688015908)
#('16:00:00', 5.1412472631395394)
#('18:00:00', 5.1899792963709128)
#('20:00:00', 5.2352777393214946)
#'22:00:00', 5.2775474204568633)

#PROD kernel:
#('00:00:00', 4.9402348514265819)
#('04:00:00', 5.2830080416976957)
#('06:00:00', 5.4445756830943601)
#('08:00:00', 5.5970589228312653)
#('10:00:00', 5.7386992688163403)
#('12:00:00', 5.8679321974740652)
#('14:00:00', 5.9834430576649913)
#('16:00:00', 6.0842105297079812)
#('18:00:00', 6.1695345970395685)
#('20:00:00', 6.239047486186724)
#('22:00:00', 6.2927076640939434)