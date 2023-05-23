#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from datetime import datetime
from math import exp,asin



from pyspark import SparkContext
from math import radians, sin, cos, sqrt

def haversine_distance(latitude_1, long_1, latitude_2, long_2):

    # Convert lat and long to radians and caclulate distance between them 
    dlon = radians(long_2) - radians(long_1)
    dlat = radians(latitude_2) - radians(latitude_1)
    
    # Haversine formula
    a = sin(dlat/2)**2 + cos(radians(latitude_1)) * cos(radians(latitude_2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    earth_radius = 6371  # Radius of the Earth in kilometers
    distance = earth_radius * c
    
    return distance

def day_difference(day1,day2):
    #diff = abs(datetime.strptime(day1, "%Y-%m-%d").date() - datetime.strptime(day2, "%Y-%m-%d").date())
    diff = abs(day1.date() - day2.date())
    no_days = diff.days
    return no_days
def hour_diff(time1,time2):
    diff = abs(datetime.strptime(time1,"%H:%M:%S") - datetime.strptime(time2,"%H:%M:%S"))
    diff = (diff.total_seconds())/3600
    return diff
    
def sum_kernel(distance_kernel,day_kernel,time_kernel):
    res = distance_kernel + day_kernel + time_kernel
    return res

def product_kernel(distance_kernel,day_kernel,time_kernel):
    res = distance_kernel * day_kernel * time_kernel
    return res


sc = SparkContext(appName = "Lab 3 ML")

temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
station_file = sc.textFile("BDA/input/stations.csv")

temperature = temperature_file.map(lambda x : x.split(';'))
stations = station_file.map(lambda x : x.split(';'))

#temperature = temperature.map(lambda x: (x[0],(x[1],x[2], float(x[3]))))
stations = stations.map(lambda x: (x[0],x[3],x[4]))


### Value to predict
target_latitude = 57.68681
target_longitude = 11.92183
target_date = '2014-05-17'

#hyperparameters
# widths for day,distance and time 
h_dist = 100
h_day = 15
h_time = 5


# Kernel function
def gaussian_kernel(u, h):
    return np.exp(-u**2 / (2 * h**2))

# calculating distance between target point and all stations in stations rdd

stations = stations.map(lambda x: (x[0], gaussian_kernel(haversine_distance(target_latitude, target_longitude, float(x[1]), float(x[2])), h_dist)))
stations = sc.broadcast(stations.collectAsMap()).value



# filter out temperature data after the target date 
target_date = datetime.strptime(target_date,'%Y-%m-%d')
temperature_filtered = temperature.filter(lambda x : datetime.strptime(x[1],'%Y-%m-%d') < target_date)



# temperature, time ,station, kernel value of difference in days
temperature_filtered = temperature_filtered.map( lambda x :(float(x[3]),x[2],stations[(x[0])],gaussian_kernel(day_difference(target_date,datetime.strptime(x[1],'%Y-%m-%d')),h_day))).cache()



res_sum = {}
res_prod = {}
time_list = ["00:00:00", "22:00:00", "20:00:00", "18:00:00", "16:00:00", "14:00:00","12:00:00", "10:00:00", "08:00:00", "06:00:00", "04:00:00"]


# Iterate over each time in the time_list
for time in time_list:
    print("Executing for - {}".format(time))
    
    # Create a new RDD, kern_matrix, by applying a lambda function to each line in the temperature_filtered RDD
    # The lambda function extracts the relevant values from the line and applies the gaussian_kernel function
    # temperature, kern_distance, day_diff, hour_diff
    kern_matrix = temperature_filtered.map(lambda line: (line[0], line[2], line[3],
                                                                   gaussian_kernel(hour_diff(line[1], time), h_time)))
    
    # Create a new RDD, kTransform, by applying a lambda function to each line in the kern_matrix RDD
    # The lambda function calculates the sum and product using the k_sum and k_prod functions
    sum_kern = kern_matrix.map(lambda line: (line[0],sum_kernel(line[1], line[2],line[3])))
    prod_kern = kern_matrix.map(lambda line: (line[0],product_kernel(line[1],line[2],line[3])))
    
    # Perform a reduce operation on the kTransform RDD to calculate the total sum of the product and sum values
    totalSum = (sum_kern.map(lambda line: (line[1], line[0]*line[1]))).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    
    # Perform another reduce operation on the kTransform RDD to calculate the product sum of the product and sum values
    prodSum = (prod_kern.map(lambda line: (line[1], line[0]*line[1]))).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    
    # Append the calculated average sum and average product to the sumOut and prodOut lists
    res_sum[time] = (totalSum[1] / totalSum[0])
    res_prod[time] = (prodSum[1] / prodSum[0])

#res_sum.saveAsTextFile('BDA/output/SumKernelResults')
#res_prod.saveAsTextFile('BDA/output/ProdKernelResults')

# converting to rdd and saving 
rdd = sc.parallelize([(k, v) for k, v in res_prod.items()])
rdd.saveAsTextFile('BDA/output/ProdKernelResults')

rdd = sc.parallelize([(k, v) for k, v in res_sum.items()])
rdd.saveAsTextFile('BDA/output/SumKernelResults')


