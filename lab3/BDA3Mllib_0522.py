from __future__ import division
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from datetime import datetime
from math import radians, cos, sin, asin, sqrt, exp
from pyspark.broadcast import Broadcast
#For features the plan is to use the distence from lon=lat=0 as orignal point
#how many dates has pass since 2013-01-01 (need to change back to 1950-01-01 for full data)
#and the hour difference from 00:00:00
#Note, since features are standardized, I think we will have to standardized the target as well, or the prediction will be super bad

def haversine(lon1, lat1, lon2=0, lat2=0):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km

def day_diff(day1, day2="2013-01-01"):
    diff = abs(datetime.strptime(str(day1), "%Y-%m-%d") - datetime.strptime(str(day2), "%Y-%m-%d"))
    no_days = diff.days
    return no_days

def hour_diff(time1, time2="00:00:00"):
    diff = abs(datetime.strptime(time1, "%H:%M:%S") - datetime.strptime(time2, "%H:%M:%S"))
    diff = (diff.total_seconds()) / 3600
    return diff


sc = SparkContext(appName="Lab 3 ML")
target_date = '2014-12-31'
target_latitude = 58.68681
target_longitude = 15.92183

target_distance =haversine(lon1=15.92183, lat1=58.68681, lon2=0, lat2=0)
target_date_diff =day_diff(day1=target_date, day2="2013-01-01")
target_hour= hour_diff(time1="02:00:00", time2="00:00:00")


temperature_file = sc.textFile("BDA/input/temperature-readings-small.csv")
temperature_data = temperature_file.map(lambda x: x.split(';'))
print(temperature_data.take(1))
print("You are here0")

target_date_strip = datetime.strptime(target_date, '%Y-%m-%d')
prev_temp = temperature_data.filter(lambda x: datetime.strptime(x[1], '%Y-%m-%d') < target_date_strip)
print(prev_temp.take(1))
print("You are here1")

station_file = sc.textFile("BDA/input/stations.csv")
stations_data = station_file.map(lambda x: x.split(';'))

stations_distance = stations_data.map(lambda x: (x[0], haversine(float(x[3]), float(x[4]), lon2=0, lat2=0)))
print(stations_distance.take(1))
print("You are here2")
stations_distance_dict = dict(stations_distance.collect())
broadcast_stations_distance = sc.broadcast(stations_distance_dict)


#training features are, daydiff, hourdiff, distance
training_temp = prev_temp.map(lambda x:(float(x[3]),day_diff(x[1], day2="2013-01-01"),hour_diff(x[2], time2="00:00:00"),broadcast_stations_distance.value.get(x[0])))
print("You are here3")
print(training_temp.take(1))

#Uncomment below line if not using standardized
#labeledpoint = training_temp.map(lambda x:LabeledPoint(x[0],[x[1],x[2],x[3]]))


####standardized####
#Or we can try not standardized it, but I did not get it work at this point
features = training_temp.map(lambda x: x[1:])
standardizer = StandardScaler()
model = standardizer.fit(features)
features_transform = model.transform(features)
label = training_temp.map(lambda x: x[0])
standardized_data = label.zip(features_transform)
labeledpoint = standardized_data.map(lambda x:LabeledPoint(x[0],[x[1]]))
print(labeledpoint.take(1))
print("above is the labeledpoint")
#####################


linearModel = LinearRegressionWithSGD.train(labeledpoint)

prediction=linearModel.predict([float(target_date_diff),target_hour,target_distance])
print(prediction)#current output will be -5.47421713265e+43
print("You are here5")
