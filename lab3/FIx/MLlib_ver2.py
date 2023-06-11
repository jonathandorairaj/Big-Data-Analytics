from __future__ import division
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from datetime import datetime
from math import radians, cos, sin, asin, sqrt, exp
from pyspark.broadcast import Broadcast
from pyspark.mllib.tree import RandomForest

# For data handling, the plan is as below
# Use the distance difference from the geographical center of sweden
# Use how many dates have passed since 1950/01/01
# Use the hour difference from 00:00:00
# Which means the features are distance,day_diff, hour_diff 

# Function to calculate haversine distance
def haversine(lon1, lat1, lon2=16.321998712, lat2=62.38583179): #geographical center of Sweden
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km

def day_diff(day1, day2="1950-01-01"):
    diff = abs(datetime.strptime(str(day1), "%Y-%m-%d") - datetime.strptime(str(day2), "%Y-%m-%d"))
    no_days = diff.days
    return no_days

def hour_diff(time1, time2="00:00:00"):
    diff = abs(datetime.strptime(time1, "%H:%M:%S") - datetime.strptime(time2, "%H:%M:%S"))
    diff = (diff.total_seconds()) / 3600
    return diff


sc = SparkContext(appName="Lab 3 ML")
target_date = '2014-5-17'
target_latitude = 58.68681
target_longitude = 15.92183

target_distance = haversine(lon1=target_longitude, lat1=target_latitude) 
target_date_diff = day_diff(day1=target_date)

temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
temperature_data = temperature_file.map(lambda x: x.split(';'))

#filter out data 
target_date_strip = datetime.strptime(target_date, '%Y-%m-%d')
prev_temp = temperature_data.filter(lambda x: datetime.strptime(x[1], '%Y-%m-%d') <= target_date_strip)


station_file = sc.textFile("BDA/input/stations.csv")
stations_data = station_file.map(lambda x: x.split(';'))

#Same for kernel model, broadcast distance for faster access
broadcast_stations_distance = sc.broadcast(stations_data.map(lambda x: (x[0], haversine(lat1=float(x[3]), 
                                                                                        lon1=float(x[4])))).collectAsMap())


#temp,day_difference, hour_difference,station_difference
training_temp = prev_temp.map(lambda x: (
    float(x[3]), day_diff(x[1]), hour_diff(x[2]), broadcast_stations_distance.value[x[0]]))

#standardized
features = training_temp.map(lambda x: x[1:])
standardizer = StandardScaler()
model = standardizer.fit(features)
features_transform = model.transform(features)
label = training_temp.map(lambda x: x[0])
standardized_data = label.zip(features_transform)
#create training data with standardized features
train_data = standardized_data.map(lambda x: LabeledPoint(x[0], [x[1]])).cache()
#has the form [LabeledPoint(6.8, [407.396549514,0.998736575617,0.0])]


# Create 2 hours interval
hour_list = ["00:00:00", "22:00:00", "20:00:00", "18:00:00", "16:00:00", "14:00:00",
             "12:00:00", "10:00:00", "08:00:00", "06:00:00", "04:00:00"]

prediction = {}
for hour in hour_list:
    target_hour = hour_diff(time1=hour)
    target_feature = Vectors.dense([float(target_date_diff), target_hour, target_distance])
    target_features_rdd = sc.parallelize([target_feature])
    standardized_target_features = model.transform(target_features_rdd)

    #calculate the threshold to filter out the data, since the train_data is already normalized
    #so here the threshold is also normalized
    hour_threshold = standardized_target_features.first()[1]
    date_threshold = standardized_target_features.first()[0]
    
    ##Create models##
    #Like in Kernel model, using filter to keep data that is from previous day plus the data from target date before current hour
    current_train_data = train_data.filter(lambda x: x.features[1] < hour_threshold or x.features[0] < date_threshold)

    dt_model = DecisionTree.trainRegressor(current_train_data, categoricalFeaturesInfo={}, maxDepth= 2)
    rf_model = RandomForest.trainRegressor(current_train_data, categoricalFeaturesInfo={}, numTrees=2, maxDepth = 2, maxBins= 4)
    
    dt_predictions = dt_model.predict(standardized_target_features).collect()[0] 
    rf_predictions = rf_model.predict(standardized_target_features).collect()[0]

 
    prediction[hour]=(rf_predictions,dt_predictions)
       

sc.parallelize(prediction.items()).coalesce(1).sortByKey().saveAsTextFile("BDA/output/prediction")