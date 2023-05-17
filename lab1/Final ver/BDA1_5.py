from pyspark import SparkContext

sc = SparkContext(appName = 'exercise 3')
precipitaion_file = sc.textFile('BDA/input/precipitation-readings.csv')
stations_file = sc.textFile('BDA/input/stations-Ostergotland.csv')

lines = precipitaion_file.map(lambda line: line.split(';'))
stations = stations_file.map(lambda line: line.split(';'))

# extracting only the station numbers, then collecting and broadcasting to make available to all nodes to filter later
stations = stations.map(lambda x: x[0])
stations = stations.collect()
stations = sc.broadcast(stations).value

year_month_station_precip = lines.map(lambda x: ( (x[1][0:4],x[1][5:7],x[0]) , (float(x[3])) ) )
#filter for years
year_month_station_precip =year_month_station_precip.filter(lambda x : int(x[0][0])>=1993 and int(x[0][0])<=2016)

#filter for stations in Ostergotland
year_month_station_precip = year_month_station_precip.filter(lambda x : x[0][2] in stations)

#summing up to get total precipitation per month,station and year
year_month_station_precip = year_month_station_precip.reduceByKey(lambda a,b: a+b)

# remap to add count column in value
monthly_precipitation = year_month_station_precip.map(lambda x: ((x[0][0],x[0][1]),(x[1],1)))

#summing up
monthly_precipitation = monthly_precipitation.reduceByKey(lambda a,b: (a[0] + b[0],a[1] + b[1]))

#obtaining average
avg_monthly_precipitaion= monthly_precipitation.mapValues(lambda x : (x[0]/x[1]))


avg_monthly_precipitaion.saveAsTextFile('BDA/output/')