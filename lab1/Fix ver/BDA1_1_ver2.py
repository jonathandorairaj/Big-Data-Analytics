from pyspark import SparkContext

sc = SparkContext(appName = "exercise 1")
# This path is to the file on hdfs
temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
lines = temperature_file.map(lambda line: line.split(";"))

# (key, value) = (year,temperature)
year_temperature = lines.map(lambda x: (x[1][0:4], (x[0],float(x[3]))))

#filter
year_temperature = year_temperature.filter(lambda x: int(x[0])>=1950 or int(x[0])<=2014)

#Get max
max_temperatures = year_temperature.reduceByKey(lambda a, b: (a[0], a[1]) if a[1] > b[1] else (b[0],b[1]))
#max_temperatures = max_temperatures.sortBy(ascending = False, keyfunc=lambda k: k[1])


#get min 
min_temperatures = year_temperature.reduceByKey(lambda a, b: (a[0], a[1]) if a[1] < b[1] else (b[0],b[1]))
#min_temperatures = min_temperatures.sortBy(ascending = False, keyfunc=lambda k: k[1])

joineddf = max_temperatures.join(min_temperatures)
joineddf = joineddf.sortBy(ascending = False, keyfunc = lambda k : k[1][0][1])

#print(max_temperatures.collect())

# Following code will save the result into /user/ACCOUNT_NAME/BDA/output folder
joineddf.saveAsTextFile("BDA/output")