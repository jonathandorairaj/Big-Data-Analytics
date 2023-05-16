from pyspark import SparkContext

sc = SparkContext(appName = "exercise 1")
# This path is to the file on hdfs
temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
lines = temperature_file.map(lambda line: line.split(";"))

# (key, value) = (year,(station,temperature))
year_temperature = lines.map(lambda x: (x[1][0:4], (x[0],float(x[3]))))

#filter
year_temperature = year_temperature.filter(lambda x: int(x[0])>=1950 and int(x[0])<=2014)

#Get max
max_temperatures = year_temperature.reduceByKey(lambda a,b: (a[0], max(a[1], b[1])))
max_temperatures = max_temperatures.sortBy(lambda x: x[1][1], ascending=False)

#Get max
min_temperatures = year_temperature.reduceByKey(lambda a,b: (a[0], min(a[1], b[1])))
min_temperatures = min_temperatures.sortBy(lambda x: x[1][1], ascending=False)


max_temperatures_combine = max_temperatures.coalesce(1)
max_temperatures_combine = max_temperatures_combine.sortBy(lambda x: x[1][1], ascending=False)
max_temperatures_combine.saveAsTextFile("BDA/output/l1max")

min_temperatures_combine = min_temperatures.coalesce(1)
min_temperatures_combine = min_temperatures_combine.sortBy(lambda x: x[1][1], ascending=False)
min_temperatures_combine.saveAsTextFile("BDA/output/l1min")