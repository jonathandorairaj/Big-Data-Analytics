from pyspark import SparkContext

sc = SparkContext(appName = "exercise 4")
# This path is to the file on hdfs
temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
precipitation_file = sc.textFile("BDA/input/precipitation-readings.csv")

temperature_lines = temperature_file.map(lambda line: line.split(";"))
precipitation_file = precipitation_file.map(lambda line: line.split(";"))

get_temperature = temperature_lines.map(lambda x: (x[0],float(x[3])))
#(station, date, precipitation)
get_precipitation = precipitation_file.map(lambda x: ((x[0], x[1]), float(x[3])))
sumdaily_precipitation = get_precipitation.reduceByKey(lambda x, y: x + y)
#map it with (station, precipitation)
sumdaily_precipitation = sumdaily_precipitation.map(lambda x: ((x[0][0]), float(x[1]))) 

max_temp = get_temperature.reduceByKey(max)
filter_temp = max_temp.filter(lambda x : x[1]>25 and x[1]<30)


max_prec = sumdaily_precipitation.reduceByKey(max)

filter_prec = max_prec.filter(lambda x : x[1]>100 and x[1]<200)


join_output= filter_temp.join(filter_prec)
join_output = join_output.coalesce(1)
join_output_sort = join_output.sortByKey()
join_output_sort.saveAsTextFile("BDA/output/temp_prec_sort")