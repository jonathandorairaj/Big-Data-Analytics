from pyspark import SparkContext

sc = SparkContext(appName = "exercise 2")
# This path is to the file on hdfs
temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
lines = temperature_file.map(lambda line: line.split(";"))

# (key, value) = ((Year, month), temp)
year_month_temperature = lines.map(lambda x: ((x[1][0:4],x[1][5:7],x[0]),float(x[3])))


#filter
year_month_temperature = year_month_temperature.filter(lambda x: int(x[0][0]) >= 1950 and int(x[0][0]) <=2014 and x[1] > 10)
#count = year_month_temperature.map(lambda x: (x[0], 1))
count = year_month_temperature.map(lambda x: ((x[0][0],x[0][1]), 1))
count = count.reduceByKey(lambda a, b: a + b)
count = count.coalesce(1)
count_sort = count.sortByKey().sortByKey(1)
count_sort.saveAsTextFile("BDA/output/countsort")
########################



count_distinct = year_month_temperature.map(lambda x: (x[0],1)).distinct()
count_distinct = count_distinct.map(lambda x: ((x[0][0],x[0][1]), 1))
count_distinct = count_distinct.reduceByKey(lambda a, b: a + b)
count_distinct = count_distinct.coalesce(1)
count_distinct_sort = count_distinct.sortByKey().sortByKey(1)
count_distinct_sort.saveAsTextFile("BDA/output/count_distinct_sort")