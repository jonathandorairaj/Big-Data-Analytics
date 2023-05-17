from pyspark import SparkContext

sc = SparkContext(appName = "exercise 4")
# This path is to the file on hdfs
temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
precipitation_file = sc.textFile("BDA/input/precipitation-readings.csv")

temperature_lines = temperature_file.map(lambda line: line.split(";"))
precipitation_file = precipitation_file.map(lambda line: line.split(";"))

get_temperature = temperature_lines.map(lambda x: (x[0],float(x[3])))
get_percipitation = precipitation_file.map(lambda x: (x[0],float(x[3])))

max_temp = get_temperature.reduceByKey(max)
filter_temp = max_temp.filter(lambda x : x[1]>25 and x[1]<30)


max_perc = get_percipitation.reduceByKey(max)
filter_perc = max_perc.filter(lambda x : x[1]>1000 and x[1]<200)



join_output= filter_temp.join(filter_perc)
join_output = join_output.coalesce(1)
join_output_sort = join_output.sortByKey()
join_output_sort.saveAsTextFile("BDA/output/temp_perce_sort")

### Below are for testing since no output
#filter_perc = max_perc.filter(lambda x : x[1]>0 and x[1]<15) 
#filter_perc = max_perc.filter(lambda x : x[1]>100) 
#filter_temp.saveAsTextFile("BDA/output/temp_test")
#max_perc.saveAsTextFile("BDA/output/maxperc_test")
#filter_perc.saveAsTextFile("BDA/output/filterperc_test")