from pyspark import SparkContext

sc = SparkContext(appName = 'exercise 3')
temperature_file = sc.textFile('BDA/input/temperature-readings.csv')

lines = temperature_file.map(lambda line: line.split(';'))

year_month_date_station_temp = lines.map(lambda x: ( (x[1][0:4],x[1][5:7],x[1][8:],x[0]) , (float(x[3])) ) )

year_month_date_station_temp =year_month_date_station_temp.filter(lambda x : int(x[0][0])>=1960 and int(x[0][0])<=2014)

min_max_temperatures = year_month_date_station_temp.groupByKey()
min_max_temperatures = min_max_temperatures.mapValues(lambda x: (min(x),max(x)))

# calculating daily average
avg_temperature = min_max_temperatures.map(lambda x: ((x[0][0], x[0][1],x[0][3]), (x[1][0] + x[1][1]) / 2))

#add count column 
avg_temperature = avg_temperature.mapValues(lambda x : (x,1))

avg_monthly_temperature = avg_temperature.reduceByKey(lambda a,b : (a[0] + b[0],a[1] + b[1]) )
avg_monthly_temperature = avg_monthly_temperature.mapValues(lambda x : (x[0]/x[1]))


avg_monthly_temperature.saveAsTextFile('BDA/output/')