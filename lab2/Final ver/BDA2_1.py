from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql import functions as F

from pyspark.sql import HiveContext

sc = SparkContext(appName = 'exercise 1')
sqlContext = SQLContext(sc)

#sqlContext = HiveContext(sc)

# This path is to the file on hdfs
temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
lines = temperature_file.map(lambda line: line.split(";"))

# (key, value) = (year,temperature)
tempReadingsRow = lines.map(lambda p: (p[0],int(p[1].split("-")[0]),float(p[3]) ))

## Inferring schema and registering the Dataframe as a table
tempReadingsString = ["station","year","value"]

schemaTempReadings = sqlContext.createDataFrame(tempReadingsRow,tempReadingsString)

# Register the DataFrame as a table
schemaTempReadings.registerTempTable("tempReadingsTable")

#filter years 1950-2014 
schemaTempReadings = schemaTempReadings.filter((schemaTempReadings["year"]>= 1950) & (schemaTempReadings["year"]<= 2014))
#schemaTempReadings = schemaTempReadings.collect()


schemaTempReadings =  schemaTempReadings.select(['year','station','value'])

schemaTempReadingsMin = schemaTempReadings.groupBy('year','station').agg(F.min('value').alias('min'))
schemaTempReadingsMinYear = schemaTempReadingsMin.groupBy('year').agg(F.min('min').alias('min'))

year_station_mintemp = schemaTempReadingsMin.join(schemaTempReadingsMinYear, ['year', 'min']).select('year', 'station', 'min').orderBy(['min'],ascending = False)


schemaTempReadingsMax = schemaTempReadings.groupBy('year','station').agg(F.max('value').alias('max'))

schemaTempReadingsMaxYear = schemaTempReadingsMax.groupBy('year').agg(F.max('max').alias('max'))

year_station_maxtemp = schemaTempReadingsMax.join(schemaTempReadingsMaxYear, ['year', 'max']).select('year', 'station', 'max').orderBy(['max'],ascending = False)

year_station_mintemp.rdd.saveAsTextFile("BDA/output/q1_min")
year_station_maxtemp.rdd.saveAsTextFile("BDA/output/q1_max")