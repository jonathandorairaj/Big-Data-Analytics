from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql import functions as F

from pyspark.sql import HiveContext

sc = SparkContext(appName = 'exercise 3')
sqlContext = SQLContext(sc)

# This path is to the file on hdfs
temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
lines = temperature_file.map(lambda line: line.split(";"))

# (key, value) = (year,month,date,station,temperature)
tempReadingsRow = lines.map(lambda x: ( x[1][0:4],x[1][5:7],x[1][8:],x[0] , float(x[3]) ) )

## Inferring schema and registering the Dataframe as a table
tempReadingsString = ["year","month","date","station","temperature"]

schemaTempReadings = sqlContext.createDataFrame(tempReadingsRow,tempReadingsString)

# Register the DataFrame as a table
schemaTempReadings.registerTempTable("tempReadingsTable")

#filter years 1950-2014 
schemaTempReadings = schemaTempReadings.filter((schemaTempReadings["year"]>= 1960) & (schemaTempReadings["year"]<= 2014))


schemaTempReadings =  schemaTempReadings.select(['year','month','station','temperature'])

schemaTempReadingsMean = schemaTempReadings.groupBy('year','month','station').agg(F.mean('temperature').alias('avg')).orderBy(['avg'],ascending = False)

schemaTempReadingsMean.rdd.saveAsTextFile("BDA/output")