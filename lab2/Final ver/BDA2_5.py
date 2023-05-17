from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql import functions as F

from pyspark.sql import HiveContext

sc = SparkContext(appName = 'exercise 5')
sqlContext = SQLContext(sc)

precipitaion_file = sc.textFile('BDA/input/precipitation-readings.csv')
stations_file = sc.textFile('BDA/input/stations-Ostergotland.csv')

lines = precipitaion_file.map(lambda line: line.split(';'))
stations = stations_file.map(lambda line: line.split(';'))

# (key, value) = (year,month,station,precipitation)
precipReadingRow = lines.map(lambda x: ( x[1][0:4],x[1][5:7],x[0],float(x[3]) ) )
# (key,value) = (year,month,station,
stationsReadingRow = stations.map(lambda x: (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]))

stationReadingString = ['station','name','height','latitude','longitude','readingfrom','readingto','Elavtion']

## Inferring schema and registering the Dataframe as a table
precipReadingsString = ["year","month","station","precipitation"]

schemaPrecipReadings = sqlContext.createDataFrame(precipReadingRow,precipReadingsString)
schemaStations = sqlContext.createDataFrame(stationsReadingRow,stationReadingString)

# Register the DataFrame as a table
schemaPrecipReadings.registerTempTable("PrecipReadingsTable")
schemaStations.registerTempTable('StationsTable')

schemaStations = schemaStations.select(['station'])

#filter years 1993-2016 
schemaPrecipReadings = schemaPrecipReadings.filter((schemaPrecipReadings["year"]>= 1993) & (schemaPrecipReadings["year"]<= 2016))

#join with station
schemaPrecipReadings = schemaPrecipReadings.join(schemaStations,['station'])

schemaPrecipReadings =  schemaPrecipReadings.select(['year','month','station','precipitation'])

#calculate total monthly precipitation
schemaPrecipReadingsMean = schemaPrecipReadings.groupBy('year','month','station').agg(F.sum('precipitation').alias('total')).orderBy(['total'],ascending = False)

#average over stations
schemaPrecipReadingsMean = schemaPrecipReadingsMean.groupBy('year','month').agg(F.avg('total').alias('avg')).orderBy(['year','month'],ascending = False)


schemaPrecipReadingsMean.rdd.saveAsTextFile("BDA/output")