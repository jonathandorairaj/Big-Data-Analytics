#Q4
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql import functions as F

sc = SparkContext(appName="exercise 1")
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
temperature_lines = temperature_file.map(lambda line: line.split(";"))
get_temperature = temperature_lines.map(lambda x: (x[0],float(x[3])))
tempschema = StructType([
    StructField("station", StringType(), True),
    StructField("temp", FloatType(), True)
])
temperature_df = sqlContext.createDataFrame(get_temperature, tempschema)
station_max_temp = temperature_df.groupBy("station").agg(F.max("temp").alias('temp'))
filter_temp = station_max_temp.filter((station_max_temp["temp"] >= "25") & (station_max_temp["temp"] <= "30"))

precipitation_file = sc.textFile("BDA/input/precipitation-readings.csv")
precipitation_file = precipitation_file.map(lambda line: line.split(";"))
get_percipitation = precipitation_file.map(lambda x: (x[0],float(x[3])))
precschema = StructType([
    StructField("station", StringType(), True),
    StructField("prec", FloatType(), True)
])
percipitation_df = sqlContext.createDataFrame(get_percipitation, precschema)
station_max_perc = percipitation_df.groupBy("station").agg(F.max("prec").alias('prec'))
filter_perc = station_max_perc.filter((station_max_perc["prec"] >= "100") & (station_max_perc["prec"] <= "200"))

combine_temp_perc = filter_temp.join(filter_perc.alias('perc'), 'station', 'inner')

#output
combine_temp_perc_combine = combine_temp_perc.rdd.coalesce(1)
filter_temp_combine = combine_temp_perc_combine.sortBy(lambda x: x[0], ascending=False)
filter_temp_combine.saveAsTextFile("BDA/output/l2_perc_temp")


#Testoutput
#filter_temp_combine = filter_temp.rdd.coalesce(1)
#filter_temp_combine = filter_temp_combine.sortBy(lambda x: x[0], ascending=False)
#filter_temp_combine.saveAsTextFile("BDA/output/l2_temptest")

#filter_perc_combine = filter_perc.rdd.coalesce(1)
#filter_perc_combine = filter_perc_combine.sortBy(lambda x: x[0], ascending=False)
#filter_perc_combine.saveAsTextFile("BDA/output/l2_perctest")

#combine_temp_perc_combine = combine_temp_perc.rdd.coalesce(1)
#filter_temp_combine = combine_temp_perc_combine.sortBy(lambda x: x[0], ascending=False)
#filter_temp_combine.saveAsTextFile("BDA/output/l2_combtest")