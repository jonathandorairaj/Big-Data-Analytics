#Q4
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql import functions as F

sc = SparkContext(appName="exercise 1")
spark = SparkSession(sc)

# Read temperature data
temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
temperature_lines = temperature_file.map(lambda line: line.split(";"))
get_temperature = temperature_lines.map(lambda x: (x[0], float(x[3])))
tempschema = StructType([
    StructField("station", StringType(), True),
    StructField("temp", FloatType(), True)
])
temperature_df = spark.createDataFrame(get_temperature, tempschema)

# Calculate max temperature
station_max_temp = temperature_df.groupBy("station").agg(F.max("temp").alias("temp"))
filter_temp = station_max_temp.filter((station_max_temp["temp"] >= 25) & (station_max_temp["temp"] <= 30))

# Read precipitation data
precipitation_file = sc.textFile("BDA/input/precipitation-readings.csv")
precipitation_lines = precipitation_file.map(lambda line: line.split(";"))
get_precipitation = precipitation_lines.map(lambda x: (x[0], x[1], float(x[3])))
prec_schema = StructType([
    StructField("station", StringType(), True),
    StructField("date", StringType(), True),
    StructField("prec", FloatType(), True)
])
precipitation_df = spark.createDataFrame(get_precipitation, prec_schema)

# Calculate total precipitation per day
daily_precipitation = precipitation_df.groupBy("station", "date").agg(F.sum("prec").alias("total_prec"))
# Get the max precipitation of a station 
max_precipitation_per_station = daily_precipitation.groupBy("station").agg(F.max("total_prec").alias("max_total_prec"))
# Filter precipitation
filter_prec = max_precipitation_per_station.filter((max_precipitation_per_station["max_total_prec"] >= 100) & (max_precipitation_per_station["max_total_prec"] <= 200))

combine_temp_prec = filter_temp.join(filter_prec.alias('prec'), 'station', 'inner')
#output
combine_temp_prec_combine = combine_temp_prec.rdd.coalesce(1)
filter_temp_combine = combine_temp_prec_combine.sortBy(lambda x: x[0], ascending=False)
filter_temp_combine.saveAsTextFile("BDA/output/l2_prec_temp")

