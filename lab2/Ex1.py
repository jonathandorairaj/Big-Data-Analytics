#Q1
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql import functions as F

sc = SparkContext(appName="exercise 1")
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
lines = temperature_file.map(lambda line: line.split(";"))

year_temperature = lines.map(lambda x: (x[1][0:4], x[0], float(x[3])))

schema = StructType([
    StructField("year", StringType(), True),
    StructField("station", StringType(), True),
    StructField("value", FloatType(), True)
])

year_temperature_df = sqlContext.createDataFrame(year_temperature, schema)

filtered_year_temperature = year_temperature_df.filter((year_temperature_df["year"] >= "1950") & (year_temperature_df["year"] <= "2014"))

max_temperatures = filtered_year_temperature.groupBy('year', 'station').agg(F.max('value').alias('maxValue')).orderBy(['year', 'station', 'maxValue'], ascending=[False, False, True])
max_temperatures_combine = max_temperatures.rdd.coalesce(1)
max_temperatures_combine = max_temperatures_combine.sortBy(lambda x: x[2], ascending=False)
max_temperatures_combine.saveAsTextFile("BDA/output/l2max")



min_temperatures = filtered_year_temperature.groupBy('year', 'station').agg(F.min('value').alias('minValue')).orderBy(['year', 'station', 'minValue'], ascending=[False, False, True])
min_temperatures_combine = min_temperatures.rdd.coalesce(1)
min_temperatures_combine = min_temperatures_combine.sortBy(lambda x: x[2], ascending=False)
min_temperatures_combine.saveAsTextFile("BDA/output/l2min")