#Q2
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql import functions as F

sc = SparkContext(appName="exercise 1")
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
lines = temperature_file.map(lambda line: line.split(";"))

year_month_temperature = lines.map(lambda x: (x[1][0:4],x[1][5:7],x[0],float(x[3])))

schema = StructType([
    StructField("year", StringType(), True),
    StructField("month", StringType(), True),
    StructField("station", StringType(), True),
    StructField("value", FloatType(), True)
])

year_month_temperature_df = sqlContext.createDataFrame(year_month_temperature, schema)

filtered_year_month_temperature = year_month_temperature_df.filter((year_month_temperature_df["year"] >= "1950") & (year_month_temperature_df["year"] <= "2014")& (year_month_temperature_df["value"] > 10))

count_temp = filtered_year_month_temperature.groupBy(["year", "month"]).count()
sort_count = count_temp.sort("count", ascending = False)
sort_count_combine = sort_count.rdd.coalesce(1)
sort_count_combine = sort_count_combine.sortBy(lambda x: x[2], ascending=False)
sort_count_combine.saveAsTextFile("BDA/output/l2_not_distinct")

distinct_temp = filtered_year_month_temperature.select(["year", "month", "station"]).distinct()
count_dist_temp = distinct_temp.groupBy(["year", "month"]).count()
sort_count_dist = count_dist_temp.sort("count", ascending = False)
sort_count_dist = sort_count_dist.rdd.coalesce(1)
sort_count_dist = sort_count_dist.sortBy(lambda x: x[2], ascending=False)
sort_count_dist.saveAsTextFile("BDA/output/l2_distinct")
