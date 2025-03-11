from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, concat_ws, to_date, to_timestamp, date_format, lag
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.functions import(
        split, concat_ws, lit, round, count, min, max, length, when, sqrt, sum, row_number, col, expr, 
        unix_timestamp, from_unixtime, to_date, to_timestamp, date_format, weekofyear, year, greatest, lag, floor, sin, cos, atan2, radians
)

# Create a SparkSession
spark = SparkSession.builder.appName('Orange').getOrCreate()

# Read data from CSV file and change the datatype
df0 = spark.read.options(header='True', delimiter=',')\
                .csv("dataset.txt")

# create a df to store the data
df = df0.withColumn("UserID", col("UserID").cast("Integer"))\
        .withColumn("Latitude", col("Latitude").cast("Double"))\
        .withColumn("Longitude", col("Longitude").cast("Double"))\
        .withColumn("AllZero", col("AllZero").cast("Integer"))\
        .withColumn("Altitude", col("Altitude").cast("Double"))\
        .withColumn("Timestamp", col("Timestamp").cast("Double"))\


# Q1-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
this part is to convert the timestamp to the format of "yyyy-MM-dd HH:mm:ss" and add 8 hours to the timestamp, using the function "to_timestamp" and "INTERVAL 8 HOURS"
first, we use the function "concat_ws" to combine the columns "Date" and "Time" into a new column "date_time", and then convert the column "date_time" to the format of 
"yyyy-MM-dd HH:mm:ss" using the function "to_timestamp". After that, we add 8 hours to the timestamp using the function "INTERVAL 8 HOURS". The timestamp is calculated
by adding 1/3 of the time zone difference between Beijing and UTC to the original timestamp. Finally, we use the function"to_date" and "date_format" to convert the 
column "date_time" to the format of "yyyy-MM-dd" and the column "Timestamp" to the format of "HH:mm:ss" respectively.
'''

# convert the timestamp to the format of "yyyy-MM-dd HH:mm:ss" and add 8 hours to the timestamp
df1 = df.withColumn("date_time", concat_ws(" ", col("Date"), col("Time")))\
        .withColumn("date_time", to_timestamp(col("date_time")))\
        .withColumn("date_time", col("date_time") + F.expr("INTERVAL 8 HOURS"))\
        .withColumn("Timestamp", col("Timestamp") + (8 * 60) / (24 * 60))\
        .withColumn("Date", to_date(col("date_time")))\
        .withColumn("Time", date_format("date_time", "HH:mm:ss"))\
        .drop("date_time")

print("The output of Question1:")
df1.show()



# Q2-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
This part is to count the number of data points for each user and each day, and then filter the data points with count >= 5. After that, we count the number of data points
for each user and order the result by count and UserID for better readability. We use the function "groupBy" to group the data by "UserID" and "Date", and then use the
function "count" to count the number of data points for each group. After that, we use the function "where" to filter the data points with count >= 5. Then, we use the
function "groupBy" to group the data by "UserID" and count the number of data points for each user. Finally, we use the function "orderBy" to order the result by count and
UserID for better readability.
'''
# Find the users with at least 5 data points
# count the number of data points for each user and each day
df2_tmp1 = df1.groupBy("UserID", "Date").count()

# filter the data points with count >= 5
df2_tmp2 = df2_tmp1.where(col("count") >= 5)

# count the number of data points for each user
df2_tmp3 = df2_tmp2.groupby("UserID").count()

# order the result by count and UserID for better readability
df2 = df2_tmp3.orderBy(col("count").desc(), col("UserID").desc())

print("The output of Question2:")
df2.show(5)



# Q3-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
The basic idea of this part is to create a new column "YearWeek" to represent the combination of year and week of the year, and then group by "UserID" and "YearWeek" and
count the data points for each week. After that, we filter weeks with more than 100 data points and count the number of weeks with more than 100 data points. Finally, we
order the result by "UserID" for better readability. We use the function "withColumn" to create a new column "YearWeek" to represent the combination of year and week of
the year. Then, we use the function "groupBy" to group the data by "UserID" and "YearWeek" and count the data points for each week. After that, we use the function "where"
to filter weeks with more than 100 data points. Then, we use the function "groupBy" to group the data by "UserID" and count the number of weeks with more than 100 data
points. Finally, we use the function "orderBy" to order the result by "UserID" for better readability.
'''
#Find the number of weeks with more than 100 data points for each user

# Create a new column "YearWeek" to represent the combination of year and week of the year
df3_tmp1 = df1.withColumn("YearWeek", F.floor((F.floor(col("Timestamp"))-2)/7))

# Group by "UserID" and "YearWeek" and count the data points for each week
df3_tmp2 = df3_tmp1.groupBy("UserID", "YearWeek").count()

# Filter weeks with more than 100 data points
df3_tmp3 = df3_tmp2.where(col("count") > 100)

# Group by "UserID" and count the number of weeks with more than 100 data points
df3_tmp4 = df3_tmp3.groupBy("UserID").count().withColumnRenamed("count", "Weeks")

# Order the result by "UserID" for better readability
df3_tmp4 = df3_tmp4.orderBy("UserID")

print("The output of Question3:")
df3_tmp4.show(df3_tmp4.count())




# Q4-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
The general idea of this part is to find the minimum latitude for each user and then find the data points with the minimum latitude for each user. We use the function
"groupBy" to group the data by "UserID" and find the minimum latitude for each user. Then, we use the function "join" to join the data points with the minimum latitude
for each user. Finally, we use the function "orderBy" to order the result by "min_Latitude" and "Date" for better readability. We use the function "groupBy" to group the
data by "UserID" and find the minimum latitude for each user. Then, we use the function "join" to join the data points with the minimum latitude for each user. Finally,
we use the function "orderBy" to order the result by "min_Latitude" and "Date" for better readability.
'''

# Find the minimum latitude for each user
df4_tmp1 = df1.groupBy("UserID").min("Latitude")

# Find the data points with the minimum latitude for each user
df4_tmp2 = df4_tmp1.withColumnRenamed("min(Latitude)", "min_Latitude").withColumnRenamed("UserID", "ID")

# Join the data points with the minimum latitude for each user
df4_tmp3 = df4_tmp2.join(df1, (df4_tmp2.min_Latitude == df1.Latitude) & (df4_tmp2.ID == df1.UserID), "inner")

# Order the result by "min_Latitude" and "Date" for better readability
df4_tmp4 = df4_tmp3.drop("Latitude")\
                     .drop("Longitude")\
                     .drop("AllZero")\
                     .drop("Altitude")\
                     .drop("Timestamp")\
                     .drop("Time")\
                     .drop("UserID")\
                     .dropDuplicates()

# Order the result by "min_Latitude" and "Date" for better readability
df4 = df4_tmp4.orderBy(col("min_Latitude"), col("Date"))

print("The output of Question4:")
df4.show(5)



# Q5-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
The idea of this part is to find the maximum altitude and the corresponding date for each user and then find the minimum altitude and the corresponding date for each user.
After that, we calculate the span for each user and then find the maximum span for each user. Finally, we order the result by "max(Span)" for better readability. We use the
function "groupBy" to group the data by "UserID" and "Date" and find the maximum altitude and the corresponding date for each user. Then, we use the function "join" to join
the data points with the maximum altitude and the corresponding date for each user. After that, we use the function "groupBy" to group the data by "UserID" and "Date" and
find the minimum altitude and the corresponding date for each user. Then, we use the function "join" to join the data points with the minimum altitude and the corresponding
date for each user. After that, we calculate the span for each user and then find the maximum span for each user. Finally, we order the result by "max(Span)" for better
readability.
'''
# Find the maximum span for each user
# Group by "UserID" and "Date" and find the maximum altitude and the corresponding date for each user
df5_tmp1 = df1.groupBy("UserID","Date").max("Altitude")

# Find the maximum altitude and the corresponding date for each user
df5_tmp2 = df5_tmp1.withColumnRenamed("UserID", "ID").withColumnRenamed("max(Altitude)", "max_Altitude").withColumnRenamed("Date", "max_Date") 

# Group by "UserID" and "Date" and find the minimum altitude and the corresponding date for each user
df5_tmp3 = df1.groupBy("UserID","Date").min("Altitude")

#Join the data points with the minimum altitude and the corresponding date for each user
df5_tmp4 = df5_tmp2.join(df5_tmp3, (df5_tmp2.ID == df5_tmp3.UserID) & (df5_tmp2.max_Date == df5_tmp3.Date), "inner")

# Calculate the span for each user
df5_tmp5 = df5_tmp4.withColumn("Span", (col("max_Altitude") - col("min(Altitude)"))*0.3048)\
        .drop("max_Altitude")\
        .drop("min(Altitude)")\
        .drop("ID")\
        .drop("max_Date")\
        .drop("Date")

# Group by "UserID" and find the maximum span for each user
df5_tmp6 = df5_tmp5.groupBy("UserID").max("Span")

# Order the result by "max(Span)" for better readability
df5 = df5_tmp6.orderBy(col("max(Span)").desc())

print("The output of Question5:")
df5.show(5)





# Q6-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
This part contains two stages. The first stage is to calculate the distance travelled by each user on each day. The second stage is to find the most travel day and the
total distance travelled by each user. The idea of the first stage is to calculate the distance between two consecutive data points using Haverne formula for each user on 
each day. We use the function "withColumn" to calculate the distance between two consecutive data points for each user on each day. Then, we use the function "dropna" to drop the data points
with null values. The idea of the second stage is to find the most travel day and the total distance travelled by each user. We use the function "groupBy" to group the data
by "UserID" and "Date" and sum the distance for each user on each day. Then, we use the function "join" to join the data points with the most travel day and the corresponding
distance for each user. After that, we use the function "groupBy" to group the data by "UserID" and sum the distance for each user. Finally, we use the function "orderBy" to
order the result by "UserID" for better readability. We use the function "withColumn" to calculate the distance between two consecutive data points for each user on each day.
Then, we use the function "dropna" to drop the data points with null values. We use the function "groupBy" to group the data by "UserID" and "Date" and sum the distance for
each user on each day. Then, we use the function "join" to join the data points with the most travel day and the corresponding distance for each user. After that, we use the
function "groupBy" to group the data by "UserID" and sum the distance for each user. Finally, we use the function "orderBy" to order the result by "UserID" for better
readability.
'''

# Stage1
# define a function to calculate the distance between two consecutive data points for each user on each day
def calculate_distance(df):

        # Create a windowSpec to partition the data by "UserID" and "Date" and order the data by "Timestamp"
        windowSpec = Window.partitionBy("UserID","Date").orderBy("Timestamp")

        # Calculate the distance between two consecutive data points for each user on each day
        df = df.withColumn("PrevLatitude", F.lag("Latitude").over(windowSpec))
        df = df.withColumn("PrevLongitude", F.lag("Longitude").over(windowSpec))

        # Calculate the distance between two consecutive data points for each user on each day
        df = df.withColumn("DeltaLatitude", radians(col("Latitude") - col("PrevLatitude")))
        df = df.withColumn("DeltaLongitude", radians(col("Longitude") - col("PrevLongitude")))

        # Use the Haversine formula to calculate the distance between two consecutive data points for each user on each day
        df = df.withColumn('a', sin(col("DeltaLatitude") / 2) **2
                                         + cos(radians(col("PrevLatitude")))
                                         * cos(radians(col("Latitude")))
                                         * (sin(col("DeltaLongitude") / 2)) **2)
        
        df = df.withColumn('c', 2 * atan2(sqrt(col("a")), sqrt(1 - col("a"))))

        df = df.withColumn('Distance', 6373.0 * col("c")).drop('a', 'c', 'DeltaLatitude', 'DeltaLongitude')

        # Drop the data points with null values
        df = df.dropna(how = 'any').drop("PrevLatitude").drop("PrevLongitude")

        return df


df6_tmp1 = calculate_distance(df1)

# Stage2
# Part1
# Find the most travel day and the corresponding distance for each user
df6_tmp2 = df6_tmp1.groupBy("UserID", "Date").sum("Distance")\
                     .withColumnRenamed("sum(Distance)", "sum_Distance")

# Find the most travel day and the corresponding distance for each user
df6_tmp3 = df6_tmp2.groupBy("UserID").max("sum_Distance")\
                     .withColumnRenamed("max(sum_Distance)", "max_Distance")\
                     .withColumnRenamed("UserID", "ID")

# find the most travel day and the corresponding distance for each user
df6_tmp4 = df6_tmp2.join(df6_tmp3, (df6_tmp2.UserID == df6_tmp3.ID) & (df6_tmp2.sum_Distance == df6_tmp3.max_Distance), "inner")\
                     .drop("sum_Distance")\
                     .drop("ID")\
                     .drop("max_Distance")\
                     .dropDuplicates()\
                     .orderBy(col("UserID").desc())
# Part2
# Calculate the total distance travelled by each user across all days
df6_tmp5 = df6_tmp2.groupBy("UserID").sum("sum_Distance")\
                     .withColumnRenamed("sum(sum_Distance)", "total_Distance")\
                     .withColumnRenamed("UserID", "ID")
# Final
# Join the data points with the most travel day and the corresponding distance for each user
df6 = df6_tmp4.join(df6_tmp5, df6_tmp4.UserID == df6_tmp5.ID, "inner")\
               .drop("ID")\
               .dropDuplicates()\
               .withColumnRenamed("Date", "most_travel_Date")

df6 = df6.orderBy(col("UserID").asc())

# calculate the total distance travelled by all users across all days
total_distance = df6_tmp5.agg(F.sum("total_Distance")).collect()[0][0]

print("The output of Question6:")
df6.show()
print("TotalDistance: ", total_distance)





# Q7-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
The idea of this part is to calculate the speed for each data point and then find the maximum speed and the corresponding timestamp for each user and date. We use the
function "withColumn" to calculate the speed for each data point. Then, we use the function "dropna" to drop the data points with null values. After that, we use the
function "groupBy" to group the data by "UserID" and "Date" and find the maximum speed and the corresponding timestamp for each user and date. Then, we use the function
"join" to join the data points with the maximum speed and the corresponding timestamp for each user and date. Finally, we use the function "drop" to drop the unnecessary
columns and use the function "dropDuplicates" to drop the duplicate rows. We use the function "withColumn" to calculate the speed for each data point. Then, we use the
function "dropna" to drop the data points with null values. After that, we use the function "groupBy" to group the data by "UserID" and "Date" and find the maximum speed
and the corresponding timestamp for each user and date. Then, we use the function "join" to join the data points with the maximum speed and the corresponding timestamp for
each user and date. Finally, we use the function "drop" to drop the unnecessary columns and use the function "dropDuplicates" to drop the duplicate rows.
'''

# define a function to calculate the speed for each data point
def calculate_speed(df):
        
        # Create a windowSpec to partition the data by "UserID" and "Date" and order the data by "Timestamp"
        windowSpec = Window.partitionBy("UserID","Date").orderBy("Timestamp")
        
        # Calculate the time lag between two consecutive data points for each user on each day
        df = df.withColumn("PrevTimestamp", F.lag("Timestamp").over(windowSpec))
        
        # Calculate the time lag between two consecutive data points for each user on each day
        df = df.withColumn("TimeLag", (col("Timestamp") - col("PrevTimestamp"))*24)
        
        # Calculate the speed for each data point
        df = df.withColumn("Speed", (col("Distance") / (col("TimeLag"))))
        
        # Drop the data points with null values
        df = df.dropna(how = 'any').drop("PrevTimestamp").drop("TimeLag")
        
        return df

df7_tmp1 = calculate_speed(df6_tmp1)


# Find the maximum speed and the corresponding timestamp for each user and date
df7_tmp2 = df7_tmp1.groupBy("UserID", "Date").max("Speed").withColumnRenamed("max(Speed)", "max_Speed")

# Join the data points with the maximum speed and the corresponding timestamp for each user and date
df7_tmp2 = df7_tmp2.withColumnRenamed("UserID", "ID").withColumnRenamed("Date", "max_Speed_Date")

# Join the data points with the maximum speed and the corresponding timestamp for each user and date
df7_tmp3 = df7_tmp1.join(df7_tmp2, (df7_tmp1.UserID == df7_tmp2.ID) & (df7_tmp1.Date == df7_tmp2.max_Speed_Date) & (df7_tmp1.Speed == df7_tmp2.max_Speed), "inner")\
                .drop("Latitude")\
                .drop("Longitude")\
                .drop("AllZero")\
                .drop("Altitude")\
                .drop("Timestamp")\
                .drop("Time")\
                .drop("Timestamp")\
                .drop("Distance")\
                .drop("ID")\
                .dropDuplicates()

# find the maximum speed and the corresponding timestamp for each user and date
df7_tmp4 = df7_tmp3.groupBy("UserID").max("max_Speed").withColumnRenamed("max(max_Speed)", "MaxSpeed")\
                .withColumnRenamed("UserID", "ID").orderBy(col("ID").asc())

# Join the data points with the maximum speed and the corresponding timestamp for each user and date
df7 = df7_tmp4.join(df7_tmp3, (df7_tmp3.UserID == df7_tmp4.ID) & (df7_tmp3.max_Speed == df7_tmp4.MaxSpeed), "inner")\
               .drop("ID")\
               .drop("max_Speed")\
               .drop("Date")\
               .drop("Speed")\
               .dropDuplicates()\
               .orderBy(col("UserID").asc())

print("The output of Question7:")
df7.show()

