from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, concat_ws, to_date, to_timestamp, date_format, lag
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.functions import(
        split, concat_ws, lit, round, count, min, max, length, when, sqrt, sum, row_number, col, expr, 
        unix_timestamp, from_unixtime, to_date, to_timestamp, date_format, weekofyear, year, greatest, lag, floor, sin, cos, atan2, radians
)

spark = SparkSession.builder.appName('Orange').getOrCreate()

# Read data from CSV file and change the datatype
df0 = spark.read.options(header='True', delimiter=',')\
                .csv("G:\Ass1\dataset.txt")
df = df0.withColumn("UserID", col("UserID").cast("Integer"))\
        .withColumn("Latitude", col("Latitude").cast("Double"))\
        .withColumn("Longitude", col("Longitude").cast("Double"))\
        .withColumn("AllZero", col("AllZero").cast("Integer"))\
        .withColumn("Altitude", col("Altitude").cast("Double"))\
        .withColumn("Timestamp", col("Timestamp").cast("Double"))\

# Q1
'''
The general idea of this question is to use the built-in function +/- expr() of pyspark to increase or decrease the time 
efficiently by merging the "Date" and "Time" columns to one column and converting the data type to timestamp. For the 
increasing of column "Timestamp", just use regular math method. Finally I use drop() function to drop the temporary 
merging column "date_time".
'''
df1 = df.withColumn("date_time", concat_ws(" ", col("Date"), col("Time")))\
        .withColumn("date_time", to_timestamp(col("date_time")))\
        .withColumn("date_time", col("date_time") + F.expr("INTERVAL 8 HOURS"))\
        .withColumn("Timestamp", col("Timestamp") + (8 * 60) / (24 * 60))\
        .withColumn("Date", to_date(col("date_time")))\
        .withColumn("Time", date_format("date_time", "HH:mm:ss"))\
        .drop("date_time")
print("---------------------------------------------------------------------------------------------------------------")
print("The output of Part1 Question1:")
df1.show()
print("---------------------------------------------------------------------------------------------------------------")


# Q2
'''
The general idea of this question is mainly based on groupBy() Function. First group by "UserID" adn "Date" column to 
form a dataframe that only contains 3 columns: "UserID", "Date" and "count", count is the count of data points for each 
date of each user id. Then I use where() function to filter the date with the count of data points less than 2 out. Then 
I group the dataframe before by column "UserID" and count the number of dates of each user. Finally, I user orderBy()
function to order the whole dataset by descending order of column "count" and "UserID", in order to find the the top 10 
user IDs and output the user with the larger ID as well.
'''
df2_temp1 = df1.groupBy("UserID", "Date").count()
df2_temp2 = df2_temp1.where(col("count") >= 5)
df2_temp3 = df2_temp2.groupby("UserID").count()
df2 = df2_temp3.orderBy(col("count").desc(), col("UserID").desc())
print("---------------------------------------------------------------------------------------------------------------")
print("The output of Part1 Question2:")
df2.show(10)
print("---------------------------------------------------------------------------------------------------------------")


# Q3
'''
This question is almost the same as question2, and I can use the temporary dataframe in question2 directly. Just use 
where() function to filter the date with the count of data points more than 150, then group the dataframe by column
"UserID" and count the number of dates of each user. To output all rows in the dataframe, just use df.count() in show()
function.
'''

# Create a new column "YearWeek" to represent the combination of year and week of the year
df3_temp1 = df1.withColumn("YearWeek", F.floor((F.floor(col("Timestamp"))-2)/7))

# Group by "UserID" and "YearWeek" and count the data points for each week
df3_temp2 = df3_temp1.groupBy("UserID", "YearWeek").count()

# Filter weeks with more than 100 data points
df3_temp3 = df3_temp2.where(col("count") > 100)

# Group by "UserID" and count the number of weeks with more than 100 data points
df3_modified = df3_temp3.groupBy("UserID").count().withColumnRenamed("count", "WeeksWithMoreThan100Points")

# Order the result by "UserID" for better readability
df3_modified = df3_modified.orderBy("UserID")

print("---------------------------------------------------------------------------------------------------------------")
print("The output of Modified Part1 Question3:")
df3_modified.show(df3_modified.count())
print("---------------------------------------------------------------------------------------------------------------")



# Q4
'''
The most important part of this question is to add date that user achieved the northern most point. First I use 
groupBy() and max() function to generate a dataframe with just two columns: "UserID" and "max_Latitude". Then I let this
df inner join with the original df, drop the columns I don't need and then drop the duplicate rows, then I get the goal
dataframe. Finally I ordered the df by descending order of column "max_Latitude" and "Date" to output the top 10 user 
IDs and the latest such a day of one user.
'''
df4_temp1 = df1.groupBy("UserID").min("Latitude")
df4_temp2 = df4_temp1.withColumnRenamed("min(Latitude)", "min_Latitude").withColumnRenamed("UserID", "ID")
df4_temp3 = df4_temp2.join(df1, (df4_temp2.min_Latitude == df1.Latitude) & (df4_temp2.ID == df1.UserID), "inner")
df4_temp4 = df4_temp3.drop("Latitude")\
                     .drop("Longitude")\
                     .drop("AllZero")\
                     .drop("Altitude")\
                     .drop("Timestamp")\
                     .drop("Time")\
                     .drop("UserID")\
                     .dropDuplicates()
df4 = df4_temp4.orderBy(col("min_Latitude"), col("Date"))
print("---------------------------------------------------------------------------------------------------------------")
print("The output of Part1 Question4:")
df4.show(10)
print("---------------------------------------------------------------------------------------------------------------")


# Q5
'''
The general idea of this question is based on groupBy().max()/min() function. Use these two functions we can get two
dataframes, each of them consists of two columns: "UserID" and "max/min(Altitude)". Then we use inner join function to 
merge these two dataframes because if a user just has a max/min altitude, then we cannot use this user, and any other
user has a max altitude must also has a min altitude. Then we generate a new column use max(Altitude) - min(Altitude),
this is the "Span" column. Then we drop every column we don't need and order this df by descending order of column 
"Span" and get the goal df.
'''
df5_temp1 = df1.groupBy("UserID","Date").max("Altitude")
df5_temp2 = df5_temp1.withColumnRenamed("UserID", "ID").withColumnRenamed("max(Altitude)", "max_Altitude").withColumnRenamed("Date", "max_Date") 
df5_temp3 = df1.groupBy("UserID","Date").min("Altitude")
df5_temp4 = df5_temp2.join(df5_temp3, (df5_temp2.ID == df5_temp3.UserID) & (df5_temp2.max_Date == df5_temp3.Date), "inner")
df5_temp5 = df5_temp4.withColumn("Span", (col("max_Altitude") - col("min(Altitude)"))*0.3048)\
        .drop("max_Altitude")\
        .drop("min(Altitude)")\
        .drop("ID")\
        .drop("max_Date")\
        .drop("Date")
df5_temp6 = df5_temp5.groupBy("UserID").max("Span")
df5 = df5_temp6.orderBy(col("max(Span)").desc())
print("---------------------------------------------------------------------------------------------------------------")
print("The output of Part1 Question5:")
df5.show(5)
print("---------------------------------------------------------------------------------------------------------------")




# Q6
'''
This question needs to be completed in two stages, and the second stage should be divided into two parts. Stage1 is 
compute the climbed height between every pairs of consecutive data points. First I use lag() function in window to 
create a new column for the original dataframe, for each user, stores the altitude of one data point before. Then I 
removed the row with value null in column "One_day_before", because actually they are all the first line of each user,
so they cannot be computed. Then I create a new Column called "Climb" to store the differences in altitudes between 
two consecutive data points and dropped the "One_day_before" column. For Stage1 we get a dataframe with all the columns
in original dataframe and one more column "Climb". Stage2 is divided into two parts. Part1 is to get the user and the 
(latest) day they climbed the most. First I use the goal dataframe from Stage1 and group it by column "UserID" and 
"Date", and get the sum of climbed height for each day by each user. Then I further group this dataframe also by
column "UserID" to get the max day-climbed height for each user. Then I inner join these two groupByed dataframe and 
drop all the columns I don't need anymore, inorder to get the date of the most climbed height for each person. For Part1
We get a dataframe with just two columns: "UserID" and "Date", the date is the (latest) day (order by date in descending 
order) they climbed the most. Part2 is just to compute overall total height climbed by all users across all days, I use 
the first groupByed dataframe in Part1, group it by column "UserID" and calculate the sum of climbed height. Finally,
I let the goal dataframe of Part1 and Part2 inner join, drop the column I don't need and generate the final dataframe.
'''
# Stage1
def calculate_distance(df):

        windowSpec = Window.partitionBy("UserID","Date").orderBy("Timestamp")

        df = df.withColumn("PrevLatitude", F.lag("Latitude").over(windowSpec))
        df = df.withColumn("PrevLongitude", F.lag("Longitude").over(windowSpec))

        df = df.withColumn("DeltaLatitude", radians(col("Latitude") - col("PrevLatitude")))
        df = df.withColumn("DeltaLongitude", radians(col("Longitude") - col("PrevLongitude")))

        df = df.withColumn('a', sin(col("DeltaLatitude") / 2) **2
                                         + cos(radians(col("PrevLatitude")))
                                         * cos(radians(col("Latitude")))
                                         * (sin(col("DeltaLongitude") / 2)) **2)
        
        df = df.withColumn('c', 2 * atan2(sqrt(col("a")), sqrt(1 - col("a"))))

        df = df.withColumn('Distance', 6373.0 * col("c")).drop('a', 'c', 'DeltaLatitude', 'DeltaLongitude')

        df = df.dropna(how = 'any').drop("PrevLatitude").drop("PrevLongitude")

        return df


df6_temp1 = calculate_distance(df1)

# Stage2
# Part1
df6_temp4 = df6_temp1.groupBy("UserID", "Date").sum("Distance")\
                     .withColumnRenamed("sum(Distance)", "sum_Distance")

df6_temp5 = df6_temp4.groupBy("UserID").max("sum_Distance")\
                     .withColumnRenamed("max(sum_Distance)", "max_Distance")\
                     .withColumnRenamed("UserID", "ID")

df6_temp6 = df6_temp4.join(df6_temp5, (df6_temp4.UserID == df6_temp5.ID) & (df6_temp4.sum_Distance == df6_temp5.max_Distance), "inner")\
                     .drop("sum_Distance")\
                     .drop("ID")\
                     .drop("max_Distance")\
                     .dropDuplicates()\
                     .orderBy(col("UserID").desc())
# Part2
df6_temp7 = df6_temp4.groupBy("UserID").sum("sum_Distance")\
                     .withColumnRenamed("sum(sum_Distance)", "total_Distance")\
                     .withColumnRenamed("UserID", "ID")
# Final
df6 = df6_temp6.join(df6_temp7, df6_temp6.UserID == df6_temp7.ID, "inner")\
               .drop("ID")\
               .dropDuplicates()\
               .withColumnRenamed("Date", "most_travel_Date")

df6 = df6.orderBy(col("UserID").asc())
# calculate the total distance travelled by all users across all days
total_distance = df6_temp7.agg(F.sum("total_Distance")).collect()[0][0]
print("---------------------------------------------------------------------------------------------------------------")
print("The output of Part1 Question6:")
df6_temp1.show()
print(total_distance)
print("---------------------------------------------------------------------------------------------------------------")






def calculate_speed(df):
        
        windowSpec = Window.partitionBy("UserID","Date").orderBy("Timestamp")
        
        df = df.withColumn("PrevTimestamp", F.lag("Timestamp").over(windowSpec))
        
        df = df.withColumn("TimeLag", col("Timestamp") - col("PrevTimestamp"))
        
        df = df.withColumn("Speed", (col("Distance") / (col("TimeLag")) / 24))
        
        df = df.dropna(how = 'any').drop("PrevTimestamp").drop("TimeLag")
        
        return df

df7_temp1 = calculate_speed(df6_temp1)



print("---------------------------------------------------------------------------------------------------------------")
df7_temp1.show()
print("---------------------------------------------------------------------------------------------------------------")

# Find the maximum speed and the corresponding timestamp for each user and date
df7_temp2 = df7_temp1.groupBy("UserID", "Date").max("Speed").withColumnRenamed("max(Speed)", "max_Speed")
df7_temp2 = df7_temp2.withColumnRenamed("UserID", "ID").withColumnRenamed("Date", "max_Speed_Date")
df7_temp3 = df7_temp1.join(df7_temp2, (df7_temp1.UserID == df7_temp2.ID) & (df7_temp1.Date == df7_temp2.max_Speed_Date) & (df7_temp1.Speed == df7_temp2.max_Speed), "inner")\
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
df7_temp3.show()
df7_temp4 = df7_temp3.groupBy("UserID").max("max_Speed").withColumnRenamed("max(max_Speed)", "max_speed")\
                .withColumnRenamed("UserID", "ID").orderBy(col("ID").asc())

df7 = df7_temp4.join(df7_temp3, df7_temp3.UserID == df7_temp4.ID, "inner")\
               .drop("max_speed")\
               .drop("Date")\
               .drop("Speed")\
               .dropDuplicates()\
               .orderBy(col("UserID").asc())

df7_temp4.show()
df7.show()

