from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
)
from pyspark.sql.functions import from_json, col, current_timestamp
import logging
import psycopg2
from pyspark.sql import DataFrame
from pyspark.sql.functions import from_utc_timestamp

username = "airflow"
password_postgres = "airflow"
POSTGRES_URL = f"jdbc:postgresql://postgres:5432/traffic-data" # Follow databse name, Change postgres to localhost and port to 5342 if testing locally
POSTGRES_PROPERTIES = {
    "user": username,
    "password": password_postgres,
    "driver": "org.postgresql.Driver",
}


def create_spark_session():
    '''
    This function creates a Spark session to consume data from Kafka
    '''

    spark = SparkSession.builder.appName("TrafficData").config(
            "spark.jars.packages",
            "org.postgresql:postgresql:42.5.4,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        ).getOrCreate()
    return spark


def create_initial_dataframe(spark_session):
    """
    Reads the streaming data and creates the initial dataframe accordingly.
    """
    try:
        # Gets the streaming data from topic random_names
        df = (
            spark_session.readStream.format("kafka")
            .option("kafka.bootstrap.servers", "data_pulling-kafka-1:9092") # "localhost:9094" if running locally
            .option("subscribe", "traffic_data")
            .option("startingOffsets", "earliest")
            .load()
        )
        logging.info("Initial dataframe created successfully")
    except Exception as e:
        logging.warning(f"Initial dataframe couldn't be created due to exception: {e}")
        raise

    return df

def structure_data(df):
    '''
    This function structures the data from the dataframe
    '''
    
    keys_dict_types = {
        "LinkID": StringType(),
        "RoadName": StringType(),
        "RoadCategory": StringType(),
        "SpeedBand": IntegerType(),
        "MinimumSpeed": StringType(),
        "MaximumSpeed": StringType(),
        "StartLon": StringType(),
        "StartLat": StringType(),
        "EndLon": StringType(),
        "EndLat": StringType()
    }

    schema = StructType(
        [StructField(field_name, type, True) for field_name, type in keys_dict_types.items()]
    )

    # df_out = (
    #     df.selectExpr("CAST(value AS STRING)")
    #     .select(from_json(col("value"), schema).alias("data"))
    #     .select("data.*")
    #     .withColumn("data_collected", current_timestamp())
    # )

    df_out = (
        df.selectExpr("CAST(value AS STRING)", "timestamp")  # Extract value and Kafka's message timestamp
        .select(from_json(col("value"), schema).alias("data"), "timestamp")  # Parse the JSON data
        .select("data.*", "timestamp")  # Include all fields and the timestamp from Kafka
        .withColumnRenamed("timestamp", "data_collected") 
        .withColumn("data_collected", from_utc_timestamp(col("data_collected"), "Asia/Singapore")) 
    )      

    return df_out


def upsert_to_postgres(batch_df: DataFrame, batch_id: int):
    """
    Custom upsert function to write data to PostgreSQL with ON CONFLICT.
    """
    # Convert the batch dataframe to Pandas for easier processing
    batch_data = batch_df.toPandas()

    # Establish a connection to PostgreSQL
    connection = psycopg2.connect(
        dbname="traffic-data",
        user="airflow",
        password="airflow",
        host="data_pulling-postgres-1",
        port="5432",
    )
    cursor = connection.cursor()

    upsert_query = """
    INSERT INTO traffic_data (
        LinkID, 
        RoadName, 
        RoadCategory, 
        SpeedBand, 
        MinimumSpeed, 
        MaximumSpeed, 
        StartLon, 
        StartLat, 
        EndLon, 
        EndLat, 
        data_collected
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (LinkID, data_collected) 
    DO NOTHING;
    """


    # Loop through the batch rows and execute the query for each row
    for index, row in batch_data.iterrows():
        cursor.execute(upsert_query, (
            row['LinkID'], 
            row['RoadName'], 
            row['RoadCategory'], 
            row['SpeedBand'], 
            row['MinimumSpeed'], 
            row['MaximumSpeed'], 
            row['StartLon'], 
            row['StartLat'], 
            row['EndLon'], 
            row['EndLat'], 
            row['data_collected']  # This is the new column
        ))


    # Commit and close the connection
    connection.commit()
    cursor.close()
    connection.close()

def insert_data():
    '''
    This function inserts the data from kafka to postgres
    '''

    # Create a Spark session
    spark = create_spark_session()

    # Extract the data
    df = create_initial_dataframe(spark)

    # Tranform the data
    df_out = structure_data(df)

    # Load the data
    # query = df_out.writeStream \
    #     .outputMode("append") \
    #     .foreachBatch(
    #         lambda batch_df, _: (
    #             batch_df.show(),  # Prints to console
    #             batch_df.write.jdbc(
    #                 url=POSTGRES_URL,
    #                 table="traffic_data",
    #                 mode="append",
    #                 properties=POSTGRES_PROPERTIES
    #             )
    #         )
    #     ) \
    #     .start()

    # Now use this in the foreachBatch function
    query = df_out.writeStream \
        .outputMode("append") \
        .foreachBatch(upsert_to_postgres) \
        .trigger(once=True) \
        .start()
        
    return query.awaitTermination()


    



if __name__ == "__main__":
   insert_data()


