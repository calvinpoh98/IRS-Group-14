# Traffic Data Pipeline

This project demonstrates a pipeline for pulling traffic data from the LTA DataMall API, processing it using Kafka and Spark, and loading it into PostgreSQL for persistent storage. The pipeline allows for efficient streaming, transformation, and storage of traffic data, including speed bands, minimum and maximum speeds, for further analysis and use.

## Project Overview

1. **Data Source**: Traffic data is pulled from the LTA DataMall API, specifically focusing on speed band information (minimum and maximum speeds).
2. **Data Ingestion**: Kafka is used to ingest the traffic data. The data is temporarily stored in Kafka, allowing consumers to consume the data efficiently.
3. **Data Transformation**: Apache Spark processes the incoming Kafka streams. It transforms the data into the appropriate structure before loading it into PostgreSQL.
4. **Data Storage**: The transformed traffic data is stored in a PostgreSQL database, which is designed to handle large-scale, structured data.

## Components

- **Apache Kafka**: Handles the streaming of traffic data, enabling asynchronous data processing.
- **Apache Spark**: Consumes data from Kafka, transforms it, and loads it into PostgreSQL.
- **PostgreSQL**: Stores the processed traffic data for long-term access.
- **Airflow**: Manages the orchestration of the ETL (Extract, Transform, Load) process.
- **Kafka UI**: Provides an interface to monitor Kafka topics.
- **pgAdmin**: Allows access to the PostgreSQL database for data inspection and management.

## Project Architecture

- **LTA DataMall API** → **Kafka Producer** → **Kafka Topics** → **Spark Consumer** → **PostgreSQL**.

## Accessing Services on localhost

- **Airflow Web Interface**: [http://localhost:8080](http://localhost:8080)  
  Use this URL to access the Airflow web interface where you can trigger DAGs, monitor jobs, and manage tasks.
  
- **Kafka UI**: [http://localhost:8800](http://localhost:8800)  
  Access this UI to monitor Kafka topics, partitions, and messages. This UI allows you to check the health of your Kafka system and view messages stored in your topics.

- **pgAdmin (PostgreSQL UI)**: [http://localhost:5050](http://localhost:5050)  
  Access pgAdmin to manage and inspect your PostgreSQL database, where the transformed traffic data is stored. You can view tables, run SQL queries, and manage the database.

This data will utimately be used to train a time series model to predict the traffic condition in Singapore on an hourly basis.
