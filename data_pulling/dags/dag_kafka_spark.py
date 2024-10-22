import sys
import os

# Add /opt/airflow/dags/src to the Python path
sys.path.append('/opt/airflow/dags/src')


from src.kafka_client.kafka_stream import stream_data

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

# When the clock hits the hour, the DAG will start running
start_time = datetime.now().hour + 1
start_date = datetime.now().replace(hour=start_time, minute=0, second=0, microsecond=0)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": start_date,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="kafka_spark_dag",
    default_args=default_args,
    schedule_interval=timedelta(hours=1),
    catchup=False,
) as dag:

    kafka_stream_task = PythonOperator(
        task_id="kafka_data_stream",
        python_callable=stream_data,
        dag=dag,
    )

    spark_task = DockerOperator(
        task_id="spark_data_processing",
        image="spark_custom:latest",
        api_version="auto",
        auto_remove=True,
        environment={'SPARK_LOCAL_HOSTNAME': 'localhost'},
        command="./bin/spark-submit --master local[*] --packages org.postgresql:postgresql:42.5.4,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 ./spark_consumer.py",
        network_mode="data_pulling_airflow-kafka",
        mount_tmp_dir=False,
        dag=dag,
    )

    kafka_stream_task >> spark_task