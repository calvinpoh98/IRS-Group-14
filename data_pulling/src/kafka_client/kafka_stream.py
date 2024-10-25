import datetime
import requests
from kafka import KafkaProducer
import json
import logging

def get_data(url=str):
    ''' 
    This function gets the data from the API and returns the data in JSON format
    '''
    headers = {
    "AccountKey": "", # Add your AccountKey here
    "accept": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    response_body = response.json()

    # Filter traffic conditions
    traffic_data = response_body['value']
    print(traffic_data)
    return traffic_data

def create_producer():
    '''
    This function creates a Kafka producer
    '''
    try:
        kafka_producer = KafkaProducer(bootstrap_servers='kafka:9092') #Change kafka to localhost if running locally
    except Exception as e:
        logging.info(
            "We assume that we are running locally, so we use localhost instead of kafka and the external "
            "port 9094"
        )
        kafka_producer = KafkaProducer(bootstrap_servers=["kafka:9094"]) # This is for testing outside the docker container

    return kafka_producer

def stream_data(topic_name = "traffic_data"):
    '''
    This function streams the data to Kafka topic
    '''
    
    url = "https://datamall2.mytransport.sg/ltaodataservice/v3/TrafficSpeedBands?$skip=59000"
    traffic_data = get_data(url)

    kafka_producer = create_producer()
    for data in traffic_data:
        kafka_producer.send(topic_name, value=json.dumps(data).encode("utf-8"))


if __name__ == "__main__":
    # Stream the data to Kafka topic
    stream_data(topic_name='traffic_data')