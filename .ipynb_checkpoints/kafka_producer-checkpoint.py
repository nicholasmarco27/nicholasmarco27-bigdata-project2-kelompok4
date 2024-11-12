import time
import random
from kafka import KafkaProducer
import json
import pandas as pd

# Initialize Kafka Producer
producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Load dataset
data = pd.read_csv('/home/nicholas/ETS/DATA_TRAIN_FIX.csv')

# Stream data row-by-row
for index, row in data.iterrows():
    message = row.to_dict()
    producer.send('LoanData', message)
    print(f'Sent: {message}')
    time.sleep(random.uniform(0.01, 0.03))  # Random delay between 0.1 and 0.5 seconds

