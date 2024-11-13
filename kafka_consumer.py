from kafka import KafkaConsumer
import json
import pandas as pd
from datetime import datetime, timedelta

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    'LoanData',  # Make sure this matches your producer's topic
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    group_id='my-consumer-group'
)

# Batch and processing parameters
batch = []
batch_size = 10000  # Define your batch size
time_window = timedelta(minutes=5)
start_time = datetime.now()
batch_count = 0  # Counter to limit to 3 batches
max_batches = 3  # Limit to 3 batches

# Directory to save CSV files
save_directory = '/home/nicholas/ETS/models'

for message in consumer:
    batch.append(message.value)

    # Check if batch meets size or time window criteria
    if len(batch) >= batch_size or (datetime.now() - start_time) >= time_window:
        # Save batch as a CSV file
        df = pd.DataFrame(batch)
        df.to_csv(f'{save_directory}/batch_{batch_count+1}.csv', index=False)
        
        # Clear batch and reset time
        batch = []
        start_time = datetime.now()
        
        # Increment batch count and check limit
        batch_count += 1
        if batch_count >= max_batches:
            print("Processed 3 batches. Stopping consumer.")
            break  # Exit loop after saving 3 batches
