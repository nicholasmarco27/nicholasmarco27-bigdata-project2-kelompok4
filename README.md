# Big Data Stream Processing with Kafka and Apache Spark
## Sistem ini dibangun untuk memproses dan menganalisis data terkait permohonan pinjaman online (pinjol) menggunakan arsitektur Big Data yang melibatkan **Apache Kafka** dan **Apache Spark**. Dataset yang digunakan adalah dataset yang diambil dari [Loan - Bank Dataset](https://www.kaggle.com/datasets/zaurbegiev/my-dataset).


## Anggota Kelompok 4

| Nama                      | NRP |
|---------------------------|-----------------------|
| Ryan Wardana               | 5027221022            |
| Nicholas Marco Weinandra   | 5027221042            |
| Stephanie Hebrina          | 5027221069            |

## Fitur Utama

### 1. **Kafka Producer**
   Mengirimkan data secara berurutan (sequential) ke Kafka Server. Kafka Producer membaca dataset dan mengirimkan setiap baris data ke Kafka Server dengan memberikan jeda acak di antara setiap pengiriman, untuk mensimulasikan pemrosesan data stream secara real-time.

### 2. **Kafka Consumer**
   Membaca data yang diterima oleh Kafka Server dan mengelompokkan data menjadi batch. Pengelompokan data ini dapat dilakukan berdasarkan:
- **Jumlah Data**: 10.000
- **Rentang Waktu (Window)**: 5
     

### 3. **Apache Spark (Training Model)**
   Melakukan training model berdasarkan batch data yang diterima. Apache Spark digunakan untuk memproses data yang sudah dikelompokkan dan membuat beberapa model berdasarkan jumlah data atau rentang waktu yang dipilih. 
-  **Skema 1**: Data dibagi secara proporsional (1/3 data pertama, 1/3 data kedua, 1/3 data ketiga).

### 4. **API Endpoint**
   Menyediakan API untuk menerima request dari pengguna dan memberikan hasil berdasarkan model yang telah dibuat.


## Streaming 

### 1. Setup Kafka

a) **Jalankan Zookeper**
```
bin/zookeeper-server-start.sh config/zookeeper.properties
```

b) **Start kafka**
```
bin/kafka-server-start.sh config/server.properties
```

c) **Make topic**
```
bin/kafka-topics.sh --create --topic kafka2 --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### 2. Running Producer & Consumer

a) **Running Consumer**
```
python3 kafka_consumer.py
```

b) **Running Producer**
```
python3 kafka_onsumer.py
```

### 3. Training model with SparkML

a) **Running Training semua model**
```
python3 training.py
```

### 4. API dan Endpoint

a) **Run api.py**
```
python3 api.py
```




