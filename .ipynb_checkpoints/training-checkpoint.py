from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

# Initialize Spark session
spark = SparkSession.builder.appName("SingleBatchClassificationTraining").getOrCreate()

# Specify the path for the target batch file and the output model directory
batch_file = '/home/nicholas/ETS/models/batch_1.csv'
output_model_directory = '/home/nicholas/ETS/models/spark_rf_models'

# Create output directory if it doesn’t exist
os.makedirs(output_model_directory, exist_ok=True)

# Load the batch data as a Spark DataFrame
data = spark.read.csv(batch_file, header=True, inferSchema=True)

# Preprocess data
# Step 1: Directly use the "Loan Status" column as the label (since it's already numeric)
# Step 2: Remove rows with null values in feature columns
feature_columns = [col for col in data.columns if col != "Loan Status"]
data_cleaned = data.na.drop(subset=feature_columns)

# Display the size of the cleaned data
print(f"Size of cleaned data for batch_1: {data_cleaned.count()} rows")

# Step 3: Assemble features (exclude "Loan Status" column)
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
assembled_data = assembler.transform(data_cleaned).select("features", "Loan Status")  # "Loan Status" is used directly as label

# Display some samples of the assembled data
print(f"Sample data from batch_1 (assembled features):")
assembled_data.show(5, truncate=False)  # Display the first 5 rows of assembled data

# Split the data into training and validation sets
train_data, valid_data = assembled_data.randomSplit([0.8, 0.2], seed=42)

# Define the RandomForest model with updated parameters
rf = RandomForestClassifier(
    labelCol="Loan Status",  # Directly use "Loan Status" as the label
    featuresCol="features", 
    numTrees=400,  # Updated number of trees (n_estimators)
    maxDepth=30,   # Updated max depth
    minInstancesPerNode=20,  # Corresponds to min_samples_leaf
    bootstrap=False,  # Updated bootstrap parameter
)

# Train the model
model = rf.fit(train_data)

# Make predictions on the validation set
predictions = model.transform(valid_data)

# Evaluate the model
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="Loan Status", predictionCol="prediction", metricName="f1")
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="Loan Status", predictionCol="prediction", metricName="accuracy")

f1_score = evaluator_f1.evaluate(predictions)
accuracy = evaluator_accuracy.evaluate(predictions)

print(f"Batch 1:")
print(f"F1 Score: {f1_score}")
print(f"Accuracy: {accuracy}")

# Save the trained model
model_path = f"{output_model_directory}/rf_model_batch_1"
model.write().overwrite().save(model_path)
print(f"RandomForest model for batch 1 saved at {model_path}")

print(feature_columns)  # This should list the feature columns

# Stop the Spark session after processing all batches
spark.stop()
