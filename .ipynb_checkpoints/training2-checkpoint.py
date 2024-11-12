from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os

# Initialize Spark session
spark = SparkSession.builder.appName("BatchRegressionTraining").getOrCreate()

# Specify the path for the target batch file and the output model directory
batch_file = '/home/nicholas/ETS/models/batch_2.csv'
output_model_directory = '/home/nicholas/ETS/models/spark_rf_models'

# Create output directory if it doesn’t exist
os.makedirs(output_model_directory, exist_ok=True)

# Load the batch data as a Spark DataFrame
data = spark.read.csv(batch_file, header=True, inferSchema=True)

# Preprocess data
# Step 1: Remove rows with null values in feature columns (exclude "Credit Score" column)
feature_columns = [col for col in data.columns if col not in ["Credit Score", "Loan Status"]]
data_cleaned = data.na.drop(subset=feature_columns)

# Display the size of the cleaned data
print(f"Size of cleaned data for batch_2: {data_cleaned.count()} rows")

# Step 2: Assemble features (exclude "Credit Score" column)
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(data_cleaned).select("features", "Credit Score")  # "Credit Score" is used directly as label

# Display some samples of the assembled data
print(f"Sample data from batch_2 (assembled features):")
assembled_data.show(5, truncate=False)  # Display the first 5 rows of assembled data

# Split the data into training and validation sets
train_data, valid_data = assembled_data.randomSplit([0.8, 0.2], seed=42)

# Define the RandomForestRegressor model with specified parameters
rf = RandomForestRegressor(
    labelCol="Credit Score", 
    featuresCol="features", 
    numTrees=300  # Adjust parameters as per your requirements
)

# Train the model
model = rf.fit(train_data)

# Make predictions on the validation set
predictions = model.transform(valid_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="Credit Score", predictionCol="prediction")

# Calculate evaluation metrics
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"Batch 2:")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")

# Save the trained model
model_path = f"{output_model_directory}/rf_model_batch_2"
model.write().overwrite().save(model_path)
print(f"RandomForest regression model for batch 2 saved at {model_path}")

print(feature_columns)  # This should list the feature columns

# Stop the Spark session after processing
spark.stop()
