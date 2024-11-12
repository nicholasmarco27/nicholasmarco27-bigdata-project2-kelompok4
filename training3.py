from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os

# Initialize Spark session
spark = SparkSession.builder.appName("BatchRegressionTraining").getOrCreate()

# Specify the path for the target batch file and the output model directory
batch_file = '/home/nicholas/ETS/models/batch_3.csv'
output_model_directory = '/home/nicholas/ETS/models/spark_rf_models'

# Create output directory if it doesn’t exist
os.makedirs(output_model_directory, exist_ok=True)

# Load the batch data as a Spark DataFrame
data = spark.read.csv(batch_file, header=True, inferSchema=True)

# Preprocess data
# Step 1: Remove columns 'Loan Status' and 'Tax Liens' from the DataFrame
final_data = data.drop("Loan Status", "Tax Liens")

# Display some rows of the DataFrame after dropping columns
print("Sample data after dropping 'Loan Status' and 'Tax Liens':")
final_data.show(5)

# Step 2: Prepare features and target
feature_columns = [col for col in final_data.columns if col != "Maximum Open Credit"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(final_data).select("features", "Maximum Open Credit")

# Display some samples of the assembled data
print(f"Sample data from batch_3 (assembled features):")
assembled_data.show(5, truncate=False)

# Split the data into training and validation sets
train_data, valid_data = assembled_data.randomSplit([0.8, 0.2], seed=42)

# Define the RandomForestRegressor model with specified parameters
rf = RandomForestRegressor(
    featuresCol="features", 
    labelCol="Maximum Open Credit", 
    numTrees=300  # Adjust parameters as per your requirements
)

# Train the model
model = rf.fit(train_data)

# Make predictions on the validation set
predictions = model.transform(valid_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="Maximum Open Credit", predictionCol="prediction")

# Calculate evaluation metrics
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print("Batch 3:")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Save the trained model
model_path = f"{output_model_directory}/rf_model_batch_3"
model.write().overwrite().save(model_path)
print(f"RandomForest regression model for batch 3 saved at {model_path}")

print(feature_columns)  # This should list the feature columns

# Stop the Spark session after processing
spark.stop()
