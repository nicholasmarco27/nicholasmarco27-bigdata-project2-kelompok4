from flask import Flask, request, jsonify
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

# Initialize Flask app and Spark session
app = Flask(__name__)
spark = SparkSession.builder.appName("PredictionAPI").getOrCreate()

# Load the models for each prediction task
loan_status_model = RandomForestClassificationModel.load('/home/nicholas/ETS/models/spark_rf_models/rf_model_batch_1')
credit_score_model = RandomForestRegressionModel.load('/home/nicholas/ETS/models/spark_rf_models/rf_model_batch_2')
max_open_credit_model = RandomForestRegressionModel.load('/home/nicholas/ETS/models/spark_rf_models/rf_model_batch_3')

# Define required feature sets for each model
loan_status_features = ["Current Loan Amount", "Credit Score", "Annual Income", "Monthly Debt", 
                        "Years of Credit History", "Number of Open Accounts", 
                        "Current Credit Balance", "Maximum Open Credit"]

credit_score_features = ["Current Loan Amount", "Annual Income", "Monthly Debt", 
                         "Years of Credit History", "Number of Open Accounts", 
                         "Current Credit Balance", "Maximum Open Credit"]

max_open_credit_features = ["Current Loan Amount", "Credit Score", "Annual Income", "Monthly Debt", 
                            "Years of Credit History", "Number of Open Accounts", 
                            "Current Credit Balance"]

# Helper function to prepare features in the required order
def prepare_features(input_data, feature_list):
    try:
        return [input_data[feature] for feature in feature_list]
    except KeyError as e:
        missing_feature = e.args[0]
        raise KeyError(f"Missing feature: {missing_feature}")

# Endpoint to predict Loan Status
@app.route('/loanstatus', methods=['POST'])
def predict_loan_status():
    data = request.json
    try:
        features = prepare_features(data, loan_status_features)
        df = spark.createDataFrame([(Vectors.dense(features),)], ["features"])
        prediction = loan_status_model.transform(df).select("prediction").collect()[0]["prediction"]
        return jsonify({"prediction": prediction})
    except KeyError as e:
        return jsonify({"error": str(e)}), 400

# Endpoint to predict Credit Score
@app.route('/creditscore', methods=['POST'])
def predict_credit_score():
    data = request.json
    try:
        features = prepare_features(data, credit_score_features)
        df = spark.createDataFrame([(Vectors.dense(features),)], ["features"])
        prediction = credit_score_model.transform(df).select("prediction").collect()[0]["prediction"]
        return jsonify({"prediction": prediction})
    except KeyError as e:
        return jsonify({"error": str(e)}), 400

# Endpoint to predict Maximum Open Credit
@app.route('/maxopencredit', methods=['POST'])
def predict_max_open_credit():
    data = request.json
    try:
        features = prepare_features(data, max_open_credit_features)
        df = spark.createDataFrame([(Vectors.dense(features),)], ["features"])
        prediction = max_open_credit_model.transform(df).select("prediction").collect()[0]["prediction"]
        return jsonify({"prediction": prediction})
    except KeyError as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
