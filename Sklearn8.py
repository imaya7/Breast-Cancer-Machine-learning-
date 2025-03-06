# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    # Load the breast cancer dataset
    data = load_breast_cancer()
    x = data.data  # Features
    y = data.target  # Target labels

    # Display basic information about the dataset
    print(f"Dataset shape: {x.shape}")
    print(f"Feature names: {data.feature_names}")
    print(f"Target names: {data.target_names}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split the data into training and testing sets (80% train, 20% test)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the features to have mean 0 and variance 1
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)

except Exception as e:
    print(f"An error occurred during data loading or preprocessing: {e}")
    raise

# Function to evaluate a model's performance
def evaluateModel(model, xTrain, xTest, yTrain, yTest, modelName):
    try:
        # Train the model
        model.fit(xTrain, yTrain)
        
        # Make predictions
        yPred = model.predict(xTest)
        
        # Get the probabilities for ROC-AUC if possible
        if hasattr(model, "predict_proba"):
            yProb = model.predict_proba(xTest)[:, 1]
        else:
            # For models that don't have predict_proba, use decision_function or predictions
            yProb = model.decision_function(xTest) if hasattr(model, "decision_function") else yPred
        
        # Calculate metrics
        accuracy = accuracy_score(yTest, yPred)
        precision = precision_score(yTest, yPred)
        recall = recall_score(yTest, yPred)
        f1 = f1_score(yTest, yPred)
        rocAuc = roc_auc_score(yTest, yProb)
        confMatrix = confusion_matrix(yTest, yPred)
        
        # Print results
        print(f"\n{modelName} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {rocAuc:.4f}")
        print("Confusion Matrix:")
        print(confMatrix)
        print("\nClassification Report:")
        print(classification_report(yTest, yPred, target_names=data.target_names))
        
        return {
            "modelName": modelName,
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rocAuc": rocAuc,
            "confMatrix": confMatrix
        }
    except Exception as e:
        print(f"An error occurred while evaluating the model {modelName}: {e}")
        return None

try:
    # Train and evaluate different models

    # Logistic Regression Model
    logRegModel = LogisticRegression(max_iter=1000, random_state=42)
    logRegResults = evaluateModel(logRegModel, xTrainScaled, xTestScaled, yTrain, yTest, "Logistic Regression")

    # Random Forest Model
    rfModel = RandomForestClassifier(n_estimators=100, random_state=42)
    rfResults = evaluateModel(rfModel, xTrainScaled, xTestScaled, yTrain, yTest, "Random Forest")

    # Decision Tree Model
    dtModel = DecisionTreeClassifier(random_state=42)
    dtResults = evaluateModel(dtModel, xTrainScaled, xTestScaled, yTrain, yTest, "Decision Tree")

    # Combine the results and calculate the overall best model
    results = [logRegResults, rfResults, dtResults]

    # Calculate an overall score for each model by averaging the metrics
    for result in results:
        if result is not None:
            result["overallScore"] = (result["accuracy"] + result["precision"] + result["recall"] + result["f1"] + result["rocAuc"]) / 5

    # Find the best model based on the overall score
    bestModel = max(results, key=lambda x: x["overallScore"] if x is not None else float('-inf'))

    # Also, find the best models for each individual metric
    bestAccuracyModel = max(results, key=lambda x: x["accuracy"] if x is not None else float('-inf'))
    bestPrecisionModel = max(results, key=lambda x: x["precision"] if x is not None else float('-inf'))
    bestRecallModel = max(results, key=lambda x: x["recall"] if x is not None else float('-inf'))
    bestF1Model = max(results, key=lambda x: x["f1"] if x is not None else float('-inf'))
    bestRocAucModel = max(results, key=lambda x: x["rocAuc"] if x is not None else float('-inf'))

    # Print out the best models for each metric
    print("\nBest Models by Metric:")
    print(f"Best Overall Model: {bestModel['modelName']} (Score: {bestModel['overallScore']:.4f})")
    print(f"Best Accuracy: {bestAccuracyModel['modelName']} ({bestAccuracyModel['accuracy']:.4f})")
    print(f"Best Precision: {bestPrecisionModel['modelName']} ({bestPrecisionModel['precision']:.4f})")
    print(f"Best Recall: {bestRecallModel['modelName']} ({bestRecallModel['recall']:.4f})")
    print(f"Best F1 Score: {bestF1Model['modelName']} ({bestF1Model['f1']:.4f})")
    print(f"Best ROC-AUC: {bestRocAucModel['modelName']} ({bestRocAucModel['rocAuc']:.4f})")

    # Analyze feature importance for the best model
    print(f"\nFeature Importance for {bestModel['modelName']}:")

    # Logistic Regression uses coefficients as feature importance
    if bestModel['modelName'] == "Logistic Regression":
        coefficients = pd.DataFrame(
            {'feature': data.feature_names,
             'coefficient': np.abs(bestModel['model'].coef_[0])}  # Absolute value of coefficients
        ).sort_values('coefficient', ascending=False)
        
        print("\nTop 10 Important Features (Logistic Regression):")
        print(coefficients.head(10))

    # Random Forest and Decision Tree models use feature importances
    elif bestModel['modelName'] in ["Random Forest", "Decision Tree"]:
        featureImportances = pd.DataFrame(
            {'feature': data.feature_names,
             'importance': bestModel['model'].feature_importances_}
        ).sort_values('importance', ascending=False)
        
        modelType = "Random Forest" if bestModel['modelName'] == "Random Forest" else "Decision Tree"
        print(f"\nTop 10 Important Features ({modelType}):")
        print(featureImportances.head(10))

except Exception as e:
    print(f"An error occurred during model training or evaluation: {e}")
    raise