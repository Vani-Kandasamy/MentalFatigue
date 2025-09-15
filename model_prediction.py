import pickle
import pandas as pd
import numpy as np

def load_model_and_predict(features_csv_path, model_path):
    # Load the model
    with open(model_path, "rb") as file:
        final_model = pickle.load(file)
    print("Model is loaded.")

    # Load the extracted features
    features_filtered = pd.read_csv(features_csv_path)

    # Extract features and separate true labels
    y_true = features_filtered['label']
    test_data_list = features_filtered.drop(columns=['label'])

    # Making Predictions
    y_pred = []
    for row_data in test_data_list.values:
        prediction = final_model.predict(np.array([row_data]))
        y_pred.append(prediction[0])

    # Map the encoded predictions back to "active" and "passive"
    label_encoding = {0: "passive", 1: "active"}  # Adjust based on your model's encoding
    y_pred_mapped = [label_encoding[pred] for pred in y_pred]

    # Count the number of each type of task
    total_tasks = len(y_pred_mapped)
    active_count = y_pred_mapped.count("active")
    passive_count = y_pred_mapped.count("passive")

    # Calculate percentages
    active_percentage = (active_count / total_tasks) * 100
    passive_percentage = (passive_count / total_tasks) * 100

    print(f"Percentage of Active Tasks: {active_percentage:.2f}%")
    print(f"Percentage of Passive Tasks: {passive_percentage:.2f}%")

    return active_percentage, passive_percentage