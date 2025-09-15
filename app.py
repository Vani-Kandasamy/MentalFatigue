import streamlit as st
import pandas as pd
from feature_extraction import extract_features_from_csv
from model_prediction import load_model_and_predict
import os


def main():
    st.title("Feature Extraction and Prediction App")

    # Upload files
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        # Extract features from the uploaded files
        features_df = extract_features_from_csv(uploaded_files)

        # Display the extracted features if needed
        st.write("Extracted Features:")
        st.write(features_df)

        # Cache the extracted features
        features_csv_path = "time_series_features_extracted.csv"
        features_df.to_csv(features_csv_path, index=False)

        # Path to the pre-trained model
        model_path = "/content/drive/MyDrive/1:1_Alessia_Brinzarea/Results/sub-NDARAC904DMU/best_model_cE1_sNDARAC904DMU"

        # Make predictions
        active_percentage, passive_percentage = load_model_and_predict(features_csv_path, model_path)

        # Show the results
        st.write(f"Percentage of Active Tasks: {active_percentage:.2f}%")
        st.write(f"Percentage of Passive Tasks: {passive_percentage:.2f}%")


if __name__ == "__main__":
    main()