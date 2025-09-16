import streamlit as st
import pandas as pd
from feature_extractor import extract_features_from_csv
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
        
        # Path to the pre-trained model
        model_path = "best_model_cE1_sNDARAC904DMU"

        # Make predictions
        active_percentage, passive_percentage = load_model_and_predict(features_df, model_path)

        # Show the results
        st.write(f"Percentage of Active Tasks: {active_percentage:.2f}%")
        st.write(f"Percentage of Passive Tasks: {passive_percentage:.2f}%")

        
        # Create a pie chart
        labels = ['Active Fatigue', 'Passive Fatigue']
        sizes = [active_percentage, passive_percentage]
        colors = ['#ff9999','#66b3ff']
        explode = (0.1, 0)  # explode 1st slice

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Display the pie chart in Streamlit
        st.pyplot(fig1)
        # Suggest actions based on fatigue levels
        if active_percentage > passive_percentage:
            st.write("### Suggestions to Reduce Active Fatigue:")
            st.write("- Ensure proper hydration and nutrition.")
            st.write("- Take regular breaks and pace yourself during tasks.")
            st.write("- Incorporate proper warm-up and cool-down exercises if applicable.")
        elif passive_percentage > active_percentage:
            st.write("### Suggestions to Reduce Passive Fatigue:")
            st.write("- Implement regular movement breaks to stay active.")
            st.write("- Engage in varied tasks and activities to avoid monotony.")
            st.write("- Maintain an ergonomic and comfortable workstation setup.")
        else:
            st.write("### General Fatigue Management Suggestions:")
            st.write("- Maintain a balanced lifestyle with adequate physical activity.")
            st.write("- Ensure mental engagement through varied activities.")
            st.write("- Take regular intervals for relaxation and refreshment.")

if __name__ == "__main__":
    main()
