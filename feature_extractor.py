# feature_extraction.py

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

def extract_features_from_csv(files):
    # Read all uploaded CSVs and concatenate them
    data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    
    # Get unique IDs and prepare data for feature extraction
    ids = data["id"].unique()
    final_seq = data.drop("label", axis=1).copy()
    final_seq.reset_index(drop=True, inplace=True)

    # Create ID to label mapping
    MAPPING_DICT = defaultdict(list)
    for id_ in tqdm(ids, desc="Mapping IDs to labels"):
        section_df = data[data["id"] == id_]
        section_label = section_df["label"].unique()
        MAPPING_DICT["ID"].append(id_)
        MAPPING_DICT["Label"].append(section_label[0])

    # Create and sort the mapping dataframe
    MAPPING_DF = pd.DataFrame(MAPPING_DICT)
    labels = MAPPING_DF.sort_values(by="ID")
    labels.reset_index(drop=True, inplace=True)

    # Extract features using tsfresh
    extracted_features = extract_features(
        final_seq,
        column_id="id",
        column_sort="Time",
        default_fc_parameters=EfficientFCParameters()
    )
    
    # Handle missing values
    impute(extracted_features)
    
    # Select labels which are in extracted features
    labels_filter = labels[labels["ID"].isin(extracted_features.index)]
    
    # Select features based on labels
    features_filtered = select_features(
        extracted_features, 
        np.array(labels_filter["Label"].to_list())
    )
    
    # Get the final unique IDs and their corresponding labels
    final_ids = features_filtered.index
    label_list = [labels_filter.loc[labels_filter['ID'] == f_id, "Label"].values[0] 
                 for f_id in final_ids]
    
    # Attach the labels to the filtered features
    features_filtered["label"] = label_list
    
    return features_filtered