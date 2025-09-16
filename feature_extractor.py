# feature_extraction.py

# -*- coding: utf-8 -*-
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
import dask.dataframe as dd
from dask.distributed import Client
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
import numpy as np

def extract_features_from_csv(files):
   
    # Read all uploaded CSVs and concatenate them
    dataframes = [pd.read_csv(file) for file in files]
    data = pd.concat(dataframes, ignore_index=True)

    # Drop the label column for feature extraction
    final_seq = data.copy()
    final_seq.drop("label", axis=1, inplace=True)
    final_seq.reset_index(drop=True, inplace=True)

    # Create a mapping of IDs to their respective labels
    ids = data["id"].unique()
    mapping_dict = {"ID": [], "Label": []}
    for id_ in tqdm(ids, desc="Mapping IDs to labels"):
        label = data.loc[data["id"] == id_, "label"].iloc[0]
        mapping_dict["ID"].append(id_)
        mapping_dict["Label"].append(label)
    mapping_df = pd.DataFrame(mapping_dict)

   ''' 
   # Setup Dask client for parallel processing and extract features
    client = Client(n_workers=4)
    ddf = dd.from_pandas(final_seq, npartitions=2)
    extracted_features = extract_features(ddf, column_id="id", column_sort="Time", n_jobs=0, default_fc_parameters=EfficientFCParameters()).compute()
    '''
   # Extract features using tsfresh
    extracted_features = extract_features(
        final_seq,
        column_id="id",
        column_sort="Time",
        default_fc_parameters=EfficientFCParameters()
    )

    # Impute missing feature values and filter relevant features
    impute(extracted_features)
    labels_filter = mapping_df[mapping_df["ID"].isin(extracted_features.index)]
    features_filtered = select_features(extracted_features, labels_filter["Label"].values)

    # Attach the labels to the filtered features
    features_filtered["label"] = features_filtered.index.map(lambda x: labels_filter.loc[labels_filter["ID"] == x, "Label"].iloc[0])


    return features_filtered
