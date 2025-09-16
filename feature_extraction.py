import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from collections import defaultdict
import numpy as np
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import numpy as np
def extract_features_from_csv(files):
    # Read all uploaded CSVs and concatenate them
    dataframes = [pd.read_csv(file) for file in files]
    data = pd.concat(dataframes, ignore_index=True)

    # Select channels
    NUMBER_CHANELS_TO_SELECT = 5
    chanels = [f"E{chanel_no + 1}" for chanel_no in range(NUMBER_CHANELS_TO_SELECT)]
    chanels.extend(["id", "label", "Time"])
    data = data[chanels]

    # Prepare data for feature extraction
    final_seq = data.copy()
    final_seq.drop("label", axis=1, inplace=True)
    final_seq.reset_index(drop=True, inplace=True)

    # Map IDs to labels
    ids = data["id"].unique()
    MAPPING_DICT = defaultdict(list)
    for id_ in ids:
        section_df = data[data["id"] == id_]
        section_label = section_df["label"].unique()
        MAPPING_DICT["ID"].append(id_)
        MAPPING_DICT["Label"].append(section_label[0])
    MAPPING_DF = pd.DataFrame(MAPPING_DICT)

    # Extract features
    extracted_features = extract_features(final_seq, column_id="id", column_sort="Time", default_fc_parameters=EfficientFCParameters())

    # Impute and filter features
    impute(extracted_features)
    labels_filter = MAPPING_DF[MAPPING_DF["ID"].isin(list(extracted_features.index))]
    features_filtered = select_features(extracted_features, np.array(labels_filter["Label"].to_list()))

    # Attach labels
    final_ids = list(features_filtered.index)
    label_list = []
    for f_id in final_ids:
        churn_label = list(labels_filter.loc[labels_filter['ID'] == f_id, "Label"].values)[0]
        label_list.append(churn_label)
    features_filtered["label"] = label_list

    return features_filtered
