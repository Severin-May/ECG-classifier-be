# TODO: check is everything is working
"""
@brief: this file is responsible for creating a CustomDataset by
        generating tensors from ECG data_handlers stored in CSV files.
"""
import torch
import pandas as pd
import numpy as np
import os

"""
@brief: creates 2 Tensors from one CSV file (one row in csv represents one heartbeat)
@param: isWithLabels - data_handlers provide can be with labels or without
@return: labels, data_handlers
"""


def create_tensor_from_csv(is_with_labels=True, csv_file_path="./ecg_csv_files/100_heartbeats_1.csv"):
    data_frame = pd.read_csv(csv_file_path)
    labels, data = None, None  # Initialize as None

    if is_with_labels:
        labels = torch.tensor(data_frame.iloc[:, 0].values, dtype=torch.float32)
        data = torch.tensor(data_frame.iloc[:, 2:].values, dtype=torch.float32)
    else:
        data = torch.tensor(data_frame.values, dtype=torch.float32)

    return labels, data


# TODO: write doc
REC_STATISTICS_FILE = "record_statistics.txt"
def process_csv(file_path):
    df = pd.read_csv(file_path)

    healthy_beats = df.loc[df.iloc[:, 0] == 0.0, df.columns[0:]].to_numpy()
    unhealthy_beats = df.loc[df.iloc[:, 0] == 1.0, df.columns[0:]].to_numpy()

    # with open(REC_STATISTICS_FILE, 'a') as file:
    #     file.write(f"File:{file_path}, #healthy: {len(healthy_beats)},     #unhealthy: {len(unhealthy_beats)}\n")
    # nan_locations = df[df.isna().any(axis=1)]
    # if not nan_locations.empty:
    #     print(f"file {file_path} The following rows contain NaN values:")
    #     print(nan_locations)
    # else:
    #     print("No NaN values found in the DataFrame.")
    return healthy_beats, unhealthy_beats


# TODO: write doc
def process_multiple_csv(folder_path):
    all_healthy_beats = np.empty((0, 302))
    all_unhealthy_beats = np.empty((0, 302))

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            healthy_beats, unhealthy_beats = process_csv(file_path)

            all_healthy_beats = np.concatenate((all_healthy_beats, healthy_beats), axis=0)
            all_unhealthy_beats = np.concatenate((all_unhealthy_beats, unhealthy_beats), axis=0)

    return all_healthy_beats, all_unhealthy_beats


# healthy_beats, unhealthy_beats = process_multiple_csv("./ecg_csv_files")
# print("Healthy Beats Tensor Shape:", healthy_beats.shape)
# print("Unhealthy Beats Tensor Shape:", unhealthy_beats.shape)

# Healthy Beats Tensor Shape: (74696, 302) -> 74696*0.8=59756(training), 14940(test)
# Unhealthy Beats Tensor Shape: (6824, 302) -> 6824*0.8=5459(training), 1365(test)
# Balanced training dataset: 5459 healthy and 5459 unhealthy
# Unbalanced training dataset: 59756 unhealthy and 5459 healthy beats

testHealthyDataEndIndex = 14940
testUnhealthyDataEndIndex = 1365


# TODO: make a doc
def create_balanced_training_datasetOld():
    healthy_beats, unhealthy_beats = process_multiple_csv("./ecg_csv_files")
    balanced_beats = np.concatenate((healthy_beats[testHealthyDataEndIndex:testHealthyDataEndIndex + 84],
                                     unhealthy_beats[testUnhealthyDataEndIndex:testUnhealthyDataEndIndex + 84]))
    np.random.shuffle(balanced_beats)
    labels = torch.tensor(balanced_beats[:, 0])
    data = torch.tensor(balanced_beats[:, 2:302], dtype=torch.float32)
    return labels, data

# TODO: make a doc
def create_unbalanced_training_datasetOld():
    healthy_beats, unhealthy_beats = process_multiple_csv("./ecg_csv_files")
    unbalanced_beats = np.concatenate((healthy_beats[testHealthyDataEndIndex:testHealthyDataEndIndex + 58596],
                                       unhealthy_beats[testUnhealthyDataEndIndex:testUnhealthyDataEndIndex + 84]))
    np.random.shuffle(unbalanced_beats)
    labels = torch.tensor(unbalanced_beats[:, 0])
    data = torch.tensor(unbalanced_beats[:, 2:302], dtype=torch.float32)
    return labels, data

# TODO: make a doc
def create_test_datasetOld():
    healthy_beats, unhealthy_beats = process_multiple_csv("./ecg_csv_files")
    test_beats = np.concatenate((healthy_beats[:14650],
                                     unhealthy_beats[:21]))
    np.random.shuffle(test_beats)
    labels = torch.tensor(test_beats[:, 0])
    data = torch.tensor(test_beats[:, 2:302], dtype=torch.float32)
    return labels, data


""" please ignore above function except process_csv """

"""
    @brief: Process multiple CSV files from the specified folder and separate healthy and unhealthy beats.

    @param: folder_path (str): Path to the folder containing CSV files.

    @return:
    - all_healthy_beats (numpy.ndarray): Concatenated array of healthy beats.
    - all_unhealthy_beats (numpy.ndarray): Concatenated array of unhealthy beats.
    - all_test_beats (numpy.ndarray): Concatenated array of beats from test files.
"""

def process_multiple_csv_separate(folder_path):
    all_healthy_beats = np.empty((0, 302))
    all_unhealthy_beats = np.empty((0, 302))
    all_test_beats = np.empty((0, 302))

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            healthy_beats, unhealthy_beats = process_csv(file_path)

            if is_test_file(file_name):
                all_test_beats = np.concatenate((all_test_beats, healthy_beats), axis=0)
                all_test_beats = np.concatenate((all_test_beats, unhealthy_beats), axis=0)
            else:
                all_healthy_beats = np.concatenate((all_healthy_beats, healthy_beats), axis=0)
                all_unhealthy_beats = np.concatenate((all_unhealthy_beats, unhealthy_beats), axis=0)

    return all_healthy_beats, all_unhealthy_beats, all_test_beats

TEST_FILE_RECORDS = [106, 116, 215, 228, 231, 212, 112, 107, 201]

"""
@brief: check if the given filename corresponds to a test file based on predefined record numbers.

@param:
- filename (str): Name of the file.

@return:
- bool: True if the file is a test file, False otherwise.
"""

def is_test_file(filename):
    return any(str(rec) in filename for rec in TEST_FILE_RECORDS)

"""
    @brief: create a balanced training dataset by combining a subset of healthy beats and all unhealthy beats.

    @return:
    - labels (torch.Tensor): Labels for the dataset.
    - data (torch.Tensor): ECG data for the dataset.
"""
def create_balanced_training_dataset(device=None):
    healthy_beats, unhealthy_beats, _ = process_multiple_csv_separate("ECG_pytorch/ecg_csv_files")
    balanced_beats = np.concatenate((healthy_beats[:5450], unhealthy_beats))
    balanced_beats = balanced_beats.astype(np.float32)
    np.random.shuffle(balanced_beats)
    labels = torch.tensor(balanced_beats[:, 0], dtype=torch.float32, device=device)
    data = torch.tensor(balanced_beats[:, 2:302], dtype=torch.float32, device=device)
    return labels, data


"""
@breif: create an unbalanced training dataset by combining a subset of healthy beats and all unhealthy beats.

@return:
- labels (torch.Tensor): Labels for the dataset.
- data (torch.Tensor): ECG data for the dataset.
"""
def create_unbalanced_training_dataset(device=None):
    healthy_beats, unhealthy_beats, _ = process_multiple_csv_separate("ECG_pytorch/ecg_csv_files")
    unbalanced_beats = np.concatenate((healthy_beats[5450:], unhealthy_beats))
    unbalanced_beats = unbalanced_beats.astype(np.float32)
    np.random.shuffle(unbalanced_beats)
    labels = torch.tensor(unbalanced_beats[:, 0], dtype=torch.float32, device=device)
    data = torch.tensor(unbalanced_beats[:, 2:302], dtype=torch.float32, device=device)
    return labels, data


"""
    @brief: create a test dataset using beats from specified test files.

    @return:
    - labels (torch.Tensor): Labels for the dataset.
    - data (torch.Tensor): ECG data for the dataset.
"""
def create_test_dataset(device=None):
    _, _, test_beats = process_multiple_csv_separate("ECG_pytorch/ecg_csv_files")
    labels = torch.tensor(test_beats[:, 0], dtype=torch.float32, device=device)
    data = torch.tensor(test_beats[:, 2:302], dtype=torch.float32, device=device)
    return labels, data