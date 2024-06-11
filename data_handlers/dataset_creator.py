import torch
import numpy as np


# Healthy Beats Tensor Shape: (74670, 302) -> 74670*0.8=60631(training), 14039(test)
# Unhealthy Beats Tensor Shape: (6824, 302) -> 6824*0.8=5450(training), 1374(test)
# Balanced training dataset: 5450 healthy and 5450 unhealthy
# Unbalanced training dataset: 60631 healthy and 5450 unhealthy beats

"""
    @brief: create a balanced training dataset by combining a subset of healthy beats and all unhealthy beats.

    @return:
    - labels (torch.Tensor): Labels for the dataset.
    - data (torch.Tensor): ECG data for the dataset.
"""


def create_balanced_training_dataset(healthy_beats, unhealthy_beats, device=None):

    beats = np.concatenate((healthy_beats[:5450], unhealthy_beats))
    beats = beats.astype(np.float32)
    np.random.shuffle(beats)
    labels = torch.tensor(beats[:, 0], dtype=torch.float32, device=device)
    data = torch.tensor(beats[:, 2:302], dtype=torch.float32, device=device)
    return labels, data


"""
@brief: create an unbalanced training dataset by combining a subset of healthy beats and all unhealthy beats.

@return:
- labels (torch.Tensor): Labels for the dataset.
- data (torch.Tensor): ECG data for the dataset.
"""


def create_unbalanced_training_dataset(healthy_beats, unhealthy_beats, device=None):
    beats = np.concatenate((healthy_beats[5450:], unhealthy_beats))
    beats = beats.astype(np.float32)
    np.random.shuffle(beats)
    labels = torch.tensor(beats[:, 0], dtype=torch.float32, device=device)
    data = torch.tensor(beats[:, 2:302], dtype=torch.float32, device=device)
    return labels, data


"""
    @brief: create a test dataset using beats from specified test files.

    @return:
    - labels (torch.Tensor): Labels for the dataset.
    - data (torch.Tensor): ECG data for the dataset.
"""


def create_test_dataset(test_beats, device=None):
    labels = torch.tensor(test_beats[:, 0], dtype=torch.float32, device=device)
    data = torch.tensor(test_beats[:, 2:302], dtype=torch.float32, device=device)
    return labels, data
