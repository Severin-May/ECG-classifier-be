from torch.utils.data import Dataset

"""
    Dataset class for handling ECG tensors and their corresponding labels.

    Args:
    - data (list or torch.Tensor): List of ECG tensors or a single tensor representing the input data.
    - labels (list or torch.Tensor): List of labels (0 for healthy, 1 for unhealthy) or a tensor representing the labels.

    Example usage:
    # Creating an instance of ECGDataset
    data = [...]  # List or tensor of ECG tensors
    labels = [...]  # List or tensor of corresponding labels
    ecg_dataset = ECGDataset(data, labels)

    Note:
    - Ensure that the length of the `data` and `labels` inputs is the same.

    Methods:
    - __len__(self): Returns the number of samples in the dataset.
    - __getitem__(self, idx): Retrieves a specific sample (ECG tensor and label) by index.

    Attributes:
    - data: The input ECG tensors.
    - labels: The corresponding labels (0.0 for healthy, 1.0 for unhealthy).
"""


class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        # print(len(self.data))
        return len(self.data)

    """
        Get a specific sample (ECG tensor and label) by index.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: A tuple containing the ECG tensor and its corresponding label.
    """

    def __getitem__(self, idx):
        return self.data[idx].clone().detach().requires_grad_(True), \
               self.labels[idx].clone().detach().requires_grad_(True)
