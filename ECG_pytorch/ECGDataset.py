from torch.utils.data import Dataset
import torch

"""
@brief: ECGDataset class from ECG tensors and their labels
        (label = 0 -> healthy, label = 1 -> unhealthy)
"""


class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        # print(len(self.data))
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].clone().detach().requires_grad_(True), self.labels[idx].clone().detach().requires_grad_(True)
        # return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx])
