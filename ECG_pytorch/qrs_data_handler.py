# import scipy.io
# import torch
#
# def mat_to_tensors():
#     mat_file = scipy.io.loadmat('../qrs_only_datasets/ecg_train_balanced.mat')
#
#     data_samples = torch.tensor(mat_file['samples'])
#     data_labels = torch.tensor(mat_file['labels'])
#
#     return data_samples, data_labels
#
# mat_to_tensors()