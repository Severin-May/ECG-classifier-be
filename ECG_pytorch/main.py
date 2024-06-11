import torch
from torch.utils.data import DataLoader
import data_handler
from ECG_pytorch.ECGDataset import ECGDataset
from models import VPNet8, VPNet12, CNNet, VPNet4, FCNNNet
import trainers
from torch import nn
from qrs_data_handler import mat_to_tensors
from matplotlib import pyplot as plt

def main():
    # labels, data = data_handler.create_tensor_from_csv()
    labels, data = data_handler.create_balanced_training_dataset()
    # labels, data = data_handler.create_unbalanced_training_dataset()
    # labels, data = mat_to_tensors()
    custom_dataset = ECGDataset(data, labels)

    labels_test, data_test = data_handler.create_test_dataset()
    test_dataset = ECGDataset(data_test, labels_test)


    batch_size = 250
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model = VPNet4().to(device)
    # model = VPNet8().to(device)
    # model = VPNet12().to(device)
    # model = CNNet().to(device)
    # model = FCNNNet().to(device)


    # hyperparameters (adjustable)
    learning_rate = 1e-3
    epochs = 50

    # initialize the loss function
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()

    # initialize optimizer object which encapsulates all optimization logic
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    tr_loss, tr_acc, te_loss, te_acc = trainers.train_epochs(model, dataloader, dataloader_test, 200, loss_fn, optimizer, len(dataloader), len(dataloader_test))
    plt.figure()
    plt.plot(tr_loss)
    plt.plot(te_loss)
    plt.show()

if __name__ == "__main__":
    main()