import torch
from torch.utils.data import DataLoader
from torch import nn
from matplotlib import pyplot as plt

from data_handlers.ECGDataset import ECGDataset
from data_handlers.dataset_creator import create_unbalanced_training_dataset, create_balanced_training_dataset, \
    create_test_dataset
from data_handlers.data_reader_from_file import CSVProcessor, FileProcessorContext, process_multiple_files

from models.models import VPNet4, FCNNet, VPNet12, CNNet, VPNet8
from trainers import trainers


def main():
    # device = ("cuda" if torch.cuda.is_available() else "cpu")
    """  configurations """
    device = None
    learning_rate = 1e-3
    batch_size = 200
    epochs = 100

    csv_processor = CSVProcessor()
    csv_processor_context = FileProcessorContext(csv_processor)
    healthy_beats, unhealthy_beats, test_beats = process_multiple_files("./ecg_csv_files", csv_processor_context)

    labels_u, data_u = create_unbalanced_training_dataset(healthy_beats, unhealthy_beats, device)
    labels_b, data_b = create_balanced_training_dataset(healthy_beats, unhealthy_beats, device)
    labels_t, data_t = create_test_dataset(test_beats, device)

    print(f"unbalanced: {len(labels_u)}, {len(data_u)},"
          f"balanced: {len(labels_b)}, {len(data_b)},"
          f"test: {len(labels_t)}, {len(data_t)}")

    labels_u = labels_u.to(device)
    data_u = data_u.to(device)
    training_dataset = ECGDataset(data_u, labels_u)

    labels_t = labels_t.to(device)
    data_t = data_t.to(device)
    test_dataset = ECGDataset(data_t, labels_t)

    dataloader_training = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    """ available models """
    model = VPNet4().to(device)
    # model = VPNet8().to(device)
    # model = VPNet12().to(device)
    # model = CNNet().to(device)
    # model = FCNNet(device).to(device)

    """ loss functions """
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    tr_loss, tr_acc, te_loss, te_acc = trainers.train_epochs(model, dataloader_training, dataloader_test, epochs, loss_fn,
                                                             optimizer, len(dataloader_training), len(dataloader_test))
    plt.figure()
    plt.plot(tr_loss)
    plt.show()

    plt.plot(te_loss)
    plt.show()

if __name__ == "__main__":
    main()
