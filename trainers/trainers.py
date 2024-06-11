import torch
import json

from testers.test_model import test

"""
trains one epoch
"""


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for _, (X, y) in enumerate(dataloader):
        # Predict
        pred = model(X)
        loss = loss_fn(torch.squeeze(pred), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


"""
trains pytorch model for a pre-defined number of epochs
"""


def train_epochs(model, train_data_loader, test_data_loader, epochs, loss_fcn, optimizer, tr_size, te_size,
                 print_to_std=True, load_model=True):
    tr_losses = []
    tr_accuracies = []
    te_losses = []
    te_accuracies = []
    best_loss = torch.inf

    for ep in range(epochs):
        if print_to_std:
            print("[EPOCH]: " + str(ep + 1) + "/" + str(epochs))
        train_loop(train_data_loader, model, loss_fcn, optimizer)
        print("Train metrics:")
        tr_l, tr_a, tr_prec, tr_rec, tr_f1, tr_prec_v, tr_rec_v, tr_f1_v = test(train_data_loader, model, loss_fcn, tr_size, print_to_std)
        print("Test metrics:")
        te_l, te_a, te_prec, te_rec, te_f1, te_prec_v, te_rec_v, te_f1_v = test(test_data_loader, model, loss_fcn, te_size, print_to_std)
        print(f"prev_loss: {best_loss}, new_loss: {te_l}")
        if te_l < best_loss:
            print(f"Loss score decreased: prev_loss: {best_loss}, new_loss: {te_l}")
            best_loss = te_l
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'te_accuracy': te_a,
                'tr_accuracy': tr_a,
                'te_recall': te_rec,
                'tr_recall': tr_rec,
                'te_precision': te_prec,
                'tr_precision': tr_prec,
                'te_f1': te_f1,
                'tr_f1': tr_f1,
                'te_recall_v': te_rec_v,
                'tr_recall_v': tr_rec_v,
                'te_precision_v': te_prec_v,
                'tr_precision_v': tr_prec_v,
                'te_f1_v': te_f1_v,
                'tr_f1_v': tr_f1_v
            }, f"{model.__class__.__name__}.pt")

        tr_losses.append(tr_l)
        tr_accuracies.append(tr_a)

        te_losses.append(te_l)
        te_accuracies.append(te_a)

    """ 
        we save training loss in this file just to have  
        a reference to it in case models starts to behave strangely
    """
    with open('my_list.json', 'w') as file:
        json.dump(tr_losses, file)
    return tr_losses, tr_accuracies, te_losses, te_accuracies
