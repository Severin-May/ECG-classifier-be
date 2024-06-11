"""
@brief: this file contains trainings
"""

import torch
# from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import pickle

"""
@brief: trains one epoch
"""


def train_loop(dataloader, model, loss_fn, optimizer):
    print("train loop")
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
@brief: calculates metrics of the model
"""
#
# def calculate_metrics_torch(input, target):
#     accuracy = BinaryAccuracy()
#     accuracy.update(input, target)
#     a = accuracy.compute()
#
#     f1 = BinaryF1Score()
#     f1.update(input, target)
#     f = f1.compute()
#
#     precision = BinaryPrecision()
#     precision.update(input, target)
#     p = precision.compute()
#
#     int_input = input.to(dtype=torch.int)
#     int_target = target.to(dtype=torch.int)
#     recall = BinaryRecall()
#     recall.update(int_input, int_target)
#     r = recall.compute()
#
#     return a, p, r, f

"""
@brief: tests the accuracy and calculates the loss
"""


def test(dataloader, model, loss_fn, size, print_to_std=True):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_true_labels = torch.empty(0, dtype=torch.float32)
    all_predicted_values = torch.empty(0, dtype=torch.float32)
    accuracy, recall, precision, f1, recall_v, precision_v, f1_v = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    TP, TN, FP, FN, TP_v, TN_v, FP_v, FN_v = 0, 0, 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pr = torch.squeeze(pred)
            rounded_predictions = torch.round(pr)
            rounded_preds = (pred >= 0.5).float()
            test_loss += loss_fn(pr, y).item()
            correct += (torch.round(pr) == y).type(torch.float).sum().item() / pr.shape[0]

            TP_v += ((rounded_predictions == 1.0) & (y == 1.0)).float().sum().item()
            FP_v += ((rounded_predictions == 1.0) & (y == 0.0)).float().sum().item()
            TN_v += ((rounded_predictions == 0.0) & (y == 0.0)).float().sum().item()
            FN_v += ((rounded_predictions == 0.0) & (y == 1.0)).float().sum().item()

            TN += ((rounded_predictions == 1.0) & (y == 1.0)).float().sum().item()
            FN += ((rounded_predictions == 1.0) & (y == 0.0)).float().sum().item()
            TP += ((rounded_predictions == 0.0) & (y == 0.0)).float().sum().item()
            FP += ((rounded_predictions == 0.0) & (y == 1.0)).float().sum().item()

            # all_true_labels.extend(y.cpu().numpy())
            # all_predicted_values.extend(rounded_predictions.cpu().numpy())
            all_true_labels = torch.cat((all_true_labels, y), dim=0)
            all_predicted_values = torch.cat((all_predicted_values, rounded_preds), dim=0)

    test_loss /= num_batches #save in array and plot
    correct /= size
    accuracy += (TP + TN) / (TP + TN + FP + FN)
    precision += TP / max(TP + FP, 1)
    recall += TP / max(TP + FN, 1)
    f1 += 2 * (precision * recall) / max((precision + recall), 1)

    precision_v += TP_v / max(TP_v + FP_v, 1)
    recall_v += TP_v / max(TP_v + FN_v, 1)
    f1_v += 2 * (precision_v * recall_v) / max((precision_v + recall_v), 1)

    # if print_to_std:
    #     print(f"Accuracy: {(100 * correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")

    # accuracy, precision, recall, f1 = calculate_metrics(all_true_labels, all_predicted_values)
    # accuracy, precision, recall, f1 = calculate_metrics_torch(all_predicted_values.squeeze(dim=1), all_true_labels)
    if print_to_std:
        print(f"Avg loss: {test_loss:>8f}, Accuracy: {(accuracy*100):.3f}%, Precision: {(precision*100):.3f}%, Recall: {(recall*100):.3f}%, F1 Score: {(f1*100):.3f}%\n")
        print(f"Avg loss_v: {test_loss:>8f}, Accuracy_v: {(accuracy*100):.3f}%, Precision_v: {(precision_v*100):.3f}%, Recall_v: {(recall_v*100):.3f}%, F1 Score_v: {(f1_v*100):.3f}%\n")
    return test_loss, accuracy, precision, recall, f1

"""
@brief: trains pytorch model for a pre-defined number of epochs
"""


def train_epochs(model, train_data_loader, test_data_loader, epochs, loss_fcn, optimizer, tr_size, te_size, print_to_std=True, load_model=True):
    tr_losses = []
    tr_accuracies = []
    te_losses = []
    te_accuracies = []
    best_loss = torch.inf

    # if load_model:
    #     load_checkpoint(torch.load("checkpoint.pth.tar"), model, optimizer)

    for ep in range(epochs):
        if print_to_std:
            print("[EPOCH]: " + str(ep + 1) + "/" + str(epochs))
        train_loop(train_data_loader, model, loss_fcn, optimizer)
        print("Train metrics:")
        tr_l, tr_a, tr_prec, tr_rec, tr_f1 = test(train_data_loader, model, loss_fcn, tr_size, print_to_std)
        print("Test metrics:")
        te_l, te_a, te_prec, te_rec, te_f1 = test(test_data_loader, model, loss_fcn, te_size, print_to_std)
        print(f"prev_loss: {best_loss}, new_loss: {te_l}")
        if te_l < best_loss:
            print(f"Loss score decreased: prev_loss: {best_loss}, new_loss: {te_l}")
            best_loss = te_l
            # checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            # save_checkpoint(checkpoint)
            # with open("checkpoint_results.txt", 'a') as file:
            #     file.write(f"some metrics info\n")
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "tr_accuracy": tr_a, "te_accuracy": te_a,
                          "tr_loss" : tr_losses, "te_loss" : te_losses,
                          "tr_prec" : tr_prec, "te_prec" : te_prec,
                          "tr_rec" : tr_rec, "te_rec" : te_rec,
                          "tr_f1" : tr_f1, "te_f1" : te_f1}

            with open(f"{model.__class__.__name__}.pickle", 'wb') as f:
                pickle.dump(checkpoint, f)

        tr_losses.append(tr_l)
        tr_accuracies.append(tr_a)

        te_losses.append(te_l)
        te_accuracies.append(te_a)

    # if load_model:
    #     with open(f"{model.__class__.__name__}.pickle", 'rb') as f:
    #         checkpoint = pickle.load(f)
    #         model.load_state_dict(checkpoint["state_dict"])
    #         print("The best achieved model scores:")
    #         print("Train metrics:")
    #         test(train_data_loader, model, loss_fcn, tr_size, print_to_std)
    #         print("Test metrics:")
    #         test(test_data_loader, model, loss_fcn, te_size, print_to_std)

    import json
    with open('tr_loss.json', 'w') as file:
        json.dump(tr_losses, file)
    with open('te_loss.json', 'w') as file:
        json.dump(te_losses, file)
    return tr_losses, tr_accuracies, te_losses, te_accuracies


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

#TODO: add cross-validation