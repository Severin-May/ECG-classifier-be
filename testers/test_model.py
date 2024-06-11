"""
@brief: tests the accuracy and calculates the loss
"""
import torch


def calculate_metrics(y, pr):
    correct = 0
    TP, TN, FP, FN, TP_v, TN_v, FP_v, FN_v = 0, 0, 0, 0, 0, 0, 0, 0
    rounded_pr = torch.round(pr.view(-1))
    # rounded_pr = torch.round(pr)
    # correct += (rounded_pr == y).type(torch.float).sum().item() / pr.shape[0]
    correct += (rounded_pr == y).sum().item()
    print(f"correct: {correct}")


    TP_v += ((rounded_pr == 1.0) & (y == 1.0)).float().sum().item()
    FP_v += ((rounded_pr == 1.0) & (y == 0.0)).float().sum().item()
    TN_v += ((rounded_pr == 0.0) & (y == 0.0)).float().sum().item()
    FN_v += ((rounded_pr == 0.0) & (y == 1.0)).float().sum().item()

    TP += ((rounded_pr == 0.0) & (y == 0.0)).float().sum().item()
    FP += ((rounded_pr == 0.0) & (y == 1.0)).float().sum().item()
    TN += ((rounded_pr == 1.0) & (y == 1.0)).float().sum().item()
    FN += ((rounded_pr == 1.0) & (y == 0.0)).float().sum().item()

    metrics_dict = {
        'Accuracy': correct/pr.shape[0],
        'Precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'Recall': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'F1_score': TP / (TP + 0.5 * (FP + FN)) if TP > 0 and (FP + FN) > 0 else 0,
        'Precision_v': TP_v / (TP_v + FP_v) if (TP_v + FP_v) > 0 else 0,
        'Recall_v': TP_v / (TP_v + FN_v) if (TP_v + FN_v) > 0 else 0,
        'F1_score_v': TP_v / (TP_v + 0.5 * (FP_v + FN_v)) if TP_v > 0 and (FP_v + FN_v) > 0 else 0
    }

    return metrics_dict


def test(dataloader, model, loss_fn, size, print_to_std=True):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    accuracy, recall, precision, f1, recall_v, precision_v, f1_v = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pr = torch.squeeze(pred)
            rounded_predictions = torch.round(pr)

            test_loss += loss_fn(pr, y).item()
            metrics_dict = calculate_metrics(y, pr)

            accuracy += metrics_dict['Accuracy']
            precision += metrics_dict['Precision']
            recall += metrics_dict['Recall']
            f1 += metrics_dict['F1_score']

            precision_v += metrics_dict['Precision_v']
            recall_v += metrics_dict['Recall_v']
            f1_v += metrics_dict['F1_score_v']

    test_loss /= num_batches
    correct /= size
    accuracy /= size
    precision /= num_batches
    recall /= num_batches
    f1 /= num_batches
    precision_v /= num_batches
    recall_v /= num_batches
    f1_v /= num_batches

    if print_to_std:
        print(
            f"Avg loss: {test_loss:>8f}, Accuracy: {(accuracy * 100):.3f}%, Precision: {(precision * 100):.3f}%, Recall: {(recall * 100):.3f}%, F1 Score: {(f1 * 100):.3f}%\n")
        print(
            f"Avg loss_v: {test_loss:>8f}, Accuracy_v: {(accuracy * 100):.3f}%, Precision_v: {(precision_v * 100):.3f}%, Recall_v: {(recall_v * 100):.3f}%, F1 Score_v: {(f1_v * 100):.3f}%\n")

    return test_loss, accuracy, precision, recall, f1, precision_v, recall_v, f1_v
