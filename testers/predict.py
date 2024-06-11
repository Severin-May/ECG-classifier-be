import pickle
import torch

from models.models import FCNNet, CNNet, VPNet4, VPNet8, VPNet12
from testers.test_model import calculate_metrics


def preprocess_input_data(df, labeled):
    expected_columns = 302 if labeled else 300
    if df.shape[1] != expected_columns:
        raise ValueError(f"Each row must have exactly {expected_columns} columns.")

    # if df.dtypes.any() != 'float32':
    #     raise ValueError("Data type of all columns must be float32.")

    if labeled and expected_columns == 302:
        # unique_values = df.iloc[:, 0].unique()
        # if set(unique_values) != {0.0, 1.0}:
        #     raise ValueError("The values in the first column must be either 0.0 or 1.0.")
        labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32)
        data = torch.tensor(df.iloc[:, 2:].values, dtype=torch.float32)
        return data, labels

    data_tensor = torch.tensor(df.values, dtype=torch.float32)

    selected_columns = df.iloc[:, 2:]
    # Convert selected columns to a PyTorch tensor
    # data_tensor = torch.tensor(selected_columns.values, dtype=torch.float32)

    return data_tensor, None


""" loading models with pickles """
def predict_and_evaluate_pickle(model_str, df, is_labeled):
    preprocessed_data, labels = preprocess_input_data(df, labeled=is_labeled)
    model = None

    if model_str == "FCNNet":
        model = FCNNet(None).to(None)
    elif model_str == "CNNet":
        model = CNNet().to(None)
    elif model_str == "VPNet4":
        model = VPNet4().to(None)
    elif model_str == "VPNet8":
        model = VPNet8().to(None)
    elif model_str == "VPNet12":
        model = VPNet12().to(None)

    with open(f"../{model.__class__.__name__}.pickle", 'rb') as f:
        checkpoint = pickle.load(f)
        model.load_state_dict(checkpoint["state_dict"])
        print("Using saved model for prediction")

        model.eval()
        predicted_labels = model(preprocessed_data)
        size = predicted_labels.shape[0]

        if is_labeled:
            metrics = calculate_metrics(labels, predicted_labels)
            rounded_pr = torch.round(predicted_labels)
            metrics['predictions'] = rounded_pr.tolist()
            metrics['original_labels'] = labels.tolist()

            metrics['Accuracy'] = round(metrics['Accuracy'] / size, 3) * 100
            metrics['Precision'] = round(metrics['Precision'] / size, 3) * 100
            metrics['Recall'] = round(metrics['Recall'] / size, 3) * 100
            metrics['F1_score'] = round(metrics['F1_score'] / size, 3) * 100
            metrics['Precision_v'] = round(metrics['Precision_v'] / size, 3) * 100
            metrics['Recall_v'] = round(metrics['Recall_v'] / size, 3) * 100
            metrics['F1_score_v'] = round(metrics['F1_score_v'] / size, 3) * 100

            return metrics

        results = {
            'Accuracy': (torch.round(predicted_labels) == labels).type(torch.float).sum().item() / size,
            'Predictions': predicted_labels
        }

        return results


""" loading models with torch load """
def predict_and_evaluate(model_str, df, is_labeled):
    preprocessed_data, labels = preprocess_input_data(df, labeled=is_labeled)
    model = None

    if model_str == "FCNNet":
        model = FCNNet(None).to(None)
    elif model_str == "CNNet":
        model = CNNet().to(None)
    elif model_str == "VPNet4":
        model = VPNet4().to(None)
    elif model_str == "VPNet8":
        model = VPNet8().to(None)
    elif model_str == "VPNet12":
        model = VPNet12().to(None)

    size = preprocessed_data.shape[0]
    checkpoint = torch.load(f"../{model.__class__.__name__}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    predicted_labels = model(preprocessed_data)
    rounded_pr = torch.round(predicted_labels)

    if is_labeled:
        metrics = calculate_metrics(labels, predicted_labels)
        metrics['Predictions'] = rounded_pr.tolist()
        metrics['Original_labels'] = labels.tolist()

        metrics['Accuracy'] = round(metrics['Accuracy'], 3) * 100
        metrics['Precision'] = round(metrics['Precision'], 3) * 100
        metrics['Recall'] = round(metrics['Recall'], 3) * 100
        metrics['F1_score'] = round(metrics['F1_score'], 3) * 100
        metrics['Precision_v'] = round(metrics['Precision_v'], 3) * 100
        metrics['Recall_v'] = round(metrics['Recall_v'], 3) * 100
        metrics['F1_score_v'] = round(metrics['F1_score_v'], 3) * 100

        return metrics

    results = {
        'Predictions': rounded_pr.tolist()
    }

    return results
