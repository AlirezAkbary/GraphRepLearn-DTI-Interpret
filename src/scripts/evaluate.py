import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import mlflow.pytorch
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from src.utils.metrics import mse, ci
from src.datasets.drug_target_dataset import DrugTargetAffinityDataset

def evaluate_model(model, dataloader, device, dataset_name):
    model.to(device)
    model.eval()

    total_preds = torch.Tensor().to(device)
    total_affinities = torch.Tensor().to(device)

    with torch.no_grad():
        for data_object in dataloader:
            data_object = data_object.to(device)
            output = model(data_object)
            total_preds = torch.cat((total_preds, output), 0)
            total_affinities = torch.cat((total_affinities, data_object.y.view(-1, 1)), 0)

    total_affinities = total_affinities.cpu().numpy().flatten()
    total_preds = total_preds.cpu().numpy().flatten()

    # Calculate metrics
    test_mse = mse(total_affinities, total_preds)
    test_ci = ci(total_affinities, total_preds)
    print("Test MSE: ", test_mse)
    print("Test CI: ", test_ci)

    # Classification metrics
    threshold = 7 if dataset_name == "davis" else 12.1
    preds_classification = np.where(total_preds > threshold, 1, 0)
    affinities_classification = np.where(total_affinities > threshold, 1, 0)
    tn, fp, fn, tp = confusion_matrix(affinities_classification, preds_classification).ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall_score(affinities_classification, preds_classification)
    accuracy = accuracy_score(affinities_classification, preds_classification)
    f1 = f1_score(affinities_classification, preds_classification)

    print("Existed numbers of class 1: ", np.sum(affinities_classification == 1))
    print("Predicted numbers of class 1: ", np.sum(preds_classification == 1))
    print('Test Accuracy: {:.6f}  Test Sensitivity: {:.6f}  Test Specificity: {:.6f}  Test F1: {:.6f}'.format(accuracy, sensitivity, specificity, f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drug-Target Affinity Evaluation')
    parser.add_argument('--dataset_name', type=str, default='davis', help='Name of the dataset to use.')
    parser.add_argument('--model_id', type=str, default='', help='ID of the model to be loaded for testing.')
    parser.add_argument('--artifact_root', type=str, default='mlruns/1', help='Root directory for the artifacts.')

    args, unknown = parser.parse_known_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on the {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Load test dataset
    dataset_name = args.dataset_name
    test_dataset = DrugTargetAffinityDataset(root='data', dataset=f'{dataset_name}_test')
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Load the best model from MLflow
    
    
    model_uri = f"{args.artifact_root}/{args.model_id}/artifacts/model"
    
    model = mlflow.pytorch.load_model(model_uri)

    # Evaluate the model
    evaluate_model(model, test_loader, device, dataset_name)