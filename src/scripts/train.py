import argparse

import torch
import optuna
import mlflow

# Set MLflow tracking URI and experiment
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "drug-target-affinity-exp"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.pytorch.autolog()

from src.tuning.tune import objective
from src.datasets.drug_target_dataset import DrugTargetAffinityDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drug-Target Affinity Prediction Pipeline')
    parser.add_argument('--dataset_name', type=str, default='davis', help='Name of the dataset to use.')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of Optuna trials for hyperparameter optimization.')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of epochs to train a model.')

    args, unknown = parser.parse_known_args()

    dataset_name = args.dataset_name

    # Load datasets

    train_dataset = DrugTargetAffinityDataset(root='data', dataset=f'{dataset_name}_train')
    val_dataset = DrugTargetAffinityDataset(root='data', dataset=f'{dataset_name}_validation')
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define fixed model parameters
    model_params = {
        'num_molecular_features': 78,
        'num_protein_features': 25,
        'num_filters': 32,
        'embedding_dim': 128,
        'sequence_length': 1000,
        'dropout_rate': 0.2
    }

    # Create and run the Optuna study
    study = optuna.create_study(direction='minimize')

    study.optimize(
        lambda trial: objective(
            trial, model_params, train_dataset, val_dataset, args.n_epochs 
        ), n_trials=args.n_trials
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")\

