import optuna
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch_geometric.loader import DataLoader

from src.models.model import MultiArchGraphModel
from src.utils.metrics import mse, ci

def objective(trial, model_params: dict, train_dataset, val_dataset, n_epochs=50):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        dataset (str): Dataset name ('davis' or 'kiba').
        model_params (dict): Additional model parameters.

    Returns:
        float: Validation loss.
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64])
    num_graph_layers = trial.suggest_int('num_graph_layers', 2, 6)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    arch = trial.suggest_categorical('arch', ['GIN', 'GCN', 'GAT', 'GraphSAGE'])


    # Update model parameters with suggested hyperparameters
    model_params_updated = model_params.copy()
    model_params_updated.update({
        'hidden_dim': hidden_dim,
        'num_graph_layers': num_graph_layers,
    })

    # Initialize MLflow logger for Lightning
    mlf_logger = MLFlowLogger(
        tracking_uri=mlflow.get_tracking_uri(),
        experiment_name=mlflow.get_experiment_by_name("drug-target-affinity-exp").name
    )

    # Create the model
    model = MultiArchGraphModel(
        arch=arch,
        learning_rate=learning_rate,
        task='regression',  # Ensure task is set to regression
        **model_params_updated
    )

    # Create DataLoaders with the suggested batch_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Setup PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        logger=mlf_logger,
        enable_checkpointing=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                verbose=True
            )
        ],
        log_every_n_steps=20
    )

    # Start MLflow run
    with mlflow.start_run():
        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Retrieve the best validation loss
        best_val_loss = trainer.callback_metrics.get('val_loss').item()

        # Log hyperparameters
        mlflow.log_params({
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
            'num_graph_layers': num_graph_layers,
            'batch_size': batch_size,
            'arch': arch
        })
        mlflow.log_metric('best_val_loss', best_val_loss)

    return best_val_loss
