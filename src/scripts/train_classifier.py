import argparse
import os

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

from src.datasets.drug_target_dataset import DrugTargetAffinityDataset
from src.models.model import MultiArchGraphModel


class ClassificationModule(MultiArchGraphModel):
    """
    Inherits from MultiArchGraphModel but:
    1) Freezes all layers except the classification head.
    2) Uses a threshold (for logging or test separation if needed).
    """

    def __init__(self, threshold=7.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

        # Freeze entire model EXCEPT the output_layer
        for param in self.parameters():
            param.requires_grad = False
        for param in self.output_layer.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate
        )


def make_dataloaders(dataset_name, train_batch_size, val_batch_size):
    """Utility to create the train/val/test DataLoaders."""
    train_dataset = DrugTargetAffinityDataset(root='data', dataset=f'{dataset_name}_train')
    val_dataset = DrugTargetAffinityDataset(root='data', dataset=f'{dataset_name}_validation')
    test_dataset = DrugTargetAffinityDataset(root='data', dataset=f'{dataset_name}_test')

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Drug-Target Affinity Classification via PyTorch Lightning')
    parser.add_argument('--dataset_name', type=str, default='davis', help='Name of the dataset (davis/kiba).')
    parser.add_argument('--model_id', type=str, default='',
                        help='ID of the regression model to load from MLflow.')
    parser.add_argument('--artifact_root', type=str, default='mlruns/1',
                        help='Root directory for MLflow artifacts.')
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='Number of epochs for classification training.')
    parser.add_argument('--train_batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=512, help='Batch size for validation and test.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for classification model.')
    parser.add_argument('--save_path', type=str, default='interp_models/',
                        help='Directory to save the best Lightning checkpoint.')
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.save_path, exist_ok=True)

    # -----------------------
    # 1) Prepare DataLoaders
    # -----------------------
    train_loader, val_loader, test_loader = make_dataloaders(
        dataset_name=args.dataset_name,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size
    )

    # -----------------------
    # 2) Load the old Regression Model from MLflow
    # -----------------------
    model_uri = f"{args.artifact_root}/{args.model_id}/artifacts/model"
    regression_model = mlflow.pytorch.load_model(model_uri)

    # We'll retrieve the architecture hyperparams from the regression model
    reg_hparams = dict(regression_model.hparams)  # a dictionary of hyperparams

    # Overwrite these hyperparams for classification task
    reg_hparams["task"] = 'classification'
    reg_hparams["num_classes"] = 2
    reg_hparams["learning_rate"] = args.learning_rate  # classification LR

    # The threshold we use to binarize labels (davis=7.0, kiba=12.1 etc.)
    threshold = 7.0 if args.dataset_name == 'davis' else 12.1

    # -----------------------
    # 3) Create the Classification Model
    #    Freeze everything except the classification head
    # -----------------------
    class_model = ClassificationModule(
        threshold=threshold,
        **reg_hparams
    )

    # Transfer weights from regression model -> classification model (except final layer)
    pretrained_dict = {
        k: v for k, v in regression_model.state_dict().items()
        if 'output_layer' not in k
    }
    my_dict = class_model.state_dict()
    my_dict.update(pretrained_dict)
    class_model.load_state_dict(my_dict)

    # -----------------------
    # 4) Set up PyTorch Lightning Trainer
    # -----------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_path,
        filename=f"best_{args.dataset_name}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True  # if you only want the model weights, or False if you want optimizer state
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=args.save_path,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=20
    )

    # -----------------------
    # 5) Train / Validate
    # -----------------------
    trainer.fit(class_model, train_loader, val_loader)

    # -----------------------
    # 6) Test using the best checkpoint
    # -----------------------
    print("\n=== TESTING ===")
    best_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_path}")
    best_model = ClassificationModule.load_from_checkpoint(best_path)
    trainer.test(best_model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
