from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
import pytorch_lightning as pl


# Factory function to get the correct graph convolution layer
def get_graph_conv_layer(arch: str, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
    if arch == 'GCN':
        return GCNConv(in_channels, out_channels, **kwargs)
    elif arch == 'GAT':
        # Adjust kwargs as necessary for GAT (like heads) if needed
        return GATConv(in_channels, out_channels, **kwargs)
    elif arch == 'GraphSAGE':
        return SAGEConv(in_channels, out_channels, **kwargs)
    elif arch == 'GIN':
        # For GIN, using a simple MLP as the neural network for convolution
        return GINConv(
            Sequential(
                Linear(in_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels)
            )
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

class MultiArchGraphModel(pl.LightningModule):
    def __init__(
        self,
        arch: str = 'GIN',  # Default architecture
        num_molecular_features: int = 78,
        num_protein_features: int = 25,
        num_filters: int = 32,
        embedding_dim: int = 128,
        hidden_dim: int = 32,
        output_dim: int = 128,
        num_graph_layers: int = 5,
        dropout_rate: float = 0.2,
        sequence_length: int = 1000,
        learning_rate: float = 1e-3,
        num_classes: int = 2,
        task: str = 'classification',  # 'classification' or 'regression'
        interpretability_mode: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        # Architecture selection
        self.arch = arch

        self.sequence_length = sequence_length
        self.interpretability_mode = interpretability_mode

        # Interpretability placeholders
        self.interpret_drug_gradients = None
        self.interpret_embedded_target_gradients = None
        self.interpret_conv_drug_activation = None
        self.interpret_conv_target_activation = None

        self.task = task
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize graph layers and batch norms
        self.graph_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First graph layer
        self.graph_layers.append(
            get_graph_conv_layer(self.arch, num_molecular_features, hidden_dim)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Additional graph layers
        for _ in range(num_graph_layers - 1):
            self.graph_layers.append(
                get_graph_conv_layer(self.arch, hidden_dim, hidden_dim)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Fully connected layer for graph branch
        self.fc_g = Linear(hidden_dim, output_dim)

        # Protein sequence branch
        self.embedding_xt = nn.Embedding(num_protein_features + 1, embedding_dim)
        self.embedding_xt.weight.requires_grad = True
        self.conv_xt = nn.Conv1d(embedding_dim, num_filters, kernel_size=8)
        conv_output_length = sequence_length - 8 + 1  # based on kernel_size
        self.fc_xt = nn.Linear(num_filters * conv_output_length, output_dim)

        # Fully connected layers for combined features
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)

        # Output layer and loss function
        if self.task == 'classification':
            self.output_layer = nn.Linear(256, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.task == 'regression':
            self.output_layer = nn.Linear(256, 1)
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")

        self.learning_rate = learning_rate
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Any) -> torch.Tensor:
        # Molecular graph branch
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x.requires_grad = True
        x.register_hook(lambda grad: setattr(self, 'interpret_drug_gradients', grad))

        for conv, bn in zip(self.graph_layers, self.batch_norms):
            x = conv(x, edge_index)
            x = self.relu(x)
            x = bn(x)

        self.interpret_conv_drug_activation = x
        x = global_add_pool(x, batch)
        x = self.relu(self.fc_g(x))
        x = self.dropout(x)

        # Protein sequence branch
        target = data.target 
        target = target.view(-1, self.sequence_length).long()  # Ensure target is reshaped correctly
        
        
        self.embedding_xt.weight.requires_grad = True  # Make sure it's trainable
        embedded_xt = self.embedding_xt(target)

        
        if self.interpretability_mode and not self.training:
            # We'll cast to leaf and enable grad
            embedded_xt = self.embedding_xt(target).detach().requires_grad_(True)
            embedded_xt.register_hook(lambda grad: setattr(self, 'interpret_embedded_target_gradients', grad))
        else:
            embedded_xt = self.embedding_xt(target)
        
        
        embedded_xt = embedded_xt.permute(0, 2, 1)  # Change shape for Conv1d: (batch, channels, sequence_length)
        conv_xt = self.conv_xt(embedded_xt)
        self.interpret_conv_target_activation = conv_xt

        conv_xt = self.relu(conv_xt)
        conv_xt = conv_xt.view(conv_xt.size(0), -1)
        conv_xt = self.relu(self.fc_xt(conv_xt))
        conv_xt = self.dropout(conv_xt)

        # Combine features and pass through fully connected layers
        combined = torch.cat((x, conv_xt), dim=1)
        combined = self.dropout(self.relu(self.fc1(combined)))
        combined = self.dropout(self.relu(self.fc2(combined)))

        return self.output_layer(combined)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        outputs = self(batch)
        labels = batch.y

        if self.task == 'classification':
            # Binarize labels
            labels = torch.where(labels > self.threshold, torch.tensor(1, device=labels.device), torch.tensor(0, device=labels.device))
            loss = self.loss_fn(outputs, labels.long())
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean()
            self.log(f'{stage}_acc', acc, on_step=(stage=='train'), on_epoch=True, prog_bar=True)
        else:
            loss = self.loss_fn(outputs.squeeze(), labels.float())
        
        self.log(f'{stage}_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'val')

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Optionally, add learning rate schedulers here
        return optimizer
