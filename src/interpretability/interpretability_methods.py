import argparse
import os
from copy import deepcopy
from typing import Tuple, Type

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.datasets.drug_target_dataset import DrugTargetAffinityDataset
from src.scripts.train_classifier import ClassificationModule  # <-- or wherever your Lightning class is defined


class Saliency:
    """
    Abstract base class for saliency methods.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model (nn.Module): A trained classification model in eval mode,
                               with interpretability_mode=True so it captures
                               interpret_drug_gradients & interpret_embedded_target_gradients.
        """
        self.model = model
        self.model.eval()  # We need eval mode for interpretability (no dropout)

    def generate_saliency(
        self,
        data_item: Data,
        target_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (drug_grads, target_grads).
        Subclasses must implement the gradient approach.
        """
        raise NotImplementedError("Subclasses must implement generate_saliency().")


class SaliencyMap(Saliency):
    """
    Vanilla Saliency: straight gradient w.r.t the input features.
    """

    def generate_saliency(self, data_item: Data, target_class: int):
        sample = deepcopy(data_item).to(next(self.model.parameters()).device)

        # Make sure x tracks gradients
        sample.x.requires_grad = True

        # Zero out any old gradient
        self.model.zero_grad()

        # Forward
        output = self.model(sample)  # shape: [1, num_classes], e.g. [1, 2]
        grad_outputs = torch.zeros_like(output)
        grad_outputs[:, target_class] = 1.0

        # Backward pass
        output.backward(gradient=grad_outputs, retain_graph=True)
        # print("model.interpret_drug_gradients:", self.model.interpret_drug_gradients, self.model.interpret_drug_gradients.grad)


        # Extract the stored gradients
        drug_grads = None
        target_grads = None

        if self.model.interpret_drug_gradients is not None:
            drug_grads = self.model.interpret_drug_gradients
        if self.model.interpret_embedded_target_gradients is not None:
            target_grads = self.model.interpret_embedded_target_gradients

        return drug_grads, target_grads


class GuidedBackProp(Saliency):
    """
    Guided Backprop: modifies ReLU layers so that negative gradients are clipped.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.hooks = []

    def guided_relu_hook(self, module, grad_in, grad_out):
        """
        Clamps negative gradients to 0 (guiding the backprop through positive paths only).
        """
        return (torch.clamp(grad_in[0], min=0.0),)

    def register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(self.guided_relu_hook)
                self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def generate_saliency(self, data_item: Data, target_class: int):
        sample = deepcopy(data_item).to(next(self.model.parameters()).device)
        sample.x.requires_grad = True

        self.register_hooks()
        self.model.zero_grad()

        output = self.model(sample)
        grad_outputs = torch.zeros_like(output)
        grad_outputs[:, target_class] = 1.0

        output.backward(gradient=grad_outputs, retain_graph=True)

        drug_grads = None
        target_grads = None

        if self.model.interpret_drug_gradients is not None:
            drug_grads = self.model.interpret_drug_gradients
        if self.model.interpret_embedded_target_gradients is not None:
            target_grads = self.model.interpret_embedded_target_gradients

        self.remove_hooks()

        return drug_grads, target_grads


def evaluate_saliency_method(
    model: nn.Module,
    method_class: Type[Saliency],
    test_dataset: DrugTargetAffinityDataset,
    sample_indices: np.ndarray,
    device: torch.device,
    label_of_interest: int,
    subset_name: str = "TP"
):
    """
    Evaluates a saliency method on a specified subset of test samples
    (e.g., TPs or TNs).

    Steps:
      1) Generate (drug_grads, target_grads).
      2) Find "significant" features (>= mean+std) [or use abs grads if you prefer].
      3) Zero them out and see if the prediction flips from label_of_interest -> opposite_label.
      4) Count flips, plot histogram of how many features get zeroed out.
    
    Args:
        model (nn.Module): The classification model (with interpretability hooks).
        method_class (Type[Saliency]): SaliencyMap or GuidedBackProp.
        test_dataset (DrugTargetAffinityDataset): The test dataset.
        sample_indices (np.ndarray): Indices of the subset to interpret (TP or TN).
        device (torch.device): CPU/GPU.
        label_of_interest (int): 1 if we want to see flips from 1->0, or 0 if we want flips from 0->1.
        subset_name (str): "TP" or "TN" (used in the plot filenames).
    """
    opposite_label = 1 - label_of_interest

    saliency_method = method_class(model)
    n_changed_drug = 0
    n_changed_target = 0

    sign_drug_dist = []
    sign_target_dist = []

    for idx in sample_indices:
        data_item = test_dataset[idx]

        # Generate Saliency
        drug_grads, target_grads = saliency_method.generate_saliency(data_item, target_class=label_of_interest)
        if drug_grads is None or target_grads is None:
            continue

        # You may want to use absolute gradient magnitude:
        # drug_features = drug_grads.abs().max(dim=1)[0]
        # target_features = target_grads.abs().max(dim=2)[0].squeeze(0)

        # Or if you prefer just the max in the positive direction:
        drug_features = drug_grads.abs().max(dim=1)[0]
        target_features = target_grads.abs().max(dim=2)[0].squeeze(0)

        drug_thr = drug_features.mean() + drug_features.std()
        target_thr = target_features.mean() + target_features.std()

        sig_drug_idxs = (drug_features >= drug_thr).nonzero().flatten()
        sig_target_idxs = (target_features >= target_thr).nonzero().flatten()

        sign_drug_dist.append(len(sig_drug_idxs))
        sign_target_dist.append(len(sig_target_idxs))

        # Evaluate flips in predictions
        # 1) Zero out protein
        data_modified_target = deepcopy(data_item).to(device)
        if len(sig_target_idxs) > 0:
            if data_modified_target.target.dim() == 1:
                data_modified_target.target = data_modified_target.target.unsqueeze(0)
            data_modified_target.target[0, sig_target_idxs] = 0
        out_target_mod = model(data_modified_target)
        pred_label_target = torch.argmax(out_target_mod, dim=1).item()
        if pred_label_target == opposite_label:
            n_changed_target += 1

        # 2) Zero out drug features
        data_modified_drug = deepcopy(data_item).to(device)
        if len(sig_drug_idxs) > 0:
            data_modified_drug.x[sig_drug_idxs, :] = 0
        out_drug_mod = model(data_modified_drug)
        pred_label_drug = torch.argmax(out_drug_mod, dim=1).item()
        if pred_label_drug == opposite_label:
            n_changed_drug += 1

    # Plot histograms
    plt.hist(sign_target_dist, bins=20, edgecolor='black')
    plt.title(f"{method_class.__name__} - Target Feature Dist ({subset_name})")
    plt.xlabel("Num Significant Target Features")
    plt.ylabel("Frequency")
    plt.savefig(f"{method_class.__name__}_target_{subset_name}.png")
    plt.close()

    plt.hist(sign_drug_dist, bins=20, edgecolor='black')
    plt.title(f"{method_class.__name__} - Drug Feature Dist ({subset_name})")
    plt.xlabel("Num Significant Drug Features")
    plt.ylabel("Frequency")
    plt.savefig(f"{method_class.__name__}_drug_{subset_name}.png")
    plt.close()

    print("=" * 60)
    print(f"Subset: {subset_name} | Saliency Method: {method_class.__name__}")
    print(f"Number of flips by altering PROTEIN: {n_changed_target}")
    print(f"Number of flips by altering DRUG: {n_changed_drug}")
    print(f"Sample sign_target_dist: {sign_target_dist[:10]} ...")
    print(f"Sample sign_drug_dist: {sign_drug_dist[:10]} ...")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the Lightning .ckpt file from ModelCheckpoint.")
    parser.add_argument("--dataset_name", type=str, default="davis",
                        help="Dataset name (davis/kiba).")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for generating predictions.")
    parser.add_argument("--threshold", type=float, default=7.0,
                        help="Affinity threshold: > threshold => class=1, else class=0.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load the test dataset
    test_dataset = DrugTargetAffinityDataset(root="data", dataset=f"{args.dataset_name}_test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2) Load the classification model from the Lightning checkpoint
    model = ClassificationModule.load_from_checkpoint(args.model_checkpoint)
    model.to(device)
    model.eval()
    # Enable interpretability
    model.interpretability_mode = True

    # 3) Identify TPs and TNs in the test set
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)  # shape: [batch_size, 2]
            all_preds.append(outputs.cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()  # [N, 2]
    all_labels = torch.cat(all_labels, dim=0).numpy()  # [N]

    pred_classes = np.argmax(all_preds, axis=1)              # 0 or 1
    true_classes = np.where(all_labels > args.threshold, 1, 0)  # 0 or 1

    # TPs: predicted=1 & true=1
    tp_indices = np.where((pred_classes == 1) & (true_classes == 1))[0]
    # TNs: predicted=0 & true=0
    tn_indices = np.where((pred_classes == 0) & (true_classes == 0))[0]

    print(f"Total test samples: {len(all_labels)}")
    print(f"Positive ground-truth: {np.sum(true_classes == 1)}")
    print(f"Predicted positives: {np.sum(pred_classes == 1)}")
    print(f"TRUE POSITIVES: {len(tp_indices)}")
    print(f"TRUE NEGATIVES: {len(tn_indices)}")

    # 4) Evaluate Saliency Methods for TPs => flipping from 1->0
    evaluate_saliency_method(
        model, SaliencyMap, test_dataset,
        sample_indices=tp_indices,
        device=device,
        label_of_interest=1,
        subset_name="TP"
    )
    evaluate_saliency_method(
        model, GuidedBackProp, test_dataset,
        sample_indices=tp_indices,
        device=device,
        label_of_interest=1,
        subset_name="TP"
    )

    # 5) Evaluate Saliency Methods for TNs => flipping from 0->1
    evaluate_saliency_method(
        model, SaliencyMap, test_dataset,
        sample_indices=tn_indices,
        device=device,
        label_of_interest=0,
        subset_name="TN"
    )
    evaluate_saliency_method(
        model, GuidedBackProp, test_dataset,
        sample_indices=tn_indices,
        device=device,
        label_of_interest=0,
        subset_name="TN"
    )


if __name__ == "__main__":
    main()
