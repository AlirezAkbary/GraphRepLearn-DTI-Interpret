# GraphRepLearn-DTI-Interpret

## Installation

1. **Create a virtual environment using Python 3**:
   ```bash
   virtualenv -p python3 graph-dta-env
   ```
2. **Activate the virtual environment**:
   ```bash
   source graph-dta-env/bin/activate
   ```
3. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

To download and preprocess the datasets (e.g., **Davis** or **KIBA**), run:
```bash
python -m src.scripts.prepare_datasets --datasets davis kiba
```
This command will:
1. Download the raw data files.
2. Generate a folder named `data/`.
3. Create train, validation, and test splits as PyTorch datasets under `data/processed/`.

### Understanding the Data Structure

In total, we handle five key files:

- **`ligands_can.txt`**: A JSON containing drug molecules in SMILES format along with unique IDs.
- **`proteins.txt`**: A list of protein amino acid sequences.
- **`Y`**: A matrix of binding affinities, where the \((i, j)\) entry corresponds to the affinity between the *i*-th drug in `ligands_can.txt` and the *j*-th protein in `proteins.txt`.
- **`train_fold_setting1.txt`** and **`test_fold_setting1.txt`**: Specify how to split samples into train and test sets. If `Y` is \((m,n)\), it gets flattened into an \((m \times n)\)-sized vector; the indices in these files refer to positions in that flattened vector.

> **Note**: Batching graphs is more complex than batching images or sequences. We use **PyTorch Geometric (PyG)**, which merges multiple smaller graphs into a single large, disjoint graph for each batch.

## Models

`models/model.py` implements multiple Graph Neural Network (GNN) architectures:

- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network)
- **GIN** (Graph Isomorphism Network)
- **GraphSAGE**

We rely on **PyTorch Lightning** to streamline training/validation/testing. A factory function supports easy switching among different GNN types. We also provide a **Saliency** interpretability mechanism that registers a backward hook to measure feature-level gradients.

## Training + Hyperparameter Tuning + Experiment Tracking

We use **MLflow** for experiment tracking (metrics, artifacts, hyperparameters). Start the MLflow tracking server:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```
This saves experiment logs/artifacts under `./mlruns`.

Then, run hyperparameter tuning with **Optuna**:
```bash
python -m src.scripts.train --dataset_name davis --n_trials 1
```
- `dataset_name` can be `davis` or `kiba`.
- `n_trials` specifies how many Optuna trials to explore.  
The tuning logic in `src/tuning/tune.py` searches different GNN architectures, hidden dimensions, etc.

### Note about `interpretability_mode` in `src/models/model.py`

In typical interpretability methods, you compute gradients w.r.t. the input features during inference. For sequential data (like proteins), an embedding layer is non-differentiable, so you capture gradients w.r.t. the embedding output. During evaluation (`model.eval()`), autograd is usually off to speed up inference, but for interpretability you must:

- Ensure you are **not** in `torch.no_grad()`.
- Enable `interpretability_mode=True` in your model so that the protein embeddings are leaf nodes (`requires_grad=True`), allowing backpropagation to flow into them.
- Call `model.eval()` (to deactivate dropout/batch norm training),
  

## Evaluating on the Test Set

Once you pick the best model via MLflow experiments, evaluate it by specifying the correct `model_id`:
```bash
python -m src.scripts.evaluate --dataset_name davis --model_id <MODEL_ID>
```
Replace `<MODEL_ID>` with the actual ID from MLflow (e.g., `3af64b7e7e454e1eafa99325082bbe04`).

## Interpretation

### Training a Classifier Head

For interpretability experiments, we typically use a classifier instead of a regressor. Hence, we:

1. Load the trained regression model.
2. Replace its final regression head with a binary classification head.
3. Freeze all other layers.
4. Retrain only the new classification head using a thresholded version of `Y` (e.g., `> 7.0` for Davis yields class=1, otherwise class=0).

Example:
```bash
python -m src.scripts.train_classifier --dataset_name davis --model_id <MODEL_ID>
```
This saves a classifier model (frozen GNN + new classification head) into `interp_models/`.

### Interpretability Methods

We compare **Vanilla Saliency** vs. **Guided Backpropagation**:

1. Identify the most important features for both drug and protein embeddings (via gradient magnitudes).
2. Zero out those features and measure how many predictions flip.
3. The method that causes more flips has higher explanatory power.

Example usage:
```bash
python -m src.interpretability.interpretability_methods \
  --model_checkpoint interp_models/best_davis.ckpt \
  --dataset_name davis \
  --batch_size 512 \
  --threshold 7.0
```
Where:
- `model_checkpoint` points to your `.ckpt` file from the classification step,
- `threshold` binarizes affinities (e.g., >7.0 => positive class).

## Contributing

We welcome contributions via issues or pull requests. Potential improvements:

- **Upsampling / Class Balancing**: The dataset is imbalanced. Investigate class weighting or SMOTE-like techniques.
- **Additional Interpretability Methods**: Benchmark new approaches (Integrated Gradients, DeepLIFT, etc.) against Saliency and Guided Backprop.
- **End-to-End Interpretable Models**: Instead of post-hoc interpretability, explore strategies that incorporate interpretability into the training process itself.
