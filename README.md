# Causal Interventional GNN

This repository contains the official implementation for the Bachelor's Thesis by Utku Bahçıvanoğlu, Bocconi University. The project introduces a high-fidelity, causally-informed Graph Neural Network (GNN) and demonstrates its superior robustness in the domain of financial factor investing.

## The Core Contribution

This thesis addresses a fundamental conflict between the theory of Structural Causal Models (SCMs) and the practice of GNNs:
-   **SCMs** assume that each cause-effect relationship is a distinct, modular mechanism.
-   **Standard GNNs** violate this by sharing the same message-passing parameters across all edges, learning an average, correlational effect.

This work resolves this gap by introducing two key contributions:
1.  **A High-Fidelity GNN Architecture:** The `EdgeWiseGNNLayer` is a novel GNN layer that instantiates a unique, learnable neural network for each edge in the causal graph. This directly implements the SCM principle of modular mechanisms, creating a more faithful causal analogue.
2.  **A Robust Causal Training Algorithm:** The `HybridCausalTrainer` learns interventional distributions from purely observational data. It combines a standard predictive loss with a novel **causal regularizer**. This regularizer forces the model's internal mechanisms to be self-consistent, pushing it to learn true causal dependencies instead of spurious correlations.

The central hypothesis is that this causally-rigorous design leads to a model that is more robust and generalizes better to out-of-distribution events like real-world market shocks.

## Project Structure

```
causal-interventional-gnn/
├── configs/              # YAML configuration files for experiments
├── data/                 # Placeholder for datasets
├── notebooks/            # Jupyter notebooks for prototyping, preprocessing, and analysis
├── outputs/              # Default directory for saved models and results
├── src/                  # Main source code (data loaders, models, trainers)
└── run_experiment.py     # Main script to run experiments
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/causal-interventional-gnn.git
    cd causal-interventional-gnn
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

All experiments are run via the `run_experiment.py` script and are configured by `.yaml` files in the `configs/` directory.

1.  **Prepare your data:** Place your processed data files in the `data/` directory.
2.  **Configure your experiment:** Create or edit a `.yaml` file in `configs/` to specify the data paths, model hyperparameters, and training settings.
3.  **Run the experiment:**
    ```bash
    python run_experiment.py --config configs/base_config.yaml
    ```
    Results and model checkpoints will be saved to the `outputs/` directory.

## Citing this Work

If you find this work useful in your research, please consider citing the repository:

```bibtex
@misc{Bahcivanoglu2025CausalGNN,
  author = {Bahçıvanoğlu, Utku},
  title = {Integrating Causal Inference in Graph Neural Networks: Theoretical Foundations and Empirical Evaluation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/utkubah/causal-interventional-gnn}}
}
```