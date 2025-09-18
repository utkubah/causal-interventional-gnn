# Causal Interventional GNN

This repository contains the official implementation for the Bachelor's Thesis by **Utku Bahçıvanoğlu** (Università Bocconi). The project introduces a **high-fidelity, causally-informed Graph Neural Network (GNN)** and evaluates it on both a **controlled synthetic SCM** and a **finance panel** built around a hypothesized microstructure DAG.

## The Core Contribution

This thesis addresses a fundamental conflict between **Structural Causal Models (SCMs)** and **standard GNN practice**:

- **SCMs** assume that each cause–effect relationship is a **distinct, modular mechanism** that can be surgically manipulated via the do-operator.
- **Typical GNNs** share a single message function across edges, collapsing distinct mechanisms into an **average correlational operator**.

This work closes that gap with two main ideas:

1. **Edge-wise Causal GNN Layer.**  
   `EdgeWiseGNNLayer` instantiates a **separate neural mechanism per directed edge**. This preserves **SCM modularity** and enables **graph surgery** (delete inbound edges + clamp assignments) to simulate interventions coherently.

2. **Two Part Causal Training.**  
   A two-stage trainer combines **observational fit** (standard predictive loss) with a **CXGNN-inspired causal regularizer** that encourages **self-consistency under do-operations**. On the synthetic dataset we additionally train with **true interventional samples** to benchmark **ATE fidelity**.

**Hypothesis.** Explicit SCM semantics (directionality, modularity, graph surgery) yield models that are not only accurate on factual data but also **behave correctly under interventions** and are **more robust** to shift.

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