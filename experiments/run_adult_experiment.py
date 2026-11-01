import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows OpenMP fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from src.data_adult import partition_adult_federated
from src.models import AdultMLP, clone_weights
from src.baselines.fedavg import train_fedavg
from src.unlearning.federaser import run_federaser
from src.metrics.evaluate import evaluate_all_metrics
from src.metrics.plots import generate_paper_plots

def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    print("=================================================================")
    print("      FEDERATED UNLEARNING EXPERIMENT ON ADULT CENSUS DATASET    ")
    print("=================================================================\n")

    # 1. Load and partition Adult dataset across 10 clients
    num_clients = 10
    client_loaders, _, test_loader, input_dim = partition_adult_federated(
        num_clients=num_clients, batch_size=64, random_state=seed
    )

    # Global initial model weights (M^(0)) shared across all runs for strict fairness
    base_model = AdultMLP(input_dim).to(device)
    initial_weights = clone_weights(base_model)

    # Hyperparameters
    num_rounds = 10
    local_epochs = 2
    calibration_epochs = 1
    lr = 1e-3
    unlearned_client_id = 0

    # -------------------------------------------------------------
    # Step 1: Train Original FedAvg Model (Clients 0 to 9)
    # -------------------------------------------------------------
    print("\n>>> STEP 1: Training Original FedAvg Model (All 10 Clients)...")
    orig_model, historical_updates, _, time_orig, orig_history = train_fedavg(
        client_loaders=client_loaders,
        test_loader=test_loader,
        input_dim=input_dim,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
        active_client_ids=list(range(num_clients)),
        initial_weights=initial_weights,
        device=device,
        verbose=True
    )

    # -------------------------------------------------------------
    # Step 2: Retrain From Scratch (Clients 1 to 9 - Ground Truth Baseline)
    # -------------------------------------------------------------
    retained_clients = [cid for cid in range(num_clients) if cid != unlearned_client_id]
    print(f"\n>>> STEP 2: Retraining From Scratch without Client {unlearned_client_id} (Gold Standard Baseline)...")
    retrained_model, _, _, time_retrain, retrain_history = train_fedavg(
        client_loaders=client_loaders,
        test_loader=test_loader,
        input_dim=input_dim,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
        active_client_ids=retained_clients,
        initial_weights=initial_weights,
        device=device,
        verbose=True
    )

    # -------------------------------------------------------------
    # Step 3: Run FedEraser Unlearning for Client 0
    # -------------------------------------------------------------
    print(f"\n>>> STEP 3: Running FedEraser Unlearning for Client {unlearned_client_id}...")
    unlearned_model, time_unlearn, unlearn_history = run_federaser(
        initial_weights=initial_weights,
        historical_updates=historical_updates,
        client_loaders=client_loaders,
        test_loader=test_loader,
        input_dim=input_dim,
        unlearned_client_id=unlearned_client_id,
        calibration_epochs=calibration_epochs,
        lr=lr,
        device=device,
        verbose=True
    )

    # -------------------------------------------------------------
    # Step 4: Evaluate & Compare All Metrics
    # -------------------------------------------------------------
    print("\n>>> STEP 4: Computing Evaluation Metrics & Speedup...")
    summary_metrics = evaluate_all_metrics(
        original_model=orig_model,
        retrained_model=retrained_model,
        unlearned_model=unlearned_model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        unlearned_client_id=unlearned_client_id,
        time_retrain=time_retrain,
        time_unlearn=time_unlearn,
        device=device
    )

    # -------------------------------------------------------------
    # Step 5: Generate Publication-Quality Plots
    # -------------------------------------------------------------
    print(">>> STEP 5: Generating FedEraser Evaluation Plots...")
    generate_paper_plots(
        orig_history=orig_history,
        retrain_history=retrain_history,
        unlearn_history=unlearn_history,
        summary_metrics=summary_metrics,
        output_dir="results",
        filename="federaser_evaluation_plots.png"
    )

if __name__ == "__main__":
    main()
