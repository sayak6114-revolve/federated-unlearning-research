import copy
import time
import torch
from src.models import (
    AdultMLP, clone_weights, load_weights, 
    compute_update, apply_update, compute_l2_norm
)
from src.baselines.fedavg import client_local_train, evaluate_model

def run_federaser(
    initial_weights,
    historical_updates,
    client_loaders,
    test_loader,
    input_dim,
    unlearned_client_id=0,
    calibration_epochs=1,
    lr=1e-3,
    device="cpu",
    verbose=True
):
    """
    Implements the FedEraser unlearning algorithm (Bao et al., IWQoS 2021).
    
    Args:
        initial_weights: state_dict at round 0 (M^(0))
        historical_updates: {round: {client_id: update_vector}} from original FedAvg
        client_loaders: {client_id: DataLoader}
        test_loader: DataLoader for test evaluation
        input_dim: feature dimension for AdultMLP
        unlearned_client_id: the client to remove from the model
        calibration_epochs: number of local calibration epochs (default 1)
        lr: learning rate for calibration
        device: "cpu" or "cuda"
        
    Returns:
        unlearned_model: AdultMLP model purged of target client's data
        elapsed_time: total unlearning time in seconds
    """
    all_rounds = sorted(historical_updates.keys())
    num_rounds = len(all_rounds)

    # Identify retained clients C = K \ {k_u}
    retained_client_ids = [cid for cid in client_loaders.keys() if cid != unlearned_client_id]

    # Re-normalize aggregation weights q_k for retained clients
    retained_sizes = {cid: len(client_loaders[cid].dataset) for cid in retained_client_ids}
    total_retained_samples = sum(retained_sizes.values())
    q_weights = {cid: retained_sizes[cid] / total_retained_samples for cid in retained_client_ids}

    # Initialize unlearned model at round 0
    unlearned_model = AdultMLP(input_dim).to(device)
    load_weights(unlearned_model, initial_weights)
    unlearn_history = {
        "train_acc": [], "train_loss": [], 
        "test_acc": [], "test_loss": [], 
        "cumulative_time": []
    }

    if verbose:
        print(f"\n=======================================================")
        print(f"--- Running FedEraser: Removing Client {unlearned_client_id} ---")
        print(f"Retained clients: {retained_client_ids} | Calibration epochs: {calibration_epochs}")
        print(f"=======================================================")

    start_time = time.time()

    for round_idx in all_rounds:
        current_unlearned_weights = clone_weights(unlearned_model)
        calibrated_updates = {}

        for cid in retained_client_ids:
            # Step A: Client-side calibration training (1 epoch on current unlearned model)
            cali_model = AdultMLP(input_dim).to(device)
            load_weights(cali_model, current_unlearned_weights)
            _, fresh_update = client_local_train(
                cali_model, client_loaders[cid], epochs=calibration_epochs, lr=lr, device=device
            )

            # Step B: Retrieve cached historical update
            historical_u = historical_updates[round_idx][cid]

            # Step C: Compute norms and scale historical update
            # e~ = (||U_fresh|| / ||U_hist||) * U_hist
            norm_fresh = compute_l2_norm(fresh_update)
            norm_hist = compute_l2_norm(historical_u)

            if norm_hist > 1e-8:
                scale_ratio = norm_fresh / norm_hist
            else:
                scale_ratio = 1.0

            calibrated_vector = {
                param_name: scale_ratio * historical_u[param_name]
                for param_name in historical_u
            }
            calibrated_updates[cid] = calibrated_vector

        # Step D: Calibrated update aggregation: U_e = sum(q_k * e~_k)
        aggregated_update = {k: torch.zeros_like(v) for k, v in current_unlearned_weights.items()}
        for cid in retained_client_ids:
            weight = q_weights[cid]
            for param_name in aggregated_update:
                aggregated_update[param_name] += weight * calibrated_updates[cid][param_name]

        # Step E: Advance unlearned model trajectory: M_f^(t) = M_f^(t-1) + U_e
        new_unlearned_weights = apply_update(current_unlearned_weights, aggregated_update)
        load_weights(unlearned_model, new_unlearned_weights)

        # Measure training metrics across retained clients
        train_accs, train_losses = [], []
        for cid in retained_client_ids:
            c_acc, c_loss = evaluate_model(unlearned_model, client_loaders[cid], device=device)
            train_accs.append(c_acc)
            train_losses.append(c_loss)
        avg_train_acc = sum(train_accs) / len(train_accs)
        avg_train_loss = sum(train_losses) / len(train_losses)

        test_acc, test_loss = evaluate_model(unlearned_model, test_loader, device=device)
        unlearn_history["train_acc"].append(avg_train_acc)
        unlearn_history["train_loss"].append(avg_train_loss)
        unlearn_history["test_acc"].append(test_acc)
        unlearn_history["test_loss"].append(test_loss)
        unlearn_history["cumulative_time"].append(time.time() - start_time)

        if verbose and (round_idx % 5 == 0 or round_idx == num_rounds):
            print(f"Calibration Round {round_idx:02d}/{num_rounds:02d} | Train Acc: {avg_train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

    elapsed_time = time.time() - start_time
    if verbose:
        final_acc = unlearn_history["test_acc"][-1]
        print(f"FedEraser finished in {elapsed_time:.2f}s | Final Test Acc: {final_acc:.2f}%\n")

    return unlearned_model, elapsed_time, unlearn_history
