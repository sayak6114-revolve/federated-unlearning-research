import copy
import time
import torch
import torch.nn as nn
from src.models import (
    AdultMLP, clone_weights, load_weights, 
    compute_update, apply_update
)

def client_local_train(model, dataloader, epochs=2, lr=1e-3, device="cpu"):
    """
    Performs standard local training on a client's private dataloader.
    Returns:
        trained_weights: dict of state_dict tensors
        update_vector: dict (trained_weights - initial_weights)
    """
    initial_weights = clone_weights(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    trained_weights = clone_weights(model)
    update_vector = compute_update(initial_weights, trained_weights)
    return trained_weights, update_vector

def evaluate_model(model, dataloader, device="cpu"):
    """Evaluates accuracy and loss on a given dataloader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return accuracy, avg_loss

def train_fedavg(
    client_loaders,
    test_loader,
    input_dim,
    num_rounds=15,
    local_epochs=2,
    lr=1e-3,
    active_client_ids=None,
    initial_weights=None,
    device="cpu",
    verbose=True
):
    """
    Simulates standard FedAvg training across specified active clients.
    Logs historical updates and global model checkpoints at every round.
    
    Returns:
        model: final trained model
        historical_updates: {round: {client_id: update_vector}}
        global_checkpoints: {round: state_dict}
        elapsed_time: total wall-clock time in seconds
    """
    if active_client_ids is None:
        active_client_ids = list(client_loaders.keys())

    # Calculate dataset sizes and aggregation weights p_k
    client_sizes = {cid: len(client_loaders[cid].dataset) for cid in active_client_ids}
    total_samples = sum(client_sizes.values())
    p_weights = {cid: client_sizes[cid] / total_samples for cid in active_client_ids}

    # Initialize model
    global_model = AdultMLP(input_dim).to(device)
    if initial_weights is not None:
        load_weights(global_model, initial_weights)
    else:
        initial_weights = clone_weights(global_model)

    historical_updates = {}
    global_checkpoints = {0: clone_weights(global_model)}
    train_history = {"test_acc": [], "test_loss": []}

    start_time = time.time()

    if verbose:
        print(f"--- Starting FedAvg: {len(active_client_ids)} active clients, {num_rounds} rounds, {local_epochs} local epochs ---")

    for round_idx in range(1, num_rounds + 1):
        round_updates = {}
        current_global_weights = clone_weights(global_model)

        # Clients train locally starting from current global checkpoint
        for cid in active_client_ids:
            client_model = AdultMLP(input_dim).to(device)
            load_weights(client_model, current_global_weights)
            _, update_vector = client_local_train(
                client_model, client_loaders[cid], epochs=local_epochs, lr=lr, device=device
            )
            round_updates[cid] = update_vector

        # Log historical client updates
        historical_updates[round_idx] = round_updates

        # Aggregate: M^(t) = M^(t-1) + sum(p_k * U_k^(t))
        aggregated_update = {k: torch.zeros_like(v) for k, v in current_global_weights.items()}
        for cid in active_client_ids:
            weight = p_weights[cid]
            for param_name in aggregated_update:
                aggregated_update[param_name] += weight * round_updates[cid][param_name]

        new_global_weights = apply_update(current_global_weights, aggregated_update)
        load_weights(global_model, new_global_weights)
        global_checkpoints[round_idx] = clone_weights(global_model)

        test_acc, test_loss = evaluate_model(global_model, test_loader, device=device)
        train_history["test_acc"].append(test_acc)
        train_history["test_loss"].append(test_loss)

        if verbose and (round_idx % 5 == 0 or round_idx == num_rounds):
            print(f"Round {round_idx:02d}/{num_rounds:02d} | Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

    elapsed_time = time.time() - start_time
    if verbose:
        final_acc = train_history["test_acc"][-1]
        print(f"FedAvg completed in {elapsed_time:.2f}s | Final Test Acc: {final_acc:.2f}%")

    return global_model, historical_updates, global_checkpoints, elapsed_time, train_history
