import torch
from src.baselines.fedavg import evaluate_model
from src.models import clone_weights, compute_model_l2_distance
from src.metrics.mia import evaluate_membership_inference

def evaluate_all_metrics(
    original_model,
    retrained_model,
    unlearned_model,
    client_loaders,
    test_loader,
    unlearned_client_id=0,
    time_retrain=None,
    time_unlearn=None,
    device="cpu"
):
    """
    Evaluates and formats comparative metrics across:
    1. Original FedAvg Model (trained with all clients)
    2. Retrained-from-scratch Model (ground truth baseline without client k_u)
    3. FedEraser Model (unlearned model)
    Includes Membership Inference Attack (MIA) privacy auditing.
    """
    retained_client_ids = [cid for cid in client_loaders if cid != unlearned_client_id]

    # Global Test Performance
    orig_test_acc, orig_test_loss = evaluate_model(original_model, test_loader, device=device)
    retrain_test_acc, retrain_test_loss = evaluate_model(retrained_model, test_loader, device=device)
    unlearn_test_acc, unlearn_test_loss = evaluate_model(unlearned_model, test_loader, device=device)

    # Performance on Forgotten Client's Data (client k_u)
    unlearned_loader = client_loaders[unlearned_client_id]
    orig_u_acc, orig_u_loss = evaluate_model(original_model, unlearned_loader, device=device)
    retrain_u_acc, retrain_u_loss = evaluate_model(retrained_model, unlearned_loader, device=device)
    unlearn_u_acc, unlearn_u_loss = evaluate_model(unlearned_model, unlearned_loader, device=device)

    # Membership Inference Attack (MIA) Auditing
    # Member Set: Client k_u training data | Non-Member Set: Held-out Test data
    mia_orig = evaluate_membership_inference(original_model, unlearned_loader, test_loader, device=device)
    mia_retrain = evaluate_membership_inference(retrained_model, unlearned_loader, test_loader, device=device)
    mia_unlearn = evaluate_membership_inference(unlearned_model, unlearned_loader, test_loader, device=device)

    # Average Performance on Retained Clients' Data
    def avg_retained_perf(m):
        accs, losses = [], []
        for cid in retained_client_ids:
            acc, loss = evaluate_model(m, client_loaders[cid], device=device)
            accs.append(acc)
            losses.append(loss)
        return sum(accs) / len(accs), sum(losses) / len(losses)

    orig_ret_acc, _ = avg_retained_perf(original_model)
    retrain_ret_acc, _ = avg_retained_perf(retrained_model)
    unlearn_ret_acc, _ = avg_retained_perf(unlearned_model)

    # L2 Parameter Distance to Retrained Model (Gold Standard)
    retrain_weights = clone_weights(retrained_model)
    dist_orig_to_retrain = compute_model_l2_distance(clone_weights(original_model), retrain_weights)
    dist_unlearn_to_retrain = compute_model_l2_distance(clone_weights(unlearned_model), retrain_weights)

    # Compute Speedup
    speedup = (time_retrain / time_unlearn) if (time_retrain and time_unlearn and time_unlearn > 0) else None

    # Print Formatted Results Table
    print("\n" + "=" * 78)
    print("                FEDERATED UNLEARNING EVALUATION REPORT")
    print("=" * 78)
    print(f"{'Metric':<38} | {'Original (All)':<12} | {'Retrained (GT)':<14} | {'FedEraser':<12}")
    print("-" * 78)
    print(f"{'Global Test Accuracy (%)':<38} | {orig_test_acc:<12.2f} | {retrain_test_acc:<14.2f} | {unlearn_test_acc:<12.2f}")
    print(f"{'Retained Clients Avg Accuracy (%)':<38} | {orig_ret_acc:<12.2f} | {retrain_ret_acc:<14.2f} | {unlearn_ret_acc:<12.2f}")
    print(f"{f'Forgotten Client {unlearned_client_id} Accuracy (%)':<38} | {orig_u_acc:<12.2f} | {retrain_u_acc:<14.2f} | {unlearn_u_acc:<12.2f}")
    print(f"{f'Forgotten Client {unlearned_client_id} Loss':<38} | {orig_u_loss:<12.4f} | {retrain_u_loss:<14.4f} | {unlearn_u_loss:<12.4f}")
    print("-" * 78)
    print(f"{'MIA Privacy Risk (ROC-AUC)':<38} | {mia_orig['mia_auc']:<12.4f} | {mia_retrain['mia_auc']:<14.4f} | {mia_unlearn['mia_auc']:<12.4f}")
    print(f"{'MIA Attacker Accuracy (%)':<38} | {mia_orig['mia_acc']:<12.2f} | {mia_retrain['mia_acc']:<14.2f} | {mia_unlearn['mia_acc']:<12.2f}")
    print("-" * 78)
    print(f"{'L2 Distance to Retrained Model':<38} | {dist_orig_to_retrain:<12.4f} | {'0.0000 (Ref)':<14} | {dist_unlearn_to_retrain:<12.4f}")
    if time_retrain and time_unlearn:
        print(f"{'Execution Time (seconds)':<38} | {'--':<12} | {time_retrain:<14.2f} | {time_unlearn:<12.2f}")
        print(f"{'Unlearning Speedup Ratio':<38} | {'--':<12} | {'1.00x':<14} | {speedup:<12.2f}x")
    print("=" * 78 + "\n")

    return {
        "orig_test_acc": orig_test_acc,
        "retrain_test_acc": retrain_test_acc,
        "unlearn_test_acc": unlearn_test_acc,
        "orig_ret_acc": orig_ret_acc,
        "retrain_ret_acc": retrain_ret_acc,
        "unlearn_ret_acc": unlearn_ret_acc,
        "orig_u_acc": orig_u_acc,
        "retrain_u_acc": retrain_u_acc,
        "unlearn_u_acc": unlearn_u_acc,
        "orig_u_loss": orig_u_loss,
        "retrain_u_loss": retrain_u_loss,
        "unlearn_u_loss": unlearn_u_loss,
        "mia_orig": mia_orig,
        "mia_retrain": mia_retrain,
        "mia_unlearn": mia_unlearn,
        "dist_orig_to_retrain": dist_orig_to_retrain,
        "dist_unlearn_to_retrain": dist_unlearn_to_retrain,
        "time_retrain": time_retrain,
        "time_unlearn": time_unlearn,
        "speedup": speedup
    }
