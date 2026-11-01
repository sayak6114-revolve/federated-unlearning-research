import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_sample_losses_and_confidences(model, dataloader, device="cpu"):
    """
    Computes per-sample cross-entropy loss and prediction confidence (max softmax probability)
    for a given dataset.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = []
    confidences = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            probs = torch.softmax(logits, dim=1)
            max_prob, _ = torch.max(probs, dim=1)

            losses.extend(loss.cpu().numpy())
            confidences.extend(max_prob.cpu().numpy())

    return np.array(losses), np.array(confidences)

def evaluate_membership_inference(model, member_loader, non_member_loader, device="cpu"):
    """
    Performs a Loss-Based Membership Inference Attack (MIA).
    
    Members: Samples that were part of Client 0's private training data (label = 1).
    Non-Members: Samples from held-out test data never seen during training (label = 0).
    
    Attack Logic:
        Because neural networks fit training data, member samples typically have
        lower loss than non-member samples. The attacker predicts membership based on
        negative loss (-loss): the lower the loss, the more likely it's a member.
        
    Evaluation:
        - ROC-AUC: 
            * 1.0 = Perfect privacy leak (attacker perfectly detects all member records).
            * 0.5 = Random guessing / Perfect privacy protection (attacker is blind).
        - Attack Accuracy: The accuracy of the attacker at the optimal threshold.
    """
    member_losses, _ = compute_sample_losses_and_confidences(model, member_loader, device)
    non_member_losses, _ = compute_sample_losses_and_confidences(model, non_member_loader, device)

    # Ensure balanced sizes between members and non-members for fair evaluation
    min_size = min(len(member_losses), len(non_member_losses))
    member_losses = member_losses[:min_size]
    non_member_losses = non_member_losses[:min_size]

    # Ground truth: 1 for members (Client 0 data), 0 for non-members (test data)
    y_true = np.concatenate([np.ones(min_size), np.zeros(min_size)])

    # Attack score: lower loss indicates membership, so score = -loss
    scores = np.concatenate([-member_losses, -non_member_losses])

    # Compute ROC-AUC
    mia_auc = roc_auc_score(y_true, scores)

    # Compute Best Decision Threshold Accuracy
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    preds = (scores >= optimal_threshold).astype(int)
    mia_acc = (preds == y_true).mean() * 100.0

    return {
        "mia_auc": mia_auc,
        "mia_acc": mia_acc,
        "fpr": fpr,
        "tpr": tpr
    }
