import os
import matplotlib.pyplot as plt
import numpy as np

def generate_paper_plots(
    orig_history,
    retrain_history,
    unlearn_history,
    summary_metrics,
    output_dir="results",
    filename="federaser_evaluation_plots.png"
):
    """
    Generates a 4-panel publication-quality plot matching the FedEraser paper style:
    1. Global Test Accuracy Trajectory across Rounds
    2. Utility vs. Forgetting Performance (Retained vs. Forgotten client)
    3. Parameter Distance (L2 norm to Retrained reference)
    4. Wall-Clock Execution Time & Speedup Comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FedEraser Federated Unlearning Evaluation on Adult Dataset", fontsize=16, fontweight='bold', y=0.98)

    # -------------------------------------------------------------
    # Subplot 1: Convergence Trajectory (Accuracy across Rounds)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    rounds = range(1, len(orig_history["test_acc"]) + 1)
    ax1.plot(rounds, orig_history["test_acc"], 'o-', color='#1f77b4', label="Original FedAvg (10 Clients)", linewidth=2, markersize=5)
    ax1.plot(rounds, retrain_history["test_acc"], 's--', color='#2ca02c', label="Retrained from Scratch (9 Clients)", linewidth=2, markersize=5)
    ax1.plot(rounds, unlearn_history["test_acc"], '^-.', color='#d62728', label="FedEraser (Unlearned Client 0)", linewidth=2.5, markersize=6)
    ax1.set_title("1. Accuracy Trajectory across Rounds", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Global Round / Calibration Round", fontsize=11)
    ax1.set_ylabel("Global Test Accuracy (%)", fontsize=11)
    ax1.set_ylim(bottom=min(min(orig_history["test_acc"]), min(unlearn_history["test_acc"])) - 1, top=88)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # -------------------------------------------------------------
    # Subplot 2: Utility Preservation vs Forgetting
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    categories = ["Global Test Acc", "Retained Clients Acc", "Forgotten Client 0 Acc"]
    x = np.arange(len(categories))
    width = 0.26

    vals_orig = [summary_metrics["orig_test_acc"], summary_metrics["orig_ret_acc"], summary_metrics["orig_u_acc"]]
    vals_retrain = [summary_metrics["retrain_test_acc"], summary_metrics["retrain_ret_acc"], summary_metrics["retrain_u_acc"]]
    vals_unlearn = [summary_metrics["unlearn_test_acc"], summary_metrics["unlearn_ret_acc"], summary_metrics["unlearn_u_acc"]]

    rects1 = ax2.bar(x - width, vals_orig, width, label="Original", color='#1f77b4', alpha=0.9)
    rects2 = ax2.bar(x, vals_retrain, width, label="Retrained (GT)", color='#2ca02c', alpha=0.9)
    rects3 = ax2.bar(x + width, vals_unlearn, width, label="FedEraser", color='#d62728', alpha=0.9)

    ax2.set_title("2. Utility vs. Forgetting Comparison", fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(75, 90)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Add value labels on top of bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # -------------------------------------------------------------
    # Subplot 3: Membership Inference Attack (MIA) ROC Curves
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    mia_orig = summary_metrics["mia_orig"]
    mia_retrain = summary_metrics["mia_retrain"]
    mia_unlearn = summary_metrics["mia_unlearn"]

    ax3.plot(mia_orig["fpr"], mia_orig["tpr"], color='#1f77b4', linewidth=2,
             label=f"Original (AUC = {mia_orig['mia_auc']:.3f})")
    ax3.plot(mia_retrain["fpr"], mia_retrain["tpr"], color='#2ca02c', linewidth=2, linestyle='--',
             label=f"Retrained Ref (AUC = {mia_retrain['mia_auc']:.3f})")
    ax3.plot(mia_unlearn["fpr"], mia_unlearn["tpr"], color='#d62728', linewidth=2.5, linestyle='-.',
             label=f"FedEraser (AUC = {mia_unlearn['mia_auc']:.3f})")
    ax3.plot([0, 1], [0, 1], 'k:', linewidth=1.5, label="Random Guess (AUC = 0.500)")

    ax3.set_title("3. Membership Inference Attack (MIA) Privacy Audit", fontsize=13, fontweight='bold')
    ax3.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax3.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.legend(loc="lower right", frameon=True, fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # -------------------------------------------------------------
    # Subplot 4: Computational Efficiency & Speedup
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    time_categories = ["Retrain from Scratch\n(Gold Standard)", "FedEraser\n(Update Calibration)"]
    times = [summary_metrics["time_retrain"], summary_metrics["time_unlearn"]]
    time_colors = ['#2ca02c', '#d62728']

    bars_t = ax4.bar(time_categories, times, color=time_colors, width=0.45, alpha=0.85)
    ax4.set_title(f"4. Wall-Clock Efficiency ({summary_metrics['speedup']:.2f}x Speedup)", fontsize=13, fontweight='bold')
    ax4.set_ylabel("Execution Time (seconds)", fontsize=11)
    ax4.set_ylim(0, max(times) * 1.3)
    ax4.grid(True, linestyle='--', alpha=0.6)

    for bar in bars_t:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"\n[Visual Plots Saved]: {filepath}")
    return filepath
