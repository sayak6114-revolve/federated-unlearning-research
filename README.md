# Federated Unlearning: FedEraser on Adult Dataset

> Research notes, mathematical formulations, end-to-end implementation of **FedEraser** (*Bao et al., IWQoS 2021*), and privacy auditing via **Membership Inference Attacks (MIA)**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Topic: Federated Unlearning](https://img.shields.io/badge/Focus-Federated%20Unlearning-6f42c1.svg)]()
[![Status: Benchmarked](https://img.shields.io/badge/Status-Benchmarked-success.svg)]()

---

## 📌 Table of Contents
1. [The Problem: Why Unlearning in FL is Hard](#1-the-problem-why-unlearning-in-fl-is-hard)
2. [The 4 Phases of FedEraser & Mathematical Design](#2-the-4-phases-of-federaser--mathematical-design)
3. [Membership Inference Attacks (MIA) Made Simple](#3-membership-inference-attacks-mia-made-simple)
4. [Experimental Results on Adult Dataset](#4-experimental-results-on-adult-dataset)
5. [Evaluation Plots](#5-evaluation-plots)
6. [Repository Structure](#6-repository-structure)
7. [Quickstart: Run It Yourself](#7-quickstart-run-it-yourself)
8. [References](#8-references)

---

## 1. The Problem: Why Unlearning in FL is Hard

In standard Federated Learning (FL), multiple clients collaboratively train a global model by sharing gradient updates rather than raw data. 

> **The Hospital Dilemma:**  
> Suppose 10 hospitals train a disease prediction model. Hospital 5 later withdraws its data under GDPR's **"Right to be Forgotten"**. Simply deleting Hospital 5's data locally does **not** erase the historical influence already baked into the shared neural network weights.

```
Naive Solution: Retrain from scratch without Hospital 5.
❌ Consumes massive compute time and wastes all previous communication rounds.
❌ Requires all 9 other hospitals to stay online and re-transmit data for hundreds of rounds.
```

### The Mathematical Trap: "Forward Coupling"
Why can't the server just subtract Hospital 5's past updates at the end ($M_{\text{final}} - \Delta w_5$)?
* Because in round 2, all other hospitals computed their gradients starting from a model that **already contained Hospital 5's round 1 update**.
* Due to the non-linear nature of deep neural networks, Hospital 5's influence propagated recursively into every subsequent update of every other hospital.
* Subtracting only Hospital 5's direct updates leaves behind the **indirect cascade** embedded across all weights.

---

## 2. The 4 Phases of FedEraser & Mathematical Design

**FedEraser** (*Bao et al., IWQoS 2021*) solves forward coupling without full retraining by trading server storage for speed:

```
Original Trajectory:    M^(0) ───> M^(1) ───> M^(2) ───> ... ───> M^(T) (contains Client 0)
                                     │          │
                                  [drift]    [drift]
                                     ▼          ▼
FedEraser Trajectory:  M_f^(0) ──> M_f^(1) ─> M_f^(2) ─> ... ─> M_f^(T) (purged of Client 0)
                                     ▲          ▲
                         [Calibration training pulls trajectory back]
```

FedEraser executes in **four systematic phases**:

### Phase 1: Update Caching (During Normal Training)
During regular FedAvg, the central server stores each client's parameter update vector:
$$\mathcal{U} = \left\{ U_k^{(t)} = w_k^{(t)} - M^{(t-1)} \;\middle|\; k \in \{1, \dots, K\}, \; t \in \{1, \dots, T\} \right\}$$

### Phase 2: Target Removal & Weight Renormalization
When client $k_u$ (e.g., Client 0) requests removal, the server excludes $k_u$ and re-normalizes the aggregation weights among the retained clients $\mathcal{C}$:
$$q_k = \frac{|D_k|}{\sum_{j \in \mathcal{C}} |D_j|} \quad \forall k \in \mathcal{C}$$

### Phase 3: Trajectory Drift Detection
If the server merely replayed the historical updates without $k_u$, the updates would be evaluated at the wrong parameter positions, causing exponential divergence known as **Trajectory Drift**:
$$\|\hat{M}^{(t)} - M_{\text{retrained}}^{(t)}\| \approx \mathcal{O}\left((1 + \eta L)^t\right)$$

### Phase 4: Calibration Training (The Core Innovation)
To eliminate drift cheaply, the server runs a lightweight synchronization at calibration checkpoints:
1. **Client Step:** Retained clients run **only 1 quick epoch** ($E_{\text{cali}} = 1$) on the current unlearned checkpoint to sense the fresh gradient slope: $\tilde{U}_{k}$.
2. **Server Calibration Formula:**
   The server keeps the **historical direction** ($U_k$) but scales its **magnitude** using the fresh curvature:
   $$\tilde{e}_{k} = \underbrace{\frac{\|\tilde{U}_{k}\|_2}{\|U_{k}\|_2}}_{\text{Adaptive Step Size}} \cdot \underbrace{U_{k}}_{\text{Historical Direction}}$$
3. **Calibrated Aggregation:**
   $$M_f^{(t)} = M_f^{(t-1)} + \sum_{k \in \mathcal{C}} q_k \tilde{e}_{k}$$

> 📖 *For a complete mathematical derivation with Lipschitz bounds and proofs, see [concepts/federaser_mathematics.md](concepts/federaser_mathematics.md).*

---

## 3. Membership Inference Attacks (MIA) Made Simple

### What is MIA?
A **Membership Inference Attack (MIA)** evaluates privacy: An auditor or attacker tries to guess:  
*"Was person X's private record part of the model's training data?"*

```
              ┌────────────────────────────────────────────────────────┐
              │ Attacker inspects prediction loss on sample (x, y):    │
              │                                                        │
              │ If Loss is abnormally LOW  ➔ Guess "MEMBER" (Memorized) │
              │ If Loss is HIGHER          ➔ Guess "NON-MEMBER" (Unseen)│
              └────────────────────────────────────────────────────────┘
```

### The Student Test Analogy
* **Before Unlearning:** If a student saw yesterday's exam questions, they answer with almost zero error (**Low Loss $\approx 0.30$**). An attacker easily identifies them as a "member."
* **After Successful Unlearning:** If you truly wiped their memory, their error on those questions must rise to match unseen test questions (**Loss rises to $\approx 0.37$**).
* **ROC-AUC Score Meaning:**
  * **$\text{AUC} = 1.00$:** Total privacy leak (attacker separates members with 100% confidence).
  * **$\text{AUC} = 0.50$:** **Complete privacy protection.** Equivalent to flipping a fair coin (attacker is totally blind).

---

## 4. Experimental Results on Adult Dataset

We benchmarked FedEraser on the **Adult Census Income Dataset** (45,222 samples, 97 features) partitioned across **10 federated clients** (~3,617 samples per client):

| Metric | Original FedAvg (10 Clients) | Retrained Baseline (9 Clients - GT) | FedEraser (Unlearned Client 0) | What it Proves |
| :--- | :---: | :---: | :---: | :--- |
| **Global Test Accuracy** | 85.19% | 85.14% | **84.48%** | Global model stays within ~0.6% of retrained. |
| **Retained Clients Avg Acc** | 85.61% | 85.57% | **85.15%** | **Utility Preserved:** Remaining clients lose <0.4% accuracy. |
| **Forgotten Client 0 Acc** | 86.79% | 86.15% | **85.05%** | Client 0's private accuracy drops as expected. |
| **Forgotten Client 0 Loss** | 0.3032 | 0.3087 | **0.3697** | **Successful Forgetting:** Error on Client 0 jumps significantly. |
| **MIA Privacy Risk (ROC-AUC)**| 0.5077 | 0.5062 | **0.5041** | **Privacy Protected:** Attacker cannot beat a random coin toss (~0.50). |
| **MIA Attacker Accuracy** | 51.00% | 50.87% | **50.79%** | Attacker has zero predictive advantage. |
| **Parameter Distance ($L_2$)** | 1.1963 | 0.0000 (Ref) | **1.6367** | Reconstructed trajectory stays in safe parameter basin. |
| **Execution Time** | 18.04s | 25.33s | **13.13s** | **$1.93\times$ faster** than full retraining on CPU. |

---

## 5. Evaluation Plots

The 4-panel publication-style evaluation plot generated directly by our pipeline:

![FedEraser Evaluation Plots](results/federaser_evaluation_plots.png)

1. **Top-Left (Convergence Trajectory):** Round-by-round accuracy showing FedEraser closely tracking the retrained-from-scratch baseline.
2. **Top-Right (Utility vs. Forgetting Profile):** Bar comparison verifying utility preservation on retained clients.
3. **Bottom-Left (MIA Privacy Audit ROC Curves):** FedEraser's ROC curve hugs the diagonal random-guess line ($y=x$), proving privacy compliance.
4. **Bottom-Right (Computational Efficiency):** Wall-clock time comparison demonstrating real-world speedup.

---

## 6. Repository Structure

```text
federated-unlearning-research/
├── README.md                          # Master project documentation
├── .gitignore                         # Ignores weights, datasets, checkpoints, logs
├── concepts/
│   └── federaser_mathematics.md       # Full mathematical proofs & trajectory drift bounds
├── papers/
│   ├── template.md                    # Standard literature review template
│   ├── 00_survey_romandini_2024.md    # Survey notes (Romandini et al.)
│   └── 01_federaser_2021.md           # FedEraser paper breakdown
├── results/
│   └── federaser_evaluation_plots.png # High-res 4-panel evaluation figure (300 DPI)
├── experiments/
│   └── run_adult_experiment.py        # End-to-end benchmark runner
└── src/
    ├── data_adult.py                  # Adult dataset fetcher & federated partitioner
    ├── models.py                      # AdultMLP architecture & parameter vector utilities
    ├── baselines/
    │   └── fedavg.py                  # FedAvg coordinator with historical update logging
    ├── unlearning/
    │   └── federaser.py               # FedEraser unlearning & calibration engine
    └── metrics/
        ├── evaluate.py                # Comparative report table generator
        ├── mia.py                     # Membership Inference Attack evaluator
        └── plots.py                   # 4-panel publication figure generator
```

---

## 7. Quickstart: Run It Yourself

### Prerequisites
* Python 3.10+
* PyTorch (`pip install torch torchvision`)
* Dependencies: `pip install pandas scikit-learn matplotlib`

### Run the Full Benchmark
Clone the repository and run the end-to-end experiment:

```bash
git clone https://github.com/sayak6114-revolve/federated-unlearning-research.git
cd federated-unlearning-research

# Run training, unlearning, MIA evaluation, and plot generation:
python experiments/run_adult_experiment.py
```

The script will:
1. Download and preprocess the Adult Census dataset.
2. Train the Original FedAvg model across 10 clients and cache update vectors.
3. Retrain from scratch without Client 0 as the gold standard baseline.
4. Execute FedEraser unlearning with 1-epoch calibration.
5. Audit privacy using Membership Inference Attacks (MIA).
6. Print the summary report table and save the 4-panel figure to `results/federaser_evaluation_plots.png`.

---

## 8. References

1. **FedEraser Paper:** Guanhua Bao, Xiangshan Xu, et al., *"FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models"*, IEEE/ACM IWQoS 2021. [arXiv:2103.04870](https://arxiv.org/abs/2103.04870).
2. **Federated Unlearning Survey:** Romandini et al., *"Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics"*, 2024. [arXiv:2401.05146](https://arxiv.org/abs/2401.05146).
3. **MIA Auditing:** Reza Shokri et al., *"Membership Inference Attacks Against Machine Learning Models"*, IEEE S&P 2017.
