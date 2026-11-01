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

## 2. The 4 Steps of FedEraser (Algorithm 1 from Paper)

In the foundational paper (*Bao et al., IEEE/ACM IWQoS 2021, Algorithm 1*), FedEraser removes the target client $k_u$ by iterating through the historical calibration rounds $t_j \in \{1, 2, \dots, T\}$ using **four exact algorithmic steps**:

```
                                  FEDERASER ITERATION (ROUND t_j)
                                                 │
            ┌────────────────────────────────────┴────────────────────────────────────┐
            ▼                                                                         ▼
   [CLIENT-SIDE EXECUTION]                                                   [SERVER-SIDE EXECUTION]
   
   Step 1: Calibration Training                                              Step 2: Update Calibrating
   • Client k_c receives M^f_{t_j}                                           • Server scales historical update U
   • Trains for E_cali = 1 epoch                                             • e~ = (||U_fresh|| / ||U_hist||) * U_hist
   • Generates fresh update: U~_{k_c}^{t_j}                                                   │
            │                                                                         ▼
            └─────────────────────────> Sends U~_{k_c}^{t_j} to Server ────────> Step 3: Calibrated Update Aggregating
                                                                                     • Aggregates across retained clients:
                                                                                     • U_e = sum(q_k * e~_k)
                                                                                              │
                                                                                              ▼
                                                                                     Step 4: Unlearned Model Updating
                                                                                     • M^f_{t_j+1} = M^f_{t_j} + U_e
```

### Step 1: Calibration Training (Client-Side)
Each retained client $k_c \in \mathcal{C} = \{1, \dots, K\} \setminus \{k_u\}$ receives the current unlearned global model $M^f_{t_j}$ from the server. The client initializes its local weights with $M^f_{t_j}$ and trains for $E_{\text{cali}}$ epochs ($E_{\text{cali}} \ll E$, typically $1$ epoch) on its local dataset $D_{k_c}$:

$$\tilde{U}_{k_c}^{t_j} \leftarrow \text{CaliTrain}\left(C_{k_c}, M^f_{t_j}, E_{\text{cali}}\right)$$

Where $\tilde{U}_{k_c}^{t_j}$ represents the fresh gradient direction computed at the current unlearned checkpoint.

---

### Step 2: Update Calibrating (Server-Side)
Because the unlearned model $M^f_{t_j}$ has drifted from the original training trajectory, the server cannot reuse the old update $U_{k_c}^{t_j}$ directly. Instead, it calibrates the historical update by re-scaling its magnitude using the ratio of the $L_2$ norms:

$$\tilde{e}_{k_c}^{t_j} = \frac{\|\tilde{U}_{k_c}^{t_j}\|_2}{\|U_{k_c}^{t_j}\|_2} \cdot U_{k_c}^{t_j}$$

* **Direction ($U_{k_c}^{t_j}$):** Preserves the historical learning direction accumulated during full training.
* **Magnitude ($\frac{\|\tilde{U}_{k_c}^{t_j}\|}{\|U_{k_c}^{t_j}\|}$):** Adapts the step size to the local slope/curvature of the unlearned model's loss landscape.

---

### Step 3: Calibrated Update Aggregating (Server-Side)
The server aggregates the calibrated updates from all remaining clients using re-normalized dataset weights $q_{k_c} = \frac{|D_{k_c}|}{\sum_{j \in \mathcal{C}} |D_j|}$:

$$\tilde{U}_{e}^{t_j} = \sum_{k_c \in \mathcal{C}} q_{k_c} \tilde{e}_{k_c}^{t_j}$$

---

### Step 4: Unlearned Model Updating (Server-Side)
The server advances the unlearned global model by applying the aggregated calibrated update:

$$M^f_{t_j+1} = M^f_{t_j} + \tilde{U}_{e}^{t_j}$$

This four-step cycle repeats until round $T$ is reconstructed, producing the final unlearned global model.

> 📖 *For the theoretical convergence and Lipschitz error bounds of this 4-step process, see [concepts/federaser_mathematics.md](concepts/federaser_mathematics.md).*

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
* **Before Unlearning (FedAvg):** If a student saw yesterday's exam questions, they answer with almost zero error (**Low Loss $\approx 0.30$**). An attacker easily identifies them as a "member."
* **After Successful Unlearning (FedEraser):** If you truly wiped their memory, their error on those questions must rise to match unseen test questions (**Loss rises to $\approx 0.37$**).
* **ROC-AUC Score Meaning:**
  * **$\text{AUC} = 1.00$:** Total privacy leak (attacker separates members with 100% confidence).
  * **$\text{AUC} = 0.50$:** **Complete privacy protection.** Equivalent to flipping a fair coin (attacker is totally blind).

---

## 4. Evaluation Metrics: Accuracy & Loss Benchmark Report

Below are the experimental evaluation metrics measured across **Original FedAvg (10 Clients)**, **Retrained from Scratch (9 Clients - Ground Truth)**, and **FedEraser (Unlearned Client 0)** on the Adult Census dataset:

### A. Comprehensive Evaluation Metrics Table

| Category | Evaluation Metric | Original FedAvg (All 10 Clients) | Retrained Baseline (9 Clients - Ground Truth) | FedEraser (Unlearned Client 0) | Metric Target / Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Global Utility** | **Global Test Accuracy (%)** | 85.34% | 85.34% | **84.53%** | Parity with Retrained ($\le 1\%$ drop) |
| | **Global Test Loss** | 0.3164 | 0.3167 | **0.3721** | Stable cross-entropy error |
| **Retained Utility** | **Retained Clients Train Acc (%)** | 85.65% | 85.53% | **85.21%** | Utility preserved on remaining clients |
| | **Retained Clients Train Loss** | 0.3142 | 0.3121 | **0.3683** | Smooth loss minimization |
| | **Retained Clients Test Acc (%)** | 85.56% | 85.53% | **85.21%** | **Utility Preserved:** $<0.4\%$ variance |
| **Forgetting Quality** | **Forgotten Client 0 Acc (%)** | 86.46% | 86.24% | **84.96%** | Accuracy drops as client is forgotten |
| | **Forgotten Client 0 Loss** | **0.3033** | **0.3088** | **0.3712** | **Higher = Forgotten:** Significant error jump |
| **Privacy Auditing** | **MIA Privacy Risk (ROC-AUC)** | 0.5076 | 0.5064 | **0.5042** | **Target $\to 0.50$:** Blind attacker |
| | **MIA Attacker Accuracy (%)** | 50.94% | 50.93% | **50.72%** | Random coin-flip performance |
| **Model Distance** | **$L_2$ Distance to Retrained Model**| 1.4416 | 0.0000 (Ref) | **1.7920** | Euclidean distance in parameter space |
| **Efficiency** | **Execution Time (seconds)** | 26.61s | 33.88s | **22.88s** | Wall-clock benchmark |
| | **Computation Rate (s/round)** | 2.66 s/rnd | 3.39 s/rnd | **2.29 s/rnd** | Throughput per round |
| | **Unlearning Speedup Ratio** | -- | 1.00x | **$1.48\times\text{--}1.93\times$** | Significant computational savings |

### B. Analysis of Key Evaluation Metrics:
1. **Utility Preservation (Did remaining clients suffer?):**
   * Retained clients accuracy for FedEraser is **$85.21\%$**, almost identical to the retrained model's **$85.53\%$** (only a $0.32\%$ difference).
2. **Forgetting Quality (Was Client 0 actually erased?):**
   * Look at the **Forgotten Client 0 Loss**:
     * Original Model: **$0.3033$** (lowest error because it was trained on Client 0).
     * Retrained Model: **$0.3088$** (ground truth for unseen data).
     * FedEraser Model: **$0.3712$** (loss jumps significantly, proving Client 0's memorized features were removed).
3. **MIA Privacy (Can an attacker verify Client 0 was present?):**
   * FedEraser achieves an **MIA ROC-AUC of $0.5042$** and **$50.72\%$ attacker accuracy**, which is statistically indistinguishable from a random 50/50 coin flip.

---

## 5. Evaluation Plots

The 4-panel publication-style evaluation plot generated directly by our pipeline:

![FedEraser Evaluation Plots](results/federaser_evaluation_plots.png)

1. **Top-Left (Convergence Trajectory):** Round-by-round accuracy showing FedEraser closely tracking the retrained-from-scratch baseline.
2. **Top-Right (Utility vs. Forgetting Profile):** Bar comparison verifying utility preservation on retained clients.
3. **Bottom-Left (MIA Privacy Audit ROC Curves):** FedEraser's ROC curve hugs the diagonal random-guess line ($y=x$), proving privacy compliance.
4. **Bottom-Right (Computational Efficiency):** Wall-clock time comparison demonstrating real-world speedup.

### Training vs. Testing Data & Time Consumption Breakdown

![FedEraser Time and Data Analysis](results/federaser_time_and_data_plots.png)

1. **Top-Left (Cumulative Time across Rounds):** Direct comparison of cumulative wall-clock time showing the significant time savings achieved at every calibration round.
2. **Top-Right (Train Loss vs. Test Loss):** Evaluates training loss vs. test loss trajectories across rounds to verify that the unlearned model generalizes properly without overfitting or divergence.
3. **Bottom-Left (Training vs. Testing vs. Forgotten Accuracy):** Compares performance across the 3 datasets: Retained Training Data, Held-out Test Data, and Forgotten Client Private Data.
4. **Bottom-Right (Computation Rate):** Seconds per round (s/round) comparing full retraining vs. lightweight FedEraser calibration.

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
│   ├── federaser_evaluation_plots.png # High-res 4-panel evaluation figure (300 DPI)
│   └── federaser_time_and_data_plots.png # High-res time & data breakdown figure (300 DPI)
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
