# Federated Unlearning Research

> Research notes, paper reviews, implementations, and evaluation of federated unlearning methods.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Focus](https://img.shields.io/badge/Focus-Federated%20Unlearning-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-green.svg)]()

---

## 1. Overview & Motivation

In Federated Learning (FL), multiple decentralized clients collaboratively train a shared model without directly exposing private raw training data. However, when a client withdraws (e.g., under GDPR's **"Right to be Forgotten"**, CCPA, or if flagged as a poisoned/byzantine participant), standard protocols face a fundamental dilemma:

> **The Hospital Dilemma:** Suppose 10 hospitals collaboratively train a diagnostic model. If Hospital 5 leaves the federation, deleting Hospital 5's data locally does **not** erase the historical influence embedded in the shared global model's weights.
> 
> 1. *How can Hospital 5's contribution be removed without retraining from scratch?*
> 2. *How do we empirically and verifiably prove the unlearning succeeded?*

- **Naive Baseline (Retrain-from-Scratch):** Discard the model and re-train from round $t=0$ on $\{C \setminus C_5\}$. While this represents the gold standard for privacy, it requires all remaining clients to be online and wastes massive computational and network resources.
- **Federated Unlearning (FU) Goal:** Update the global parameters $w \to w_{\text{unlearned}}$ to eradicate the influence of the target client while preserving model utility on retained clients at a fraction of retraining cost.

---

## 2. Unlearning Trilemma & Evaluation Metrics

A successful federated unlearning method must balance three competing objectives:

```
                      The Unlearning Trilemma
                 
                      Completeness / Privacy
                      (MIA AUC -> 50%, Backdoor ASR -> 0%)
                                 /\
                                /  \
                               /    \
                              /      \
                             /________\
               Model Utility            Efficiency
    (Retained Acc, L2 Distance)      (Speedup, Comm. Rounds)
```

### A. Completeness & Privacy Metrics (Did it actually forget?)
- **Membership Inference Attack (MIA) AUC / Accuracy:** An auditor or attacker trains shadow models to determine if samples from the target client were used in training. Successful unlearning reduces MIA accuracy to random chance ($\approx 50\%$ or $\text{AUC} \approx 0.5$).
- **Backdoor Attack Success Rate (ASR):** If the removed client injected a trigger/watermark, ASR on that trigger must drop to $\approx 0\%$ after unlearning.
- **Forgotten Client Loss & Accuracy:** Error rate on the forgotten client's private test distribution should match that of a model that has never seen it.

### B. Utility Metrics (Is the model still useful?)
- **Retained Clients' Test Accuracy:** Accuracy across remaining clients $\{C \setminus C_i\}$ must remain intact ($\le 1\text{--}2\%$ drop).
- **Weight Distance ($L_2$ Norm):** $\|w_{\text{unlearned}} - w_{\text{retrained}}\|_2 \to 0$.
- **Representation Distance (CKA / KL-Divergence):** Verification that internal representations on retained data mirror the retrained reference.

### C. Efficiency Metrics (Is it practical?)
- **Speedup Ratio:** $\frac{\text{Time}(\text{Retrain from scratch})}{\text{Time}(\text{Unlearn})}$.
- **Communication Rounds:** Communication rounds required relative to full retraining.

---

## 3. Core Mechanisms for Client Removal

```
                    Federated Unlearning Methods
                                 |
    +----------------------------+----------------------------+
    |                            |                            |
1. Update Calibration      2. Gradient Inversion        3. Subspace Projection
   (Historical Rollback)      (Gradient Ascent)            (Null-Space Erasure)
   e.g., FedEraser            e.g., FedAU                  e.g., Projected Gradients
```

1. **Update Calibration (FedEraser):** The server logs historical client update vectors $\Delta w_k^{(t)}$. When unlearning is triggered, the server subtracts target client vectors from historical aggregations and runs a small number of calibration rounds with retained clients to correct trajectory drift.
2. **Gradient Ascent / Reverse Learning:** Executes bounded gradient ascent on the forgotten client's data or proxy distribution to invert learned weights, followed by fine-tuning on retained data.
3. **Subspace & Null-Space Projection:** Projects global updates onto the orthogonal complement of the target client's gradient subspace, preventing weight updates along client-sensitive directions.

---

## 4. Reading Roadmap & Literature Reviews

| Paper | Key Contribution | Notes Link |
| :--- | :--- | :--- |
| **Federated Unlearning: A Survey** (Romandini et al., 2024) | Comprehensive taxonomy, design guidelines, and evaluation metrics. | [papers/00_survey_romandini_2024.md](papers/00_survey_romandini_2024.md) |
| **FedEraser** (Bao et al., IWQoS 2021) | Pioneer paper on client-level removal via cached update calibration. | [papers/01_federaser_2021.md](papers/01_federaser_2021.md) |
| **VeriFi** (Gao et al., INFOCOM 2022) | Verifiable federated unlearning using cryptographic & statistical proofs. | [papers/template.md](papers/template.md) |

---

## 5. Repository Structure

```text
federated-unlearning-research/
├── README.md               # Main project roadmap, problem formulation & metrics
├── .gitignore              # Ignores weights, datasets, checkpoints, logs
├── concepts/               # Mathematical design and core formulations
│   └── federaser_mathematics.md # Mathematical breakdown of FedEraser
├── papers/                 # Literature reviews and reading notes
│   ├── template.md         # Reusable reading note template
│   ├── 00_survey_romandini_2024.md
│   └── 01_federaser_2021.md
└── src/                    # Python package for reproduction & experiments
    ├── baselines/          # FedAvg & Retrain-from-scratch benchmark
    ├── unlearning/         # FedEraser, Gradient Ascent algorithms
    └── metrics/            # MIA auditing, Backdoor ASR, L2 distance
```

---

## 6. Research Checklist & Progress

- [x] Formulate core research objectives and threat models.
- [x] Design evaluation protocol (MIA, ASR, Retained Accuracy, Speedup).
- [x] Initialize structured repository and paper note workflows.
- [ ] Complete study of Romandini et al. Survey (2024).
- [ ] Complete deep-dive into FedEraser algorithm and calibration equations.
- [ ] Implement simulated 10-client FedAvg baseline on toy dataset (MNIST / CIFAR-10).
- [ ] Implement Retrain-from-Scratch reference baseline.
- [ ] Implement FedEraser unlearning algorithm and benchmark speedup vs. retraining.
- [ ] Implement MIA auditor to measure forgetting privacy.
