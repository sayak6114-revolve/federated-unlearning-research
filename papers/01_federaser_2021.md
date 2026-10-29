# FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models

- **Authors:** Guanhua Bao, Xiangshan Xu, Shi-Qi Shen, et al.
- **Venue & Year:** IEEE/ACM IWQoS 2021
- **Paper Link:** [https://arxiv.org/abs/2103.04870](https://arxiv.org/abs/2103.04870)
- **Granularity:** Client-Level Unlearning

---

## 1. Summary in 3 Bullets
- First paper to formalize **client-level** unlearning in Federated Learning without requiring clients to share raw data.
- Avoids retraining from scratch by leveraging **cached historical parameter updates** stored by the central server during regular training.
- Reconstructs unlearned parameter update trajectories and performs **calibration training rounds** with retained clients to recover high utility.

---

## 2. Core Problem & Motivation
- **The Gap:** When a client withdraws its data or requests deletion under regulations like GDPR, retraining the FL model from round $0$ without that client is the only naive exact baseline. However, retraining consumes enormous wall-clock time and requires all remaining clients to be active and re-transmit data.
- **Goal:** Approximate the parameters of a model retrained from scratch while drastically reducing training time and communication overhead.

---

## 3. Threat Model & Setting
- **Unlearning Request:** A client $C_i$ requests data removal.
- **Server Knowledge:** The server caches client update vectors $\Delta w_k^{(t)}$ sent during past rounds $t \in [1, T]$.
- **Data Privacy:** The server does *not* need access to raw client data.

---

## 4. Proposed Algorithm: How FedEraser Works

FedEraser operates in four key phases:

1. **Historical Update Logging:** During normal FedAvg training, the server caches the weight updates $\Delta w_k^{(t)} = w_k^{(t)} - w^{(t-1)}$ received from each client $k$ at round $t$.
2. **Target Update Subtraction:** When client $C_i$ requests unlearning, the server removes $C_i$'s contribution from the aggregated historical updates:
   $$\Delta w_{-i}^{(t)} = \sum_{k \neq i} \frac{|D_k|}{\sum_{j \neq i} |D_j|} \Delta w_k^{(t)}$$
3. **Unlearned Model Reconstruction:** The server reconstructs an approximate historical global model trajectory $\tilde{w}^{(t)}$ by iteratively applying $\Delta w_{-i}^{(t)}$.
4. **Calibration Training:** Because removing client updates creates cumulative drift from the true retraining trajectory, the server selects a small subset of rounds and asks retained clients to compute a small number of calibration steps to realign the model trajectory.

---

## 5. Evaluation & Metrics Used

- **Completeness / Forgetting:**
  - **Backdoor Attack Success Rate (ASR):** Evaluated against a scenario where the unlearned client was a backdoor attacker. FedEraser eradicated the backdoor effect (ASR dropped to ~0%).
- **Model Utility:**
  - **Retained Test Accuracy:** Unlearned model accuracy was within $\approx 1\%$ of the retrained-from-scratch model.
  - **$L_2$ Parameter Distance:** Measured $\|w_{\text{FedEraser}} - w_{\text{Retrain}}\|_2$ to verify proximity in weight space.
- **Efficiency:**
  - **Speedup:** Achieved up to $4\times$ to $10\times$ speedup in wall-clock time compared to retraining from scratch.
  - **Communication Overhead:** Reduced the required communication rounds by up to $80\%$.

---

## 6. Critical Analysis & Open Questions
- **Strengths:** Intuitive, clear baseline, preserves client data privacy, significantly faster than full retraining.
- **Limitations:** Storing historical updates $\Delta w_k^{(t)}$ for all rounds across many clients introduces server storage overhead; performance can degrade in highly Non-IID heterogeneous settings.
