# Mathematical Design and Formulation of FedEraser

This document provides a rigorous breakdown of the mathematical framework behind **FedEraser** (*Bao et al., IEEE/ACM IWQoS 2021*), explaining why standard unlearning fails in Federated Learning (FL), the mathematical formulation of trajectory drift, and how the calibration mechanism reconstructs an unlearned model.

---

## 1. Background: Standard Federated Learning (FedAvg)

Let $K$ denote the total number of clients participating in the federated network. Each client $k \in \{1, \dots, K\}$ possesses a private local dataset $D_k$ with sample size $n_k = |D_k|$. The total dataset size across all clients is:

$$N = \sum_{k=1}^K n_k$$

The relative aggregation weight of client $k$ in standard FedAvg is defined as:

$$p_k = \frac{n_k}{N} = \frac{|D_k|}{\sum_{j=1}^K |D_j|}$$

### Global Training Iterations

At global round $t \in \{1, 2, \dots, T\}$:

1. **Broadcast:** The central server broadcasts the current global parameter vector $M^{(t-1)}$ to selected clients.
2. **Local Update:** Each client $k$ initializes its local weights $w_k^{(t, 0)} = M^{(t-1)}$ and performs $E$ epochs of stochastic gradient descent (SGD) on local loss $\mathcal{L}_k$:
   $$w_k^{(t, \tau)} = w_k^{(t, \tau-1)} - \eta \nabla \mathcal{L}_k\left(w_k^{(t, \tau-1)}; \xi\right)$$
   where $\eta$ is the learning rate and $\xi \subset D_k$ is a mini-batch.
3. **Parameter Difference (Update Vector):** The local update submitted by client $k$ back to the server is:
   $$U_k^{(t)} = w_k^{(t)} - M^{(t-1)}$$
4. **Server Aggregation:** The server aggregates these updates to produce the next global model:
   $$M^{(t)} = M^{(t-1)} + \sum_{k=1}^K p_k U_k^{(t)}$$

---

## 2. The Mathematical Dilemma: Forward Coupling

Suppose client $k_u \in \{1, \dots, K\}$ issues an unlearning request to remove all influence of its private dataset $D_{k_u}$.

### Why Naive Update Subtraction Fails

A naive approach would be to directly subtract the accumulated updates of client $k_u$ from the final global model $M^{(T)}$:

$$M_{\text{naive}} \stackrel{?}{=} M^{(T)} - \sum_{t=1}^T p_{k_u} U_{k_u}^{(t)}$$

**Proof of Failure (Forward Coupling):**

The global model at round $t$ is recursively defined as:

$$M^{(t)} = M^{(0)} + \sum_{\tau=1}^t \sum_{k=1}^K p_k U_k^{(\tau)}$$

Notice that for any retained client $k_c \neq k_u$ at round $\tau > 1$, its update vector $U_{k_c}^{(\tau)}$ is explicitly conditioned on the model received at round $\tau-1$:

$$U_{k_c}^{(\tau)} = \Phi\left(M^{(\tau-1)}; D_{k_c}\right)$$

where $\Phi(\cdot)$ represents the non-linear local SGD operator. Because $M^{(\tau-1)}$ already contains the contributions of $k_u$ from all preceding rounds $\{1, 2, \dots, \tau-1\}$, we have:

$$\frac{\partial U_{k_c}^{(\tau)}}{\partial U_{k_u}^{(s)}} \neq 0 \quad \text{for all } s < \tau$$

Therefore, client $k_u$'s data influence is irreversibly entangled into the gradient updates of all other benign clients. Subtracting only $k_u$'s direct vectors leaves behind the **indirect cascade** embedded in the weights of all retained clients.

---

## 3. FedEraser Algorithm & Mathematical Formulation

FedEraser resolves forward coupling by **reconstructing the parameter trajectory** using cached historical updates and recalibrating with minimal retained client computation.

```
Original Trajectory:      M^(0) ──> M^(1) ──> M^(2) ──> ... ──> M^(T) (contains k_u)
                                      │         │
                                   [drift]   [drift]
                                      ▼         ▼
Reconstructed Trajectory: M_f^(0) ─> M_f^(1) ─> M_f^(2) ─> ... ─> M_f^(T) (purged of k_u)
                                      ▲         ▲
                           [Calibration training pulls back trajectory]
```

### Phase 1: Update Caching (Normal Training)
During standard FedAvg, the server stores all client updates in a historical cache:

$$\mathcal{U} = \left\{ U_k^{(t)} \;\middle|\; k \in \{1, \dots, K\}, \; t \in \{1, \dots, T\} \right\}$$

---

### Phase 2: Target Removal & Weight Renormalization
When client $k_u$ is removed, the set of retained (calibrating) clients is:

$$\mathcal{C} = \{1, 2, \dots, K\} \setminus \{k_u\}$$

The server re-normalizes the aggregation weights among the remaining clients so they sum to 1:

$$q_k = \frac{n_k}{\sum_{j \in \mathcal{C}} n_j} = \frac{|D_k|}{\sum_{j \in \mathcal{C}} |D_j|} \quad \forall k \in \mathcal{C}$$

---

### Phase 3: Trajectory Drift Formulation

If the server simply re-aggregated the cached historical updates of retained clients without client $k_u$:

$$\hat{U}^{(t)} = \sum_{k \in \mathcal{C}} q_k U_k^{(t)}$$

$$\hat{M}^{(t)} = \hat{M}^{(t-1)} + \hat{U}^{(t)}$$

The error between this uncalibrated reconstruction $\hat{M}^{(t)}$ and the true retrained model $M_{\text{retrain}}^{(t)}$ is bounded by:

$$\|\hat{M}^{(t)} - M_{\text{retrain}}^{(t)}\| \le \sum_{\tau=1}^t \sum_{k \in \mathcal{C}} q_k \left\| U_k^{(\tau)}\left(M^{(\tau-1)}\right) - U_k^{(\tau)}\left(M_{\text{retrain}}^{(\tau-1)}\right) \right\|$$

Under an $L$-Lipschitz smoothness assumption on local gradients $\|\nabla \mathcal{L}_k(w_1) - \nabla \mathcal{L}_k(w_2)\| \le L \|w_1 - w_2\|$, the difference grows exponentially with $t$:

$$\|\hat{M}^{(t)} - M_{\text{retrain}}^{(t)}\| \approx \mathcal{O}\left( (1 + \eta L)^t \right)$$

This exponential divergence is known as **Trajectory Drift**.

---

### Phase 4: Calibration Training Mechanism

To eliminate trajectory drift without full retraining, FedEraser selects a set of calibration time steps $t_j \in \{t_1, t_2, \dots, t_S\} \subseteq \{1, \dots, T\}$.

#### Step A: Client-Side Calibration Training
At calibration round $t_j$, the server broadcasts the current unlearned global model $M_{t_j}^f$ to all retained clients $k_c \in \mathcal{C}$.

Each client computes a fresh local update starting from $M_{t_j}^f$ for a very small number of epochs ($E_{\text{cali}} \ll E$, typically $E_{\text{cali}} = 1$):

$$\tilde{U}_{k_c}^{t_j} \leftarrow \text{LocalSGD}\left(M_{t_j}^f, D_{k_c}, E_{\text{cali}}\right)$$

#### Step B: Server-Side Update Calibration Equation
The server adjusts the cached historical update $U_{k_c}^{t_j}$ using the newly received gradient information $\tilde{U}_{k_c}^{t_j}$:

$$\tilde{e}_{k_c}^{t_j} = \frac{\|\tilde{U}_{k_c}^{t_j}\|_2}{\|U_{k_c}^{t_j}\|_2} \cdot U_{k_c}^{t_j}$$

#### Mathematical Insight of this Equation:
1. **Direction Component ($U_{k_c}^{t_j}$):** Preserves the original gradient direction computed during extensive local training rounds. It represents the historical trajectory of feature learning.
2. **Magnitude Ratio ($\frac{\|\tilde{U}_{k_c}^{t_j}\|_2}{\|U_{k_c}^{t_j}\|_2}$):** Acts as a dynamic curvature scaling factor.
   - If $\|\tilde{U}_{k_c}^{t_j}\| < \|U_{k_c}^{t_j}\|$, the unlearned model is closer to a local minimum or in a flatter region of the loss surface for client $k_c$, so the step size is safely compressed.
   - If $\|\tilde{U}_{k_c}^{t_j}\| > \|U_{k_c}^{t_j}\|$, the unlearned model is on a steep slope, scaling up the step size to accelerate convergence toward the retrained trajectory.

#### Step C: Calibrated Aggregation
The server computes the aggregated calibrated update for the round:

$$\tilde{U}_e^{t_j} = \sum_{k_c \in \mathcal{C}} q_{k_c} \tilde{e}_{k_c}^{t_j}$$

#### Step D: Global Model Update
The unlearned global model advances to the next calibration checkpoint:

$$M_{t_j+1}^f = M_{t_j}^f + \tilde{U}_e^{t_j}$$

This process repeats until round $T$ is reconstructed, producing the final unlearned model $M_{\text{unlearn}} = M_T^f$.

---

## 4. Theoretical and Complexity Comparison

| Dimension | Retrain-from-Scratch (Gold Standard) | FedEraser (Bao et al.) |
| :--- | :--- | :--- |
| **Initial Parameter State** | Random weights $w_0 \sim \mathcal{N}(0, \sigma^2)$ | Starts from $M^{(0)}$ using cached vectors |
| **Local Epochs per Round** | Full training epochs ($E \ge 5$) | Calibration epochs ($E_{\text{cali}} = 1$) |
| **Communication Rounds** | Full rounds $T$ (e.g., 100–500) | Calibration subset rounds $S \ll T$ |
| **Client Computational Cost** | $\mathcal{O}(T \cdot E \cdot |D_k|)$ | $\mathcal{O}(S \cdot E_{\text{cali}} \cdot |D_k|)$ |
| **Server Storage Overhead** | $\mathcal{O}(|\Theta|)$ (only current model) | $\mathcal{O}(K \cdot T \cdot |\Theta|)$ (historical update cache) |
| **Wall-Clock Speedup** | $1\times$ (Baseline) | **$4\times \text{ to } 10\times$ faster** |

---

## 5. Summary of Mathematical Takeaways

1. **Why unlearning in FL is hard:** Forward coupling entangles client gradients across time via recursive non-linear parameter dependence: $U_k^{(t)} = \Phi(M^{(t-1)}(U_{k_u}^{(t-1)}))$.
2. **Why naive subtraction fails:** It ignores the indirect cascade of updates already propagated to retained clients.
3. **How FedEraser solves it:** Reconstructs the unlearned trajectory from round 0 using historical vectors scaled by real-time loss curvature:
   $$\tilde{e}_{k_c}^{t_j} = \frac{\|\tilde{U}_{k_c}^{t_j}\|_2}{\|U_{k_c}^{t_j}\|_2} \cdot U_{k_c}^{t_j}$$
