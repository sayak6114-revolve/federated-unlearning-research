# [Paper Title]

- **Authors:** 
- **Venue & Year:** 
- **Paper Link:** 
- **Official Code:** 

---

## 1. Summary in 3 Bullets
- 
- 
- 

---

## 2. Core Problem & Motivation
- **Granularity:** Client-level / Sample-level / Class-level
- **The Gap:** Why does standard retraining or naive unlearning fail here?
- **Setting & Assumptions:** (IID vs Non-IID data, communication constraints, server storage)

---

## 3. Threat Model & Unlearning Formulation
- **Trigger:** (GDPR request / poisoned client / malicious actor)
- **Server Knowledge:** (What is retained on the server? e.g., historical updates, client data, proxy dataset)
- **Goal Definition:** (Exact unlearning vs. $\epsilon$-approximate unlearning vs. empirical unlearning)

---

## 4. Proposed Method / Algorithm
1. **Step 1:** 
2. **Step 2:** 
3. **Step 3:** 

*Key Equations / Formula:*
$$ $$

---

## 5. Evaluation & Verification Metrics
- **Forgetting / Privacy Metrics:**
  - [ ] Membership Inference Attack (MIA) AUC/Accuracy: 
  - [ ] Backdoor Attack Success Rate (ASR): 
  - [ ] Target Client Accuracy / Loss: 
- **Utility Preservation Metrics:**
  - [ ] Retained Clients' Test Accuracy: 
  - [ ] Weight Distance ($L_2$ norm to retrained model): 
- **Efficiency Metrics:**
  - [ ] Speedup vs Retraining from Scratch: 
  - [ ] Communication Rounds: 

---

## 6. Critical Analysis
- **Strengths:**
- **Weaknesses / Real-world Hurdles:**
- **Takeaways for our implementation:**
