# Federated Unlearning: A Survey

## Paper Information

- **Title:** Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics
- **Authors:** Romandini et al.
- **Paper Link:** [https://arxiv.org/abs/2401.05146](https://arxiv.org/abs/2401.05146)
- **Status:** Currently studying
- **Research focus:** Client-level unlearning and evaluation metrics

---

## 1. Research Objective

Understand how to remove a client's contribution from a trained federated learning model and evaluate whether the removal was successful.

---

## 2. Problem Statement (Hospital Motivation)

Suppose ten hospitals collaboratively train a machine learning model using federated learning. The hospitals keep their patient data locally.

If Hospital 5 leaves the collaboration, the system must address two separate questions:

1. **How can the influence of Hospital 5 be removed from the trained model?**
2. **How can we evaluate whether the unlearning process worked?**

Deleting Hospital 5's data from storage does not automatically remove its previous influence from the global model.

---

## 3. Key Concepts

### Federated Learning
Multiple clients train a shared model by computing local updates and sharing those updates with a coordinating server rather than directly sharing raw training data.

### Client-Level Unlearning
An approach to removing the contribution of one participating client from an already-trained federated model.

### Retraining Baseline
A reference model trained from scratch without the client that requested removal. It represents the gold-standard ground truth for unlearning.

### Unlearning Evaluation
The process of measuring forgetting effectiveness, retained model utility, privacy leakage, and computational cost.

---

## 4. Evaluation Metrics to Investigate

- **Retained test accuracy and loss:** Ensuring remaining clients' performance doesn't degrade.
- **Forget-set behavior:** Accuracy and loss degradation specifically on the target client's private distribution.
- **Model similarity to the retrained reference:** $L_2$ parameter distance $\|w_u - w_r\|_2$ and representation similarity.
- **Membership inference attack performance:** Shadow model AUC dropping towards random guess ($\approx 0.5$).
- **Unlearning time compared with full retraining:** Speedup ratio and communication rounds saved.

These metrics evaluate different properties and must be interpreted together.

---

## 5. Questions to Answer While Reading

- What are the main categories of federated unlearning methods?
- What information must be stored during federated training?
- How do existing methods remove a client's influence?
- How is successful unlearning evaluated?
- What are the limitations of the evaluation metrics?
- Can a system provide verifiable evidence that unlearning occurred?

---

## 6. Research Progress

- [x] Define the research objective.
- [x] Create project repository structure.
- [ ] Complete the survey paper.
- [ ] Understand the unlearning problem formulation.
- [ ] Study unlearning algorithms (FedEraser, Gradient Ascent).
- [ ] Study evaluation metrics (MIA, Backdoor ASR).
- [ ] Implement a baseline experiment.

---

## References

Romandini et al., *Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics*. [arXiv:2401.05146](https://arxiv.org/abs/2401.05146).
