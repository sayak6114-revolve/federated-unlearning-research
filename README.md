# federated-unlearning-research
Research notes, paper reviews, implementations, and evaluation of federated unlearning methods.
# Federated Unlearning Research

## Overview

This repository documents my study and implementation of federated unlearning, focusing on removing a client's contribution from a trained federated learning model and evaluating whether the unlearning process worked.

## Research Objectives

- Understand federated learning and client-level unlearning.
- Study and compare federated unlearning algorithms.
- Evaluate unlearning using appropriate metrics and retraining baselines.
- Implement experiments with simulated clients.
- Explore verifiable unlearning and potential blockchain integration.

## Learning Roadmap

1. Study the fundamentals of federated learning.
2. Read *[Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics](url)*.
3. Understand client-level unlearning algorithms, including FedEraser.
4. Study evaluation metrics for forgetting, model utility, and efficiency.
5. Implement federated learning and client-removal experiments.
6. Compare the unlearned model with a model retrained without the removed client.
7. Investigate verification mechanisms.

## Evaluation Plan

The initial experiments will investigate:

- Accuracy and loss on retained clients' test data.
- Behavior on the removed client's data.
- Similarity to a retrained reference model.
- Membership inference attack performance.
- Unlearning time compared with full retraining.

The exact metrics and evaluation protocol will be selected based on the research papers.

## Research Papers

### Federated Unlearning Survey

**Title:** Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics

**Paper:** [https://arxiv.org/abs/2401.05146](url)

**Status:** Reading and taking notes.

## Repository Structure

- `papers/` — Research paper summaries.
- `concepts/` — Fundamental concepts and equations.
- `experiments/` — Experimental implementations.
- `results/` — Evaluation results and plots.

## Current Progress

- [x] Define the research objective.
- [ ] Complete the survey paper.
- [ ] Implement a baseline federated learning model.
- [ ] Implement a client-level unlearning method.
- [ ] Evaluate unlearning against retraining.

## References

Research papers are linked to their original sources. Notes and code in this repository will be developed as part of my own study.
