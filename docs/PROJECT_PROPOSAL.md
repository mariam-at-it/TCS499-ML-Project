# Simulation-Based Evaluation of Federated Learning for Smart City Traffic and Energy Prediction

## Project Proposal

**Prepared by:** 
**Date:** February 2026
**Institution:** 

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Objectives](#2-project-objectives)
3. [Project Significance](#3-project-significance)
4. [Supportive Literature](#4-supportive-literature)
5. [Research Methodology](#5-research-methodology)
6. [Research Scope](#6-research-scope)
7. [Project Outcomes](#7-project-outcomes)
8. [Project Timeline](#8-project-timeline)

---

## 1. Problem Statement

### 1.1 Context

Smart cities generate massive volumes of data from distributed infrastructure - traffic sensors, smart meters, environmental monitors, and connected devices. This data can power machine learning models that optimize traffic flow and forecast energy demand.

However, traditional centralized machine learning requires aggregating raw data from all sources into a single location. In modern smart cities, this is increasingly impractical for three reasons:

**1. Privacy and Regulation.** The GDPR (Articles 5, 6, and 44–49) restricts how personal data - including movement patterns and household energy usage - can be collected, processed, and transferred across organizational boundaries. The EU AI Act (Regulation (EU) 2024/1689, with phased applicability from February 2, 2025 through August 2, 2027 depending on the requirement category) adds compliance requirements for AI systems used in critical infrastructure such as traffic management. Together, these regulations create strong incentives for approaches that minimize raw data movement.

**2. Data Silos.** Smart city data is owned by different organizations: municipal transport agencies, private utility companies, and commercial IoT providers. These entities are often unwilling or legally unable to share raw data with each other, even when collaboration would improve model quality.

**3. Communication Cost.** Transmitting high-frequency sensor data from hundreds of devices to a central server is expensive in terms of bandwidth. For example, 325 traffic sensors reporting at 5-minute intervals generate continuous data streams that must be stored, transmitted, and processed centrally - a cost that scales linearly with the number of sensors and reporting frequency (though, as discussed in Section 5.10, FL's communication cost has its own scaling characteristics).

### 1.2 The Problem

> How can multiple smart city stakeholders collaboratively train useful ML models for traffic prediction and energy forecasting **without sharing their raw data**?

### 1.3 Proposed Solution

**Federated Learning (FL)** addresses this by training models locally on each data source and sharing only model updates, never raw data. A central server aggregates these updates into a global model.

However, FL introduces its own challenges: accuracy degrades under non-IID (non-identically distributed) data, differential privacy mechanisms add noise that further reduces performance, and the communication overhead of exchanging model updates across many rounds is nontrivial. Whether these trade-offs are acceptable for real smart city tasks is largely unknown - most FL research benchmarks on image classification datasets (CIFAR, MNIST) rather than urban time-series data.

This project applies FL to two concrete smart city domains - **traffic flow prediction** and **energy demand forecasting** - and rigorously evaluates the accuracy, communication cost, privacy trade-offs, and system-level bottlenecks through simulation.

---

## 2. Project Objectives

### 2.1 Goal

Apply existing Federated Learning techniques to smart city traffic and energy prediction tasks, systematically evaluate performance under realistic non-IID conditions and privacy constraints, and report what works, what doesn't, and what practical challenges arise.

### 2.2 Specific Objectives

| ID | Objective | How We'll Know It Worked |
|----|-----------|--------------------------|
| **O1** | Implement FL-based traffic flow prediction using FedAvg and FedProx on the Flower framework | Working pipeline that trains across simulated clients and produces predictions |
| **O2** | Implement FL-based energy demand forecasting on distributed household data | Working pipeline comparable to O1 for energy domain |
| **O3** | Measure the accuracy gap between FL and centralized training under realistic non-IID data | Published accuracy comparison tables across multiple heterogeneity levels (bin-and-Dirichlet β ∈ {0.1, 0.5, 1.0, 10.0}) |
| **O4** | Profile system-level costs: communication overhead, wall-clock training time, memory usage, and convergence behavior | Quantified cost tables comparing FL vs. centralized training across all metrics |
| **O5** | Evaluate differential privacy (via Flower) and document privacy-accuracy trade-offs | Privacy-accuracy curves across ε ∈ {2, 4, 8} with formal DP guarantees |

### 2.3 Research Questions

1. **RQ1:** How much accuracy do we lose compared to centralized training when data is non-IID across city zones, and at what heterogeneity level does FL become impractical?
2. **RQ2:** What is the practical privacy-accuracy trade-off when adding differential privacy to the FL pipeline?
3. **RQ3:** What are the system-level costs (communication bytes, wall-clock time, memory) of FL training compared to centralized training for urban time-series models?
4. **RQ4:** What are the main practical bottlenecks in FL for smart city tasks - communication overhead, non-IID convergence degradation, straggler effects, or privacy noise?

---

## 3. Project Significance

### 3.1 Practical Significance

Most FL research stays in simulation on image datasets and stops at reporting accuracy. This project focuses on **what actually happens when you apply FL to realistic urban time-series prediction tasks** - the accuracy gaps under heterogeneous city data, the communication overhead, and the privacy costs. This kind of systems-aware empirical evaluation is exactly what cities and practitioners need to make informed adoption decisions.

Specific practical contributions:
- **Deployment-readiness assessment** documenting the conditions under which FL is viable for traffic and energy prediction
- **Honest benchmarks** showing where FL works well and where it struggles compared to centralized training
- **Communication and compute cost analysis** quantifying the real overhead of FL for urban time-series workloads

### 3.2 Societal Significance

Federated Learning lets cities build intelligent systems without mass data collection. If this project demonstrates that FL achieves acceptable accuracy for traffic and energy prediction, it provides a privacy-respecting alternative to the current paradigm of centralizing citizen data. If FL falls short, documenting where and why is equally valuable - it prevents cities from adopting an unsuitable technology based on hype.

### 3.3 Academic Significance

While FL algorithms are well-studied, **systems-aware evaluations on real smart city datasets are scarce**. Most papers benchmark on CIFAR/MNIST and report only accuracy. This project contributes empirical evidence on:
- How non-IID urban data affects FL convergence in practice (not just in theory)
- The actual communication cost of FL vs. centralized training for time-series models
- Privacy-accuracy trade-offs with formal DP guarantees on urban datasets
- Cross-domain comparison (traffic vs. energy) revealing whether FL viability is task-dependent

---

## 4. Supportive Literature

### 4.1 Federated Learning Foundations

| Reference | Key Contribution | Relevance to This Project |
|-----------|------------------|---------------------------|
| McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS.* | Introduced FedAvg - the core FL algorithm | We use FedAvg as our primary aggregation strategy |
| Li et al. (2020). "Federated Optimization in Heterogeneous Networks." *MLSys.* | FedProx - handles non-IID data and stragglers via proximal term | We use FedProx to handle heterogeneous city data |
| Kairouz et al. (2021). "Advances and Open Problems in Federated Learning." *Foundations and Trends in ML.* | Comprehensive survey of FL challenges | Frames our research questions within known open problems |

### 4.2 FL for Smart Cities and IoT

| Reference | Key Contribution | Relevance |
|-----------|------------------|-----------|
| Ghadi et al. (2023). "Integration of FL with IoT for Smart Cities." *PeerJ Computer Science*, 9:e1657. | Maps FL onto IoT architectures for urban sensing | Informs our system architecture design |
| Delgado Fernández et al. (2025). "Scaling Smart Cities with FL: Building Energy Performance Prediction." *BISE.* | Validated FL on 25,000+ buildings for energy prediction | Directly comparable to our energy forecasting objective |
| Li et al. (2025). "FedGDAN: Privacy-preserving traffic flow prediction." *Scientific Reports* (Nature). | FL for traffic using graph neural networks | Shows FL viability for traffic prediction; we build on this |
| Belal et al. (2024). "Survey of Federated Learning Models for Spatial-Temporal Mobility Applications." *ACM Trans. Spatial Algorithms and Systems.* | Survey of FL for mobility data | Validates our dataset and methodology choices |

### 4.3 Non-IID Data and Practical Challenges

| Reference | Key Contribution | Relevance |
|-----------|------------------|-----------|
| Jimenez et al. (2024). "Non-IID Data in FL: Survey with Taxonomy, Metrics, and Methods." *arXiv:2411.12377.* | Taxonomy of data heterogeneity types | Guides our non-IID simulation methodology |
| Jian & Liu (2025). "Widening the Network Mitigates Data Heterogeneity on FedAvg." *ICML 2025.* | Wider networks help under non-IID conditions | Informs our model architecture choices |
| Zakerinia et al. (2024). "Communication-Efficient FL With Data and Client Heterogeneity." *AISTATS.* | Handles async updates + compression | Contextualizes our communication cost analysis |

### 4.4 Privacy in Federated Learning

| Reference | Key Contribution | Relevance |
|-----------|------------------|-----------|
| Abadi et al. (2016). "Deep Learning with Differential Privacy." *CCS.* | DP-SGD and moments accountant | Foundation for our DP integration |
| Zhu et al. (2019). "Deep Leakage from Gradients." *NeurIPS.* | Gradient inversion attacks on FL | Motivates our DP evaluation and threat model |

### 4.5 Research Gap

The gap we address: **While individual studies have evaluated FL for traffic prediction (Li et al., 2025) or energy forecasting (Delgado Fernández et al., 2025), no existing work provides a unified cross-domain evaluation that systematically compares FL against centralized training across multiple heterogeneity levels, privacy budgets, and system-level cost metrics for both traffic and energy tasks simultaneously.** This project fills that gap with an honest, systems-aware empirical evaluation.

---

## 5. Research Methodology

### 5.1 Approach

We use a **build-measure-report** methodology:
1. **Build** FL pipelines for traffic and energy prediction using Flower
2. **Measure** accuracy, communication cost, privacy trade-offs, convergence behavior, and system-level overhead
3. **Report** findings including negative results and practical lessons

### 5.2 Framework and Tools

| Tool | Role | Why This One |
|------|------|--------------|
| **Flower (flwr)** | FL framework | Mature, supports large-scale simulation via Virtual Client Engine (VCE), built-in FedAvg/FedProx/DP wrappers |
| **PyTorch** | Model training | Standard; compatible with Flower |
| **FedSTGraph** | Federated traffic benchmark | Provides partitioning logic for METR-LA/PEMS-BAY into federated tasks |
| **psutil / PyTorch Profiler** | System profiling | Measures CPU/RAM/wall-clock during FL rounds |

### 5.3 Datasets

| Domain | Dataset | Size | Access | Client Formation |
|--------|---------|------|--------|------------------|
| **Traffic** | METR-LA (via FedSTGraph) | 207 sensors, 4 months (Mar–Jun 2012), 5-min intervals | Public | Grouped by district (cross-silo) or 1 sensor = 1 client (ablation) |
| **Traffic** | PEMS-BAY (via FedSTGraph) | 325 sensors, ~5 months (Jan–May 2017), 5-min intervals | Public | Grouped by district (cross-silo) or 1 sensor = 1 client (ablation) |
| **Energy** | CER Irish Smart Metering | ~4,000–5,000+ residential households (exact count depends on filtering), 30-min intervals, ~75 weeks (Jul 2009–Dec 2010) | Public (request form) | Grouped by area (cross-silo) or 1 household = 1 client (ablation) |

**Energy dataset rationale:** We use the CER (Commission for Energy Regulation) Irish Smart Metering dataset as our primary energy dataset. Unlike the UCI Individual Household Electric Power dataset (which contains only a single household and cannot meaningfully represent multi-client FL), CER contains thousands of distinct residential households with natural inter-household heterogeneity. In our experiments, we will report the exact number of households retained after preprocessing/quality filters.

**Fallback:** If CER access is delayed, the Low Carbon London dataset (5,567 households, 2011–2014, 30-min intervals, UK Power Networks) is a viable alternative. Both require only a request form.

**Prediction targets:**
- **Traffic:** Predict traffic speed 15/30/60 minutes ahead for each sensor
- **Energy:** Predict household energy consumption 1/6/24 hours ahead

**Data Splitting:** All datasets use chronological train/validation/test splits (e.g., first 70% of timestamps for training, next 15% for validation, final 15% for testing). We do not shuffle across time to avoid data leakage in time-series prediction.

### 5.4 Models

We use standard, well-understood architectures - the goal is to evaluate FL, not invent new models:

| Task | Model | Rationale |
|------|-------|-----------|
| Traffic prediction | GRU (2-layer, 64 hidden) | Proven for traffic time series; efficient for large-scale FL simulation |
| Traffic prediction | STGCN (centralized baseline only) | Captures spatial dependencies between sensors; not evaluated in our FL setting without additional cross-client feature sharing |
| Energy forecasting | LSTM (2-layer, 64 hidden) | Standard for energy time series |

**Note on STGCN:** STGCN is included as a centralized-only baseline. In the FL setups we evaluate, clients hold disjoint subsets of sensors/households. For graph models like STGCN, a client would typically need neighbor node features at each timestep to compute graph convolutions across edges that cross client boundaries. Supporting that would require an additional protocol (e.g., cross-client feature sharing, split learning, or a different client formation where each client owns a subgraph), which is out of scope for this proposal.

### 5.5 FL Configuration

#### 5.5.1 Client Construction
We evaluate two distinct regimes:
1.  **Cross-silo (Primary):** Raw data sources are grouped into 10–20 clients based on geographic districts (traffic) or areas/tariff groups (energy). Each client represents an institution or zone holding data for multiple sensors/households.
2.  **Cross-device-scale (Ablation):** Each sensor (traffic) or household (energy) is treated as an individual client. This tests scalability to hundreds/thousands of clients.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Aggregation** | FedAvg (primary), FedProx (secondary) | Standard algorithms - we're evaluating FL viability, not inventing new methods |
| **Clients** | 10–20 per experiment (cross-silo) | Realistic for cross-silo city deployment where participants are institutions |
| **Local epochs** | 1, 3, 5 (compared) | Trade-off between communication rounds and local computation |
| **Rounds** | Up to 200 | Until convergence or budget exhausted |
| **Client fraction** | 100% (Default), 50% (Ablation) | Default assumes reliable institutional clients; ablation tests straggler effects |
| **Simulation engine** | Flower VCE | Multiplexes clients on available hardware; controls per-client CPU/GPU allocation |

**Convergence Rule:** Training stops when the global model's validation MAE improves by less than 0.1% over 10 consecutive rounds, or when 200 rounds are reached, whichever comes first.

### 5.6 Non-IID Simulation

Smart city data is naturally non-IID. We simulate and measure heterogeneity using:

- **Bin-and-Dirichlet partitioning:** For regression tasks, we first discretize targets into K=10 equal-frequency (quantile) bins. We then apply Dirichlet partitioning (β ∈ {0.1, 0.5, 1.0, 10.0}) over these bins to control the degree of label heterogeneity across clients.
- **Geographic partitioning:** Assigning sensors to clients based on spatial proximity.
- **Natural partitioning:** For CER energy data, each household is inherently a different distribution.

**Heterogeneity quantification:** We measure the degree of non-IID using the Earth Mover's Distance (EMD) computed on the histogram of target values (using the same K=10 quantile bins). This gives a scalar measure of how different each client's target distribution is from the global distribution.

### 5.7 Threat Model and Privacy Evaluation

#### 5.7.1 Threat Model
We assume a **trusted aggregator** threat model. The aggregator is trusted to coordinate the learning process and see the raw model updates before noise is added (server-side DP). The privacy goal is to protect the final **published model** from leaking information about any individual client's dataset to external observers.

Known external threats include:
- **Gradient inversion attacks** (Zhu et al., 2019): Reconstructing training data from model updates (e.g., if per-round updates are exposed via logging, debugging, or a server compromise).
- **Membership inference attacks**: Determining whether specific data points were in the training set.

**Privacy Unit:** The unit of privacy is **client-level** (update-level DP). This means the DP mechanism bounds what can be learned about any single client's entire dataset from the published model.

#### 5.7.2 Differential Privacy Configuration

We use **server-side (central) differential privacy** via Flower's `DifferentialPrivacyServerSideFixedClipping` strategy: each participating client's update is L2-clipped, then Gaussian noise is added to the aggregated update before publishing/saving the global model.

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| **Privacy reporting** | Target ε ∈ {2, 4, 8} (reported post-hoc) | We will compute the achieved ε for a fixed δ given the training schedule (rounds, sampling) and the configured noise multiplier |
| **δ** | 10⁻⁵ | Fixed value used for all reported (ε, δ) pairs |
| **Clipping norm (S)** | 1.0 | Bounds each client's update L2 norm before aggregation |
| **Noise mechanism** | Gaussian (server-side) | Noise added to the aggregated update after clipping |
| **Noise multiplier (z)** | Swept to hit target ε values | Flower config parameter; determines noise scale relative to S |
| **Clients sampled per round (m)** | Determined by client fraction × total clients | Required input to privacy accounting (sampling rate depends on m) |

**Accounting:** Because Flower's strategy is typically configured via a noise multiplier (not directly by ε), we will compute and report the achieved privacy budget using an RDP accountant given: clipping norm, noise multiplier, number of rounds, and client sampling rate (including the 50% client-fraction ablation).

### 5.8 Centralized Baseline

To ensure fair comparison, the centralized baseline uses:
- **Same model architecture** (identical hyperparameters)
- **Same total data** (union of all clients' datasets)
- **Same preprocessing** (identical normalization, windowing, train/val/test splits)
- **Same random seeds** (3 seeds per experiment; we report mean ± std)

**Training Budget:** We will report two comparisons:
1. **Best-achievable:** Both FL and centralized models train until convergence (early stopping with patience = 10).
2. **Compute-matched (sanity check):** Centralized training is limited to a comparable optimization budget (matching total local update steps) to ensure conclusions are not driven purely by extra centralized compute.

**Hyperparameters:** Hyperparameters (learning rate, batch size, hidden dimensions) are tuned on the centralized baseline using the validation set, then held fixed for all FL experiments. This ensures FL vs. centralized comparisons reflect the effect of federation, not hyperparameter differences.

### 5.9 Evaluation Metrics

| What We Measure | Metric | Why |
|-----------------|--------|-----|
| Prediction accuracy | MAE, RMSE | Standard for traffic/energy regression tasks |
| FL vs. centralized gap | Δ accuracy (absolute and relative) | Core question: how much do we lose from federation? |
| Communication cost | Total bytes uploaded + downloaded across all rounds | Quantifies the real bandwidth overhead of FL |
| Privacy cost | Accuracy at ε = {2, 4, 8} vs. no-DP baseline | Privacy-accuracy trade-off curve |
| Convergence speed | Rounds to reach 95% of final accuracy | How quickly does FL converge? |
| Training time | Wall-clock time per round, total training time | Practical feasibility on standard compute |
| Resource usage | Peak RAM per client, total CPU-hours | Simulation cost; proxy for deployment requirements |
| Fairness | Macro-averaged client-level MAE (mean, min, max, std) | Assess whether FL disproportionately benefits or harms specific zones or households |

### 5.10 Communication Cost Analysis

FL communication cost is **not** inherently lower than centralized training - it depends on model size, number of rounds, and client participation.

- **FL cost:** We compute FL communication cost as: model_size_bytes × 2 (upload + download) × participating_clients × rounds.
- **Centralized cost:** We compare this against the one-time cost of centralizing raw data.

We compute and compare both quantities explicitly. FL may transfer **more** total bytes than centralized training for small datasets with large models. The advantage of FL is that it avoids transferring **raw data**, which is the privacy-relevant comparison - but we report both the byte count and the privacy distinction honestly.

### 5.11 Compute Feasibility

All experiments run on a single workstation. Estimated compute budget:

| Experiment | Clients | Rounds | Est. Time per Run | Runs | Total |
|------------|---------|--------|-------------------|------|-------|
| Traffic (GRU, FedAvg) | 20 | 200 | ~2 hours | 12 (4 β × 3 seeds) | ~24 hrs |
| Traffic (GRU, FedProx) | 20 | 200 | ~2 hours | 12 | ~24 hrs |
| Energy (LSTM, FedAvg) | 20 | 200 | ~1.5 hours | 12 | ~18 hrs |
| Energy (LSTM, FedProx) | 20 | 200 | ~1.5 hours | 12 | ~18 hrs |
| DP experiments (primary: traffic GRU FedAvg) | 20 | 200 | ~2 hours | 9 (3 ε × 3 seeds) | ~18 hrs |
| DP spot-checks (energy LSTM, ε=4 only) | 20 | 200 | ~1.5 hours | 3 (1 ε × 3 seeds) | ~5 hrs |
| Large-scale (207/325 sensors) | 207/325 | 100 | ~6 hours | 6 | ~36 hrs |
| Centralized baselines | - | - | ~1 hour | 15 | ~15 hrs |
| **Total estimated** | | | | | **~158 hrs** |

This is feasible on a single GPU workstation over ~2–3 weeks of compute.

### 5.12 Reproducibility

- All code open-sourced on GitHub
- Hydra/YAML configs for every experiment
- Fixed random seeds (3 per experiment)
- Docker containers for environment consistency
- FedSTGraph provides deterministic data partitioning for traffic datasets
- **Data Access:** Datasets are not redistributed with the repository. We provide download instructions and preprocessing scripts. Users must obtain CER and Low Carbon London datasets through their respective request forms.

---

## 6. Research Scope

### 6.1 What This Project Does

- Applies **existing** FL algorithms (FedAvg, FedProx) to traffic and energy prediction
- Uses **existing** privacy mechanisms (Flower's DP wrapper)
- Evaluates **all experiments in simulation** using Flower's Virtual Client Engine
- Measures system-level costs (communication, compute, memory) alongside accuracy
- Reports honestly, including negative results
- Produces a **deployment-readiness assessment** for practitioners

### 6.2 What This Project Does NOT Do

| Excluded | Why |
|----------|-----|
| Invent new FL algorithms | Not the focus - we evaluate existing methods |
| Formal convergence proofs | Applied project, not theoretical |
| Video or image data | Scope control - we focus on time-series sensor data |
| Hardware deployment (edge devices) | All evaluation is simulation-based; hardware validation is future work |
| Byzantine fault tolerance | Assumes trusted institutional participants (cross-silo) |
| Secure Aggregation | Adds complexity without advancing our core questions |
| Vertical FL | Different setting; out of scope |
| Communication compression techniques | We measure baseline FL cost; compression is an optimization layer, not our research question |

### 6.3 Assumptions

1. Participants are **known institutions** (city agencies, utility companies) - not anonymous devices
2. Participants are **honest** (no adversarial behavior - justified by cross-silo trust model)
3. In the primary cross-silo experiments, all clients participate each round (100% availability). We run separate ablation experiments with partial participation (50% client fraction) to assess straggler and availability effects (RQ4).
4. Simulation accurately represents the computational workload (model training, aggregation). Communication overhead is measured as serialized payload bytes; network latency and bandwidth constraints are not modeled.

### 6.4 Limitations

1. We use public datasets partitioned to simulate silos - not actual multi-stakeholder data from a real city
2. All results are simulation-based; real-world deployment may surface additional challenges (network variability, hardware constraints, operational issues)
3. Privacy evaluation uses basic DP (Gaussian mechanism with server-side clipping) - more sophisticated methods (Secure Aggregation, Homomorphic Encryption) are out of scope
4. Results may not generalize to all cities, data types, or FL configurations beyond those tested
5. STGCN is evaluated only as a centralized baseline because graph convolutions generally require neighbor node features across edges; supporting those dependencies across client boundaries would require additional protocols beyond the FL setups evaluated here.

---

## 7. Project Outcomes

### 7.1 Deliverables

| # | Deliverable | Description | Month |
|---|-------------|-------------|-------|
| D1 | FL Training Pipeline | Working Flower-based FL pipeline for traffic and energy prediction, with Hydra configs | 8 |
| D2 | Evaluation Report | Comprehensive accuracy, communication, privacy, and convergence benchmarks across all experimental conditions | 14 |
| D3 | Deployment-Readiness Assessment | Analysis of compute/bandwidth requirements, non-IID sensitivity, and DP impact - with concrete recommendations on when FL is/isn't viable for smart city tasks | 16 (draft), 18 (final) |
| D4 | Open-Source Repository | All code, configs, experiment logs, and analysis scripts on GitHub with documentation | 18 |

### 7.2 Publications

| # | Publication | Venue Type | Timeline |
|---|-------------|------------|----------|
| P1 | "Federated Learning for Smart City Traffic and Energy Prediction: A Systems-Aware Evaluation" | Conference (SenSys / IoTDI / BuildSys) | Submit Month 14 |
| P2 | "How Viable is Federated Learning for Urban Time-Series Prediction? Lessons from Simulation" | Workshop (FL@FM / FedVision / Urban Computing) | Submit Month 16 |

Two publications is realistic for an 18-month research project. A third (journal extension) may follow if results warrant it.

### 7.3 Expected Results

We **do not promise specific accuracy numbers** - we report what we find. However, based on prior work, we expect:

| Metric | Expected Range | Based On |
|--------|----------------|----------|
| FL vs. centralized accuracy gap (near-IID, β=10) | 1–3% | Delgado Fernández et al. (2025) |
| FL vs. centralized accuracy gap (extreme non-IID, β=0.1) | 5–15% | Li et al. (2020), Jimenez et al. (2024) |
| Accuracy with DP (ε=4) | 5–15% additional degradation | Abadi et al. (2016) |
| Communication cost | Computed from actual model size (reported from implementation) | Framework analysis |
| Convergence | 50–100 rounds to 95% final accuracy | Typical for FedAvg on time-series |

If results are worse than expected, **we report that too** - negative results about where FL fails are equally valuable for practitioners.

### 7.4 Contributions

This is not an algorithmic contribution. Our contributions are:

1. **Empirical evidence** on FL performance for traffic and energy prediction under realistic non-IID conditions
2. **Systems-level cost analysis** quantifying the communication, compute, and convergence overhead of FL vs. centralized training
3. **Privacy-accuracy trade-off data** with formal DP guarantees for cities evaluating privacy-preserving ML
4. **Cross-domain comparison** revealing whether FL viability differs between traffic and energy prediction tasks
5. **Deployment-readiness assessment** with concrete recommendations for practitioners
6. **Open-source benchmark** that others can reproduce and extend

---

## 8. Project Timeline

### 8.1 Overview

```
Q1         Q2         Q3         Q4         Q5         Q6
|----------|----------|----------|----------|----------|-----|

Phase 1: Setup & Literature
██████████
[W1]

Phase 2: Build FL Pipelines
       ██████████████████
       [W2-W3]

Phase 3: Core Experiments
                  ██████████████████████
                  [W4-W6]

Phase 4: Systems Analysis & Ablations
                            ██████████████████
                            [W6-W7]

Phase 5: Writing & Release (overlaps with 3 & 4)
                                 ██████████████████
                                 [W5-W8]

Milestones:
    ◆ W2:  Environment ready, datasets loaded, centralized baselines
    ◆ W4:  FL pipelines working for both domains
    ◆ W6: Core experiments complete, P1 submitted
    ◆ W7: Systems analysis complete, P2 submitted
    ◆ W8: Assessment published, repository released
```

### 8.2 Detailed Plan

| Phase | Weeks | What Happens | Deliverable |
|-------|--------|--------------|-------------|
| **1. Setup** | 1 | Literature review; set up Flower + FedSTGraph; acquire CER dataset; establish centralized baselines for traffic and energy; validate data loading and preprocessing | Centralized baselines; working dev environment |
| **2. Build** | 2–3 | Implement FedAvg and FedProx pipelines for traffic (GRU) and energy (LSTM); integrate Flower DP wrapper; test on small scale (5 clients, 20 rounds); validate correctness against centralized | D1: Working FL pipeline |
| **3. Experiments** | 4–6 | Run core experiments: non-IID sweep (4 β values × 3 seeds), DP sweep (3 ε values), centralized comparisons; fairness analysis across clients | D2: Evaluation report; P1 submitted |
| **4. Systems Analysis** | 6–7 | Communication cost analysis; convergence profiling; large-scale experiments (207/325 sensors); compute budget measurements; ablation studies (local epochs, client fraction) | D3: Deployment-readiness assessment |
| **5. Write & Release** | 7–8 | Write deployment-readiness assessment; prepare workshop paper; open-source release; documentation; reproducibility verification | D4: Repository; P2 submitted |

### 8.3 Milestones

| Milestone | Week | Go / No-Go |
|-----------|-------|------------|
| **Environment ready** | 2 | If CER dataset unavailable → use Low Carbon London fallback |
| **FL pipelines working** | 4 | If accuracy gap >25% → investigate root cause and document as finding |
| **Core experiments complete** | 6 | Submit paper regardless of whether results are positive or negative |
| **Systems analysis complete** | 7 | If compute budget exceeded → reduce experiment grid and document |

### 8.4 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CER / energy dataset access delayed | Medium | Low | Low Carbon London is a ready fallback; both require only a request form |
| FL accuracy much worse than centralized | Low | Medium | This is a finding - report it honestly |
| DP causes non-convergence at ε=2 | Medium | Low | Expected; we also test ε=4 and ε=8 to show the trade-off curve |
| Compute budget insufficient for full grid | Low | Medium | Prioritize core experiments; reduce seeds from 3 to 2 if needed |
| Timeline slip | Medium | Medium | Phases overlap; can compress Phase 5 if needed |
| Simulation results may not transfer to real deployment | Inherent | Medium | Acknowledged as limitation; we scope claims to simulation results |

---

## References

1. McMahan, H.B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.

2. Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks. *MLSys*.

3. Kairouz, P., et al. (2021). Advances and Open Problems in Federated Learning. *Foundations and Trends in ML*.

4. Ghadi, Y.Y., et al. (2023). Integration of FL with IoT for Smart Cities. *PeerJ Computer Science*, 9:e1657.

5. Delgado Fernández, J., et al. (2025). Scaling Smart Cities with FL: Building Energy Performance Prediction. *BISE*.

6. Li, X., et al. (2025). FedGDAN: Privacy-preserving traffic flow prediction. *Scientific Reports* (Nature).

7. Belal, Y., et al. (2024). Survey of Federated Learning Models for Spatial-Temporal Mobility Applications. *ACM Trans. Spatial Algorithms and Systems*.

8. Jimenez, D.M., et al. (2024). Non-IID Data in FL: Survey with Taxonomy, Metrics, and Methods. *arXiv:2411.12377*.

9. Abadi, M., et al. (2016). Deep Learning with Differential Privacy. *ACM CCS*.

10. Mironov, I. (2017). Rényi Differential Privacy. *IEEE CSF*.

11. Jian, L. & Liu, D. (2025). Widening the Network Mitigates Data Heterogeneity on FedAvg. *ICML 2025*.

12. Zakerinia, H., et al. (2024). Communication-Efficient FL With Data and Client Heterogeneity. *AISTATS*.

13. Zhu, L., et al. (2019). Deep Leakage from Gradients. *NeurIPS*.

14. CER Smart Metering Project (data access via ISSDA). Commission for Energy Regulation, Ireland. https://www.ucd.ie/issda/data/commissionforenergyregulationcer/

15. Flower Framework documentation. https://flower.ai/docs/

16. FedSTGraph repository. https://github.com/yang-pz/FedSTGraph

17. Regulation (EU) 2024/1689 (Artificial Intelligence Act). Official Journal of the European Union / EUR-Lex. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
