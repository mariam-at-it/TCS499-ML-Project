# Simulation-Based Evaluation of Federated Learning for Smart City Traffic and Energy Prediction

## Project Proposal

**Prepared by:** [Your Name]
**Date:** February 2026
**Institution:** [Your Institution]

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

Smart cities generate massive volumes of data from distributed infrastructure — traffic sensors, smart meters, environmental monitors, and connected devices. This data can power machine learning models that optimize traffic flow and forecast energy demand.

However, traditional centralized machine learning requires aggregating raw data from all sources into a single location. In modern smart cities, this is increasingly impractical for three reasons:

**1. Privacy and Regulation.** The GDPR (Articles 5, 6, and 44–49) restricts how personal data — including movement patterns and household energy usage — can be collected, processed, and transferred across organizational boundaries. The EU AI Act (Regulation 2024/1689, with high-risk system obligations taking effect August 2026) adds compliance requirements for AI systems used in critical infrastructure such as traffic management. Together, these regulations create strong incentives for approaches that minimize raw data movement.

**2. Data Silos.** Smart city data is owned by different organizations: municipal transport agencies, private utility companies, and commercial IoT providers. These entities are often unwilling or legally unable to share raw data with each other, even when collaboration would improve model quality.

**3. Communication Cost.** Transmitting high-frequency sensor data from hundreds of devices to a central server is expensive in terms of bandwidth. For example, 325 traffic sensors reporting at 5-minute intervals generate continuous data streams that must be stored, transmitted, and processed centrally — a cost that scales linearly with the number of sensors and reporting frequency.

### 1.2 The Problem

> How can multiple smart city stakeholders collaboratively train useful ML models for traffic prediction and energy forecasting **without sharing their raw data**?

### 1.3 Proposed Solution

**Federated Learning (FL)** addresses this by training models locally on each data source and sharing only model updates (weights/gradients), never raw data. A central server aggregates these updates into a global model.

However, FL introduces its own challenges: accuracy degrades under non-IID (non-identically distributed) data, differential privacy mechanisms add noise that further reduces performance, and the communication overhead of exchanging model updates across many rounds is nontrivial. Whether these trade-offs are acceptable for real smart city tasks is largely unknown — most FL research benchmarks on image classification datasets (CIFAR, MNIST) rather than urban time-series data.

This project applies FL to two concrete smart city domains — **traffic flow prediction** and **energy demand forecasting** — and rigorously evaluates the accuracy, communication cost, privacy trade-offs, and system-level bottlenecks through simulation.

---

## 2. Project Objectives

### 2.1 Goal

Apply existing Federated Learning techniques to smart city traffic and energy prediction tasks, systematically evaluate performance under realistic non-IID conditions and privacy constraints, and report what works, what doesn't, and what practical challenges arise.

### 2.2 Specific Objectives

| ID | Objective | How We'll Know It Worked |
|----|-----------|--------------------------|
| **O1** | Implement FL-based traffic flow prediction using FedAvg and FedProx on the Flower framework | Working pipeline that trains across simulated clients and produces predictions |
| **O2** | Implement FL-based energy demand forecasting on distributed household data | Working pipeline comparable to O1 for energy domain |
| **O3** | Measure the accuracy gap between FL and centralized training under realistic non-IID data | Published accuracy comparison tables across multiple heterogeneity levels (Dirichlet β ∈ {0.1, 0.5, 1.0, 10.0}) |
| **O4** | Profile system-level costs: communication overhead, wall-clock training time, memory usage, and convergence behavior | Quantified cost tables comparing FL vs. centralized training across all metrics |
| **O5** | Evaluate differential privacy (via Flower + Opacus) and document privacy-accuracy trade-offs | Privacy-accuracy curves across ε ∈ {2, 4, 8} with formal accounting |

### 2.3 Research Questions

1. **RQ1:** How much accuracy do we lose compared to centralized training when data is non-IID across city zones, and at what heterogeneity level does FL become impractical?
2. **RQ2:** What is the practical privacy-accuracy trade-off when adding differential privacy to the FL pipeline?
3. **RQ3:** What are the system-level costs (communication bytes, wall-clock time, memory) of FL training compared to centralized training for urban time-series models?
4. **RQ4:** What are the main practical bottlenecks in FL for smart city tasks — communication overhead, non-IID convergence degradation, straggler effects, or privacy noise?

---

## 3. Project Significance

### 3.1 Practical Significance

Most FL research stays in simulation on image datasets and stops at reporting accuracy. This project focuses on **what actually happens when you apply FL to realistic urban time-series prediction tasks** — the accuracy gaps under heterogeneous city data, the communication overhead, and the privacy costs. This kind of systems-aware empirical evaluation is exactly what cities and practitioners need to make informed adoption decisions.

Specific practical contributions:
- **Deployment-readiness assessment** documenting the conditions under which FL is viable for traffic and energy prediction
- **Honest benchmarks** showing where FL works well and where it struggles compared to centralized training
- **Communication and compute cost analysis** quantifying the real overhead of FL for urban time-series workloads

### 3.2 Societal Significance

Federated Learning lets cities build intelligent systems without mass data collection. If this project demonstrates that FL achieves acceptable accuracy for traffic and energy prediction, it provides a privacy-respecting alternative to the current paradigm of centralizing citizen data. If FL falls short, documenting where and why is equally valuable — it prevents cities from adopting an unsuitable technology based on hype.

### 3.3 Academic Significance

While FL algorithms are well-studied, **systems-aware evaluations on real smart city datasets are scarce**. Most papers benchmark on CIFAR/MNIST and report only accuracy. This project contributes empirical evidence on:
- How non-IID urban data affects FL convergence in practice (not just in theory)
- The actual communication cost of FL vs. centralized training for time-series models
- Privacy-accuracy trade-offs with formal DP accounting on urban datasets
- Cross-domain comparison (traffic vs. energy) revealing whether FL viability is task-dependent

---

## 4. Supportive Literature

### 4.1 Federated Learning Foundations

| Reference | Key Contribution | Relevance to This Project |
|-----------|------------------|---------------------------|
| McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS.* | Introduced FedAvg — the core FL algorithm | We use FedAvg as our primary aggregation strategy |
| Li et al. (2020). "Federated Optimization in Heterogeneous Networks." *MLSys.* | FedProx — handles non-IID data and stragglers via proximal term | We use FedProx to handle heterogeneous city data |
| Kairouz et al. (2021). "Advances and Open Problems in Federated Learning." *Foundations and Trends in ML.* | Comprehensive survey of FL challenges | Frames our research questions within known open problems |

### 4.2 FL for Smart Cities and IoT

| Reference | Key Contribution | Relevance |
|-----------|------------------|-----------|
| Ghadi et al. (2023). "Integration of FL with IoT for Smart Cities." *PeerJ Computer Science*, 9:e1657. | Maps FL onto IoT architectures for urban sensing | Informs our system architecture design |
| Delgado Fernández et al. (2025). "Scaling Smart Cities with FL: Building Energy Performance Prediction." *BISE.* | Validated FL on 25,000+ buildings for energy prediction | Directly comparable to our energy forecasting objective |
| Li et al. (2025). "FedGDAN: Privacy-preserving traffic flow prediction." *Scientific Reports* (Nature). | FL for traffic using graph neural networks | Shows FL viability for traffic prediction; we build on this |
| Belal et al. (2025). "Survey of FL Models for Spatial-Temporal Mobility." *ACM Computing Surveys.* | Survey of FL for mobility data | Validates our dataset and methodology choices |

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
| Mironov (2017). "Rényi Differential Privacy." *CSF.* | Tighter composition bounds for DP | We use Rényi DP accounting via Opacus |
| Zhu et al. (2019). "Deep Leakage from Gradients." *NeurIPS.* | Gradient inversion attacks on FL | Motivates our DP evaluation and threat model |
| IEEE Survey (2025). "FL for Urban Sensing: Attacks, Defences, and Applications." | Security landscape for urban FL | Contextualizes our threat model |

### 4.5 Research Gap

The gap we address: **most FL papers propose new algorithms and test on image classification benchmarks. Very few take existing FL methods, apply them to real urban time-series datasets, and rigorously evaluate the system-level trade-offs (accuracy, communication, privacy, convergence) that determine whether FL is actually viable for smart city deployment.** This project fills that gap with an honest, systems-aware empirical evaluation.

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
| **PyTorch** | Model training | Standard; compatible with Flower and Opacus |
| **Opacus** | Differential privacy | Provides per-sample gradient clipping + Rényi DP accounting; integrates with Flower |
| **FedSTGraph** | Federated traffic benchmark | Provides partitioning logic for METR-LA/PEMS-BAY into federated tasks |
| **psutil / PyTorch Profiler** | System profiling | Measures CPU/RAM/wall-clock during FL rounds |

### 5.3 Datasets

| Domain | Dataset | Size | Access | Client Formation |
|--------|---------|------|--------|------------------|
| **Traffic** | METR-LA (via FedSTGraph) | 207 sensors, 4 months (Mar–Jun 2012), 5-min intervals | Public | Each sensor = 1 client (geographic partitioning) |
| **Traffic** | PEMS-BAY (via FedSTGraph) | 325 sensors, ~5 months (Jan–May 2017), 5-min intervals | Public | Each sensor = 1 client (geographic partitioning) |
| **Energy** | CER Irish Smart Metering | 5,000+ households, 1.5 years (Jul 2009–Dec 2010), 30-min intervals | Public (request form) | Each household = 1 client |

**Energy dataset rationale:** We use the CER (Commission for Energy Regulation) Irish Smart Metering dataset as our primary energy dataset. Unlike the UCI Individual Household Electric Power dataset (which contains only a single household and cannot meaningfully represent multi-client FL), CER contains 5,000+ distinct households with natural inter-household heterogeneity — exactly the non-IID property that makes FL challenging and interesting.

**Fallback:** If CER access is delayed, the Low Carbon London dataset (5,567 households, 2011–2014, 30-min intervals, UK Power Networks) is a viable alternative. Both require only a request form.

**Prediction targets:**
- **Traffic:** Predict traffic speed 15/30/60 minutes ahead for each sensor
- **Energy:** Predict household energy consumption 1/6/24 hours ahead

### 5.4 Models

We use standard, well-understood architectures — the goal is to evaluate FL, not invent new models:

| Task | Model | Parameters | Rationale |
|------|-------|------------|-----------|
| Traffic prediction | GRU (2-layer, 64 hidden) | ~800K | Proven for traffic time series; efficient for large-scale FL simulation |
| Traffic prediction | STGCN (baseline comparison) | ~500K | Captures spatial dependencies between sensors |
| Energy forecasting | LSTM (2-layer, 64 hidden) | ~600K | Standard for energy time series |

**Note on STGCN:** STGCN requires a graph adjacency matrix. In our geographic partitioning (each sensor = one client), each client holds a single node's data. We handle this by sharing the global adjacency structure across clients (it is public topology, not private data) and training on each client's local node features. If this proves problematic (e.g., poor convergence due to missing neighbor data), we will document this as a finding and focus analysis on GRU/LSTM, which do not have this structural dependency.

### 5.5 FL Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Aggregation** | FedAvg (primary), FedProx (secondary) | Standard algorithms — we're evaluating FL viability, not inventing new methods |
| **Clients** | 10–20 per experiment (cross-silo) | Realistic for cross-silo city deployment where participants are institutions (transport agencies, utility companies) |
| **Local epochs** | 1, 3, 5 (compared) | Trade-off between communication rounds and local computation |
| **Rounds** | Up to 200 | Until convergence (defined as <0.1% improvement over 10 rounds) or budget exhausted |
| **Client fraction** | 50–100% per round | Cross-silo setting: few reliable institutional participants, most available each round |
| **Simulation engine** | Flower VCE | Multiplexes clients on available hardware; controls per-client CPU/GPU allocation |

**Scalability note:** For traffic datasets (207/325 sensors), we also run experiments treating each sensor as a client to evaluate FL at larger scale. These are labeled separately as "cross-device-scale" experiments.

### 5.6 Non-IID Simulation

Smart city data is naturally non-IID — a residential zone sensor has different traffic patterns than a highway sensor, and different households have different energy consumption profiles. We simulate and measure heterogeneity using:

- **Dirichlet partitioning** with β ∈ {0.1, 0.5, 1.0, 10.0} — from extreme non-IID (β=0.1) to near-IID (β=10.0)
- **Geographic partitioning** — assigning sensors to clients based on spatial proximity (using FedSTGraph's built-in partitioning for traffic)
- **Natural partitioning** — for CER energy data, each household is inherently a different distribution

**Heterogeneity quantification:** We measure the degree of non-IID using the Earth Mover's Distance (EMD) between client data distributions, reported alongside all accuracy results so readers can assess the severity of heterogeneity in each experiment.

### 5.7 Threat Model and Privacy Evaluation

#### 5.7.1 Threat Model

We assume a **honest-but-curious** server: the aggregation server follows the FL protocol correctly but may attempt to infer private information from received model updates. Known attack vectors include:

- **Gradient inversion attacks** (Zhu et al., 2019): reconstructing training data from gradients
- **Membership inference attacks**: determining whether specific data points were in a client's training set

We do **not** address Byzantine (malicious) clients — our cross-silo setting assumes trusted institutional participants.

**What DP protects against:** Differential privacy bounds the information any single client's data can contribute to the model, limiting the success of both gradient inversion and membership inference. It does **not** make these attacks impossible — it provides a mathematically quantifiable upper bound on leakage.

#### 5.7.2 Differential Privacy Configuration

We use **Flower's `DifferentialPrivacyServerSideFixedClipping`** wrapper combined with **Opacus** for formal privacy accounting:

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| **ε (privacy budget)** | 2, 4, 8 | Standard benchmarking range. ε=2 provides strong privacy (significant noise); ε=8 provides weak privacy (minimal noise). For context, Apple uses ε=2–8 for iOS telemetry; Google uses ε=2.3 for RAPPOR. |
| **δ** | 1/n² where n = number of clients | Ensures δ ≪ 1/n. For 20 clients: δ = 0.0025; for 207 sensors: δ ≈ 2.3×10⁻⁵ |
| **Clipping norm (S)** | 1.0 | Standard default; bounds per-update L2 norm |
| **Noise mechanism** | Gaussian (server-side) | Applied by Flower wrapper after clipping |
| **Privacy accountant** | Rényi DP (via Opacus) | Provides tighter composition bounds than basic composition theorem |

**What the ε values mean practically:** At ε=2, each client's contribution is heavily obscured — accuracy will degrade noticeably but privacy is strong. At ε=8, privacy protection is weaker but accuracy impact is minimal. We report the full trade-off curve and let the reader decide what ε is acceptable for their application. We do not claim any ε value is "sufficient" — sufficiency depends on the deployment context, threat model, and regulatory requirements.

**DP mechanism details for reproducibility:** Server-side fixed clipping clips each client's model update to L2 norm ≤ S before aggregation. Gaussian noise with σ proportional to S/ε is added to the aggregated update. The total privacy budget across T rounds is tracked via Rényi DP composition (Mironov, 2017), which yields tighter bounds than naive linear composition.

### 5.8 Centralized Baseline

To ensure fair comparison, the centralized baseline uses:
- **Same model architecture** (identical hyperparameters)
- **Same total data** (union of all clients' datasets)
- **Same total computation** (total epochs = FL rounds × local epochs × client fraction)
- **Same preprocessing** (identical normalization, windowing, train/val/test splits)
- **Same early stopping** (patience = 10 epochs on validation loss)
- **Same random seeds** (3 seeds per experiment; we report mean ± std)

This ensures the FL vs. centralized gap is attributable to federation, not training budget differences.

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
| Fairness | Accuracy variance across clients (std, min, max) | Does FL hurt some zones/households more than others? |

### 5.10 Communication Cost Analysis

FL communication cost is **not** inherently lower than centralized training — it depends on model size, number of rounds, and client participation:

- **FL cost:** Each round transfers model weights (upload + download) for participating clients. For an 800K-parameter model at FP32: ~3.2 MB per client per round × 2 (up + down) × C clients × R rounds.
- **Centralized cost:** Raw data transferred once from all sources to the central server.

We compute and compare both quantities explicitly. FL may transfer **more** total bytes than centralized training for small datasets with large models. The advantage of FL is that it avoids transferring **raw data**, which is the privacy-relevant comparison — but we report both the byte count and the privacy distinction honestly.

### 5.11 Compute Feasibility

All experiments run on a single workstation. Estimated compute budget:

| Experiment | Clients | Rounds | Est. Time per Run | Runs | Total |
|------------|---------|--------|-------------------|------|-------|
| Traffic (GRU, FedAvg) | 20 | 200 | ~2 hours | 12 (4 β × 3 seeds) | ~24 hrs |
| Traffic (GRU, FedProx) | 20 | 200 | ~2 hours | 12 | ~24 hrs |
| Traffic (STGCN) | 20 | 200 | ~3 hours | 12 | ~36 hrs |
| Energy (LSTM, FedAvg) | 20 | 200 | ~1.5 hours | 12 | ~18 hrs |
| Energy (LSTM, FedProx) | 20 | 200 | ~1.5 hours | 12 | ~18 hrs |
| DP experiments | 20 | 200 | ~2 hours | 9 (3 ε × 3 seeds) | ~18 hrs |
| Large-scale (207/325 sensors) | 207/325 | 100 | ~6 hours | 6 | ~36 hrs |
| Centralized baselines | — | — | ~1 hour | 15 | ~15 hrs |
| **Total estimated** | | | | | **~190 hrs** |

This is feasible on a single GPU workstation over ~2–3 weeks of compute. If hardware is limited, we prioritize the core experiments (FedAvg/FedProx × traffic/energy × non-IID sweep) and run the large-scale and DP experiments with fewer seeds.

### 5.12 Reproducibility

- All code open-sourced on GitHub
- Hydra/YAML configs for every experiment
- Fixed random seeds (3 per experiment)
- Docker containers for environment consistency
- FedSTGraph provides deterministic data partitioning for traffic datasets

---

## 6. Research Scope

### 6.1 What This Project Does

- Applies **existing** FL algorithms (FedAvg, FedProx) to traffic and energy prediction
- Uses **existing** privacy mechanisms (Flower's DP wrapper + Opacus accounting)
- Evaluates **all experiments in simulation** using Flower's Virtual Client Engine
- Measures system-level costs (communication, compute, memory) alongside accuracy
- Reports honestly, including negative results
- Produces a **deployment-readiness assessment** for practitioners

### 6.2 What This Project Does NOT Do

| Excluded | Why |
|----------|-----|
| Invent new FL algorithms | Not the focus — we evaluate existing methods |
| Formal convergence proofs | Applied project, not theoretical |
| Video or image data | Scope control — we focus on time-series sensor data |
| Hardware deployment (edge devices) | All evaluation is simulation-based; hardware validation is future work |
| Byzantine fault tolerance | Assumes trusted institutional participants (cross-silo) |
| Secure Aggregation | Adds complexity without advancing our core questions |
| Vertical FL | Different setting; out of scope |
| Communication compression techniques | We measure baseline FL cost; compression is an optimization layer, not our research question |

### 6.3 Assumptions

1. Participants are **known institutions** (city agencies, utility companies) — not anonymous devices
2. Participants are **honest** (no adversarial behavior — justified by cross-silo trust model)
3. All clients are available each round (cross-silo: few, reliable participants)
4. Simulation accurately represents the computation and communication patterns of a real FL deployment (standard assumption in FL research)

### 6.4 Limitations

1. We use public datasets partitioned to simulate silos — not actual multi-stakeholder data from a real city
2. All results are simulation-based; real-world deployment may surface additional challenges (network variability, hardware constraints, operational issues)
3. Privacy evaluation uses basic DP (Gaussian mechanism with server-side clipping) — more sophisticated methods (Secure Aggregation, Homomorphic Encryption) are out of scope
4. Results may not generalize to all cities, data types, or FL configurations beyond those tested
5. STGCN results may be limited by the graph partitioning challenge (see Section 5.4)

---

## 7. Project Outcomes

### 7.1 Deliverables

| # | Deliverable | Description | Month |
|---|-------------|-------------|-------|
| D1 | FL Training Pipeline | Working Flower-based FL pipeline for traffic and energy prediction, with Hydra configs | 8 |
| D2 | Evaluation Report | Comprehensive accuracy, communication, privacy, and convergence benchmarks across all experimental conditions | 14 |
| D3 | Deployment-Readiness Assessment | Analysis of compute/bandwidth requirements, non-IID sensitivity, and DP impact — with concrete recommendations on when FL is/isn't viable for smart city tasks | 18 |
| D4 | Open-Source Repository | All code, configs, experiment logs, and analysis scripts on GitHub with documentation | 18 |

### 7.2 Publications

| # | Publication | Venue Type | Timeline |
|---|-------------|------------|----------|
| P1 | "Federated Learning for Smart City Traffic and Energy Prediction: A Systems-Aware Evaluation" | Conference (SenSys / IoTDI / BuildSys) | Submit Month 14 |
| P2 | "How Viable is Federated Learning for Urban Time-Series Prediction? Lessons from Simulation" | Workshop (FL@FM / FedVision / Urban Computing) | Submit Month 16 |

Two publications is realistic for an 18-month research project. A third (journal extension) may follow if results warrant it.

### 7.3 Expected Results

We **do not promise specific accuracy numbers** — we report what we find. However, based on prior work, we expect:

| Metric | Expected Range | Based On |
|--------|----------------|----------|
| FL vs. centralized accuracy gap (near-IID, β=10) | 1–3% | Delgado Fernández et al. (2025) |
| FL vs. centralized accuracy gap (extreme non-IID, β=0.1) | 5–15% | Li et al. (2020), Jimenez et al. (2024) |
| Accuracy with DP (ε=4) | 5–15% additional degradation | Abadi et al. (2016) |
| Communication cost (FedAvg, 800K model, 200 rounds, 20 clients) | ~25 GB total transfer | Computed: 3.2MB × 2 × 20 × 200 |
| Convergence | 50–100 rounds to 95% final accuracy | Typical for FedAvg on time-series |

If results are worse than expected, **we report that too** — negative results about where FL fails are equally valuable for practitioners.

### 7.4 Contributions

This is not an algorithmic contribution. Our contributions are:

1. **Empirical evidence** on FL performance for traffic and energy prediction under realistic non-IID conditions
2. **Systems-level cost analysis** quantifying the communication, compute, and convergence overhead of FL vs. centralized training
3. **Privacy-accuracy trade-off data** with formal Rényi DP accounting for cities evaluating privacy-preserving ML
4. **Cross-domain comparison** revealing whether FL viability differs between traffic and energy prediction tasks
5. **Deployment-readiness assessment** with concrete recommendations for practitioners
6. **Open-source benchmark** that others can reproduce and extend

---

## 8. Project Timeline

### 8.1 Overview (18 Months)

```
Q1         Q2         Q3         Q4         Q5         Q6
|----------|----------|----------|----------|----------|-----|

Phase 1: Setup & Literature
██████████
[M1-M3]

Phase 2: Build FL Pipelines
       ██████████████████
       [M3-M8]

Phase 3: Core Experiments
                  ██████████████████████
                  [M7-M14]

Phase 4: Systems Analysis & Ablations
                            ██████████████████
                            [M11-M16]

Phase 5: Writing & Release (overlaps with 3 & 4)
                                 ██████████████████
                                 [M13-M18]

Milestones:
    ◆ M3:  Environment ready, datasets loaded, centralized baselines
    ◆ M8:  FL pipelines working for both domains
    ◆ M14: Core experiments complete, P1 submitted
    ◆ M16: Systems analysis complete, P2 submitted
    ◆ M18: Assessment published, repository released
```

### 8.2 Detailed Plan

| Phase | Months | What Happens | Deliverable |
|-------|--------|--------------|-------------|
| **1. Setup** | 1–3 | Literature review; set up Flower + FedSTGraph + Opacus; acquire CER dataset; establish centralized baselines for traffic and energy; validate data loading and preprocessing | Centralized baselines; working dev environment |
| **2. Build** | 3–8 | Implement FedAvg and FedProx pipelines for traffic (GRU, STGCN) and energy (LSTM); integrate Flower DP wrapper; test on small scale (5 clients, 20 rounds); validate correctness against centralized | D1: Working FL pipeline |
| **3. Experiments** | 7–14 | Run core experiments: non-IID sweep (4 β values × 3 seeds), DP sweep (3 ε values × 3 seeds), centralized comparisons; fairness analysis across clients | D2: Evaluation report; P1 submitted |
| **4. Systems Analysis** | 11–16 | Communication cost analysis; convergence profiling; large-scale experiments (207/325 sensors); compute budget measurements; ablation studies (local epochs, client fraction, model width) | D3: Deployment-readiness assessment |
| **5. Write & Release** | 13–18 | Write deployment-readiness assessment; prepare workshop paper; open-source release; documentation; reproducibility verification | D4: Repository; P2 submitted |

### 8.3 Milestones

| Milestone | Month | Go / No-Go |
|-----------|-------|------------|
| **Environment ready** | 3 | If CER dataset unavailable → use Low Carbon London fallback |
| **FL pipelines working** | 8 | If accuracy gap >25% → investigate root cause and document as finding |
| **Core experiments complete** | 14 | Submit paper regardless of whether results are positive or negative |
| **Systems analysis complete** | 16 | If compute budget exceeded → reduce experiment grid and document |

### 8.4 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CER / energy dataset access delayed | Medium | Low | Low Carbon London is a ready fallback; both require only a request form |
| FL accuracy much worse than centralized | Low | Medium | This IS a finding — report it honestly |
| DP causes non-convergence at ε=2 | Medium | Low | Expected; we also test ε=4 and ε=8 to show the trade-off curve |
| Compute budget insufficient for full grid | Low | Medium | Prioritize core experiments; reduce seeds from 3 to 2 if needed |
| STGCN fails under geographic partitioning | Medium | Low | GRU/LSTM results are the primary contribution; STGCN is a comparison |
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

7. Belal, Y., et al. (2025). Survey of FL Models for Spatial-Temporal Mobility. *ACM Computing Surveys*.

8. Jimenez, D.M., et al. (2024). Non-IID Data in FL: Survey with Taxonomy, Metrics, and Methods. *arXiv:2411.12377*.

9. Abadi, M., et al. (2016). Deep Learning with Differential Privacy. *ACM CCS*.

10. Mironov, I. (2017). Rényi Differential Privacy. *IEEE CSF*.

11. Jian, L. & Liu, D. (2025). Widening the Network Mitigates Data Heterogeneity on FedAvg. *ICML 2025*.

12. Zakerinia, H., et al. (2024). Communication-Efficient FL With Data and Client Heterogeneity. *AISTATS*.

13. Zhu, L., et al. (2019). Deep Leakage from Gradients. *NeurIPS*.

14. CER Smart Metering Project (2012). Commission for Energy Regulation, Ireland. https://www.ucd.ie/issda/data/commissionforenergyregulationcer/

15. Flower Framework (2026). https://flower.ai/docs/

16. FedSTGraph Benchmark (2025). https://github.com/yang-pz/FedSTGraph

---

**End of Proposal**
