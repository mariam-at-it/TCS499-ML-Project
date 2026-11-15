# Federated Learning for Smart City Traffic and Energy Prediction: A Deployment-Focused Study

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

Smart cities generate massive volumes of data from distributed infrastructure — traffic sensors, smart meters, environmental monitors, and connected devices. This data can power machine learning models that optimize traffic flow, forecast energy demand, and improve urban services.

However, traditional centralized machine learning requires aggregating raw data from all sources into a single location. In modern smart cities, this is increasingly impractical for three reasons:

**1. Privacy and Regulation.** Laws such as the GDPR, the EU AI Act (2025-2026), and CCPA restrict how citizen data can be collected, moved, and stored. Smart city data — movement patterns, household energy usage, health metrics — is inherently sensitive.

**2. Data Silos.** Smart city data is owned by different organizations: municipal transport agencies, private utility companies, hospital networks, and commercial IoT providers. These entities are unwilling or legally unable to share raw data with each other.

**3. Bandwidth and Latency.** Transmitting high-frequency sensor data from thousands of devices to a central cloud is expensive and slow. Real-time applications like traffic management cannot tolerate the round-trip delay.

### 1.2 The Problem

> How can multiple smart city stakeholders collaboratively train useful ML models for traffic prediction and energy forecasting **without sharing their raw data**?

### 1.3 Proposed Solution

**Federated Learning (FL)** solves this by training models locally on each data source and sharing only model updates (weights/gradients), never raw data. The central server aggregates these updates into a global model. This project applies FL to two concrete smart city domains — **traffic flow prediction** and **energy demand forecasting** — and evaluates whether it works well enough for practical deployment.

---

## 2. Project Objectives

### 2.1 Goal

Apply existing Federated Learning techniques to smart city traffic and energy prediction tasks, deploy on edge hardware, and report what works, what doesn't, and what practical challenges arise.

### 2.2 Specific Objectives

| ID | Objective | How We'll Know It Worked |
|----|-----------|--------------------------|
| **O1** | Implement FL-based traffic flow prediction using FedAvg and FedProx on the Flower framework | Working pipeline that trains across 50+ simulated clients and produces predictions |
| **O2** | Implement FL-based energy demand forecasting on distributed household data | Working pipeline comparable to O1 for energy domain |
| **O3** | Measure the accuracy gap between FL and centralized training under realistic non-IID data | Published accuracy comparison tables across multiple heterogeneity levels |
| **O4** | Deploy and benchmark the trained models on edge hardware (NVIDIA Jetson, Raspberry Pi) | Measured inference latency and resource usage on real devices |
| **O5** | Evaluate basic differential privacy (Flower's built-in DP) and document privacy-accuracy trade-offs | Privacy-accuracy curves across ε ∈ {2, 4, 8} |

### 2.3 Research Questions

1. **RQ1:** How much accuracy do we lose compared to centralized training when data is non-IID across city zones?
2. **RQ2:** What is the practical privacy-accuracy trade-off when adding differential privacy to the FL pipeline?
3. **RQ3:** Can FL models run on low-cost edge hardware (Jetson, RPi) with acceptable latency for real-time city applications?
4. **RQ4:** What are the main practical bottlenecks when deploying FL in a smart city setting?

---

## 3. Project Significance

### 3.1 Practical Significance

Most FL research stays in simulation. This project focuses on **what actually happens when you try to deploy FL for real urban tasks** — the accuracy gaps, the hardware limitations, the engineering friction. This kind of empirical deployment knowledge is exactly what cities and practitioners need but rarely find in academic papers.

Specific practical contributions:
- **Deployment guide** for cities considering FL adoption
- **Honest benchmarks** showing where FL works well and where it struggles
- **Edge deployment validation** proving (or disproving) that FL models can run on affordable hardware

### 3.2 Societal Significance

Federated Learning lets cities build intelligent systems without mass data collection. If this project demonstrates that FL achieves acceptable accuracy for traffic and energy prediction, it provides a privacy-respecting alternative to the current paradigm of centralizing citizen data.

### 3.3 Academic Significance

While FL algorithms are well-studied, **deployment-focused evaluations on real smart city datasets are scarce**. Most papers benchmark on CIFAR/MNIST and stop at simulation. This project contributes empirical evidence on:
- How non-IID urban data affects FL convergence in practice
- Real-world resource constraints on edge devices
- The gap between simulated and hardware-deployed FL performance

---

## 4. Supportive Literature

### 4.1 Federated Learning Foundations

| Reference | Key Contribution | Relevance to This Project |
|-----------|------------------|---------------------------|
| McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS.* | Introduced FedAvg — the core FL algorithm | We use FedAvg as our primary baseline |
| Li et al. (2020). "Federated Optimization in Heterogeneous Networks." *MLSys.* | FedProx — handles non-IID data and stragglers via proximal term | We use FedProx to handle heterogeneous city data |
| Kairouz et al. (2021). "Advances and Open Problems in Federated Learning." *Foundations and Trends in ML.* | Comprehensive survey of FL challenges | Frames our research questions within known open problems |

### 4.2 FL for Smart Cities and IoT

| Reference | Key Contribution | Relevance |
|-----------|------------------|-----------|
| Ghadi (2023). "Integration of FL with IoT for Smart Cities." *J. Sensor Actuator Networks.* | Maps FL onto IoT architectures for urban sensing | Informs our system architecture design |
| Fridgen et al. (2025). "Scaling Smart Cities with FL: Building Energy Performance Prediction." *BISE.* | Validated FL on 25,000+ buildings for energy prediction | Directly comparable to our energy forecasting objective |
| Li et al. (2025). "FedGDAN: Privacy-preserving traffic flow prediction." *Nature Sci. Reports.* | FL for traffic using graph neural networks | Shows FL viability for traffic prediction; we build on this |
| Belal et al. (2025). "Survey of FL Models for Spatial-Temporal Mobility." *ACM Computing Surveys.* | Survey of FL for mobility data | Validates our dataset and methodology choices |

### 4.3 Non-IID Data and Practical Challenges

| Reference | Key Contribution | Relevance |
|-----------|------------------|-----------|
| Jimenez et al. (2024). "Non-IID Data in FL: Survey with Taxonomy and Methods." | Taxonomy of data heterogeneity types | Guides our non-IID simulation methodology |
| Jian & Liu (2025). "Widening the Network Mitigates Data Heterogeneity on FedAvg." | Wider networks help under non-IID | Informs our model architecture choices |
| Zakerinia et al. (2024). "Communication-Efficient FL With Data and Client Heterogeneity." *AISTATS.* | Handles async updates + compression | Relevant if communication becomes a bottleneck |

### 4.4 Privacy in Federated Learning

| Reference | Key Contribution | Relevance |
|-----------|------------------|-----------|
| Abadi et al. (2016). "Deep Learning with Differential Privacy." *CCS.* | DP-SGD and moments accountant | Foundation for our DP integration |
| Mironov (2017). "Rényi Differential Privacy." *CSF.* | Tighter composition bounds for DP | We use Rényi DP accounting |
| IEEE Survey (2025). "FL for Urban Sensing: Attacks, Defences, and Applications." | Security landscape for urban FL | Contextualizes our threat assumptions |

### 4.5 Research Gap

The gap we address is simple: **most FL papers propose new algorithms and test on CIFAR. Very few take existing FL methods, apply them to real urban datasets, deploy on edge hardware, and report what happens.** This project fills that gap with an honest, deployment-focused evaluation.

---

## 5. Research Methodology

### 5.1 Approach

We use a **build-measure-report** methodology:
1. **Build** FL pipelines for traffic and energy prediction using Flower
2. **Measure** accuracy, communication cost, privacy trade-offs, and edge performance
3. **Report** findings including negative results and practical lessons

### 5.2 Framework and Tools

| Tool | Role | Why This One |
|------|------|--------------|
| **Flower (flwr)** | FL framework | Mature, supports simulation and real edge deployment, built-in FedAvg/FedProx/DP |
| **PyTorch** | Model training | Standard; compatible with Flower |
| **FedSTGraph** | Federated dataset benchmark | Provides pre-partitioned METR-LA/PEMS-BAY for FL |
| **NVIDIA Jetson Orin Nano** | Edge device (primary) | Representative of real smart city edge hardware |
| **Raspberry Pi 5** | Edge device (secondary) | Low-cost alternative for cost-sensitive deployments |

### 5.3 Datasets

| Domain | Dataset | Size | Access |
|--------|---------|------|--------|
| **Traffic** | METR-LA (via FedSTGraph) | 207 sensors, 4 months, 5-min intervals | Public |
| **Traffic** | PEMS-BAY (via FedSTGraph) | 325 sensors, 6 months, 5-min intervals | Public |
| **Energy** | UCI Household Power Consumption | 4 years, 1-min intervals | Public |
| **Energy** | Pecan Street Dataport | 25,000+ homes, 15-min intervals | Requires DUA |

**If Pecan Street access is denied:** We will use UCI Household Power as primary energy dataset and partition it to simulate multi-household FL. This is documented as a limitation.

### 5.4 Models

We use standard, well-understood architectures — the goal is to evaluate FL deployment, not invent new models:

| Task | Model | Parameters | Rationale |
|------|-------|------------|-----------|
| Traffic prediction | GRU (2-layer) | ~800K | Proven for traffic time series; lightweight for edge |
| Traffic prediction | STGCN (baseline comparison) | ~500K | Captures spatial dependencies between sensors |
| Energy forecasting | LSTM (2-layer) | ~600K | Standard for energy time series |

All models are small enough for edge deployment. We will quantize (INT8) for Jetson (TensorRT) and RPi (TFLite).

### 5.5 FL Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Aggregation** | FedAvg (primary), FedProx (secondary) | Standard algorithms — we're evaluating deployment, not inventing new ones |
| **Clients** | 20-50 simulated | Realistic for cross-silo city deployment |
| **Local epochs** | 1, 3, 5 (compared) | Trade-off between communication and convergence |
| **Rounds** | Up to 200 | Until convergence or budget exhausted |
| **Client fraction** | 50-100% per round | Cross-silo setting (few reliable participants) |

### 5.6 Non-IID Simulation

Smart city data is naturally non-IID (a residential zone sensor has different patterns than a highway sensor). We simulate this using:

- **Dirichlet partitioning** with β ∈ {0.1, 0.5, 1.0, 10.0} — from extreme to near-IID
- **Geographic partitioning** — assigning sensors to clients based on spatial proximity (using FedSTGraph's built-in partitioning)

### 5.7 Privacy Evaluation

We use **Flower's built-in Differential Privacy** module (`DifferentialPrivacyServerSideFixedClipping`):

| Parameter | Values Tested |
|-----------|---------------|
| **ε (privacy budget)** | 2, 4, 8 |
| **δ** | 10⁻⁵ |
| **Clipping norm** | 1.0 |

We report accuracy at each ε level and document the privacy-accuracy trade-off curve. No claims about "sufficient" privacy — we present the data and let the reader decide based on their requirements.

### 5.8 Evaluation Metrics

| What We Measure | Metric | Why |
|-----------------|--------|-----|
| Prediction accuracy | MAE, RMSE | Standard for traffic/energy regression |
| FL vs centralized gap | Δ accuracy | Core question: how much do we lose? |
| Communication cost | Total bytes transferred | Practical deployment concern |
| Privacy cost | Accuracy at ε = {2, 4, 8} | Privacy-accuracy trade-off |
| Edge inference speed | Latency (ms) | Can it run in real-time? |
| Edge resource usage | RAM, CPU%, power draw | Practical for real deployment |
| Convergence speed | Rounds to 95% of final accuracy | How long does training take? |
| Fairness | Accuracy variance across clients | Does FL hurt some zones more than others? |

### 5.9 Reproducibility

- All code open-sourced on GitHub
- Hydra/YAML configs for every experiment
- Fixed random seeds
- Docker containers for environment consistency
- FedSTGraph provides deterministic data partitioning

---

## 6. Research Scope

### 6.1 What This Project Does

- Applies **existing** FL algorithms (FedAvg, FedProx) to traffic and energy prediction
- Uses **existing** privacy mechanisms (Flower's built-in DP)
- Deploys on **real edge hardware** (Jetson, RPi)
- Measures everything and reports honestly, including negative results
- Produces a **deployment guide** for practitioners

### 6.2 What This Project Does NOT Do

| Excluded | Why |
|----------|-----|
| Invent new FL algorithms | Not the focus — we evaluate existing methods |
| Formal convergence proofs | Applied project, not theoretical |
| Video or image data | Scope control — we focus on time-series sensor data |
| Full city-scale production deployment | We validate with prototypes; production is follow-on work |
| Byzantine fault tolerance | Assumes trusted institutional participants (cross-silo) |
| Secure Aggregation | Adds complexity without advancing our core questions |
| Vertical FL | Different setting; out of scope |

### 6.3 Assumptions

1. Participants are **known institutions** (city agencies, utility companies) — not anonymous devices
2. Participants are **honest** (no adversarial behavior — this is cross-silo, not cross-device)
3. Network connectivity exists between participants and aggregator
4. Edge devices have at least Raspberry Pi 4-level compute

### 6.4 Limitations

1. We use public datasets partitioned to simulate silos — not actual multi-stakeholder data
2. Edge prototype is small-scale (2-5 devices), not a full city deployment
3. Privacy evaluation uses basic DP — more sophisticated methods (SecAgg, HE) are out of scope
4. Results may not generalize to all cities or data types

---

## 7. Project Outcomes

### 7.1 Deliverables

| # | Deliverable | Description | Month |
|---|-------------|-------------|-------|
| D1 | FL Training Pipeline | Working Flower-based FL pipeline for traffic and energy prediction | 8 |
| D2 | Evaluation Report | Comprehensive accuracy, communication, and privacy benchmarks | 14 |
| D3 | Edge Deployment Package | Quantized models running on Jetson and RPi with measured performance | 18 |
| D4 | Deployment Guide | Practical guide for cities considering FL adoption | 20 |
| D5 | Open-Source Repository | All code, configs, and experiment logs on GitHub | 20 |

### 7.2 Publications

| # | Publication | Venue Type | Timeline |
|---|-------------|------------|----------|
| P1 | "FL for Smart City Traffic and Energy Prediction: A Deployment Study" | Conference (SenSys / IoTDI / BuildSys) | Submit Month 14 |
| P2 | "Deploying FL on Edge Devices: Lessons from Smart City Applications" | Workshop (FL@NeurIPS or EdgeSys) | Submit Month 16 |

Two publications is realistic. A third (journal) may follow if results warrant it.

### 7.3 Expected Results

We **do not promise specific accuracy numbers** — we report what we find. However, based on prior work, we expect:

| Metric | Expected Range | Based On |
|--------|----------------|----------|
| FL vs centralized accuracy gap | 2-10% depending on non-IID level | Fridgen et al. (2025), FedGDAN (2025) |
| Accuracy with DP (ε=4) | 5-15% degradation | Abadi et al. (2016) |
| Edge inference latency (Jetson) | 30-80ms for GRU/LSTM models | TensorRT benchmarks |
| Edge inference latency (RPi) | 80-200ms for GRU/LSTM models | TFLite benchmarks |
| Communication reduction vs centralized | 80-95% (FL sends model updates, not raw data) | Inherent to FL design |

If results are worse than expected, **we report that too** — negative results about where FL fails are equally valuable for practitioners.

### 7.4 Contributions

This is not an algorithmic contribution. Our contributions are:

1. **Empirical evidence** on FL performance for traffic and energy prediction under realistic conditions
2. **Practical deployment validation** on edge hardware with measured resource constraints
3. **Privacy-accuracy trade-off data** for cities evaluating DP-enabled FL
4. **Deployment guide** synthesizing lessons learned for practitioners
5. **Open-source benchmark** that others can reproduce and extend

---

## 8. Project Timeline

### 8.1 Overview (20 Months)

```
Q1         Q2         Q3         Q4         Q5         Q6         Q7
|----------|----------|----------|----------|----------|----------|-----|

Phase 1: Setup & Literature
██████████
[M1-M3]

Phase 2: Build FL Pipelines
       ██████████████████
       [M3-M8]

Phase 3: Experiments & Evaluation
                  ██████████████████████
                  [M7-M14]

Phase 4: Edge Deployment
                            ██████████████████
                            [M11-M18]

Phase 5: Writing & Release (overlaps with 3 & 4)
                                 ██████████████████████
                                 [M12-M20]

Milestones:
    ◆ M3:  Environment ready, datasets loaded
    ◆ M8:  FL pipelines working for both domains
    ◆ M14: All experiments complete, paper submitted
    ◆ M18: Edge deployment validated
    ◆ M20: Guide published, repository released
```

### 8.2 Detailed Plan

| Phase | Months | What Happens | Deliverable |
|-------|--------|--------------|-------------|
| **1. Setup** | 1-3 | Literature review; set up Flower + FedSTGraph; acquire datasets; establish centralized baselines for traffic and energy | Centralized baselines; working dev environment |
| **2. Build** | 3-8 | Implement FedAvg and FedProx pipelines for traffic (GRU, STGCN) and energy (LSTM); integrate Flower DP; test on small scale | D1: Working FL pipeline |
| **3. Evaluate** | 7-14 | Run all experiments: non-IID sweep, DP sweep, communication measurement, fairness analysis; centralized vs FL comparison | D2: Evaluation report; P1 paper submitted |
| **4. Deploy** | 11-18 | Quantize models; deploy on Jetson and RPi; measure latency, memory, power; compare simulated vs real-device performance | D3: Edge deployment package |
| **5. Write** | 12-20 | Write deployment guide; prepare workshop paper; open-source release; documentation | D4: Guide; D5: Repository; P2 submitted |

### 8.3 Milestones

| Milestone | Month | Go / No-Go |
|-----------|-------|------------|
| **Environment ready** | 3 | If datasets unavailable → switch to public-only plan |
| **FL pipelines working** | 8 | If accuracy gap >20% → investigate and document as finding |
| **Experiments complete** | 14 | Submit paper regardless of whether results are positive or negative |
| **Edge deployment validated** | 18 | If latency >500ms → use lighter model and document |

### 8.4 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pecan Street data denied | Medium | Low | UCI Household Power is sufficient fallback |
| FL accuracy much worse than centralized | Low | Medium | This IS a finding — report it honestly |
| Edge hardware too slow | Low | Low | Use lighter models; quantize more aggressively |
| Timeline slip | Medium | Medium | Phases overlap; can compress Phase 5 |

---

## References

1. McMahan, H.B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.

2. Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks. *MLSys*.

3. Kairouz, P., et al. (2021). Advances and Open Problems in Federated Learning. *Foundations and Trends in ML*.

4. Ghadi, Y.Y. (2023). Integration of FL with IoT for Smart Cities. *J. Sensor and Actuator Networks*.

5. Fridgen, G., et al. (2025). Scaling Smart Cities with FL: Building Energy Performance Prediction. *BISE*.

6. Li, X., et al. (2025). FedGDAN: Privacy-preserving traffic flow prediction. *Nature Scientific Reports*.

7. Belal, Y., et al. (2025). Survey of FL Models for Spatial-Temporal Mobility. *ACM Computing Surveys*.

8. Jimenez, D.M., et al. (2024). Non-IID Data in FL: Survey with Taxonomy, Metrics, and Methods.

9. Abadi, M., et al. (2016). Deep Learning with Differential Privacy. *ACM CCS*.

10. Mironov, I. (2017). Rényi Differential Privacy. *IEEE CSF*.

11. Jian, L. & Liu, D. (2025). Widening the Network Mitigates Data Heterogeneity on FedAvg.

12. Zakerinia, H., et al. (2024). Communication-Efficient FL With Data and Client Heterogeneity. *AISTATS*.

13. Flower Framework (2026). https://flower.ai/docs/

14. FedSTGraph Benchmark (2025). https://github.com/ypz-git/FedSTGraph

---

**End of Proposal**
