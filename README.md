# Real-Time Satellite Telemetry Anomaly Detection (OPSSAT-AD)

A hands-on MLOps learning project focused on building an end-to-end anomaly detection pipeline for satellite telemetry streams.

Using real telemetry data from the European Space Agency's **OPS-SAT** mission, this repository documents my process of learning how to take a machine learning model from local experimentation to cloud deployment with the purpose of practicing continuous time-series windowing, experiment tracking with **MLflow**, stateful inference with **FastAPI** and **Redis**, containerization with **Docker**, and orchestration on **Kubernetes (AKS)**.

## Learning Goals & Project Overview

Rather than stopping at a static model in a Jupyter notebook, the goal of this project is to explore and solve practical engineering challenges that come up when running ML models on live telemetry:

* **Dynamic Polling Rates:** Telemetry channels dynamically switch between $1\text{ Hz}$ ($\text{sampling} = 1\text{s}$) and $0.2\text{ Hz}$ ($\text{sampling} = 5\text{s}$) depending on the spacecraft's operational mode.

* **Stream Discontinuities:** Orbital passes and communication gaps create jumps in timestamps ($dt > 2.5*\text{sampling}$) that distort rolling statistics if not handled properly.

* **Training-Serving Parity:** Learning how to structure code so offline training and live API inference use the exact same preprocessing logic without skew.

* **Stateful Microservices:** Understanding how to maintain a rolling time-series buffer across stateless API replicas using an in-memory store.

The planned design processes each telemetry channel independently with a sliding window ($W = 50$ readings), labeled by the state of the **latest data point** ($y_t$), to enable immediate anomaly detection without lag or lingering false alarms. The shared window feature extraction and continuity logic are implemented; the end-to-end training and serving pipeline is still planned.

## Planned System Architecture

The diagram shows the target architecture. The simulator, FastAPI service, Redis integration, model inference, and deployment components have not been implemented yet.

```
                       [ Telemetry Simulator ]
                         (Replays train == 0)
                                  │
                                  ▼
                         [ FastAPI Gateway ]
                                  │
                                  ▼
           ┌───────────────────────────────────────────────┐
           │                 Redis Cache                   │
           │  • Keys: window:{channel_id}                  │
           │  • Sliding Buffer: LTRIM -50 -1               │
           │  • Reset Trigger: dt > 2.5 × sampling         │
           └──────────────────────┬────────────────────────┘
                                  │ (50-point buffer)
                                  ▼
                    [ Feature Extraction Engine ]
                    (Rate-normalized slope, stats)
                                  │
                                  ▼
                    [ Inference Model (MLflow) ]
              (RandomForest Baseline ➔ FCNN Upgrade)
                                  │
                                  ▼
                  { "anomaly": 1/0, "confidence": 0.94 } (planned)

```

### Current Implementation and Planned Concepts

1. **Stream continuity (implemented):** Readings are continuous when the channel and sampling rate match and the timestamp gap is no greater than $2.5 \times \text{sampling}$ (2.5 seconds at sampling 1; 12.5 seconds at sampling 5). A sampling-rate change or a larger gap signals a buffer reset; comparing readings from different channels raises an error. Dataset run segmentation is planned.

2. **Shared window features (implemented):** `features.py` extracts named statistics (`last`, `mean`, `std`, `min`, `max`, quartiles, and `last_minus_mean`), rate differences, slope, sampling rate, buffer fill, and a one-hot channel encoding. `feature_dict_to_array` converts them to a NumPy vector in the canonical feature order. Training and serving are intended to reuse this logic.

3. **Leakage-free time-series validation (planned):** Use `StratifiedGroupKFold` grouped by continuous telemetry `run_id` sequences instead of random splits, to avoid leaking overlapping sliding windows between training and validation folds.

## Iterative Modeling Plan

The following target metrics are goals, not results measured by this project. Published OPSSAT-AD benchmarks report that supervised models can outperform unsupervised baselines ($F_1 \approx 0.29$ for Isolation Forest vs. $F_1 \ge 0.90$ for supervised classifiers). Instead of jumping straight to the most complex model, the planned work follows a two-step progression to practice model versioning:

| Phase | Model | Target Metrics | Learning Focus | 
| ----- | ----- | ----- | ----- | 
| **Phase 1–3 (v1)** | `RandomForestClassifier` | $F_1 \approx 0.88$, $AUC_{PR} \ge 0.90$ | Build the end-to-end pipeline quickly with fast local training and a slim Docker image ($< 300\text{ MB}$). | 
| **Phase 4 (v2)** | Fully Connected Neural Network (`FCNN`) | $F_1 \ge 0.94$, $AUC_{PR} \ge 0.97$ | Practice promoting a v2 model in the MLflow Registry, and rolling it out on Kubernetes without downtime. | 

##  Learning Roadmap

* **Phase 1: Data Exploration, Feature Engineering & Baseline Model**

  * Implement continuous run segmentation and rate-normalized windowing.

  * Validate models cleanly using `StratifiedGroupKFold` grouped by `run_id`.

  * Track experiments and metrics ($AUC_{PR}$, $F_1$, Precision, Recall, MCC) locally with MLflow.

* **Phase 2: Containerization with Docker**

  * Build a FastAPI inference service connected to Redis for sliding-window state management.

  * Package the app into a lightweight Docker image (`python:3.11-slim`) and run FastAPI + Redis together via `docker-compose`.

  * Write a Python telemetry simulator that replays `train == 0` rows chronologically over HTTP.

* **Phase 3: Local Kubernetes Orchestration**

  * Deploy FastAPI and Redis to a local Minikube cluster using Deployments, Services, and ConfigMaps.

  * Experiment with Kubernetes self-healing by deleting pods mid-stream and verifying recovery.

* **Phase 4: Cloud Deployment on Azure & CI/CD**

  * Provision Azure Container Registry (ACR) and Azure Kubernetes Service (AKS).

  * Build a GitHub Actions workflow to automate testing, Docker image builds, and rolling updates to AKS.

## Planned Tech Stack

* **Language & ML:** Python, Pandas, NumPy, Scikit-learn, PyTorch

* **Experiment Tracking:** MLflow

* **API & State Store:** FastAPI, Uvicorn, Redis

* **Containers & Orchestration:** Docker, Docker Compose, Kubernetes (Minikube / AKS)

* **Cloud & Automation:** Microsoft Azure (ACR, AKS), GitHub Actions