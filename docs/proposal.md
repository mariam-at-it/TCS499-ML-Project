Developing a Privacy-Preserving Smart Energy Prediction System for Smart Cities Using Federated Learning

1 Chapter 1: Introduction

1.1 Problem Statement
Household energy consumption is increasing due to the growing use of smart appliances, IoT devices, and digitally connected services. Smart meters generate large volumes of time-series data that can be used to forecast electricity demand, improve energy efficiency, and support sustainable energy management in smart cities. Accurate household-level forecasting helps users and energy stakeholders make better operational and planning decisions.

However, many energy prediction systems use centralized machine learning, where raw household consumption data is collected and stored on a central server for training. This approach creates privacy and security risks because energy usage patterns can reveal sensitive information such as occupancy routines, lifestyle habits, and appliance usage behavior. These risks can reduce user trust and participation.

In addition, some existing approaches rely mainly on historical energy readings and underuse contextual variables (for example, weather conditions and time-based occupancy proxies such as hour-of-day, day-of-week, and optional area-level mobility/traffic indicators). Ignoring relevant context can reduce forecast quality. Many systems also stop at prediction and do not provide actionable recommendations that help users reduce consumption or shift usage.

A privacy-preserving, context-aware prediction approach is therefore needed. Federated learning (FL) enables multiple households to collaboratively train a shared model while keeping raw data on local devices. Only model updates (not raw data) are exchanged with an aggregation server. This reduces data exposure risk, although additional safeguards (for example, secure aggregation and encrypted communication) are still required to strengthen privacy.

Therefore, this project proposes a federated learning-based smart energy consumption prediction system for smart city households that integrates contextual information, preserves data privacy, and provides actionable energy-saving recommendations.

1.2 Project Objective
The main objective of this project is to design, develop, and evaluate a privacy-preserving smart energy consumption prediction prototype for smart city households using federated learning.

Specific objectives are to:
1. Develop a federated learning pipeline that trains a shared energy prediction model without transferring raw household consumption data.
2. Integrate contextual features (for example, weather and time-based occupancy proxies, with optional mobility/traffic indicators) to improve prediction quality.
3. Build a web-based dashboard that presents forecasts, historical trends, and personalized energy-saving recommendations.
4. Evaluate the system in terms of prediction accuracy, communication efficiency, usability, and privacy-preserving design.

1.3 Scope
This project focuses on the design and implementation of a privacy-preserving smart energy consumption prediction system using federated learning. The system enables multiple households (or simulated household clients) to collaboratively train prediction models without sharing raw consumption data.

Contextual features such as weather conditions and time-based occupancy proxies are incorporated to improve prediction performance. Area-level traffic/mobility indicators may be used as optional proxy features where available and relevant. An interactive web-based dashboard presents energy forecasts, historical trends, and personalized recommendations.

The project is limited to a functional prototype and experimental evaluation. It does not include city-wide production deployment, integration with national utility infrastructure, or direct control of household appliances. The system assumes the availability of smart meter data and contextual information from public APIs, open datasets, or simulated data for testing and experimentation.

1.4 Significance
This project contributes to privacy-preserving intelligent energy management by demonstrating how federated learning can support collaborative model training while reducing raw data sharing.

For households, the system can support better energy-use decisions through forecasts and recommendations, which may improve efficiency and reduce costs. For researchers and smart city stakeholders, the project provides a practical prototype and evaluation framework for applying federated learning to residential energy analytics.

More broadly, the project supports sustainable energy practices through privacy-aware, data-driven decision support.

1.5 Project Outcomes
The project will deliver a functional prototype of a privacy-preserving smart energy consumption prediction system with the following outputs:
1. A federated learning-based prediction workflow for household energy forecasting.
2. A context-aware prediction model that uses energy history with selected contextual features.
3. A web dashboard for visualization of forecasts, trends, and recommendations.
4. An evaluation report covering prediction accuracy, communication cost, and system usability.
5. Technical documentation of architecture, requirements, and testing results.

1.6 Report Structure
This report is organized into five chapters. Chapter 1 presents the problem statement, objectives, scope, significance, and project outcomes. Chapter 2 reviews relevant literature and identifies research gaps. Chapter 3 describes the methodology, software engineering approach, system architecture, and implementation process. Chapter 4 presents the system requirements and related specifications. Chapter 5 (to be completed in the final report) presents testing, evaluation results, conclusions, and recommendations for future work.

2 Chapter 2: Literature Review

2.1 Overview
Smart energy management in residential environments relies on data generated by smart meters, IoT devices, and contextual sources such as weather services and mobility/traffic systems. Accurately forecasting energy consumption is important for demand planning, energy efficiency, and sustainable urban operations. Traditional centralized machine learning approaches aggregate sensitive household data at a central server, which raises privacy, security, and governance concerns [2], [4], [5].

Federated learning has emerged as a promising approach for privacy-aware collaborative learning. By keeping raw data localized and sharing model updates instead, FL reduces direct exposure of sensitive data. However, FL does not completely eliminate privacy risk and usually benefits from supporting mechanisms such as secure aggregation, encrypted communication, and robust authentication [3]-[6]. This chapter reviews the theoretical foundations of FL, related work in smart cities and energy forecasting, and the research gaps motivating this project.

2.2 Theoretical Background of Federated Learning
Federated learning is a distributed machine learning paradigm in which multiple clients collaboratively train a shared model without exchanging raw data [3]. In a typical FL workflow, each client trains a local model on its own dataset and transmits model updates to a coordinating server, which aggregates them into an updated global model.

Key characteristics of FL include:
- Privacy-aware training: Raw data remains on client devices; only model updates are shared.
- Data decentralization: Training can occur across multiple participants that retain local data control.
- Iterative communication: FL reduces raw dataset transfer but introduces repeated exchange of model parameters or gradients.
- Heterogeneity challenges: Client data distributions, device capabilities, and connectivity may vary significantly.

These characteristics and trade-offs are widely discussed in FL surveys, especially for IoT and distributed environments with heterogeneous participants [4], [5].

Privacy can be strengthened with techniques such as secure aggregation, differential privacy, or homomorphic encryption, depending on system constraints and performance requirements [5]-[7]. FL has been studied in domains such as healthcare, finance, IoT, and smart cities, especially where privacy and data ownership are critical [2], [4].

2.3 Related Work
Federated learning has been explored in smart city systems, energy prediction, and IoT-enabled analytics [1], [2], [4], [8]-[10].

FL in Smart Cities
Recent reviews and surveys on FL in smart city and IoT domains commonly highlight benefits such as privacy-aware collaboration and reduced centralized data collection, while also identifying challenges including communication cost, non-IID data, system heterogeneity, and security threats [2], [4], [5]. These studies support the suitability of FL for urban analytics but also show that end-to-end deployable prototypes are still limited [2].

Energy Consumption Prediction
Existing studies on FL-based energy forecasting report promising prediction performance for smart meter and smart grid scenarios [1], [9], [10]. However, many implementations focus primarily on forecasting accuracy and privacy-aware training, with limited emphasis on integrating richer contextual features, robust recommendation support, or full-stack user-facing systems [9], [10].

Traffic and IoT Integration
Research on intelligent transportation systems and urban IoT shows that mobility and environmental signals can improve urban prediction tasks [4], [8]. For household energy prediction, such signals are better treated as optional proxy variables and should be combined with stronger temporal/contextual features (for example, weather and calendar/time features) to avoid weak or misleading assumptions [2], [10].

Overall, the literature supports FL as a suitable foundation for privacy-aware energy prediction, but integrated systems that combine federated training, contextual feature engineering, and actionable recommendations remain underdeveloped [2], [9], [10].

2.4 Research Gap
Despite progress in FL-based energy forecasting, several gaps remain:
- Many studies prioritize prediction accuracy and privacy claims but provide limited implementation detail for a complete user-facing system.
- Contextual feature integration is often limited or inconsistent, especially for occupancy-related proxy variables.
- Actionable recommendation generation is frequently absent, reducing practical value for household users.
- Comparative evaluation of prediction accuracy and FL communication overhead is often insufficient in prototype systems.
- Security and privacy controls beyond basic FL (for example, secure transport, authentication, and audit logging) are not always clearly specified.

This project addresses these gaps by designing a privacy-preserving, context-aware FL prototype for household energy prediction with a web dashboard, recommendation support, and explicit system requirements for privacy, security, and evaluation [1], [5], [9], [10].

2.5 Chapter Summary
This chapter reviewed the foundations of federated learning and its relevance to smart cities and energy forecasting. The literature shows strong potential for privacy-aware collaborative learning but also highlights gaps in contextual feature integration, full-stack prototype design, and user-oriented recommendations. These findings justify the proposed system and guide the methodology and requirements presented in subsequent chapters.

3 Chapter 3: Methodology

3.1 Overview
This chapter describes the methodology used to design and develop the privacy-preserving smart energy consumption prediction system. It outlines the software engineering method, system architecture, data sources, model development, implementation process, and evaluation approach.

The project uses an iterative development process suitable for a prototype that combines software engineering, data engineering, and machine learning experimentation. The methodology integrates federated learning with contextual data processing to improve prediction quality while preserving privacy.

3.2 Software Engineering Method (Agile, Scrum-Inspired Iterative SDLC)
This project adopts an Agile, Scrum-inspired iterative SDLC. A purely linear model is less suitable because the project includes experimentation (feature selection, model tuning, FL configuration) and user-interface refinement, which benefit from short feedback cycles.

The development process is organized into short iterations (sprints), each producing a testable increment. Typical sprint activities include:
1. Sprint Planning: Define sprint goals, scope, and acceptance criteria.
2. Requirements Refinement: Clarify functional and non-functional requirements based on findings from previous iterations.
3. Design and Implementation: Build or improve system modules (data pipeline, FL server/client, dashboard, recommendation logic).
4. Testing and Review: Perform unit/integration tests and evaluate model and system behavior.
5. Retrospective: Identify improvements for the next sprint.

Key Agile artifacts used in this project include a product backlog (features and technical tasks), sprint backlog, acceptance criteria, and test results. This approach provides flexibility while maintaining engineering discipline and traceability.

3.3 Development Phases
Within the Agile process, the technical work follows these high-level phases:
1. Requirements Analysis:
Stakeholder needs, data availability, privacy constraints, and evaluation criteria are defined.
2. System Design:
Architecture, data flow, API contracts, database schema, and FL workflow are designed.
3. Prototype Development:
Frontend, backend, FL orchestration, feature engineering, and recommendation components are implemented incrementally.
4. Testing and Validation:
Functional testing, integration testing, and model evaluation are performed across iterations.
5. Deployment and Evaluation:
The prototype is deployed in a controlled environment for demonstration and final evaluation.

3.4 System Architecture
The proposed system uses a three-tier architecture to support modularity, privacy-aware processing, and maintainability.

- Presentation Layer:
A web-based dashboard where users view predicted energy consumption, historical trends, alerts, and recommendations.

- Application Layer:
A backend service that manages authentication, API requests, contextual data integration, recommendation generation, and FL coordination.

- Data and FL Layer:
Household datasets remain on client devices or simulated client nodes. A central FL coordinator aggregates model updates to produce global model versions. The server stores metadata, user profiles, logs, and prediction outputs, but not raw household training data.

This architecture enables collaborative learning while reducing the need for centralized raw data collection.

3.5 Data Collection and Sources
The system uses multiple data sources to improve prediction quality.

- Household Energy Data:
Smart meter time-series datasets containing historical electricity consumption values for each household/client.

- Weather Data:
Contextual weather variables (for example, temperature, humidity, rainfall, or apparent temperature) from public APIs or open datasets.

- Temporal and Occupancy Proxy Features:
Derived features such as hour-of-day, day-of-week, weekend/weekday, and holiday indicators. Optional area-level traffic/mobility indicators may be included if they are available and relevant.

- Simulated Data (if required):
Synthetic or partitioned datasets for FL experimentation when sufficient real distributed data is unavailable.

Data preprocessing includes time alignment, missing-value handling, normalization/scaling, outlier checks, and feature engineering.

3.6 System Design
The system is designed to support privacy-preserving collaborative learning while remaining usable and maintainable.

Key design components include:
- Secure user authentication and authorization
- Federated learning coordinator/server and client nodes
- Context-aware feature engineering and prediction engine
- Recommendation module for actionable guidance
- Interactive dashboard for visualization and user feedback
- Logging and monitoring for auditability and debugging

Unified Modeling Language (UML) diagrams, including use case, sequence, and class diagrams, will be used to visualize system interactions and structure.

3.7 Federated Learning Model Development
The prediction model is developed using a federated learning approach in which multiple clients train local models using local household data and contextual features. Model updates are periodically sent to a coordinating server, where they are aggregated using a federated optimization strategy (baseline: Federated Averaging / FedAvg) to produce an updated global model.

The training process is iterative and continues for a defined number of communication rounds or until performance converges based on validation metrics. Model evaluation compares FL performance against a centralized or local-only baseline (where feasible) to assess trade-offs.

To improve robustness and accuracy, the development process includes:
- Baseline model selection (for example, linear regression / tree-based model or a simple neural network)
- Contextual feature ablation tests (with vs without contextual features)
- Hyperparameter tuning (learning rate, local epochs, batch size, number of rounds)
- Client heterogeneity handling and dropout tolerance in simulations

3.8 System Implementation
The system is implemented as a web-based platform for accessibility and demonstration.

The frontend provides dashboards for forecasts, trends, recommendations, and model status. The backend handles API requests, authentication, data management, contextual feature processing, and FL orchestration. A relational database stores application data and metadata. The FL framework manages distributed training across simulated or actual client instances.

Containerization may be used to simplify reproducible deployment and testing.

3.9 Testing and Evaluation
Testing and evaluation are conducted throughout development and during final validation.

1. Unit Testing:
Validate individual modules such as authentication, data preprocessing, prediction logic, recommendation generation, and dashboard components.

2. Integration Testing:
Verify communication between frontend, backend APIs, database, contextual data services, and FL clients/server.

3. Model Evaluation:
Measure forecasting performance using metrics such as MAE, RMSE, and (where appropriate) MAPE. Assess FL-specific metrics such as communication rounds, total bytes transferred, training time, and client participation rate.

4. Security and Privacy Verification:
Check transport security configuration, authentication/authorization behavior, and confirm that raw household training data is not transmitted to the FL coordinator.

5. User Acceptance Testing (UAT):
Evaluate usability, clarity of visualizations, and usefulness of recommendations against defined acceptance criteria.

3.10 Tools and Technologies (Recommended Software Stack)

Component	Recommended Technology	Reason
Frontend	React + TypeScript (Vite)	Fast UI development, maintainable typed code, strong ecosystem for dashboards
Visualization	Plotly.js or Recharts	Interactive time-series charts for energy trends and forecasts
Backend API	Python FastAPI	High performance, clear API design, excellent Pydantic validation for data schemas
Database	PostgreSQL	Reliable relational database for users, metadata, logs, and prediction records
ORM/Data Access	SQLAlchemy (or SQLModel)	Structured database access and maintainability
FL Framework	Flower (FLwr)	Practical federated learning orchestration, framework-agnostic, suitable for prototypes and simulation
ML / Modeling	PyTorch + scikit-learn	Flexible modeling plus strong utilities for baselines, preprocessing, and metrics
Data Processing	Pandas + NumPy	Standard data cleaning, time-series preprocessing, and feature engineering
Authentication	JWT + password hashing (bcrypt/Argon2)	Secure session/auth handling for web users and API access
Testing	pytest (+ frontend test framework if used)	Unit/integration testing and reproducible validation
Deployment	Docker + Docker Compose	Reproducible local deployment of frontend, backend, DB, and FL services
Version Control	Git + GitHub/GitLab	Traceability, collaboration, and change management
External APIs	Weather API (+ optional traffic/mobility API)	Contextual features for prediction

3.11 Ethical Considerations
This project prioritizes privacy-by-design. Raw household energy data remains on client devices/nodes and is not transferred to the aggregation server for model training. The system should use encrypted communication channels (for example, HTTPS/TLS) and access controls to reduce unauthorized access risks.

Any external datasets used should come from legally accessible/public sources, and simulated data may be used where necessary to avoid exposing sensitive information. The prototype is developed for academic/research purposes and should avoid collecting personally identifiable information beyond what is necessary for user account management.

3.12 Chapter Summary
This chapter presented the methodology for developing the privacy-preserving smart energy consumption prediction system. It described the Agile, Scrum-inspired iterative SDLC, system architecture, data sources, FL model development, implementation strategy, and testing approach. The methodology is designed to support a practical, privacy-aware prototype with measurable evaluation outcomes.

4 Chapter 4: Requirements

4.1 Introduction
This chapter defines the requirements for the privacy-preserving smart energy consumption prediction system using federated learning. Requirements are derived from the needs of household users, researchers/developers, and system administrators. They address privacy-preserving collaborative training, contextual feature integration, forecasting quality, usability, and maintainability.

4.2 Requirement Gathering
Requirements are identified using the following methods:
- Stakeholder Analysis: Identify primary stakeholders (household users, researchers/developers, administrators, and supervisors/evaluators).
- Literature Review: Review common capabilities and limitations of smart energy forecasting and FL systems.
- Technical Feasibility Analysis: Assess available datasets, APIs, computing resources, and prototype constraints.
- Iterative Validation: Refine requirements through sprint reviews and prototype testing.

These methods help ensure that the requirements remain feasible, testable, and aligned with project objectives.

4.3 User Requirements
- Users shall be able to securely log in and access only their own dashboard data.
- Users shall be able to view predicted energy consumption for selected time horizons (for example, next hour/day depending on model design).
- Users shall be able to view historical consumption trends and compare them with predictions.
- Users shall receive understandable recommendations to improve energy efficiency or shift usage patterns.
- Users shall be able to view key contextual indicators associated with predictions (for example, weather conditions or peak-time periods).
- Administrators/researchers shall be able to monitor FL training rounds, model versions, and system status.

4.4 System Requirements
- The system shall support federated model training across multiple clients without transferring raw household training data to the server.
- The system shall integrate historical energy data with contextual features (weather and temporal features; optional mobility/traffic indicators).
- The system shall provide a web-based dashboard accessible through modern browsers on desktop and tablet devices.
- The system shall store user accounts, configuration, prediction outputs, and logs in a persistent database.
- The system shall support model versioning and tracking of training/evaluation runs.
- The system shall handle partial client participation (for example, offline clients) during FL rounds in simulation/prototype mode.

4.5 Functional Requirements
FR1. User Authentication and Authorization
- Register/login users securely.
- Enforce role-based access (for example, user and admin/researcher roles).

FR2. Data Ingestion and Validation
- Ingest household energy data from files, datasets, or local client stores.
- Fetch contextual data from configured APIs or load local datasets.
- Validate data format, timestamps, units, and missing fields before training/prediction.

FR3. Data Preprocessing and Feature Engineering
- Align energy and contextual data on a common time index.
- Handle missing values and outliers using defined preprocessing rules.
- Generate temporal features (hour/day/weekend/holiday) and optional occupancy proxy features.

FR4. Federated Learning Coordination
- Initialize global model parameters and distribute them to participating clients.
- Trigger or schedule FL rounds.
- Receive client model updates and aggregate them using FedAvg (baseline).
- Record training metadata (round number, participants, timestamps, metrics).

FR5. Local Client Training
- Train a local model on each client using only local data.
- Return model updates and local metrics without sending raw training data.
- Support configurable local training parameters (epochs, batch size, learning rate).

FR6. Prediction Generation
- Generate forecasts for configured time horizons using the latest approved model.
- Store prediction outputs and associated timestamps/model version.

FR7. Recommendation Module
- Generate actionable recommendations based on forecasts, trends, and predefined rules (for example, reduce peak-hour usage, shift non-critical loads).
- Present recommendations in clear, user-friendly language.

FR8. Dashboard and Visualization
- Display historical consumption, predicted consumption, and contextual indicators.
- Display recommendation summaries and, where applicable, model status information.
- Provide filtering by date/time range.

FR9. Monitoring and Logging
- Log system events, API errors, training rounds, and important actions for debugging and audit purposes.
- Provide basic admin/researcher visibility into training progress and failures.

FR10. Evaluation and Reporting
- Compute and display/store evaluation metrics (MAE, RMSE, optional MAPE).
- Export evaluation results or reports for project documentation.

4.6 Non-Functional Requirements
- Privacy: Raw household training data shall remain on client devices/nodes; only model updates and approved metadata may be transmitted.
- Security: The system shall use authenticated access, secure password storage, and encrypted communication (TLS/HTTPS in deployed environments).
- Performance: Prediction requests should return within an acceptable prototype response time (target to be defined based on deployment hardware); FL round duration shall be monitored and reported.
- Reliability: The system shall handle API failures, missing contextual data, and temporary client unavailability with graceful error handling and retries where applicable.
- Usability: The dashboard shall present forecasts and recommendations clearly and be usable by non-technical household users.
- Scalability: The architecture shall support increasing numbers of simulated clients for experimentation without major redesign.
- Maintainability: Components shall be modular, documented, and testable to support iterative improvement.
- Interoperability: The system shall expose clear API interfaces and support common data formats (for example, CSV/JSON).
- Reproducibility: Model training/evaluation configurations, data versions, and metrics shall be recorded to support repeatable experiments.

4.7 Hardware Requirements
- User Device: Laptop/desktop/tablet with modern web browser and internet/network access.
- FL Client Node (prototype): Personal computer or virtual instance capable of local preprocessing and model training for assigned data partitions.
- Application/FL Server: Machine or cloud instance with sufficient CPU/RAM to run backend APIs, database, and FL coordination services.
- Optional GPU: Helpful for faster model training if deep learning models are used, but not mandatory for a prototype.

4.8 Software Requirements (Recommended)
- Operating System: Windows, Linux, or macOS for development; Linux recommended for deployment.
- Frontend: React + TypeScript (Vite) with charting library (Plotly.js or Recharts).
- Backend API: Python FastAPI.
- FL Framework: Flower (FLwr).
- ML Libraries: PyTorch, scikit-learn.
- Data Processing: Pandas, NumPy.
- Database: PostgreSQL.
- Authentication/Security: JWT library, password hashing library (bcrypt/Argon2), TLS/HTTPS for deployment.
- Testing: pytest (backend), plus frontend test tooling if implemented.
- Containerization/Deployment: Docker and Docker Compose.
- Version Control: Git.

4.9 Constraints and Assumptions
Constraints
- Availability and quality of smart meter and contextual datasets may limit model performance and evaluation realism.
- Public API limits, latency, or missing data may affect contextual feature collection.
- Prototype scope excludes production-grade utility integration, large-scale deployment, and advanced privacy mechanisms unless time/resources permit.
- Computing resources may limit the number of FL clients, model complexity, and experiment duration.

Assumptions
- Household data can be partitioned per client (real or simulated) for FL experiments.
- Users consent to using their data locally for model training in the prototype setting.
- Time synchronization and unit consistency can be achieved during preprocessing.

4.10 Chapter Summary
This chapter defines the requirements for the privacy-preserving smart energy prediction system, including user, system, functional, non-functional, hardware, and software requirements. It also clarifies constraints and assumptions. These requirements provide a stronger and more testable foundation for implementing and evaluating the proposed federated learning prototype.

REFERENCES
[1] M. N. Fekri, K. Grolinger, and S. Mir, "Asynchronous adaptive federated learning for distributed load forecasting with smart meter data," International Journal of Electrical Power & Energy Systems, vol. 153, p. 109285, 2023, doi: 10.1016/j.ijepes.2023.109285.

[2] S. Pandya et al., "Federated learning for smart cities: A comprehensive survey," Sustainable Energy Technologies and Assessments, vol. 55, p. 102987, 2023, doi: 10.1016/j.seta.2022.102987.

[3] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 2017, pp. 1273-1282.

[4] D. C. Nguyen, M. Ding, P. N. Pathirana, A. Seneviratne, J. Li, and H. V. Poor, "Federated learning for Internet of Things: A comprehensive survey," IEEE Communications Surveys & Tutorials, vol. 23, no. 3, pp. 1622-1658, 2021, doi: 10.1109/COMST.2021.3075439.

[5] V. Mothukuri, R. M. Parizi, S. Pouriyeh, Y. Huang, A. Dehghantanha, and G. Srivastava, "A survey on security and privacy of federated learning," Future Generation Computer Systems, vol. 115, pp. 619-640, 2021, doi: 10.1016/j.future.2020.10.007.

[6] K. Bonawitz et al., "Practical secure aggregation for privacy-preserving machine learning," Cryptology ePrint Archive, Paper 2017/281, 2017. [Online]. Available: https://eprint.iacr.org/2017/281

[7] M. Moshawrab and M. Adda, "Securing Federated Learning: Approaches, Mechanisms and Opportunities," Electronics, vol. 13, no. 18, p. 3675, 2024, doi: 10.3390/electronics13183675.

[8] C. Zhang, C. Song, Y. Yang, and W. Wang, "A Survey on Federated Learning in Intelligent Transportation Systems," arXiv preprint arXiv:2403.07444, 2024. [Online]. Available: https://arxiv.org/abs/2403.07444

[9] M. M. Badr et al., "Privacy-Preserving Federated-Learning-Based Net-Energy Forecasting," in SoutheastCon 2022, Mobile, AL, USA, 2022, pp. 133-139, doi: 10.1109/SoutheastCon48659.2022.9764093.

[10] R. Wang et al., "An adaptive federated learning system for community building energy load forecasting and anomaly prediction," Energy and Buildings, vol. 295, p. 113215, 2023, doi: 10.1016/j.enbuild.2023.113215.
