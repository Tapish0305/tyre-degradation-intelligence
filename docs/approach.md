# F1 Tyre Degradation Intelligence: Detailed Technical Approach

This document explains the end-to-end architecture and logic behind the F1 Tyre Degradation Intelligence project. The goal of this system is to accurately extract the "true" underlying tyre wear (degradation) from noisy Formula 1 lap times by decoupling confounding factors like fuel burn, traffic, track evolution, and weather using Bayesian State-Space Models (SSM).

## 🚀 Unique Selling Proposition (USP) & Value Additions

While traditional time-series models (like ARIMA) simply forecast historical lap times without understanding the physical context, and standard track-side heuristics guess degradation using linear slopes, **this project introduces a probabilistic, physics-informed AI approach:**

1. **Unconfounding the True Tyre Pace:** We don't just look at lap times; we isolate and mathematically strip away the effects of fuel burn, dirty air (traffic), track evolution (rubbering in), and thermal weather variations to reveal the *true latent capability* of the tyre at any given lap.
2. **Pre-Race Predictive Power (Practice-to-Race Projection):** Instead of only analyzing data post-race, our Hierarchical SSM learns the physical wear profiles of tyre compounds (Hard, Medium, Soft) from Free Practice 2 (FP2). It then forward-projects these learned distributions onto race day, giving strategists a robust predictive baseline *before the lights go out*.
3. **Live Weather Dynamics:** The degradation rate isn't a static constant. Our model uses Bayesian Inference to continuously update the wear penalty based on standardized weather covariates, recognizing that a hotter track surface directly accelerates physical tyre wear.
4. **Actionable Strategist UI:** We translate complex Bayesian Monte Carlo posteriors (e.g., 95% Credible Intervals, R-hat convergence) into plain-English, real-time strategic actions (e.g., "Optimal Pit Window", "Wear Cliff Reached") through interactive dashboards.

## System Architecture

The codebase is structured around several core modules:

1. **`data_pipeline.py`**: Ingestion, cleaning, and feature engineering (including telemetry for style metrics and weather extraction).
2. **`model.py`**: Bayesian State-Space Modeling (SSM) and predictive inference.
3. **`driver_style.py`**: Computes a Driver Style / Aggression Indicator based on vehicle telemetry (jerk).
4. **`dashboard_data.py`**: Analytical backend, business logic, and heuristics for the UI.
5. **UI & Launchers** (`run_dashboard.py`, `app.py`, `streamlit_app.py`): Frontends for data visualization (Flask and Streamlit) and the unified entry point.
6. **`fit_and_validate.py`**: Pipeline runner for end-to-end validation against benchmarks.

---

## 1. Data Ingestion & Preprocessing (`data_pipeline.py`)

This module leverages the `fastf1` API to fetch real telemetry and timing data, and prepares it for modeling.

### Core Functions:
- **`setup_fastf1_cache(cache_dir)`**: Enables local caching of the `fastf1` requests to prevent rate-limiting and speed up consecutive runs.
- **`load_f1_session(year, grand_prix, session_type, cache_dir)`**: Retrieves the requested F1 session (e.g., Practice 2, Race).
- **`compute_gap_to_car_ahead(laps_df)`**: Calculates the time gap to the car directly ahead on track. This is crucial for modeling "dirty air" (traffic). If a driver is in free air, the gap is capped at 10 seconds.
- **`compute_fuel_load(df, is_race, total_race_laps)`**: Estimates the fuel weight (in kg) carried by the car. 
  - *Race*: Assumes a linear burn from 110.0kg at Lap 1 down to 0.0kg at the end of the race.
  - *Practice*: Estimates fuel based on the length of the stint (1.7kg/lap + 5.0kg buffer).
- **`compute_session_timeline(df)`**: Normalizes the elapsed session time from 0 to 1. This is used to model "track evolution" (how track grip improves as rubber is laid down over the session).
- **`compute_weather_features(df, weather_df)`**: Attaches per-lap weather covariates (track temperature, air temperature, humidity, wind speed) to capture thermal tyre degradation effects.
- **`extract_clean_laps(session, drivers, is_race, outlier_threshold)`**: The core cleaning function. It filters out pit-in/pit-out laps, safety car laps, and severe outliers (laps significantly slower than the stint median). It applies the gap, fuel, timeline, and weather computations.
- **`prepare_dataset_for_ssm(df, driver_order)`**: Converts the pandas DataFrame into structured `numpy` arrays. It encodes categorical variables (Driver IDs, Compound IDs) and creates arrays indicating stint resets, which are required for the JAX/NumPyro models.
- **`load_event_dataset(...)`**: A high-level convenience wrapper that loads, cleans, and structures both Practice and Race datasets simultaneously.
- **`load_event_dataset_with_driver_style(...)`**: Extends the data loading by parsing detailed high-frequency telemetry during Practice to compute the Driver Style profile alongside the SSM dataset.

---

## 2. Bayesian State-Space Modeling (`model.py`)

This module uses `NumPyro` and `JAX` for ultra-fast Markov Chain Monte Carlo (MCMC) inference. It models lap times as a combination of latent tyre pace and known physical effects.

### Modeling Functions:
- **`baseline_ssm_model(...)`**: 
  - *Equation*: $y_t = \alpha_t + \gamma \cdot \text{fuel}_t + \epsilon_t$
  - A baseline State-Space Model that only accounts for fuel effect ($\gamma$). The latent tyre pace ($\alpha_t$) follows a random walk with drift (the degradation rate $\nu$).
- **`traffic_ssm_model(...)`**: 
  - *Equation*: $y_t = \alpha_t + \gamma \cdot \text{fuel}_t + \delta \cdot \exp(-\text{gap}_t / \tau) + \epsilon_t$
  - Adds a non-linear exponential decay penalty for traffic. When the gap to the car ahead is small, the penalty $\delta$ spikes, but it quickly decays based on the scale $\tau$.
- **`hierarchical_ssm_model(...)`**: 
  - The flagship competition model. It pools data across multiple drivers simultaneously.
  - *Features*:
    - **Track Evolution**: $\rho \cdot \text{session\_time}$ (Shared across drivers).
    - **Compound Hierarchy**: Learns specific degradation rates for Hard, Medium, and Soft compounds ($\mu_{\nu\_hard}, \mu_{\nu\_med}, \mu_{\nu\_soft}$).
    - **Driver Base Pace**: Each driver has an intrinsic base pace ($\alpha_{base}$).
    - **Compound Offsets**: Pace delta for Hard (slower) and Soft (faster) relative to Medium.

### Inference & Validation Functions:
- **`fit_ssm(...)`**: The main harness to fit a selected model using the NUTS (No-U-Turn Sampler) kernel. It extracts posterior samples and generates summary statistics (mean, std, 95% credible intervals).
- **`predict_race_from_practice_samples(practice_samples, race_dataset)`**: 
  - Takes the degradation rates ($\nu$) learned purely from Practice sessions (where fuel and traffic were decoupled).
  - Projects them forward onto the Race dataset using the race fuel burn curve and race traffic gaps.
  - Generates Monte Carlo predictions for race lap times.
- **`compute_rmspe(...)` & `compute_crps_samples(...)`**: Calculates Root Mean Squared Predictive Error (RMSPE) and Continuous Ranked Probability Score (CRPS) to quantify predictive accuracy and uncertainty calibration.

---

## 3. Driver Style & Aggression (`driver_style.py`)

This module introduces a jerk-based driving style indicator derived from high-frequency car telemetry (acceleration). It quantifies how aggressively a driver interacts with the tyre.

### Core Metrics:
- **Longitudinal & Lateral DSI**: Standard deviations of the longitudinal ($j_x$) and lateral ($j_y$) jerk. Higher values indicate more abrupt changes in acceleration (aggressive).
- **Combined DSI**: A Euclidean combination of longitudinal and lateral jerk to provide a single, unified "aggression index" for the dashboard.
- **Secondary Metrics**: Tracks heavy braking frequency, braking/cornering intensity, and throttle application rate. Optionally distills these into a PCA composite score.
- **`compute_driver_style_profile(...)`**: Aggregates lap-by-lap style metrics into a robust session-level profile (using medians) for each driver.

---

## 4. UI Dashboard Engine & Frontends

The analytics are surfaced via multiple interactive frontends (`app.py`, `streamlit_app.py`), orchestrated by a central launcher (`run_dashboard.py`). The analytical backend (`dashboard_data.py`) sits between the SSM/driver style engines and the UI. It processes the raw posterior samples and kinematics into interpretable strategic intelligence.

### Core Functions (`dashboard_data.py`):
- **`get_or_fit_dashboard_data(...)`**: Orchestrates the entire pipeline. It fits the hierarchical model on FP2 data, runs the forward race prediction, fits benchmark models (ARIMA and Naive Linear), and returns a comprehensive dictionary of results.
- **`compute_question1_pit_analytics(...)`**:
  - *Purpose*: Determines the Optimal Pit Window.
  - Calculates the cumulative time lost due to degradation lap-over-lap.
  - Compares the "net pit benefit": staying out and losing time to wear vs. pitting (incurring a ~22s penalty) but gaining fresh tyre pace.
  - Identifies the "Crossover Lap" where pitting becomes mathematically faster.
- **`compute_question2_checkpoints(...)`**:
  - *Purpose*: Breaks down exact lap time deficits at regular checkpoint intervals (e.g., every 5 laps).
  - Deconstructs the observed lap time into isolated buckets: Latent Tyre Wear (s), Fuel Effect (s), and Traffic Penalty (s).
  - Calculates Remaining Tyre Life percentage.
- **`compute_question3_trust_metrics(...)`**:
  - *Purpose*: Quantifies model trustworthiness.
  - Calculates how many actual race laps fell inside the Bayesian 90% and 95% Credible Intervals (Calibration Coverage).
  - Computes a weighted `Trust Index` (0-100%) based on MCMC convergence, calibration coverage, and out-of-session RMSPE.

---

## 5. End-to-End Validation (`fit_and_validate.py`)

This script serves as a reproducible pipeline runner to prove the model's validity against scientific benchmarks. It operates in four phases and outputs publication-grade charts to the `plots/` directory:

1. **Phase 1**: Replicates baseline paper results (Single Driver, Fuel-only).
2. **Phase 2**: Evaluates the isolated impact of the Traffic penalty exponential decay curve.
3. **Phase 3**: Fits the full Hierarchical model on a multi-driver cohort to extract track evolution and compound-specific degradation distributions.
4. **Phase 4**: Runs the Predictive Validation Harness comparing the Hierarchical Practice-to-Race model against ARIMA time-series and naive linear extrapolations, outputting RMSPE and CRPS scores.
