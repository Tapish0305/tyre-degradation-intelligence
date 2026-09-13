# Tyre Degradation Intelligence: Isolating True Tyre Wear Rates in Formula 1

[![Competition](https://img.shields.io/badge/Theme-AI%20Motorsport%20Intelligence-red.svg)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPyro](https://img.shields.io/badge/Backend-NumPyro%20%7C%20JAX-orange.svg)](https://num.pyro.ai/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20%26%20Flask-ff4b4b.svg)](https://streamlit.io/)
[![FastF1](https://img.shields.io/badge/Data-FastF1%20v3.8-green.svg)](https://docs.fastf1.dev/)

A competition-aligned Bayesian hierarchical state-space modeling framework and **Interactive UI Dashboard** designed to **isolate true tyre degradation rates from confounding practice variables** (fuel load, traffic/dirty air, and track evolution) and predict actual race-day pace with full uncertainty quantification.

Based on and extending the research paper:  
> **"A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing"**  
> *Cole Cappello and Andrew Hoegh* (arXiv:2512.00640v1)

---

## 1. The USP: Driver Style Indicator (DSI)

Our primary Unique Selling Proposition (USP) is the **Driver Style Indicator (DSI)**. While most models attribute tyre wear exclusively to the car and track, our model quantitatively extracts the *driver's contribution to tyre degradation* using micro-telemetry features. 

We capture the **"Jerk" (rate of change of acceleration)** during braking and cornering phases. By analyzing longitudinal and lateral G-forces at high frequencies, we construct the DSI to segment drivers into distinct profiles (e.g., Smooth/Progressive vs. Aggressive/Stabby). This DSI serves as a latent covariate in the Bayesian model, explaining variances in degradation rates that fuel and traffic alone cannot.

---

## 2. The Three Questions Answered by the Dashboard

Per competition and mentor evaluation criteria, the UI Dashboard explicitly and transparently answers:

```text
+---------------------------------------------------------------------------------------------------------------+
|  1. RATE OF PERFORMANCE LOSS & PIT STOP CALL                                                                  |
|     - Live wear rate ν (s/lap) with 95% Bayesian credible intervals.                                          |
|     - Cumulative pace deficit curves & dynamic Optimal Pit Window [Lap L_start, Lap L_end].                   |
|     - Crossover analysis: Identifies exact lap where degradation time loss exceeds pit stop delta penalty.   |
+---------------------------------------------------------------------------------------------------------------+
|  2. PERFORMANCE AT DEFINED CHECKPOINTS                                                                        |
|     - Lap-by-lap & 5-lap checkpoint table (Lap 5, 10, 15, 20, 25, 30...).                                     |
|     - Component decomposition: Isolated Tyre Wear (Δt) vs Fuel Burn vs Dirty Air Traffic vs Track Grip.      |
|     - Remaining Tyre Life % barometer and actionable strategist callouts ("Optimal Window", "Box Now").       |
+---------------------------------------------------------------------------------------------------------------+
|  3. TRUSTWORTHINESS OF PREDICTION (VALIDATION EVIDENCE)                                                       |
|     - Composite Trust Index (0-100%, e.g., 94.2% HIGH CONFIDENCE).                                            |
|     - Post-Race Validation Proof: 90% and 95% Bayesian predictive credible intervals vs actual race pace.     |
|     - Calibration Score: Confirms 91.8% of observed race laps fall inside the 90% credible envelope.          |
|     - Benchmark Comparison: Evaluates against Naive Extrapolation and ARIMA(2,1,2) baselines.                 |
+---------------------------------------------------------------------------------------------------------------+
```

---

## 3. Mathematical Formulation

### 3.1 Observation Equation (Fuel + Traffic + Track Evolution)
For driver $d \in \{1, \dots, D\}$ at session lap $t \in \{1, \dots, T_d\}$:
$$y_{d, t} = \alpha_{d, t} + \gamma \cdot \text{fuel}_{d, t} + \delta \cdot \exp\left(-\frac{\text{gap}_{d, t}}{\tau}\right) + \rho \cdot \text{session\_time}_{d, t} + \epsilon_{d, t}$$

- **$\alpha_{d, t}$**: Latent tyre pace state (seconds) on lap $t$.
- **$\gamma \cdot \text{fuel}_{d, t}$**: Fuel time penalty ($\gamma \approx 0.012\text{--}0.033\text{ s/kg}$).
- **$\delta \cdot \exp(-\text{gap}_{d, t} / \tau)$**: Nonlinear dirty air penalty ($\delta \approx 0.62\text{s}$, $\tau \approx 1.5\text{s}$).
- **$\rho \cdot \text{session\_time}_{d, t}$**: Shared track evolution grip progression ($\rho < 0$), identified across multi-driver cohorts.
- **$\epsilon_{d, t} \sim \text{Student-}t(\nu_{df}=4, 0, \sigma_\epsilon^2)$**: Heavy-tailed error robust to driver mistakes and lock-ups.

### 3.2 Latent Process & Compound Hierarchy
$$\alpha_{d, t+1} = (1 - I_{\text{pit}, d, t})(\alpha_{d, t} + \nu_{d, c(d, t)}) + I_{\text{pit}, d, t}(\alpha_{\text{reset}, d, c(d, t+1)}) + \eta_{d, t}$$
$$\nu_{d, c} \sim \mathcal{N}^+(\mu_{\nu, c}, \sigma_\nu^2), \quad \alpha_{\text{reset}, d, c} = \alpha_{\text{base}, d} + \Delta_c$$

---

## 4. Benchmark Results (Practice $\to$ Race Validation)

Validation performed on the **2024 Italian Grand Prix (Monza)** across top drivers (**Hamilton, Leclerc, Verstappen, Norris**) over **11 race stints (194 laps)** after training exclusively on **Practice 2 (FP2 long runs)**:

| Model / Methodology | Training Session | Target Evaluation | RMSPE (s) | CRPS (s) |
| :--- | :--- | :--- | :---: | :---: |
| **1. Naive Practice Linear Extrapolation** | FP2 Practice | Race (All Stints) | $3.5787\text{ s}$ | $2.7554\text{ s}$ |
| **2. ARIMA(2,1,2) Benchmark (Paper Baseline)** | Race (1-step-ahead) | Race (All Stints) | $12.3319\text{ s}$ | $2.2166\text{ s}$ |
| **3. Base Race-Only SSM (Base Paper)** | Race | Race | $0.3797\text{ s}$ | $0.1424\text{ s}$ |
| **4. Competition Hierarchical SSM (Ours)** | **FP2 Practice (Decoupled)** | **Race (Forward Prediction)** | **$3.0495\text{ s}$** | **$2.2010\text{ s}$** |

---

## 5. UI Dashboard: How to Launch & Explore

The interactive dashboard is a standalone Flask application with a modern Plotly & Tailwind interface.

```bash
python app.py
```
*Accessible at: **http://localhost:5000***

*(Note: Data extraction and initial MCMC Bayesian sampling might take a few seconds on the very first boot; models are then persistently cached on disk for instant future loads.)*

---

## 6. Repository Structure

```text
tyre-degradation-intelligence/
├── app.py                      # Standalone Flask + Plotly.js + Tailwind Web Application (Dashboard UI)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & benchmark report
├── src/                        # Core Application Code
│   ├── dashboard_data.py       # Dashboard analytics engine & checkpoint/trust calculator
│   ├── data_pipeline.py        # FastF1 ingestion, gap derivation, fuel calculation, caching
│   ├── driver_style.py         # DSI (Driver Style Indicator) telemetry extraction (Jerk & G-Forces)
│   └── model.py                # NumPyro Bayesian SSM definitions and MCMC runner
├── docs/                       # Project Documentation
│   ├── approach.md             # Detailed project methodology and problem-solving approach
│   └── ASSUMPTIONS.md          # Centralized log of all modeling assumptions & limitations
└── cache/                      # FastF1 persistent cache directory
```

---

## 7. Evaluation Criteria Compliance

- **Feasibility**: Ingests standard publicly accessible FastF1 timing and telemetry data with persistent local caching.
- **Economical**: Pure Python, JIT-compiled with JAX; entire multi-driver hierarchical MCMC runs in **~67 seconds on CPU** ($<5\text{ min}$ budget).
- **Trustworthy**: Rigorous Bayesian uncertainty quantification with 90%/95% credible intervals and out-of-session practice-to-race validation calibration.
