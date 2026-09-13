# Modeling Assumptions & Limitations: Tyre Degradation Intelligence

## 1. Executive Summary & Purpose
This document provides a comprehensive, centralized record of all modeling assumptions, mathematical formulations, data derivation proxies, identifiability constraints, and known limitations in the **Tyre Degradation Intelligence** system. By decoupling true tire wear from external track and operational variables (fuel load, traffic dirty air, and session track evolution), this project delivers clean tire performance degradation curves and a practice-to-race validation harness aligned with the *AI Motorsport Intelligence* competition brief.

---

## 2. Observation Model & Covariate Specifications

### 2.1 Formula
The target observation equation for driver $d \in \{1, \dots, D\}$ at session lap $t \in \{1, \dots, T_d\}$ is:
$$y_{d, t} = \alpha_{d, t} + \gamma \cdot \text{fuel}_{d, t} + \delta \cdot f(\text{gap}_{d, t}) + \rho \cdot \text{session\_time}_{d, t} + \epsilon_{d, t}$$

where:
- $\alpha_{d, t}$: Latent tire pace state (seconds) on lap $t$ for driver $d$.
- $\gamma \cdot \text{fuel}_{d, t}$: Linear lap time penalty due to car fuel mass.
- $\delta \cdot f(\text{gap}_{d, t})$: Nonlinear aerodynamic and thermal penalty induced by traffic / dirty air.
- $\rho \cdot \text{session\_time}_{d, t}$: Linear track evolution term representing grip / rubber accumulation over the session.
- $\epsilon_{d, t}$: Observation error capturing driver variance, transient mistakes, and micro-sector noise.

---

## 3. Detailed Component Assumptions & Priors

### 3.1 Fuel Weight Modeling ($\gamma \cdot \text{fuel}_{d, t}$)
- **Race Sessions**:
  - **Starting Fuel**: Assumed to start at $110\text{ kg}$ on Lap 1 (matching FIA maximum fuel regulation capacity and the base paper specification).
  - **Linear Fuel Consumption**: Assumed to burn linearly down to $0\text{ kg}$ on the final scheduled lap $T_{\text{total}}$:
    $$\text{fuel}_{\text{race}}(t) = 110 \times \left(1 - \frac{t - 1}{T_{\text{total}} - 1}\right)\text{ kg}$$
  - **Rationale**: While starting fuel varies slightly by engine efficiency (~100–108 kg), engineers minimize excess fuel to negligible levels (~1–2 kg FIA mandatory inspection sample). Linear decay represents constant average fuel burn rate per lap.
- **Practice Sessions (FP1/FP2/FP3)**:
  - In practice sessions, cars are fueled for specific run lengths rather than the full 60-minute session.
  - Fuel load per stint $s$ is estimated using stint duration:
    $$\text{fuel}_{\text{practice}}(t) = \kappa \times (\text{StintLength}_s - \text{lap\_in\_stint}) + \text{fuel}_{\text{buffer}}$$
    where $\kappa = 1.7\text{ kg/lap}$ (standard Grand Prix lap fuel burn) and $\text{fuel}_{\text{buffer}} = 5.0\text{ kg}$ (minimum safety buffer).
- **Fuel Penalty Prior ($\gamma$)**:
  $$\gamma \sim \mathcal{N}^+(0.033, 0.005^2) \quad [\text{seconds / kg}]$$
  Centered at $0.033\text{ s/kg}$ (approx. $0.33\text{ s}$ per $10\text{ kg}$), matching empirical F1 engineering rules of thumb.

### 3.2 Traffic Penalty Modeling ($\delta \cdot f(\text{gap}_{d, t})$)
- **Functional Form**:
  $$f(\text{gap}) = \exp\left(-\frac{\text{gap}}{\tau}\right)$$
- **Decay Constant ($\tau$)**:
  $$\tau \sim \mathcal{N}^+(1.5, 0.5^2) \quad [\text{seconds}]$$
  - At $\text{gap} = 0\text{s}$, $f(\text{gap}) = 1.0$ (full dirty air penalty $\delta$).
  - At $\text{gap} = 1.5\text{s}$, $f(\text{gap}) = \exp(-1) \approx 0.368$.
  - At $\text{gap} = 3.0\text{s}$, $f(\text{gap}) = \exp(-2) \approx 0.135$.
  - At $\text{gap} \ge 5.0\text{s}$, $f(\text{gap}) < 0.04 \approx 0.0$ (free air threshold).
- **Traffic Penalty Prior ($\delta$)**:
  $$\delta \sim \mathcal{N}^+(0.80, 0.30^2) \quad [\text{seconds}]$$
  Represents peak lap time deficit from aerodynamic wash (loss of downforce) and tire surface overheating when closely following a competitor.

### 3.3 Track Evolution ($\rho \cdot \text{session\_time}_{d, t}$) & Cohort Identifiability
- **Identifiability Constraint**:
  - In single-driver data, tire degradation ($\nu > 0$, lap times increasing) and track rubber build-up ($\rho < 0$, lap times decreasing) are strictly collinear and unidentifiable.
  - **Hierarchical Cohort Resolution**: By pooling **multiple drivers** in the same session running stints at staggered start times, the shared parameter $\rho$ is uniquely identified.
- **Session Time Normalization**:
  $$\text{session\_time}_{d, t} = \frac{\text{LapStartTime}_{d, t} - T_{\text{session\_start}}}{T_{\text{session\_end}} - T_{\text{session\_start}}} \in [0, 1]$$
- **Track Evolution Prior ($\rho$)**:
  $$\rho \sim \mathcal{N}(-0.50, 0.25^2) \quad [\text{seconds / session}]$$
  Negative sign enforces that track evolution improves grip and lowers lap times over the course of a 60-minute session.

### 3.4 State Dynamics & Hierarchical Compound Prior Structure
- **Process Equation**:
  $$\alpha_{d, t+1} = (1 - I_{\text{pit}, d, t})(\alpha_{d, t} + \nu_{d, c(d,t)}) + I_{\text{pit}, d, t}(\alpha_{\text{reset}, d, c(d,t+1)}) + \eta_{d, t}$$
  $$\eta_{d, t} \sim \mathcal{N}(0, \sigma_\eta^2), \quad \sigma_\eta \sim \mathcal{N}^+(0.10, 0.05^2)$$
- **Hierarchical Compound Degradation ($\nu_{d, c}$)**:
  $$\nu_{d, c} \sim \mathcal{N}^+(\mu_{\nu, c}, \sigma_\nu^2)$$
  - Hard Compound: $\mu_{\nu, \text{Hard}} \sim \mathcal{N}^+(0.03, 0.02^2)$
  - Medium Compound: $\mu_{\nu, \text{Medium}} \sim \mathcal{N}^+(0.05, 0.02^2)$
  - Soft Compound: $\mu_{\nu, \text{Soft}} \sim \mathcal{N}^+(0.08, 0.03^2)$
  - Variance: $\sigma_\nu \sim \mathcal{N}^+(0.02, 0.01^2)$
- **Compound Pace Intercepts ($\alpha_{\text{reset}, d, c}$)**:
  $$\alpha_{\text{reset}, d, c} = \alpha_{\text{base}, d} + \Delta_c$$
  - Driver Baseline Pace: $\alpha_{\text{base}, d} \sim \mathcal{N}(\mu_{\text{base}}, 1.0^2)$
  - Compound Delta (relative to Medium reference $\Delta_{\text{Medium}} = 0.0$):
    - $\Delta_{\text{Hard}} \sim \mathcal{N}(+0.50, 0.20^2)$ (Hard is slower initially)
    - $\Delta_{\text{Soft}} \sim \mathcal{N}(-0.50, 0.20^2)$ (Soft is faster initially)

### 3.5 Observation Likelihood & Robustness
- **Gaussian Likelihood**:
  $$\epsilon_{d, t} \sim \mathcal{N}(0, \sigma_\epsilon^2), \quad \sigma_\epsilon \sim \mathcal{N}^+(0.30, 0.10^2)$$
- **Student-t Heavy-Tailed Likelihood**:
  $$\epsilon_{d, t} \sim \text{Student-}t(\nu_{df}=4, 0, \sigma_\epsilon^2)$$
  Provides robustness against minor driver errors, momentary off-tracks, and traffic interference without skewing the latent wear slope.

---

## 4. Ingestion Cleaning & Fallback Protocols

### 4.1 Excluded Lap Types
- **In-Laps / Out-Laps**: Pit in/out laps do not reflect representative tire degradation and are filtered using FastF1 flags `PitInTime` and `PitOutTime`.
- **Yellow Flags / Safety Car (SC) / Virtual Safety Car (VSC)**: Laps with delta-time restrictions (`TrackStatus` != '1') are excluded.
- **Outlier / In-lap Threshold**: Laps with lap times $\ge 107\%$ of driver median pace are filtered as slow cool-down or traffic-aborted laps.

### 4.2 Missing Gap-to-Car-Ahead Fallback
- **Derivation**: Gaps are computed from session timestamps of consecutive cars crossing sector timing loops.
- **Fallback Rule**: When car telemetry or timing stream is incomplete for a specific lap, the gap is assigned to $\ge 10.0\text{s}$ ($f(\text{gap}) = 0.0$), treating the lap as clean air.
- **Data Integrity Guarantee**: Values are never silently fabricated.

---

## 5. Practice $\to$ Race Validation Methodology

1. **Training Phase**: The hierarchical model is fitted exclusively on **Practice Data (FP2 long runs)** for the driver cohort, producing posterior estimates for clean wear rates $\nu_{d, c}$ and baseline pace $\alpha_{\text{reset}, d, c}$.
2. **Race Projection Phase**: For each race stint, the posterior predictive distribution of race lap times $\hat{y}_{d, t}^{\text{race}}$ is generated by applying the isolated practice wear rates to the race fuel curve and stint lengths:
   $$\hat{y}_{d, t}^{\text{race}} = \alpha_{\text{reset}, d, c(d,t)} + (t - t_{\text{stint\_start}}) \cdot \nu_{d, c(d,t)} + \gamma \cdot \text{fuel}_{\text{race}}(t) + \delta \cdot f(\text{gap}_{d, t}^{\text{race}}) + \rho \cdot \text{session\_time}_{d, t}^{\text{race}}$$
3. **Scoring Metrics**:
   - **RMSPE** (Root Mean Squared Predictive Error):
     $$\text{RMSPE} = \sqrt{\frac{1}{N} \sum_{t=1}^N (y_{d, t}^{\text{race}} - \hat{y}_{d, t}^{\text{race}})^2}$$
   - **CRPS** (Continuous Ranked Probability Score):
     $$\text{CRPS}(F, y) = \int_{-\infty}^\infty (F(z) - \mathbb{I}(z \ge y))^2 dz$$
     evaluates full probabilistic calibration and forecast sharpness.

---

## 6. Out-of-Scope Items for v1
- **Safety Car Tire Cooling Dynamics**: Safety car periods alter tire carcass temperature and pressure; v1 treats safety cars by lap exclusion rather than thermal modeling.
- **Wet / Intermediate Conditions**: Only dry slick compounds (Soft, Medium, Hard) are modeled in v1.
