"""
Bayesian State-Space Models for F1 Tyre Degradation Intelligence.
Implemented in NumPyro with JAX for ultra-fast, robust MCMC inference.
Includes:
1. Baseline single-driver fuel model (Paper replication)
2. Traffic-corrected SSM (Nonlinear gap decay)
3. Multi-driver Hierarchical SSM (Track evolution + Compound hierarchy + Pooling + Weather)
4. Practice-to-Race Predictive Validation Engine with RMSPE & CRPS scoring
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from typing import Dict, Any, Tuple, Optional, List


# --------------------------------------------------------------------------
# 1. NumPyro Model Definitions
# --------------------------------------------------------------------------

def baseline_ssm_model(
    lap_time: jnp.ndarray,
    fuel: jnp.ndarray,
    is_reset: jnp.ndarray,
    lap_in_stint: jnp.ndarray,
    num_laps: int
):
    """
    Phase 1 Baseline Model: Single-driver, race-only, fuel-corrected SSM.
    Observation: y_t = alpha_t + gamma * fuel_t + eps_t
    Process: alpha_t = alpha_reset + lap_in_stint * nu + random_walk_eta
    """
    # Priors
    sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.30))
    sigma_eta = numpyro.sample('sigma_eta', dist.HalfNormal(0.10))
    gamma = numpyro.sample('gamma', dist.Normal(0.033, 0.005))
    nu = numpyro.sample('nu', dist.HalfNormal(0.10))
    
    # Prior for initial fresh-tyre pace intercept
    mean_pace = jnp.median(lap_time) - 0.033 * jnp.median(fuel)
    alpha_reset = numpyro.sample('alpha_reset', dist.Normal(mean_pace, 1.5))
    
    # Process innovations (random walk)
    eta = numpyro.sample('eta', dist.Normal(0.0, 1.0), sample_shape=(num_laps,))
    eta_scaled = eta * sigma_eta
    
    # Accumulate innovations within stints
    # Using non-centered parameterization for efficient MCMC sampling
    def step_fn(prev_innov, reset_innov_pair):
        reset, raw_innov = reset_innov_pair
        current_innov = jnp.where(reset > 0, 0.0, prev_innov + raw_innov)
        return current_innov, current_innov
    
    _, innov_accum = jax.lax.scan(step_fn, 0.0, (is_reset, eta_scaled))
    
    alpha = alpha_reset + lap_in_stint * nu + innov_accum
    numpyro.deterministic('alpha', alpha)
    
    mu_y = alpha + gamma * fuel
    numpyro.deterministic('mu_y', mu_y)
    
    # Robust Student-t observation likelihood (handles minor driver errors / lock-ups)
    with numpyro.plate('data', num_laps):
        numpyro.sample('y_obs', dist.StudentT(df=4.0, loc=mu_y, scale=sigma_eps), obs=lap_time)


def traffic_ssm_model(
    lap_time: jnp.ndarray,
    fuel: jnp.ndarray,
    gap: jnp.ndarray,
    is_reset: jnp.ndarray,
    lap_in_stint: jnp.ndarray,
    num_laps: int
):
    """
    Phase 2 Model: Adds nonlinear traffic decay penalty delta * exp(-gap / tau).
    Observation: y_t = alpha_t + gamma * fuel_t + delta * exp(-gap_t / tau) + eps_t
    """
    sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.30))
    sigma_eta = numpyro.sample('sigma_eta', dist.HalfNormal(0.10))
    gamma = numpyro.sample('gamma', dist.Normal(0.033, 0.005))
    nu = numpyro.sample('nu', dist.HalfNormal(0.10))
    
    # Traffic parameters
    delta = numpyro.sample('delta', dist.HalfNormal(0.80))
    tau = numpyro.sample('tau', dist.TruncatedNormal(loc=1.5, scale=0.5, low=0.5, high=5.0))
    
    mean_pace = jnp.median(lap_time) - 0.033 * jnp.median(fuel)
    alpha_reset = numpyro.sample('alpha_reset', dist.Normal(mean_pace, 1.5))
    
    eta = numpyro.sample('eta', dist.Normal(0.0, 1.0), sample_shape=(num_laps,))
    eta_scaled = eta * sigma_eta
    
    def step_fn(prev_innov, reset_innov_pair):
        reset, raw_innov = reset_innov_pair
        current_innov = jnp.where(reset > 0, 0.0, prev_innov + raw_innov)
        return current_innov, current_innov
    
    _, innov_accum = jax.lax.scan(step_fn, 0.0, (is_reset, eta_scaled))
    
    alpha = alpha_reset + lap_in_stint * nu + innov_accum
    numpyro.deterministic('alpha', alpha)
    
    traffic_penalty = delta * jnp.exp(-gap / tau)
    numpyro.deterministic('traffic_penalty', traffic_penalty)
    
    mu_y = alpha + gamma * fuel + traffic_penalty
    numpyro.deterministic('mu_y', mu_y)
    
    with numpyro.plate('data', num_laps):
        numpyro.sample('y_obs', dist.StudentT(df=4.0, loc=mu_y, scale=sigma_eps), obs=lap_time)


def hierarchical_ssm_model(
    lap_time: jnp.ndarray,
    fuel: jnp.ndarray,
    gap: jnp.ndarray,
    session_time: jnp.ndarray,
    driver_id: jnp.ndarray,
    compound_id: jnp.ndarray,
    is_reset: jnp.ndarray,
    lap_in_stint: jnp.ndarray,
    num_laps: int,
    num_drivers: int,
    num_compounds: int = 3,
    track_temp_z: Optional[jnp.ndarray] = None,
    air_temp_z: Optional[jnp.ndarray] = None,
    humidity_z: Optional[jnp.ndarray] = None,
    wind_speed_z: Optional[jnp.ndarray] = None
):
    """
    Phase 3 & 4 Hierarchical Multivariate Model:
    - Multi-driver pooling with driver-specific base pace alpha_base[d]
    - Compound-specific hierarchical degradation rates nu[d, c]
    - Compound offset relative to Medium: Delta[Hard] > 0 (slower), Delta[Soft] < 0 (faster)
    - Session-shared track evolution rho * session_time
    - Session-shared traffic penalty delta * exp(-gap / tau)
    - Fuel correction gamma * fuel
    - Weather covariates (all pre-standardized, z-scored, per-lap):
        * track_temp_z modulates the DEGRADATION RATE nu multiplicatively via kappa_temp,
          since hotter track surface accelerates the physical wear process itself, not
          just instantaneous grip.
        * air_temp_z, humidity_z, wind_speed_z enter additively on lap time via
          beta_air, beta_humid, beta_wind, since these mostly affect grip/engine
          cooling/aero on a per-lap basis rather than cumulative wear.
      All four are optional and default to zero-vectors so the model still runs on
      sessions with no weather data (e.g. missing FastF1 weather channel).
    """
    sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.30))
    sigma_eta = numpyro.sample('sigma_eta', dist.HalfNormal(0.10))
    gamma = numpyro.sample('gamma', dist.Normal(0.033, 0.005))
    
    # Shared traffic parameters
    delta = numpyro.sample('delta', dist.HalfNormal(0.80))
    tau = numpyro.sample('tau', dist.TruncatedNormal(loc=1.5, scale=0.5, low=0.5, high=5.0))
    
    # Shared track evolution parameter (rho < 0 means track rubbers in / grips up over session)
    rho = numpyro.sample('rho', dist.Normal(-0.50, 0.25))
    
    # --- Weather covariates ---
    # Default to zero arrays (i.e. no-op) when weather channels are unavailable.
    if track_temp_z is None:
        track_temp_z = jnp.zeros(num_laps)
    if air_temp_z is None:
        air_temp_z = jnp.zeros(num_laps)
    if humidity_z is None:
        humidity_z = jnp.zeros(num_laps)
    if wind_speed_z is None:
        wind_speed_z = jnp.zeros(num_laps)
    
    # kappa_temp: fractional change in degradation rate per std-dev of track temp.
    # Weakly-informative, centered at 0 (no effect) but allows meaningful swings
    # since track temp is a well-established driver of thermal degradation.
    kappa_temp = numpyro.sample('kappa_temp', dist.Normal(0.0, 0.15))
    
    # Additive per-lap pace effects, kept small relative to fuel/traffic (second-order
    # corrections for grip/cooling/aero rather than primary confounds).
    beta_air = numpyro.sample('beta_air', dist.Normal(0.0, 0.05))
    beta_humid = numpyro.sample('beta_humid', dist.Normal(0.0, 0.05))
    beta_wind = numpyro.sample('beta_wind', dist.Normal(0.0, 0.05))
    
    # Hierarchical compound degradation rates: Hard (0), Medium (1), Soft (2)
    mu_nu_hard = numpyro.sample('mu_nu_hard', dist.HalfNormal(0.06))
    mu_nu_med = numpyro.sample('mu_nu_med', dist.HalfNormal(0.08))
    mu_nu_soft = numpyro.sample('mu_nu_soft', dist.HalfNormal(0.12))
    mu_nu_vec = jnp.stack([mu_nu_hard, mu_nu_med, mu_nu_soft])
    
    sigma_nu = numpyro.sample('sigma_nu', dist.HalfNormal(0.02))
    
    # Driver-compound specific degradation rates nu[d, c]
    nu_raw = numpyro.sample('nu_raw', dist.Normal(0.0, 1.0), sample_shape=(num_drivers, num_compounds))
    nu_dc = jnp.maximum(0.001, mu_nu_vec[None, :] + nu_raw * sigma_nu)
    numpyro.deterministic('nu_dc', nu_dc)
    
    # Compound delta pace intercepts (relative to Medium = 0)
    delta_hard = numpyro.sample('delta_hard', dist.Normal(0.50, 0.20))
    delta_soft = numpyro.sample('delta_soft', dist.Normal(-0.50, 0.20))
    compound_delta = jnp.stack([delta_hard, 0.0, delta_soft])
    
    # Driver baseline pace
    overall_median = jnp.median(lap_time) - 0.033 * jnp.median(fuel)
    mu_base = numpyro.sample('mu_base', dist.Normal(overall_median, 1.5))
    sigma_base = numpyro.sample('sigma_base', dist.HalfNormal(0.50))
    
    base_raw = numpyro.sample('base_raw', dist.Normal(0.0, 1.0), sample_shape=(num_drivers,))
    alpha_base = mu_base + base_raw * sigma_base
    numpyro.deterministic('alpha_base', alpha_base)
    
    # Calculate reset pace for each lap based on driver and compound
    # alpha_reset[d, c] = alpha_base[d] + compound_delta[c]
    lap_alpha_reset = alpha_base[driver_id] + compound_delta[compound_id]
    
    # Get active degradation rate for each lap, modulated by track temperature.
    # (1 + kappa_temp * track_temp_z) keeps the effect multiplicative and centered
    # at 1.0 when track_temp_z == 0 (average conditions), floored so nu stays positive.
    lap_nu_base = nu_dc[driver_id, compound_id]
    lap_nu = jnp.maximum(0.0005, lap_nu_base * (1.0 + kappa_temp * track_temp_z))
    numpyro.deterministic('lap_nu_effective', lap_nu)
    
    # Random walk innovations
    eta = numpyro.sample('eta', dist.Normal(0.0, 1.0), sample_shape=(num_laps,))
    eta_scaled = eta * sigma_eta
    
    def step_fn(prev_innov, reset_innov_pair):
        reset, raw_innov = reset_innov_pair
        current_innov = jnp.where(reset > 0, 0.0, prev_innov + raw_innov)
        return current_innov, current_innov
    
    _, innov_accum = jax.lax.scan(step_fn, 0.0, (is_reset, eta_scaled))
    
    alpha = lap_alpha_reset + lap_in_stint * lap_nu + innov_accum
    numpyro.deterministic('alpha', alpha)
    
    traffic_term = delta * jnp.exp(-gap / tau)
    track_evo_term = rho * session_time
    fuel_term = gamma * fuel
    weather_term = beta_air * air_temp_z + beta_humid * humidity_z + beta_wind * wind_speed_z
    
    numpyro.deterministic('traffic_term', traffic_term)
    numpyro.deterministic('track_evo_term', track_evo_term)
    numpyro.deterministic('weather_term', weather_term)
    
    mu_y = alpha + fuel_term + traffic_term + track_evo_term + weather_term
    numpyro.deterministic('mu_y', mu_y)
    
    with numpyro.plate('data', num_laps):
        numpyro.sample('y_obs', dist.StudentT(df=4.0, loc=mu_y, scale=sigma_eps), obs=lap_time)


# --------------------------------------------------------------------------
# 2. Inference & Fitting Harness
# --------------------------------------------------------------------------

def compute_lap_in_stint(dataframe: pd.DataFrame) -> np.ndarray:
    """Computes zero-indexed lap count within current stint."""
    laps_in_stint = []
    for (driver, stint), group in dataframe.groupby(['Driver', 'Stint'], sort=False):
        laps_in_stint.extend(list(range(len(group))))
    return np.array(laps_in_stint, dtype=np.float32)


def _zscore(raw: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Standardizes a raw covariate array. Returns (z_array, mean, std).
    Falls back to std=1.0 if the raw data is degenerate (e.g. constant/missing),
    so downstream coefficients don't blow up.
    """
    raw = np.asarray(raw, dtype=np.float64)
    mean = float(np.nanmean(raw)) if np.isfinite(raw).any() else 0.0
    std = float(np.nanstd(raw))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    z = (np.nan_to_num(raw, nan=mean) - mean) / std
    return z.astype(np.float32), mean, std


def fit_ssm(
    model_name: str,
    dataset: Dict[str, Any],
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Fits the requested SSM using NumPyro NUTS MCMC.
    
    Parameters
    ----------
    model_name : str
        'baseline', 'traffic', or 'hierarchical'.
    dataset : dict
        Dataset dict from data_pipeline.prepare_dataset_for_ssm.
    num_warmup : int
        Number of warmup / burn-in iterations.
    num_samples : int
        Number of posterior samples to draw.
    num_chains : int
        Number of MCMC chains.
    seed : int
        Random seed for JAX PRNG.

    Notes on weather (hierarchical model only)
    -------------------------------------------
    Raw weather covariates (track_temp, air_temp, humidity, wind_speed) are
    standardized HERE, on whatever dataset is being fit (typically Practice).
    The (mean, std) used are stored in the returned dict under 'weather_norm_stats'
    so that `predict_race_from_practice_samples` can apply the *same* transform
    to the Race dataset's raw weather values — this is required for the learned
    beta/kappa coefficients to mean the same thing at prediction time as they did
    during fitting.
    """
    df = dataset['dataframe']
    lap_in_stint = compute_lap_in_stint(df)
    
    rng_key = jax.random.PRNGKey(seed)
    nuts_kernel = NUTS(
        baseline_ssm_model if model_name == 'baseline' else
        traffic_ssm_model if model_name == 'traffic' else
        hierarchical_ssm_model
    )
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    
    weather_norm_stats = None
    
    if model_name == 'baseline':
        mcmc.run(
            rng_key,
            lap_time=jnp.array(dataset['lap_time']),
            fuel=jnp.array(dataset['fuel']),
            is_reset=jnp.array(dataset['is_reset']),
            lap_in_stint=jnp.array(lap_in_stint),
            num_laps=dataset['num_laps']
        )
    elif model_name == 'traffic':
        mcmc.run(
            rng_key,
            lap_time=jnp.array(dataset['lap_time']),
            fuel=jnp.array(dataset['fuel']),
            gap=jnp.array(dataset['gap']),
            is_reset=jnp.array(dataset['is_reset']),
            lap_in_stint=jnp.array(lap_in_stint),
            num_laps=dataset['num_laps']
        )
    elif model_name == 'hierarchical':
        track_temp_z, tt_mean, tt_std = _zscore(dataset.get('track_temp', np.zeros(dataset['num_laps'])))
        air_temp_z, at_mean, at_std = _zscore(dataset.get('air_temp', np.zeros(dataset['num_laps'])))
        humidity_z, hu_mean, hu_std = _zscore(dataset.get('humidity', np.zeros(dataset['num_laps'])))
        wind_z, wi_mean, wi_std = _zscore(dataset.get('wind_speed', np.zeros(dataset['num_laps'])))
        
        weather_norm_stats = {
            'track_temp': (tt_mean, tt_std),
            'air_temp': (at_mean, at_std),
            'humidity': (hu_mean, hu_std),
            'wind_speed': (wi_mean, wi_std),
        }
        
        mcmc.run(
            rng_key,
            lap_time=jnp.array(dataset['lap_time']),
            fuel=jnp.array(dataset['fuel']),
            gap=jnp.array(dataset['gap']),
            session_time=jnp.array(dataset['session_time']),
            driver_id=jnp.array(dataset['driver_id']),
            compound_id=jnp.array(dataset['compound_id']),
            is_reset=jnp.array(dataset['is_reset']),
            lap_in_stint=jnp.array(lap_in_stint),
            num_laps=dataset['num_laps'],
            num_drivers=dataset['num_drivers'],
            num_compounds=dataset['num_compounds'],
            track_temp_z=jnp.array(track_temp_z),
            air_temp_z=jnp.array(air_temp_z),
            humidity_z=jnp.array(humidity_z),
            wind_speed_z=jnp.array(wind_z)
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    samples = mcmc.get_samples()
    
    # Compute summary statistics
    summary = {}
    for param_name, vals in samples.items():
        if vals.ndim == 1:
            summary[param_name] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'q025': float(np.percentile(vals, 2.5)),
                'q50': float(np.percentile(vals, 50.0)),
                'q975': float(np.percentile(vals, 97.5))
            }
            
    result = {
        'model_name': model_name,
        'mcmc': mcmc,
        'samples': samples,
        'summary': summary,
        'dataset': dataset
    }
    if weather_norm_stats is not None:
        result['weather_norm_stats'] = weather_norm_stats
        
    return result


# --------------------------------------------------------------------------
# 3. Practice -> Race Validation Engine & Metrics
# --------------------------------------------------------------------------

def compute_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Predictive Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_crps_samples(y_true: np.ndarray, y_samples: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score (CRPS) computed empirically from MCMC predictive samples.
    y_true: shape (N,)
    y_samples: shape (S, N) where S is number of posterior samples.
    """
    # CRPS(F, y) = E|Y - y| - 0.5 * E|Y - Y'|
    s, n = y_samples.shape
    term1 = np.mean(np.abs(y_samples - y_true[None, :]), axis=0) # (N,)
    
    # Compute E|Y - Y'| efficiently via sorting
    sorted_samples = np.sort(y_samples, axis=0) # (S, N)
    weights = (2.0 * np.arange(1, s + 1) - s - 1)[:, None] # (S, 1)
    term2 = (2.0 / (s * (s - 1))) * np.sum(weights * sorted_samples, axis=0)
    
    crps_per_lap = term1 - 0.5 * term2
    return float(np.mean(crps_per_lap))


def _apply_norm_stats(raw: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Applies a previously-fit (mean, std) transform to a new raw array."""
    raw = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=mean)
    std = std if (np.isfinite(std) and std > 1e-6) else 1.0
    return ((raw - mean) / std).astype(np.float32)


def predict_race_from_practice_samples(
    practice_samples: Dict[str, np.ndarray],
    race_dataset: Dict[str, Any],
    weather_norm_stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Forward extrapolation engine:
    Takes posterior parameter samples fitted on Practice sessions (isolating true wear rates nu),
    and predicts race lap times across all race stints using the race fuel burn curve.

    Parameters
    ----------
    weather_norm_stats : dict, optional
        The (mean, std) per weather covariate learned from the PRACTICE dataset
        (returned by `fit_ssm` as `result['weather_norm_stats']`). Race weather is
        standardized using these same stats -- not the race dataset's own mean/std --
        so that the practice-fitted beta/kappa coefficients apply consistently.
        If omitted, weather terms are treated as zero (no-op), matching pre-weather
        behaviour.
    """
    df_race = race_dataset['dataframe']
    lap_in_stint = compute_lap_in_stint(df_race)
    num_laps = race_dataset['num_laps']
    driver_ids = race_dataset['driver_id']
    compound_ids = race_dataset['compound_id']
    fuel = race_dataset['fuel']
    gap = race_dataset['gap']
    session_time = race_dataset['session_time']
    y_true = race_dataset['lap_time']
    
    # Extract posterior samples
    num_samples = len(practice_samples['sigma_eps'])
    
    gamma_samples = practice_samples.get('gamma', np.full(num_samples, 0.033))
    delta_samples = practice_samples.get('delta', np.full(num_samples, 0.80))
    tau_samples = practice_samples.get('tau', np.full(num_samples, 1.5))
    rho_samples = practice_samples.get('rho', np.full(num_samples, -0.50))
    sigma_eps_samples = practice_samples.get('sigma_eps', np.full(num_samples, 0.30))
    
    # Weather posterior samples (absent -> zero effect, fully backward compatible)
    kappa_temp_samples = practice_samples.get('kappa_temp', np.zeros(num_samples))
    beta_air_samples = practice_samples.get('beta_air', np.zeros(num_samples))
    beta_humid_samples = practice_samples.get('beta_humid', np.zeros(num_samples))
    beta_wind_samples = practice_samples.get('beta_wind', np.zeros(num_samples))
    
    if weather_norm_stats is not None:
        track_temp_z = _apply_norm_stats(
            race_dataset.get('track_temp', np.zeros(num_laps)), *weather_norm_stats['track_temp']
        )
        air_temp_z = _apply_norm_stats(
            race_dataset.get('air_temp', np.zeros(num_laps)), *weather_norm_stats['air_temp']
        )
        humidity_z = _apply_norm_stats(
            race_dataset.get('humidity', np.zeros(num_laps)), *weather_norm_stats['humidity']
        )
        wind_z = _apply_norm_stats(
            race_dataset.get('wind_speed', np.zeros(num_laps)), *weather_norm_stats['wind_speed']
        )
    else:
        track_temp_z = np.zeros(num_laps, dtype=np.float32)
        air_temp_z = np.zeros(num_laps, dtype=np.float32)
        humidity_z = np.zeros(num_laps, dtype=np.float32)
        wind_z = np.zeros(num_laps, dtype=np.float32)
    
    if 'nu_dc' in practice_samples:
        nu_dc_samples = practice_samples['nu_dc'] # shape (S, D, C)
    else:
        nu_val = practice_samples.get('nu', np.full(num_samples, 0.05))
        nu_dc_samples = np.tile(nu_val[:, None, None], (1, race_dataset['num_drivers'], 3))
        
    if 'alpha_base' in practice_samples:
        alpha_base_samples = practice_samples['alpha_base'] # shape (S, D)
        delta_hard = practice_samples.get('delta_hard', np.full(num_samples, 0.50))
        delta_soft = practice_samples.get('delta_soft', np.full(num_samples, -0.50))
        comp_delta = np.stack([delta_hard, np.zeros(num_samples), delta_soft], axis=1) # (S, 3)
    else:
        alpha_reset_val = practice_samples.get('alpha_reset', np.full(num_samples, 70.0))
        alpha_base_samples = np.tile(alpha_reset_val[:, None], (1, race_dataset['num_drivers']))
        comp_delta = np.zeros((num_samples, 3))
        
    # Generate race lap predictions for each posterior sample
    pred_y_samples = np.zeros((num_samples, num_laps), dtype=np.float32)
    clean_alpha_samples = np.zeros((num_samples, num_laps), dtype=np.float32)
    
    for s in range(num_samples):
        # Driver base pace + compound offset
        resets = alpha_base_samples[s, driver_ids] + comp_delta[s, compound_ids]
        # Active degradation rate for each lap, modulated by race-day track temperature
        # using the SAME kappa_temp learned from practice.
        nus_base = nu_dc_samples[s, driver_ids, compound_ids]
        nus = np.maximum(0.0005, nus_base * (1.0 + kappa_temp_samples[s] * track_temp_z))
        
        # Clean degradation trajectory (now temperature-adjusted for race conditions)
        alpha_s = resets + lap_in_stint * nus
        clean_alpha_samples[s] = alpha_s
        
        # Add race-day covariates
        fuel_s = gamma_samples[s] * fuel
        traffic_s = delta_samples[s] * np.exp(-gap / tau_samples[s])
        track_evo_s = rho_samples[s] * session_time
        weather_s = (
            beta_air_samples[s] * air_temp_z
            + beta_humid_samples[s] * humidity_z
            + beta_wind_samples[s] * wind_z
        )
        
        # Observation noise
        noise_s = np.random.standard_t(df=4.0, size=num_laps) * sigma_eps_samples[s]
        
        pred_y_samples[s] = alpha_s + fuel_s + traffic_s + track_evo_s + weather_s + noise_s
        
    pred_mean = np.mean(pred_y_samples, axis=0)
    pred_q025 = np.percentile(pred_y_samples, 2.5, axis=0)
    pred_q05 = np.percentile(pred_y_samples, 5.0, axis=0)
    pred_q50 = np.percentile(pred_y_samples, 50.0, axis=0)
    pred_q95 = np.percentile(pred_y_samples, 95.0, axis=0)
    pred_q975 = np.percentile(pred_y_samples, 97.5, axis=0)

    # Posterior mean + 95% CI of the CLEAN decoupled latent tyre pace itself
    # (zero fuel/traffic/observation-noise), so dashboards can plot the
    # model's actual fitted alpha trajectory + uncertainty instead of
    # resynthesizing a simplified mean-only line from a single nu estimate.
    clean_alpha_q025 = np.percentile(clean_alpha_samples, 2.5, axis=0)
    clean_alpha_q975 = np.percentile(clean_alpha_samples, 97.5, axis=0)
    
    overall_rmspe = compute_rmspe(y_true, pred_mean)
    overall_crps = compute_crps_samples(y_true, pred_y_samples)
    
    # Stint-level metrics
    stint_metrics = []
    for (driver, stint_num), group in df_race.groupby(['Driver', 'Stint']):
        indices = group.index.values
        stint_y_true = y_true[indices]
        stint_y_pred = pred_mean[indices]
        stint_y_samples = pred_y_samples[:, indices]
        
        rmspe = compute_rmspe(stint_y_true, stint_y_pred)
        crps = compute_crps_samples(stint_y_true, stint_y_samples)
        stint_metrics.append({
            'Driver': driver,
            'Stint': int(stint_num),
            'Compound': group['Compound'].iloc[0],
            'Laps': len(group),
            'RMSPE': rmspe,
            'CRPS': crps
        })
        
    return {
        'pred_mean': pred_mean,
        'pred_q05': pred_q05,
        'pred_q95': pred_q95,
        'pred_q025': pred_q025,
        'pred_q975': pred_q975,
        'pred_samples': pred_y_samples,
        'clean_alpha_mean': np.mean(clean_alpha_samples, axis=0),
        'clean_alpha_q025': clean_alpha_q025,
        'clean_alpha_q975': clean_alpha_q975,
        'overall_rmspe': overall_rmspe,
        'overall_crps': overall_crps,
        'stint_metrics': pd.DataFrame(stint_metrics),
        'dataframe': df_race
    }