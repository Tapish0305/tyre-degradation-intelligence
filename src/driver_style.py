"""
Driver Style / Aggression Indicator for F1 Tyre Degradation Intelligence.

Implements a jerk-based driving style indicator derived from car telemetry,
following the published methodology of computing longitudinal and lateral
jerk distributions from vehicle acceleration signals:

    j_x = d(a_x)/dt          (longitudinal jerk)
    j_y = d(a_y)/dt          (lateral jerk)

    Longitudinal DSI = std(j_x)
    Lateral DSI       = std(j_y)

IMPORTANT LABELING NOTE (read before using these outputs anywhere user-facing):
--------------------------------------------------------------------------
The published research defines the driving-style indicator as the two
SEPARATE components above (longitudinal and lateral jerk standard deviation).
This module additionally computes a Euclidean combination of the two,

    DSI_combined = sqrt(sigma_jx^2 + sigma_jy^2)

purely as a single-number convenience for a dashboard tile. DSI_combined is
NOT part of the original published methodology and must always be labeled
as "a combined project representation of the two published components" --
never as an official F1 metric or as the research metric itself. The two
underlying components (Longitudinal DSI, Lateral DSI) should always be
retained and surfaced alongside the combined figure.

The overall indicator produced by this module should be presented to users as:

    "Driver Style / Aggression Indicator"   (preferred)
    "Driver Tyre-Usage Aggression Index"    (alternative, tyre-usage framing)

...and NEVER as an official F1 driver characteristic or rating.

Interpretation
--------------
Higher DSI  -> more abrupt changes in acceleration -> more aggressive car usage.
Lower DSI   -> smoother inputs -> more measured/conservative car usage.

Supporting (non-jerk) metrics
------------------------------
Alongside the jerk-based DSI (which remains the PRIMARY research-backed
metric), this module also computes:
  - heavy braking frequency
  - braking intensity
  - cornering intensity
  - throttle application behaviour
  - acceleration (traction) behaviour

These are reported as separate, independent columns -- they are NEVER
manually weighted or collapsed into a single score by this module. If a
single supplementary number is wanted, `compute_pca_composite_style_score`
derives weights from the data itself (first principal component of the
standardized secondary metrics) rather than assigning them by hand. That
composite is explicitly optional, exploratory, and secondary to DSI_combined.

Data requirements
------------------
Telemetry + position channels are required, which are NOT loaded by the
default `data_pipeline.load_f1_session(...)` call (it uses telemetry=False
to keep the SSM pipeline fast). Load sessions for this module with:

    session = load_f1_session(year, gp, session_type, load_telemetry=True)

fastf1's `Lap.get_telemetry()` requires this to return merged car + position
(X, Y) channels; without it, position data will be unavailable and lateral
jerk cannot be computed.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Any

try:
    import fastf1
except ImportError:  # pragma: no cover - fastf1 is a hard dependency elsewhere in the project
    fastf1 = None


DSI_COMBINED_LABEL = (
    "Driver Style / Aggression Indicator (combined project representation of "
    "published longitudinal + lateral jerk DSI; not an official F1 metric)"
)

# Minimum number of uniformly-resampled telemetry points required to attempt
# two rounds of differentiation (raw signal -> accel -> jerk) with any
# numerical stability. Short/aborted laps below this are skipped.
_MIN_SAMPLES = 30


# --------------------------------------------------------------------------
# 1. Low-level kinematics: filter -> differentiate -> filter -> differentiate
# --------------------------------------------------------------------------

def _resample_uniform(
    time_s: np.ndarray,
    signals: Dict[str, np.ndarray],
    dt: float = 0.10
) -> Optional[Dict[str, np.ndarray]]:
    """
    Interpolates one or more telemetry signals onto a uniform time grid.
    Required because raw FastF1 telemetry samples are not evenly spaced,
    and numerical differentiation (twice, for jerk) is unstable on an
    irregular grid.

    Returns None if there isn't enough usable data.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    valid = np.isfinite(time_s)
    if valid.sum() < _MIN_SAMPLES:
        return None

    order = np.argsort(time_s[valid])
    t_valid = time_s[valid][order]
    # Drop duplicate timestamps (can occur in telemetry merges) which break interp
    keep = np.concatenate([[True], np.diff(t_valid) > 1e-6])
    t_valid = t_valid[keep]
    if len(t_valid) < _MIN_SAMPLES:
        return None

    t0, t1 = t_valid[0], t_valid[-1]
    if t1 - t0 < dt * _MIN_SAMPLES:
        return None

    t_uniform = np.arange(t0, t1, dt)
    out = {'time_s': t_uniform}
    for name, arr in signals.items():
        arr = np.asarray(arr, dtype=np.float64)[valid][order][keep]
        out[name] = np.interp(t_uniform, t_valid, arr)
    return out


def _smooth(x: np.ndarray, window: int = 21, polyorder: int = 3) -> np.ndarray:
    """
    Savitzky-Golay smoothing to suppress telemetry noise before/after each
    differentiation step. Window is clamped to be odd and no larger than the
    signal itself, so short laps still produce a (lightly) smoothed result
    instead of raising.
    """
    n = len(x)
    w = min(window, n - (1 - n % 2))  # ensure w <= n and odd
    if w < polyorder + 2:
        return x.copy()
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        return x.copy()
    return savgol_filter(x, window_length=w, polyorder=polyorder, mode='interp')


def compute_lap_kinematics(
    lap: "fastf1.core.Lap",
    resample_dt: float = 0.10,
    smooth_window: int = 21,
    smooth_polyorder: int = 3
) -> Optional[Dict[str, np.ndarray]]:
    """
    Derives longitudinal/lateral acceleration and jerk from a single lap's
    telemetry, following filter -> differentiate -> filter -> differentiate:

      1. Pull merged car + position telemetry (Speed, X, Y, Throttle, Brake).
      2. Resample onto a uniform time grid (required for stable differentiation).
      3. Smooth Speed and (X, Y) to suppress sensor/logging noise.
      4. a_x = d(speed_m_s)/dt                         (longitudinal accel)
         a_y = speed_m_s^2 * path curvature            (lateral accel, from
                                                          heading change rate)
      5. Smooth a_x, a_y (second filtering pass, before differentiating again).
      6. j_x = d(a_x)/dt, j_y = d(a_y)/dt               (jerk)

    Returns None if the lap has insufficient/unusable telemetry (e.g. an
    in/out lap with a very short telemetry window, or missing position data).
    """
    try:
        telem = lap.get_telemetry()
    except Exception:
        return None

    if telem is None or len(telem) < _MIN_SAMPLES:
        return None
    if 'X' not in telem.columns or 'Y' not in telem.columns:
        return None  # position data required for lateral acceleration/jerk

    time_s = telem['Time'].dt.total_seconds().values
    signals = {
        'speed_kmh': telem['Speed'].values if 'Speed' in telem.columns else np.full(len(telem), np.nan),
        'x': telem['X'].values,
        'y': telem['Y'].values,
        'throttle': telem['Throttle'].values if 'Throttle' in telem.columns else np.full(len(telem), np.nan),
        'brake': telem['Brake'].astype(float).values if 'Brake' in telem.columns else np.full(len(telem), np.nan),
    }

    grid = _resample_uniform(time_s, signals, dt=resample_dt)
    if grid is None:
        return None

    t = grid['time_s']
    speed_mps = _smooth(grid['speed_kmh'] / 3.6, window=smooth_window, polyorder=smooth_polyorder)
    x_s = _smooth(grid['x'], window=smooth_window, polyorder=smooth_polyorder)
    y_s = _smooth(grid['y'], window=smooth_window, polyorder=smooth_polyorder)

    # --- Longitudinal acceleration: derivative of smoothed speed ---
    a_x = np.gradient(speed_mps, t)
    a_x = _smooth(a_x, window=smooth_window, polyorder=smooth_polyorder)

    # --- Lateral acceleration: v^2 * curvature, curvature from heading rate ---
    dx = np.gradient(x_s, t)
    dy = np.gradient(y_s, t)
    heading = np.unwrap(np.arctan2(dy, dx))
    ds = np.sqrt(dx ** 2 + dy ** 2)  # instantaneous path speed (units/s), matches speed_mps
    ds_safe = np.where(ds < 1e-3, np.nan, ds)
    curvature = np.gradient(heading, t) / ds_safe  # d(heading)/ds = d(heading)/dt / (ds/dt)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    a_y = (speed_mps ** 2) * curvature
    a_y = _smooth(a_y, window=smooth_window, polyorder=smooth_polyorder)

    # --- Jerk: second differentiation, on the already-smoothed accelerations ---
    j_x = np.gradient(a_x, t)
    j_y = np.gradient(a_y, t)

    return {
        'time_s': t,
        'speed_mps': speed_mps,
        'a_x': a_x,
        'a_y': a_y,
        'j_x': j_x,
        'j_y': j_y,
        'throttle': np.interp(t, grid['time_s'], grid['throttle']) if np.isfinite(grid['throttle']).any() else np.full(len(t), np.nan),
        'brake': np.interp(t, grid['time_s'], grid['brake']) if np.isfinite(grid['brake']).any() else np.full(len(t), np.nan),
    }


# --------------------------------------------------------------------------
# 2. Per-lap style metrics
# --------------------------------------------------------------------------

def compute_lap_style_metrics(
    lap: "fastf1.core.Lap",
    resample_dt: float = 0.10,
    smooth_window: int = 21,
    smooth_polyorder: int = 3,
    heavy_brake_g_threshold: float = -1.5
) -> Optional[Dict[str, float]]:
    """
    Computes the jerk-based DSI (primary, research-backed) plus supporting
    braking/cornering/throttle/acceleration descriptors for a single lap.

    All supporting metrics are returned as SEPARATE fields -- none are
    combined into a single number here. See module docstring re: labeling
    of DSI_combined.
    """
    kin = compute_lap_kinematics(lap, resample_dt, smooth_window, smooth_polyorder)
    if kin is None:
        return None

    a_x, a_y = kin['a_x'], kin['a_y']
    j_x, j_y = kin['j_x'], kin['j_y']
    throttle, brake = kin['throttle'], kin['brake']

    sigma_jx = float(np.std(j_x))
    sigma_jy = float(np.std(j_y))
    dsi_combined = float(np.sqrt(sigma_jx ** 2 + sigma_jy ** 2))

    # --- Braking descriptors ---
    braking_mask = a_x < -0.5  # decelerating meaningfully (m/s^2), independent of Brake channel noise
    if brake is not None and np.isfinite(brake).any():
        # Prefer the actual brake signal when available (more direct than inferring from a_x)
        braking_mask = braking_mask | (brake > 0.5)

    # Count discrete braking *events* (rising edges) per lap, normalized to
    # a per-minute rate so laps of slightly different length are comparable.
    braking_edges = np.diff(braking_mask.astype(int))
    n_brake_events = int(np.sum(braking_edges == 1))
    lap_duration_s = float(kin['time_s'][-1] - kin['time_s'][0]) if len(kin['time_s']) > 1 else np.nan
    heavy_braking_freq_per_min = (
        n_brake_events / (lap_duration_s / 60.0) if lap_duration_s and lap_duration_s > 0 else np.nan
    )

    braking_ax = a_x[braking_mask]
    braking_intensity_mean = float(np.mean(braking_ax)) if braking_ax.size > 0 else np.nan
    braking_intensity_p90 = float(np.percentile(braking_ax, 10)) if braking_ax.size > 0 else np.nan
    # (10th percentile of a_x during braking == the strongest ~10% of braking events,
    #  since a_x is negative under braking)

    # --- Cornering descriptor ---
    cornering_intensity_p90 = float(np.percentile(np.abs(a_y), 90)) if a_y.size > 0 else np.nan

    # --- Throttle descriptors ---
    if throttle is not None and np.isfinite(throttle).any():
        throttle_full_pct = float(np.mean(throttle >= 99.0) * 100.0)
        dthrottle = np.gradient(throttle, kin['time_s'])
        rising = dthrottle[dthrottle > 0]
        throttle_application_rate = float(np.mean(rising)) if rising.size > 0 else np.nan
    else:
        throttle_full_pct = np.nan
        throttle_application_rate = np.nan

    # --- Acceleration (traction) descriptor ---
    accel_mask = a_x > 0.5
    accel_intensity_mean = float(np.mean(a_x[accel_mask])) if np.any(accel_mask) else np.nan

    return {
        'Longitudinal_DSI': sigma_jx,
        'Lateral_DSI': sigma_jy,
        'DSI_combined': dsi_combined,
        'HeavyBrakingFreq_per_min': heavy_braking_freq_per_min,
        'BrakingIntensity_mean_ms2': braking_intensity_mean,
        'BrakingIntensity_p90_ms2': braking_intensity_p90,
        'CorneringIntensity_p90_ms2': cornering_intensity_p90,
        'ThrottleFullPct': throttle_full_pct,
        'ThrottleApplicationRate': throttle_application_rate,
        'AccelIntensity_mean_ms2': accel_intensity_mean,
        'LapDuration_s': lap_duration_s,
    }


# --------------------------------------------------------------------------
# 3. Per-driver profile across a session (or a set of pre-cleaned laps)
# --------------------------------------------------------------------------

def compute_driver_style_profile(
    session: "fastf1.core.Session",
    drivers: List[str],
    clean_laps_df: Optional[pd.DataFrame] = None,
    resample_dt: float = 0.10,
    smooth_window: int = 21,
    smooth_polyorder: int = 3,
    min_valid_laps: int = 3
) -> pd.DataFrame:
    """
    Builds a per-driver Driver Style / Aggression Indicator profile across a
    session.

    Parameters
    ----------
    session : fastf1.core.Session
        Must have been loaded with telemetry=True (see load_f1_session in
        data_pipeline.py, `load_telemetry=True`).
    drivers : list of str
        Driver codes/numbers to profile.
    clean_laps_df : pd.DataFrame, optional
        If provided (e.g. the output of data_pipeline.extract_clean_laps),
        restricts analysis to those already-cleaned laps (excludes pit
        in/out laps, SC/VSC laps, and stint outliers) so the style profile
        isn't skewed by non-representative laps. If omitted, all of the
        driver's laps in the session are attempted.
    min_valid_laps : int
        Minimum number of laps with usable telemetry required to report a
        driver's profile; drivers below this are reported with NaNs and a
        note, rather than a profile built on too little data.

    Aggregation uses the MEDIAN across a driver's laps for every metric
    (robust to a single scruffy or traffic-affected lap), never a mean of
    means or any hand-weighted blend.

    Returns
    -------
    pd.DataFrame indexed by Driver, one row per driver, with median values
    of every metric from compute_lap_style_metrics, plus NumLapsUsed and
    NumLapsAttempted.
    """
    rows = []

    for driver in drivers:
        driver_laps = session.laps.pick_drivers(driver) if hasattr(session.laps, 'pick_drivers') else session.laps.pick_driver(driver)

        if clean_laps_df is not None:
            allowed_lap_numbers = set(
                clean_laps_df.loc[clean_laps_df['Driver'].astype(str).str.upper() == str(driver).upper(), 'LapNumber']
            )
            driver_laps = driver_laps[driver_laps['LapNumber'].isin(allowed_lap_numbers)]

        per_lap_metrics = []
        for _, lap in driver_laps.iterrows():
            metrics = compute_lap_style_metrics(
                lap, resample_dt=resample_dt, smooth_window=smooth_window, smooth_polyorder=smooth_polyorder
            )
            if metrics is not None:
                per_lap_metrics.append(metrics)

        num_attempted = len(driver_laps)
        num_used = len(per_lap_metrics)

        if num_used < min_valid_laps:
            rows.append({
                'Driver': driver,
                'NumLapsAttempted': num_attempted,
                'NumLapsUsed': num_used,
                'Note': f'Fewer than {min_valid_laps} laps with usable telemetry -- profile not computed.'
            })
            continue

        metrics_df = pd.DataFrame(per_lap_metrics)
        profile_row = {'Driver': driver, 'NumLapsAttempted': num_attempted, 'NumLapsUsed': num_used, 'Note': ''}
        for col in metrics_df.columns:
            profile_row[col] = float(metrics_df[col].median(skipna=True))
        rows.append(profile_row)

    profile_df = pd.DataFrame(rows).set_index('Driver')
    return profile_df


# --------------------------------------------------------------------------
# 4. Optional data-driven composite of the SECONDARY (non-jerk) metrics
# --------------------------------------------------------------------------

_SECONDARY_METRIC_COLS = [
    'HeavyBrakingFreq_per_min',
    'BrakingIntensity_p90_ms2',   # already negative-signed (harder braking = more negative)
    'CorneringIntensity_p90_ms2',
    'ThrottleFullPct',
    'ThrottleApplicationRate',
    'AccelIntensity_mean_ms2',
]


def compute_pca_composite_style_score(
    profile_df: pd.DataFrame,
    secondary_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    OPTIONAL, exploratory: derives a single composite score from the
    secondary (non-jerk) style metrics using PCA -- i.e. weights come from
    the data's own variance structure via the first principal component,
    not from manually-assigned coefficients.

    This is intentionally kept separate from DSI_combined. The jerk-based
    DSI remains the primary, research-backed aggression metric; this
    composite is a supplementary, data-driven summary of the braking/
    cornering/throttle descriptors only, for situations where a dashboard
    wants one extra number instead of five separate bars.

    Sign is NOT guaranteed by PCA (PC1 could point either direction), so the
    component is oriented so that higher = more aggressive by flipping sign
    if BrakingIntensity_p90_ms2's loading (which is more negative for
    harder braking) would otherwise make the score run backwards.

    Returns profile_df with two extra columns:
      - SecondaryStyleScore_PC1
      - SecondaryStyleScore_ExplainedVarianceRatio (same value repeated per
        row, so callers can display "PC1 explains X% of variance" next to
        the score without a second return value)
    """
    cols = secondary_cols if secondary_cols is not None else _SECONDARY_METRIC_COLS
    available_cols = [c for c in cols if c in profile_df.columns]

    valid_rows = profile_df.dropna(subset=available_cols)
    out = profile_df.copy()
    out['SecondaryStyleScore_PC1'] = np.nan
    out['SecondaryStyleScore_ExplainedVarianceRatio'] = np.nan

    if len(available_cols) < 2 or len(valid_rows) < 3:
        # Not enough drivers or metrics to fit a meaningful PCA -- leave as NaN
        # rather than fabricate a composite from too little data.
        return out

    X = valid_rows[available_cols].values.astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    X_z = (X - mean) / std

    # PCA via SVD (no sklearn dependency)
    u, s, vt = np.linalg.svd(X_z, full_matrices=False)
    pc1_loadings = vt[0]
    pc1_scores = u[:, 0] * s[0]

    # Orient sign: braking intensity is stored as a more-negative-is-harder
    # value, so a "more aggressive" driver should have a MORE NEGATIVE
    # BrakingIntensity_p90_ms2 but a HIGHER composite score. Flip if needed.
    if 'BrakingIntensity_p90_ms2' in available_cols:
        idx = available_cols.index('BrakingIntensity_p90_ms2')
        if pc1_loadings[idx] > 0:
            pc1_scores = -pc1_scores
            pc1_loadings = -pc1_loadings

    explained_var_ratio = float((s[0] ** 2) / np.sum(s ** 2))

    out.loc[valid_rows.index, 'SecondaryStyleScore_PC1'] = pc1_scores
    out.loc[valid_rows.index, 'SecondaryStyleScore_ExplainedVarianceRatio'] = explained_var_ratio

    return out