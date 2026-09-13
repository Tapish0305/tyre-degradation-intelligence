"""
Data Ingestion and Preprocessing Pipeline for F1 Tyre Degradation Intelligence.
Extracts FastF1 timing and telemetry data, computes fuel weight, traffic gaps,
track evolution timelines, and structures datasets for Bayesian State-Space Models.
"""

import os
import fastf1
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


def setup_fastf1_cache(cache_dir: str = 'cache') -> None:
    """Enable FastF1 caching in the specified directory."""
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def load_f1_session(
    year: int,
    grand_prix: str,
    session_type: str,
    cache_dir: str = 'cache',
    load_telemetry: bool = False
) -> fastf1.core.Session:
    """
    Loads a FastF1 session with local caching.

    load_telemetry : bool
        The SSM pipeline (extract_clean_laps / prepare_dataset_for_ssm) only needs
        lap-level timing + weather, so this defaults to False to keep those loads
        fast. Set True when you also need car telemetry (Speed/Throttle/Brake) and
        position (X/Y) channels -- e.g. for driver_style.py's jerk-based aggression
        indicators, which require per-sample acceleration derived from telemetry.
    """
    setup_fastf1_cache(cache_dir)
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load(laps=True, telemetry=load_telemetry, weather=True, messages=False)
    return session


def compute_gap_to_car_ahead(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the gap (in seconds) to the car ahead on track for every lap.
    If the driver is in free air or is leading, or if timing data is missing,
    clips the gap to 10.0 seconds (clean air threshold, f(gap) -> 0).
    """
    df = laps_df.copy()
    if 'Time' not in df.columns or df['Time'].isna().all():
        df['gap_ahead_s'] = 10.0
        return df

    df['Time_s'] = df['Time'].dt.total_seconds()
    
    # Sort chronologically within each lap number
    df_sorted = df.sort_values(by=['LapNumber', 'Time_s'])
    
    gaps = {}
    for lap_num, group in df_sorted.groupby('LapNumber'):
        valid_group = group[group['Time_s'].notna()].sort_values(by='Time_s')
        if len(valid_group) > 0:
            time_diffs = valid_group['Time_s'].diff()
            for idx, diff in time_diffs.items():
                if pd.isna(diff) or diff <= 0:
                    # Race leader on this lap or single car
                    gaps[idx] = 10.0
                else:
                    gaps[idx] = float(np.clip(diff, 0.0, 10.0))
                    
    df['gap_ahead_s'] = df.index.map(gaps).fillna(10.0).astype(float)
    return df


def compute_fuel_load(
    df: pd.DataFrame,
    is_race: bool = False,
    total_race_laps: Optional[int] = None
) -> pd.DataFrame:
    """
    Computes estimated fuel load in kilograms for each lap.
    
    - Race: Linear burn from 110.0 kg on Lap 1 down to 0.0 kg on the final lap.
    - Practice: Stint-length-based fuel load: 1.7 kg/lap * (StintLength - lap_in_stint) + 5.0 kg buffer.
    """
    df = df.copy()
    if is_race:
        max_laps = total_race_laps if total_race_laps is not None else int(df['LapNumber'].max())
        if max_laps <= 1:
            max_laps = 70
        df['fuel_kg'] = 110.0 * (1.0 - (df['LapNumber'] - 1.0) / max(max_laps - 1.0, 1.0))
        df['fuel_kg'] = df['fuel_kg'].clip(lower=0.0, upper=110.0)
    else:
        fuel_loads = []
        for (driver, stint), group in df.groupby(['Driver', 'Stint']):
            stint_len = len(group)
            for i, (_, row) in enumerate(group.iterrows()):
                laps_remaining = max(stint_len - (i + 1), 0)
                fuel_est = 1.7 * laps_remaining + 5.0
                fuel_loads.append((row.name, fuel_est))
        if fuel_loads:
            fuel_df = pd.DataFrame(fuel_loads, columns=['idx', 'fuel_kg']).set_index('idx')
            df['fuel_kg'] = fuel_df['fuel_kg']
        else:
            df['fuel_kg'] = 15.0
            
    return df


def compute_session_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes normalized elapsed session time in [0, 1] to capture track evolution.
    """
    df = df.copy()
    if 'Time_s' not in df.columns:
        if 'Time' in df.columns and df['Time'].notna().any():
            df['Time_s'] = df['Time'].dt.total_seconds()
        else:
            df['Time_s'] = df['LapNumber'] * 90.0
            
    min_time = df['Time_s'].min()
    max_time = df['Time_s'].max()
    time_range = max(max_time - min_time, 1.0)
    df['session_time'] = (df['Time_s'] - min_time) / time_range
    return df


def compute_weather_features(
    df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Attaches per-lap weather covariates by matching each lap's session timestamp
    to the nearest FastF1 weather sample (weather is polled roughly every ~60s
    through a session, independent of any single car).

    Produces four raw (non-standardized) columns:
      - track_temp   (deg C)  -- primary driver of thermal tyre degradation
      - air_temp     (deg C)
      - humidity     (%)
      - wind_speed   (km/h)

    Standardization (z-scoring) is deliberately NOT done here -- it happens in
    model.py's fit_ssm/predict harness, so that Practice-fitted normalization
    stats can be reapplied consistently to Race data at prediction time.

    If weather data is unavailable (session loaded with weather=False, or the
    channel is empty for this session), columns are filled with NaN and the
    caller/model treats them as a no-op (zero effect after z-scoring).
    """
    df = df.copy()
    weather_cols_map = {
        'TrackTemp': 'track_temp',
        'AirTemp': 'air_temp',
        'Humidity': 'humidity',
        'WindSpeed': 'wind_speed'
    }
    out_cols = list(weather_cols_map.values())

    if weather_df is None or len(weather_df) == 0 or 'Time' not in weather_df.columns:
        for col in out_cols:
            df[col] = np.nan
        return df

    w = weather_df.copy()
    w['Time_s'] = w['Time'].dt.total_seconds()
    w = w.dropna(subset=['Time_s']).sort_values('Time_s')
    available_src_cols = [c for c in weather_cols_map if c in w.columns]

    if len(w) == 0 or not available_src_cols:
        for col in out_cols:
            df[col] = np.nan
        return df

    if 'Time_s' not in df.columns:
        if 'Time' in df.columns and df['Time'].notna().any():
            df['Time_s'] = df['Time'].dt.total_seconds()
        else:
            df['Time_s'] = df['LapNumber'] * 90.0

    # merge_asof requires sorted, non-null keys on both sides
    df_valid = df[df['Time_s'].notna()].sort_values('Time_s')
    df_invalid = df[df['Time_s'].isna()]

    merged = pd.merge_asof(
        df_valid,
        w[['Time_s'] + available_src_cols],
        on='Time_s',
        direction='nearest'
    )
    merged.index = df_valid.index

    df = pd.concat([merged, df_invalid], axis=0).sort_index()
    df = df.rename(columns=weather_cols_map)

    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            # Fill any residual gaps (e.g. laps before the first weather sample,
            # or laps with no adjacent weather row) with the session median so a
            # handful of missing rows don't destabilize downstream z-scoring.
            df[col] = df[col].fillna(df[col].median())

    return df


def extract_clean_laps(
    session: fastf1.core.Session,
    drivers: Optional[List[str]] = None,
    is_race: bool = False,
    outlier_threshold: float = 1.07
) -> pd.DataFrame:
    """
    Filters and cleans session laps:
    - Restricts to requested drivers.
    - Removes pit in-laps and pit out-laps.
    - Removes laps under abnormal track status (SC, VSC, Red flag).
    - Removes standing start Lap 1 for race sessions.
    - Removes slow out-of-trajectory laps (> outlier_threshold of driver stint median).
    - Derives fuel load, traffic gap, and normalized session timeline.
    """
    laps = session.laps.copy()
    
    if drivers is not None:
        drivers = [str(d).upper() for d in drivers]
        laps = laps[laps['Driver'].astype(str).str.upper().isin(drivers)]
        
    # Convert LapTime to seconds
    laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()
    
    # Filter pit in/out laps and missing times
    clean = laps[
        laps['PitInTime'].isna() & 
        laps['PitOutTime'].isna() & 
        laps['LapTime_s'].notna() &
        (laps['LapTime_s'] > 40.0)
    ].copy()
    
    # Filter track status if available (1 = clear track)
    if 'TrackStatus' in clean.columns:
        clean = clean[clean['TrackStatus'].astype(str).isin(['1', '1.0', ''])]
        
    if is_race:
        # Drop Lap 1 standing start
        clean = clean[clean['LapNumber'] > 1]
        
    # Standardize tire compound
    compound_map = {
        'HARD': 'HARD', 'MEDIUM': 'MEDIUM', 'SOFT': 'SOFT',
        'C1': 'HARD', 'C2': 'HARD', 'C3': 'MEDIUM', 'C4': 'SOFT', 'C5': 'SOFT'
    }
    clean['Compound'] = clean['Compound'].astype(str).str.upper().map(lambda c: compound_map.get(c, 'MEDIUM'))
    
    # Filter extreme outliers (> outlier_threshold of stint median)
    valid_indices = []
    for (driver, stint), group in clean.groupby(['Driver', 'Stint']):
        if len(group) >= 3:
            median_pace = group['LapTime_s'].median()
            valid_group = group[group['LapTime_s'] <= median_pace * outlier_threshold]
            valid_indices.extend(valid_group.index.tolist())
        else:
            valid_indices.extend(group.index.tolist())
            
    clean = clean.loc[valid_indices].sort_values(by=['Driver', 'LapNumber']).copy()
    
    # Calculate traffic gaps
    clean = compute_gap_to_car_ahead(clean)
    
    # Calculate fuel load
    total_laps = int(session.laps['LapNumber'].max()) if len(session.laps) > 0 else 71
    clean = compute_fuel_load(clean, is_race=is_race, total_race_laps=total_laps)
    
    # Calculate session timeline
    clean = compute_session_timeline(clean)
    
    # Attach weather covariates (track/air temp, humidity, wind speed).
    # session.weather_data is only populated if load_f1_session() was called
    # with weather=True; compute_weather_features degrades gracefully (NaN
    # columns) otherwise.
    weather_df = getattr(session, 'weather_data', None)
    clean = compute_weather_features(clean, weather_df)
    
    return clean


def prepare_dataset_for_ssm(
    df: pd.DataFrame,
    driver_order: Optional[List[str]] = None
) -> Dict[str, Union[np.ndarray, List[str], int, pd.DataFrame]]:
    """
    Converts a cleaned DataFrame into structured numpy/JAX arrays indexed for SSM modeling.
    """
    df = df.sort_values(by=['Driver', 'LapNumber']).reset_index(drop=True)
    
    if driver_order is None:
        unique_drivers = sorted(df['Driver'].unique().tolist())
    else:
        unique_drivers = [d for d in driver_order if d in df['Driver'].unique()]
        for d in df['Driver'].unique():
            if d not in unique_drivers:
                unique_drivers.append(d)
                
    driver_to_id = {d: i for i, d in enumerate(unique_drivers)}
    compound_to_id = {'HARD': 0, 'MEDIUM': 1, 'SOFT': 2}
    
    df['driver_id'] = df['Driver'].map(driver_to_id)
    df['compound_id'] = df['Compound'].map(lambda c: compound_to_id.get(c, 1))
    
    # Calculate pit / stint reset indicators per driver
    is_stint_reset = []
    for driver_id in range(len(unique_drivers)):
        driver_df = df[df['driver_id'] == driver_id]
        if len(driver_df) == 0:
            continue
        stint_series = driver_df['Stint'].values
        resets = [1] + [1 if stint_series[i] != stint_series[i-1] else 0 for i in range(1, len(stint_series))]
        is_stint_reset.extend(resets)
        
    df['is_reset'] = is_stint_reset
    
    # Raw (non-standardized) weather covariates. Missing entirely if
    # compute_weather_features() couldn't attach them (e.g. weather=False at
    # session load, or an empty weather channel) -- fall back to zeros so the
    # dataset shape is always consistent; model.py's z-scoring treats an
    # all-zero / degenerate column as a no-op.
    weather_cols = ['track_temp', 'air_temp', 'humidity', 'wind_speed']
    weather_arrays = {}
    for col in weather_cols:
        if col in df.columns:
            weather_arrays[col] = df[col].values.astype(np.float32)
        else:
            weather_arrays[col] = np.zeros(len(df), dtype=np.float32)

    dataset = {
        'lap_time': df['LapTime_s'].values.astype(np.float32),
        'fuel': df['fuel_kg'].values.astype(np.float32),
        'gap': df['gap_ahead_s'].values.astype(np.float32),
        'session_time': df['session_time'].values.astype(np.float32),
        'driver_id': df['driver_id'].values.astype(np.int32),
        'compound_id': df['compound_id'].values.astype(np.int32),
        'is_reset': df['is_reset'].values.astype(np.int32),
        'lap_number': df['LapNumber'].values.astype(np.int32),
        'stint': df['Stint'].values.astype(np.int32),
        'num_laps': len(df),
        'num_drivers': len(unique_drivers),
        'num_compounds': 3,
        'driver_names': unique_drivers,
        'dataframe': df
    }
    dataset.update(weather_arrays)
    return dataset


def load_event_dataset(
    year: int,
    grand_prix: str,
    practice_session: str,
    drivers: List[str],
    cache_dir: str = 'cache'
) -> Tuple[Dict, Dict]:
    """
    Convenience function to load and clean both Practice and Race datasets for an event.
    """
    p_session = load_f1_session(year, grand_prix, practice_session, cache_dir)
    r_session = load_f1_session(year, grand_prix, 'Race', cache_dir)
    
    p_clean = extract_clean_laps(p_session, drivers=drivers, is_race=False)
    r_clean = extract_clean_laps(r_session, drivers=drivers, is_race=True)
    
    p_dataset = prepare_dataset_for_ssm(p_clean, driver_order=drivers)
    r_dataset = prepare_dataset_for_ssm(r_clean, driver_order=drivers)
    
    return p_dataset, r_dataset


def load_event_dataset_with_driver_style(
    year: int,
    grand_prix: str,
    practice_session: str,
    drivers: List[str],
    cache_dir: str = 'cache',
    compute_pca_composite: bool = False,
    min_valid_laps: int = 3
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Extends load_event_dataset(...) with a per-driver Driver Style / Aggression
    Indicator profile (see driver_style.py) computed from the practice
    session's telemetry.

    This is a SEPARATE function from load_event_dataset (rather than a new
    argument on it) so existing callers of load_event_dataset are unaffected --
    this version does the extra work of loading practice with telemetry=True
    and running the jerk-based kinematics pipeline per lap, which is
    meaningfully slower than the lap-timing-only path.

    Parameters
    ----------
    compute_pca_composite : bool
        If True, also attaches the optional data-driven secondary-metrics
        composite (driver_style.compute_pca_composite_style_score) to the
        returned profile. Off by default since the jerk-based DSI columns
        (Longitudinal_DSI, Lateral_DSI, DSI_combined) are the primary,
        research-backed metric and the PCA composite is a supplementary
        summary of the OTHER (non-jerk) descriptors only.
    min_valid_laps : int
        Passed through to driver_style.compute_driver_style_profile -- drivers
        with fewer usable-telemetry laps than this get a NaN profile row
        rather than one built on too little data.

    Returns
    -------
    (practice_dataset, race_dataset, driver_style_profile)
        practice_dataset, race_dataset : same shape as load_event_dataset(...)
        driver_style_profile : pd.DataFrame indexed by Driver (see
            driver_style.compute_driver_style_profile for columns), computed
            on the SAME cleaned practice laps used for the SSM (pit in/out,
            SC/VSC, and stint-outlier laps already excluded), so the
            aggression profile and the degradation model are describing the
            same underlying laps.
    """
    try:
        from src.driver_style import compute_driver_style_profile, compute_pca_composite_style_score
    except ImportError as exc:
        raise ImportError(
            "load_event_dataset_with_driver_style requires driver_style.py "
            "(and its scipy dependency) to be importable alongside data_pipeline.py."
        ) from exc

    # Practice needs telemetry=True for jerk-based kinematics; Race only feeds
    # the SSM's lap-timing pipeline here, so it stays on the faster default load.
    p_session = load_f1_session(year, grand_prix, practice_session, cache_dir, load_telemetry=True)
    r_session = load_f1_session(year, grand_prix, 'Race', cache_dir)
    
    p_clean = extract_clean_laps(p_session, drivers=drivers, is_race=False)
    r_clean = extract_clean_laps(r_session, drivers=drivers, is_race=True)
    
    p_dataset = prepare_dataset_for_ssm(p_clean, driver_order=drivers)
    r_dataset = prepare_dataset_for_ssm(r_clean, driver_order=drivers)
    
    style_profile = compute_driver_style_profile(
        p_session, drivers=drivers, clean_laps_df=p_clean, min_valid_laps=min_valid_laps
    )
    if compute_pca_composite:
        style_profile = compute_pca_composite_style_score(style_profile)
    
    return p_dataset, r_dataset, style_profile


def compute_exclusion_summary(
    session: fastf1.core.Session,
    drivers: Optional[List[str]] = None,
    is_race: bool = False,
    outlier_threshold: float = 1.07
) -> pd.DataFrame:
    """
    Lap-exclusion transparency audit.

    Re-applies the SAME filter cascade as extract_clean_laps() -- pit in/out
    & missing/short laps, non-clear track status (SC/VSC/Red Flag), the
    standing-start Lap 1 (race only), then stint-pace outliers -- one stage
    at a time, and reports how many laps are dropped at each stage plus how
    many survive to feed the SSM.

    This is read-only and purely additive: it does not alter
    extract_clean_laps() or its output in any way, it only audits it, so
    existing callers of extract_clean_laps are completely unaffected.

    Returns
    -------
    pd.DataFrame with one row per cascade stage: 'Stage', 'Laps Remaining',
    'Laps Dropped This Stage'.
    """
    laps = session.laps.copy()

    if drivers is not None:
        drivers_u = [str(d).upper() for d in drivers]
        laps = laps[laps['Driver'].astype(str).str.upper().isin(drivers_u)]

    rows = []
    n0 = len(laps)
    rows.append({'Stage': 'Raw laps (selected drivers)', 'Laps Remaining': n0, 'Laps Dropped This Stage': 0})

    laps = laps.copy()
    laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()

    stage1 = laps[
        laps['PitInTime'].isna() &
        laps['PitOutTime'].isna() &
        laps['LapTime_s'].notna() &
        (laps['LapTime_s'] > 40.0)
    ]
    rows.append({
        'Stage': 'Pit in/out & missing/invalid-time laps',
        'Laps Remaining': len(stage1),
        'Laps Dropped This Stage': n0 - len(stage1)
    })

    stage2 = stage1
    if 'TrackStatus' in stage2.columns:
        stage2 = stage2[stage2['TrackStatus'].astype(str).isin(['1', '1.0', ''])]
    rows.append({
        'Stage': 'Non-clear track status (SC/VSC/Red Flag)',
        'Laps Remaining': len(stage2),
        'Laps Dropped This Stage': len(stage1) - len(stage2)
    })

    stage3 = stage2
    if is_race:
        stage3 = stage3[stage3['LapNumber'] > 1]
    rows.append({
        'Stage': 'Standing-start Lap 1 (race only)',
        'Laps Remaining': len(stage3),
        'Laps Dropped This Stage': len(stage2) - len(stage3)
    })

    compound_map = {
        'HARD': 'HARD', 'MEDIUM': 'MEDIUM', 'SOFT': 'SOFT',
        'C1': 'HARD', 'C2': 'HARD', 'C3': 'MEDIUM', 'C4': 'SOFT', 'C5': 'SOFT'
    }
    stage3 = stage3.copy()
    if len(stage3) > 0:
        stage3['Compound'] = stage3['Compound'].astype(str).str.upper().map(lambda c: compound_map.get(c, 'MEDIUM'))

    valid_indices = []
    for (driver, stint), group in stage3.groupby(['Driver', 'Stint']):
        if len(group) >= 3:
            median_pace = group['LapTime_s'].median()
            valid_group = group[group['LapTime_s'] <= median_pace * outlier_threshold]
            valid_indices.extend(valid_group.index.tolist())
        else:
            valid_indices.extend(group.index.tolist())
    stage4 = stage3.loc[valid_indices] if len(stage3) > 0 else stage3

    rows.append({
        'Stage': f'Stint-pace outliers (> {outlier_threshold:.2f}x stint median)',
        'Laps Remaining': len(stage4),
        'Laps Dropped This Stage': len(stage3) - len(stage4)
    })

    rows.append({
        'Stage': 'Laps used by the SSM (final)',
        'Laps Remaining': len(stage4),
        'Laps Dropped This Stage': 0
    })

    return pd.DataFrame(rows)