"""
Series Profiling
================

Quantitative pattern measurement for time series.

This module measures patterns observed visually in EDA:
- Trend and seasonal strength (STL decomposition)
- Autocorrelation (ACF lag-1, lag-52)
- Intermittency (ADI, CV², Syntetos-Boylan classification)
- Distribution shape (skewness, kurtosis)
- Volatility (heteroscedasticity)
- Outliers (IQR method)

Usage
-----
Single series:
    >>> profile = profile_series(series, period=52)
    >>> print(profile['trend_strength'], profile['demand_class'])

Full DataFrame:
    >>> profiles = profile_dataframe(df, id_col='unique_id', value_col='y', period=52)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# =============================================================================
# TREND & SEASONALITY (STL Decomposition)
# =============================================================================

def calc_stl_strength(
    series: pd.Series,
    period: int = 52
) -> Tuple[float, float]:
    """
    Calculate trend and seasonal strength using STL decomposition.
    
    Based on Wang-Hyndman-Talagala formulas:
        Trend strength    = max(0, 1 - Var(resid) / Var(trend + resid))
        Seasonal strength = max(0, 1 - Var(resid) / Var(seasonal + resid))
    
    Parameters
    ----------
    series : pd.Series
        Time series values (no NaNs, no index gaps)
    period : int
        Seasonal period (52 for weekly, 12 for monthly)
    
    Returns
    -------
    tuple of (trend_strength, seasonal_strength)
        Both range from 0 (none) to 1 (dominant)
        Returns (nan, nan) if decomposition fails
    
    Examples
    --------
    >>> trend_str, seasonal_str = calc_stl_strength(series, period=52)
    >>> if trend_str > 0.6:
    ...     print("Strong trend — include trend component")
    """
    try:
        from statsmodels.tsa.seasonal import STL
        
        if len(series) < period * 2:
            return np.nan, np.nan
        if series.std() == 0:
            return np.nan, np.nan
        
        stl = STL(series, period=period, robust=True).fit()
        
        var_resid = np.var(stl.resid)
        trend_strength = max(0, 1 - var_resid / np.var(stl.trend + stl.resid))
        seasonal_strength = max(0, 1 - var_resid / np.var(stl.seasonal + stl.resid))
        
        return trend_strength, seasonal_strength
    except Exception:
        return np.nan, np.nan


# =============================================================================
# AUTOCORRELATION
# =============================================================================

def calc_acf_metrics(
    series: pd.Series,
    nlags: int = 52
) -> Tuple[float, float]:
    """
    Calculate key ACF values at lag-1 and lag-52.
    
    Parameters
    ----------
    series : pd.Series
        Time series values
    nlags : int
        Maximum lag to compute (must be >= 52 for lag-52)
    
    Returns
    -------
    tuple of (acf_lag1, acf_lag52)
        Autocorrelation values; returns (nan, nan) if calculation fails
    
    Examples
    --------
    >>> lag1, lag52 = calc_acf_metrics(series)
    >>> if abs(lag1) > 0.3:
    ...     print("Strong lag-1 — include AR(1) or lag features")
    """
    try:
        from statsmodels.tsa.stattools import acf
        
        clean = series.dropna()
        if len(clean) < nlags + 10 or clean.std() == 0:
            return np.nan, np.nan
        
        acf_vals = acf(clean, nlags=nlags, fft=True)
        lag1 = acf_vals[1]
        lag52 = acf_vals[min(52, nlags)]
        
        return lag1, lag52
    except Exception:
        return np.nan, np.nan


# =============================================================================
# INTERMITTENCY
# =============================================================================

# Syntetos-Boylan thresholds
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


def calc_intermittency_metrics(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate intermittency metrics and Syntetos-Boylan classification.
    
    Parameters
    ----------
    series : pd.Series
        Time series values
    
    Returns
    -------
    dict with keys:
        - zero_pct: Proportion of zeros
        - cv: Coefficient of variation
        - cv2: CV squared
        - adi: Average demand interval
        - demand_class: 'Smooth', 'Erratic', 'Intermittent', or 'Lumpy'
    
    Examples
    --------
    >>> metrics = calc_intermittency_metrics(series)
    >>> if metrics['demand_class'] == 'Lumpy':
    ...     print("Use Croston or TSB, not standard methods")
    """
    y = series.values
    n = len(y)
    
    zero_pct = (y == 0).mean()
    mean_val = y.mean()
    cv = y.std() / mean_val if mean_val > 0 else np.nan
    cv2 = cv ** 2 if pd.notna(cv) else np.nan
    
    # ADI: average demand interval
    non_zero_idx = np.where(y > 0)[0]
    if len(non_zero_idx) > 1:
        adi = np.diff(non_zero_idx).mean()
    else:
        adi = n  # All zeros or one value
    
    # Syntetos-Boylan classification
    if adi < ADI_THRESHOLD and (cv2 or 0) < CV2_THRESHOLD:
        demand_class = 'Smooth'
    elif adi < ADI_THRESHOLD and (cv2 or 0) >= CV2_THRESHOLD:
        demand_class = 'Erratic'
    elif adi >= ADI_THRESHOLD and (cv2 or 0) < CV2_THRESHOLD:
        demand_class = 'Intermittent'
    else:
        demand_class = 'Lumpy'
    
    return {
        'zero_pct': zero_pct,
        'cv': cv,
        'cv2': cv2,
        'adi': adi,
        'demand_class': demand_class
    }


# =============================================================================
# DISTRIBUTION
# =============================================================================

def calc_distribution_metrics(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate distribution shape metrics.
    
    Parameters
    ----------
    series : pd.Series
        Time series values
    
    Returns
    -------
    dict with keys:
        - skewness: Distribution skewness (>1 = right-skewed)
        - kurtosis: Distribution kurtosis (>0 = heavy tails)
        - log_beneficial: Whether log transform reduces CV
    
    Examples
    --------
    >>> metrics = calc_distribution_metrics(series)
    >>> if metrics['skewness'] > 1:
    ...     print("Right-skewed — consider log transform")
    """
    from scipy import stats
    
    y = series.values
    y_nonzero = y[y > 0]
    
    if len(y_nonzero) < 10:
        return {'skewness': np.nan, 'kurtosis': np.nan, 'log_beneficial': np.nan}
    
    skewness = stats.skew(y_nonzero)
    kurtosis = stats.kurtosis(y_nonzero)
    
    # Would log transform help?
    cv_raw = y_nonzero.std() / y_nonzero.mean()
    log_y = np.log1p(y_nonzero)
    cv_log = log_y.std() / log_y.mean() if log_y.mean() > 0 else np.nan
    log_beneficial = cv_log < cv_raw if pd.notna(cv_log) else False
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'log_beneficial': log_beneficial
    }


# =============================================================================
# VOLATILITY
# =============================================================================

def calc_volatility_metrics(
    series: pd.Series,
    window_size: int = 13
) -> Dict[str, Any]:
    """
    Check if variance scales with level (heteroscedasticity).
    
    Splits series into windows, computes mean and std for each,
    then checks correlation between them.
    
    Parameters
    ----------
    series : pd.Series
        Time series values
    window_size : int
        Window size for computing local mean/std (default 13 = quarterly)
    
    Returns
    -------
    dict with keys:
        - level_var_corr: Correlation between level and variance
        - is_heteroscedastic: True if variance scales with level (corr > 0.5)
    
    Examples
    --------
    >>> metrics = calc_volatility_metrics(series)
    >>> if metrics['is_heteroscedastic']:
    ...     print("Use multiplicative seasonality or log transform")
    """
    y = series.values
    n_windows = len(y) // window_size
    
    if n_windows < 4:
        return {'level_var_corr': np.nan, 'is_heteroscedastic': np.nan}
    
    means = [y[i*window_size:(i+1)*window_size].mean() for i in range(n_windows)]
    stds = [y[i*window_size:(i+1)*window_size].std() for i in range(n_windows)]
    
    if np.std(means) == 0 or np.std(stds) == 0:
        return {'level_var_corr': 0, 'is_heteroscedastic': False}
    
    corr = np.corrcoef(means, stds)[0, 1]
    
    return {
        'level_var_corr': corr,
        'is_heteroscedastic': corr > 0.5
    }


# =============================================================================
# OUTLIERS
# =============================================================================

def calc_outlier_metrics(
    series: pd.Series,
    threshold_pct: float = 0.02
) -> Dict[str, Any]:
    """
    Detect outliers using IQR method.
    
    Parameters
    ----------
    series : pd.Series
        Time series values
    threshold_pct : float
        Percentage threshold for "has_outliers" flag (default 2%)
    
    Returns
    -------
    dict with keys:
        - n_outliers: Count of outlier observations
        - outlier_pct: Proportion of outliers
        - has_outliers: True if outlier_pct > threshold_pct
    
    Examples
    --------
    >>> metrics = calc_outlier_metrics(series)
    >>> if metrics['has_outliers']:
    ...     print("Consider robust methods or pre-clean data")
    """
    y = series.values
    y_nonzero = y[y > 0]
    
    if len(y_nonzero) < 10:
        return {'n_outliers': 0, 'outlier_pct': 0.0, 'has_outliers': False}
    
    q1, q3 = np.percentile(y_nonzero, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    outliers = (y_nonzero < lower) | (y_nonzero > upper)
    n_outliers = int(outliers.sum())
    outlier_pct = n_outliers / len(y_nonzero)
    
    return {
        'n_outliers': n_outliers,
        'outlier_pct': outlier_pct,
        'has_outliers': outlier_pct > threshold_pct
    }


# =============================================================================
# ALL-IN-ONE
# =============================================================================

def profile_series(
    series: pd.Series,
    period: int = 52,
    include_stl: bool = True
) -> Dict[str, Any]:
    """
    Compute full pattern profile for a single series.
    
    Parameters
    ----------
    series : pd.Series
        Time series values
    period : int
        Seasonal period for STL (52 for weekly, 12 for monthly)
    include_stl : bool
        Whether to compute STL strength (slower)
    
    Returns
    -------
    dict with all metrics:
        - trend_strength, seasonal_strength (if include_stl)
        - acf_lag1, acf_lag52
        - zero_pct, cv, cv2, adi, demand_class
        - skewness, kurtosis, log_beneficial
        - level_var_corr, is_heteroscedastic
        - n_outliers, outlier_pct, has_outliers
    
    Examples
    --------
    >>> profile = profile_series(series, period=52)
    >>> print(f"Demand class: {profile['demand_class']}")
    >>> print(f"Trend strength: {profile['trend_strength']:.2f}")
    """
    result = {}
    
    # STL strength (optional, slow)
    if include_stl:
        trend_str, seasonal_str = calc_stl_strength(series, period)
        result['trend_strength'] = trend_str
        result['seasonal_strength'] = seasonal_str
    
    # ACF
    lag1, lag52 = calc_acf_metrics(series, nlags=max(52, period))
    result['acf_lag1'] = lag1
    result['acf_lag52'] = lag52
    
    # Intermittency
    result.update(calc_intermittency_metrics(series))
    
    # Distribution
    result.update(calc_distribution_metrics(series))
    
    # Volatility
    result.update(calc_volatility_metrics(series))
    
    # Outliers
    result.update(calc_outlier_metrics(series))
    
    return result


def profile_dataframe(
    df: pd.DataFrame,
    id_col: str = 'unique_id',
    value_col: str = 'y',
    period: int = 52,
    include_stl: bool = True,
    sample_n: Optional[int] = None,
    random_state: int = 42,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Compute pattern profiles for all series in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with id_col and value_col
    id_col : str
        Column identifying each series
    value_col : str
        Column with values to analyze
    period : int
        Seasonal period for STL
    include_stl : bool
        Whether to compute STL strength (slower)
    sample_n : int, optional
        Sample N series (None = all series)
    random_state : int
        Random seed for sampling
    show_progress : bool
        Show tqdm progress bar
    
    Returns
    -------
    pd.DataFrame
        One row per series with all profile metrics
    
    Examples
    --------
    >>> profiles = profile_dataframe(df, sample_n=500)
    >>> profiles['demand_class'].value_counts(normalize=True)
    """
    unique_ids = df[id_col].unique()
    
    if sample_n is not None and sample_n < len(unique_ids):
        rng = np.random.default_rng(random_state)
        unique_ids = rng.choice(unique_ids, size=sample_n, replace=False)
    
    results = []
    iterator = unique_ids
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(unique_ids, desc="Profiling series")
        except ImportError:
            pass
    
    for uid in iterator:
        series = df[df[id_col] == uid][value_col]
        profile = profile_series(series, period=period, include_stl=include_stl)
        profile[id_col] = uid
        results.append(profile)
    
    return pd.DataFrame(results)

def acf_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    """Summarize ACF significance across portfolio."""
    return pd.DataFrame({
        'Lag': ['Lag-1', 'Lag-52'],
        'Threshold': ['> 0.3', '> 0.2'],
        'Significant': [
            f"{(profiles['acf_lag1'].abs() > 0.3).mean():.1%}",
            f"{(profiles['acf_lag52'].abs() > 0.2).mean():.1%}",
        ],
        'Implication': ['AR(1) / lag features', 'Same-week-last-year']
    })

# =============================================================================
# INTERPRETATION HELPERS
# =============================================================================

def interpret_strength(value: float) -> str:
    """Interpret trend/seasonal strength value."""
    if pd.isna(value):
        return 'Unknown'
    if value < 0.3:
        return 'Weak'
    if value < 0.6:
        return 'Moderate'
    return 'Strong'


def summarize_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Create portfolio-level summary from profiles DataFrame.
    
    Parameters
    ----------
    profiles : pd.DataFrame
        Output from profile_dataframe()
    
    Returns
    -------
    pd.DataFrame
        Summary table with metrics, values, and implications
    """
    n = len(profiles)
    
    summary = {
        'Trend': {
            'Metric': 'Strong trend (>0.6)',
            'Value': f"{(profiles['trend_strength'] > 0.6).mean():.1%}",
            'Implication': 'Include trend component'
        },
        'Seasonal': {
            'Metric': 'Strong seasonal (>0.6)',
            'Value': f"{(profiles['seasonal_strength'] > 0.6).mean():.1%}",
            'Implication': 'Period={}, holiday features'.format(
                52 if 'trend_strength' in profiles else '?'
            )
        },
        'ACF Lag-1': {
            'Metric': 'Significant lag-1 (>0.3)',
            'Value': f"{(profiles['acf_lag1'].abs() > 0.3).mean():.1%}",
            'Implication': 'AR(1) / lag features'
        },
        'ACF Lag-52': {
            'Metric': 'Significant lag-52 (>0.2)',
            'Value': f"{(profiles['acf_lag52'].abs() > 0.2).mean():.1%}",
            'Implication': 'Same-week-last-year feature'
        },
        'Intermittency': {
            'Metric': 'Lumpy demand',
            'Value': f"{(profiles['demand_class'] == 'Lumpy').mean():.1%}",
            'Implication': 'Croston/TSB for these series'
        },
        'Distribution': {
            'Metric': 'Right-skewed (>1)',
            'Value': f"{(profiles['skewness'] > 1).mean():.1%}",
            'Implication': 'Log transform'
        },
        'Volatility': {
            'Metric': 'Variance scales with level',
            'Value': f"{profiles['is_heteroscedastic'].mean():.1%}",
            'Implication': 'Multiplicative seasonality'
        },
        'Outliers': {
            'Metric': 'Has outliers (>2%)',
            'Value': f"{profiles['has_outliers'].mean():.1%}",
            'Implication': 'Robust methods or pre-clean'
        }
    }
    
    return pd.DataFrame(summary).T


__all__ = [
    # Individual metrics
    'calc_stl_strength',
    'calc_acf_metrics', 
    'calc_intermittency_metrics',
    'calc_distribution_metrics',
    'calc_volatility_metrics',
    'calc_outlier_metrics',
    # All-in-one
    'profile_series',
    'profile_dataframe',
    # Helpers
    'interpret_strength',
    'summarize_profiles',
    # Constants
    'ADI_THRESHOLD',
    'CV2_THRESHOLD',
]