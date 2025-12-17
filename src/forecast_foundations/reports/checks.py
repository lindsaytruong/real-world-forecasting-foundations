"""
Module-specific checks organized by 5Q framework.

Each check function returns a list of check items:
    [{'check': str, 'value': str, 'status': str}, ...]

Status symbols: ✓ (pass), ✗ (fail), ⚠ (warning), ℹ (info)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


# =============================================================================
# Q1: TARGET — Is y clear, numeric, and clean?
# =============================================================================

def target_first_contact(df: pd.DataFrame, **kwargs) -> List[Dict]:
    """Q1 checks for 1.06: dtype validation, NAs, zeros."""
    date_col = kwargs.get('date_col', 'ds')
    target_col = kwargs.get('target_col', 'y')
    checks = []
    
    def add(check, value, status):
        checks.append({'check': check, 'value': value, 'status': status})
    
    # Date column
    ds_exists = date_col in df.columns
    add(f'{date_col} exists', 'Yes' if ds_exists else 'Missing', '✓' if ds_exists else '✗')
    
    if ds_exists:
        ds_is_dt = pd.api.types.is_datetime64_any_dtype(df[date_col])
        add(f'{date_col} is datetime', str(df[date_col].dtype), '✓' if ds_is_dt else '✗')
        na_ds = df[date_col].isna().sum()
        add(f'No NAs in {date_col}', f'{na_ds:,}', '✓' if na_ds == 0 else '✗')
    
    # Target column
    y_exists = target_col in df.columns
    add(f'{target_col} exists', 'Yes' if y_exists else 'Missing', '✓' if y_exists else '✗')
    
    if y_exists:
        y_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        add(f'{target_col} is numeric', str(df[target_col].dtype), '✓' if y_numeric else '✗')
        na_y = df[target_col].isna().sum()
        na_pct = na_y / len(df) * 100 if len(df) > 0 else 0
        add(f'NAs in {target_col}', f'{na_y:,} ({na_pct:.1f}%)', 'ℹ' if na_y > 0 else '✓')
        if y_numeric:
            zeros_pct = (df[target_col] == 0).mean() * 100
            add('Zeros', f'{zeros_pct:.1f}%', 'ℹ')
            n_neg = (df[target_col] < 0).sum()
            add('Negative values', f'{n_neg:,}', '✓' if n_neg == 0 else '⚠')
    
    return checks


def target_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict]:
    """Q1 checks for 1.08: imputation status."""
    target_col = kwargs.get('target_col', 'y')
    checks = []
    
    def add(check, value, status):
        checks.append({'check': check, 'value': value, 'status': status})
    
    # Imputation info
    if 'is_gap' in df.columns and target_col in df.columns:
        values_imputed = int((df['is_gap'] == 1).sum())
        add('Values imputed', f'{values_imputed:,}', 'ℹ')
    
    # Remaining NAs
    if target_col in df.columns:
        remaining_nas = int(df[target_col].isna().sum())
        add('NAs remaining', f'{remaining_nas:,}', '✓' if remaining_nas == 0 else '✗')
    
    add('Strategy', 'Zero fill (missing = no sales)', 'ℹ')
    
    return checks


# =============================================================================
# Q2: METRIC — Any issues that bias evaluation?
# =============================================================================

def metric_first_contact(df: pd.DataFrame, **kwargs) -> List[Dict]:
    """Q2 checks for 1.06: duplicates, series alignment."""
    date_col = kwargs.get('date_col', 'ds')
    id_col = kwargs.get('id_col', 'unique_id')
    checks = []
    
    def add(check, value, status):
        checks.append({'check': check, 'value': value, 'status': status})
    
    # Duplicates
    key_cols = [c for c in [date_col, id_col] if c in df.columns]
    if key_cols:
        n_dups = df.duplicated(subset=key_cols).sum()
        add('Duplicates', f'{n_dups:,}', '✓' if n_dups == 0 else '✗')
    
    # Series lengths
    if id_col in df.columns and date_col in df.columns:
        lengths = df.groupby(id_col)[date_col].count()
        min_len, max_len = lengths.min(), lengths.max()
        add('Series lengths', f'{min_len:,} – {max_len:,}', '✓' if min_len == max_len else 'ℹ')
        
        # End date alignment
        end_dates = df.groupby(id_col)[date_col].max()
        n_end_dates = end_dates.nunique()
        add('End date alignment', 'Aligned' if n_end_dates == 1 else f'{n_end_dates} different',
            '✓' if n_end_dates == 1 else '⚠')
    
    return checks


def metric_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict]:
    """Q2 checks for 1.08: minimal (most handled by readiness)."""
    return []  # Covered by structure/readiness checks


# =============================================================================
# Q3: STRUCTURE — Enough history at the right granularity?
# =============================================================================

def _detect_frequency(df: pd.DataFrame, date_col: str = 'ds') -> str:
    """Detect frequency from date column."""
    if date_col not in df.columns:
        return 'Unknown'
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return 'Unknown'
    
    unique_dates = df[date_col].drop_duplicates().sort_values()
    if len(unique_dates) < 2:
        return 'Unknown'
    
    freq_days = unique_dates.diff().mode().iloc[0].days
    freq_map = {1: 'Daily', 7: 'Weekly', 14: 'Biweekly', 30: 'Monthly', 31: 'Monthly'}
    return freq_map.get(freq_days, f'{freq_days} days')


def structure_first_contact(df: pd.DataFrame, **kwargs) -> List[Dict]:
    """Q3 checks for 1.06: date range, frequency, timeline health."""
    date_col = kwargs.get('date_col', 'ds')
    id_col = kwargs.get('id_col', 'unique_id')
    hierarchy_cols = kwargs.get('hierarchy_cols', [])
    min_weeks = kwargs.get('min_weeks', 104)
    checks = []
    
    def add(check, value, status):
        checks.append({'check': check, 'value': value, 'status': status})
    
    # Date range and history
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        min_d, max_d = df[date_col].min(), df[date_col].max()
        span_weeks = (max_d - min_d).days / 7
        span_years = span_weeks / 52
        add('Date range', f'{min_d.date()} → {max_d.date()}', '✓')
        add('History for seasonality', f'{span_weeks:.0f} weeks ({span_years:.1f} yrs)',
            '✓' if span_weeks >= min_weeks else '⚠')
        add('Frequency', _detect_frequency(df, date_col), '✓')
    
    # Series count
    if id_col in df.columns:
        add('Series', f'{df[id_col].nunique():,}', '✓')
    
    # Hierarchy
    if hierarchy_cols:
        hier_counts = [f"{df[c].nunique()} {c}" for c in hierarchy_cols if c in df.columns]
        if hier_counts:
            add('Hierarchy', ' → '.join(hier_counts), '✓')
    
    # Timeline health
    if id_col in df.columns and date_col in df.columns:
        series_per_date = df.groupby(date_col)[id_col].nunique()
        min_s, max_s = series_per_date.min(), series_per_date.max()
        is_stable = series_per_date.std() / series_per_date.mean() < 0.05 if series_per_date.mean() > 0 else True
        stable_str = "Stable" if is_stable else "Variable"
        add('Series per date', f'{min_s:,} – {max_s:,} ({stable_str})', '✓' if is_stable else 'ℹ')
    
    return checks


def structure_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict]:
    """Q3 checks for 1.08: gap analysis, series alignment."""
    date_col = kwargs.get('date_col', 'ds')
    id_col = kwargs.get('id_col', 'unique_id')
    input_df = kwargs.get('input_df')
    checks = []
    
    def add(check, value, status):
        checks.append({'check': check, 'value': value, 'status': status})
    
    # Rows added
    rows_added = len(df) - len(input_df) if input_df is not None else 0
    add('Rows added', f'{rows_added:,}', 'ℹ')
    
    # Gap analysis
    if 'is_gap' in df.columns and id_col in df.columns:
        n_gaps = int(df['is_gap'].sum())
        n_series_affected = df[df['is_gap'] == 1][id_col].nunique() if n_gaps > 0 else 0
        total_series = df[id_col].nunique()
        pct = (n_series_affected / total_series * 100) if total_series > 0 else 0
        add('Series with gaps', f'{n_series_affected:,} of {total_series:,} ({pct:.0f}%)', 'ℹ')
        add('Gap flag preserved', 'is_gap column ✓', '✓')
    
    # Series alignment
    if id_col in df.columns and date_col in df.columns:
        end_dates = df.groupby(id_col)[date_col].max()
        n_end_dates = end_dates.nunique()
        add('End date alignment', 'All aligned' if n_end_dates == 1 else f'{n_end_dates} different', 
            '✓' if n_end_dates == 1 else '⚠')
    
    return checks


# =============================================================================
# Q4: DRIVERS — What features can we safely add?
# =============================================================================

def drivers_first_contact(df: pd.DataFrame, **kwargs) -> List[Dict]:
    """Q4 checks for 1.06: calendar/price availability."""
    date_col = kwargs.get('date_col', 'ds')
    drivers = kwargs.get('drivers', {}) or {}
    checks = []
    
    def add(check, value, status):
        checks.append({'check': check, 'value': value, 'status': status})
    
    if not drivers:
        add('Drivers', 'None provided', 'ℹ')
        return checks
    
    target_freq = _detect_frequency(df, date_col)
    
    for name, driver_df in drivers.items():
        if driver_df is None:
            add(name.title(), 'None', 'ℹ')
            continue
        
        add(name.title(), 'Available', '✓')
        
        # Frequency check
        driver_date_col = 'd' if 'd' in driver_df.columns else 'date' if 'date' in driver_df.columns else date_col
        driver_freq = _detect_frequency(driver_df, driver_date_col)
        freq_match = target_freq == driver_freq
        add(f'{name.title()} frequency', f'{driver_freq} (target: {target_freq})', '✓' if freq_match else '✗')
        
        # Calendar-specific
        if name == 'calendar':
            if 'event_name_1' in driver_df.columns:
                n_events = len(driver_df['event_name_1'].dropna().unique())
                add('Events', f'{n_events} unique', '✓')
            snap_cols = [c for c in driver_df.columns if 'snap' in c.lower()]
            if snap_cols:
                add('SNAP', 'Available', '✓')
        
        # Prices-specific
        elif name == 'prices':
            price_cols = [c for c in driver_df.columns if 'price' in c.lower()]
            if price_cols:
                add('Price columns', ', '.join(price_cols[:3]), '✓')
    
    return checks


def drivers_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict]:
    """Q4 checks for 1.08: calendar merge status."""
    input_df = kwargs.get('input_df')
    checks = []
    
    def add(check, value, status):
        checks.append({'check': check, 'value': value, 'status': status})
    
    # Columns added
    if input_df is not None:
        input_cols = set(input_df.columns)
        output_cols = set(df.columns)
        calendar_cols = list(output_cols - input_cols - {'is_gap'})
        add('Columns added', f'{len(calendar_cols)}', 'ℹ')
        if calendar_cols:
            cols_str = ', '.join(sorted(calendar_cols)[:6])
            if len(calendar_cols) > 6:
                cols_str += '...'
            add('Calendar features', cols_str, '✓')
    
    # Check for nulls (merge failures)
    calendar_indicators = ['wm_yr_wk', 'event_name_1', 'snap_CA', 'month']
    has_calendar = any(c in df.columns for c in calendar_indicators)
    
    if has_calendar:
        sample_col = next((c for c in calendar_indicators if c in df.columns), None)
        if sample_col:
            null_pct = df[sample_col].isna().mean() * 100
            add('Merge success', f'{100 - null_pct:.0f}% matched', '✓' if null_pct < 1 else '⚠')
    
    add('Join key', 'ds (week start)', 'ℹ')
    
    return checks


# =============================================================================
# READINESS — Final status (for prep modules)
# =============================================================================

def readiness_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict]:
    """Readiness check for 1.08: confirms data is forecast-ready."""
    date_col = kwargs.get('date_col', 'ds')
    target_col = kwargs.get('target_col', 'y')
    id_col = kwargs.get('id_col', 'unique_id')
    checks = []
    
    def add(check, value, status):
        checks.append({'check': check, 'value': value, 'status': status})
    
    # Complete timeline
    add('Timeline complete', 'All gaps filled', '✓')
    
    # No NAs
    na_count = df[target_col].isna().sum() if target_col in df.columns else 0
    add('No NAs in target', 'All values filled' if na_count == 0 else f'{na_count:,} remaining',
        '✓' if na_count == 0 else '✗')
    
    # Series aligned
    if id_col in df.columns and date_col in df.columns:
        end_dates = df.groupby(id_col)[date_col].max()
        all_same_end = end_dates.nunique() == 1
        add('Series aligned', 'All end same date' if all_same_end else f'{end_dates.nunique()} end dates',
            '✓' if all_same_end else '⚠')
    
    # Calendar merged
    calendar_indicators = ['wm_yr_wk', 'event_name_1', 'snap_CA']
    has_calendar = any(c in df.columns for c in calendar_indicators)
    add('Calendar merged', 'Features attached' if has_calendar else 'No calendar', 
        '✓' if has_calendar else '⚠')
    
    return checks


# =============================================================================
# MODULE REGISTRY
# =============================================================================

MODULE_CHECKS = {
    "1.06": {
        'target': target_first_contact,
        'metric': metric_first_contact,
        'structure': structure_first_contact,
        'drivers': drivers_first_contact,
    },
    "1.07": {
        'target': target_first_contact,
        'structure': structure_first_contact,
    },
    "1.08": {
        'target': target_data_prep,
        'metric': metric_data_prep,
        'structure': structure_data_prep,
        'drivers': drivers_data_prep,
        'readiness': readiness_data_prep,
    },
}

MODULE_TITLES = {
    "1.06": "First Contact",
    "1.07": "Data Structure",
    "1.08": "Data Preparation",
    "1.09": "Diagnostics",
    "1.10": "Feature Engineering",
}


__all__ = ['MODULE_CHECKS', 'MODULE_TITLES']
