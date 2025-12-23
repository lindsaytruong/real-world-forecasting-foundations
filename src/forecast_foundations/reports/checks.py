"""
Module-specific checks organized by the 5Q framework.

Each check function returns a list of check items:
    [{'check': str, 'value': str, 'status': str, 'key': str}, ...]

`key` is a stable identifier you can use to automate gating across modules.
Status symbols:
    ✓ pass
    ✗ fail (blocking)
    ⚠ warning
    ℹ info
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Helpers
# =============================================================================

def _add(checks: List[Dict[str, Any]], key: str, check: str, value: str, status: str) -> None:
    checks.append({"key": key, "check": check, "value": value, "status": status})


def _exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _is_datetime(df: pd.DataFrame, col: str) -> bool:
    return _exists(df, col) and pd.api.types.is_datetime64_any_dtype(df[col])


def _is_numeric(df: pd.DataFrame, col: str) -> bool:
    return _exists(df, col) and pd.api.types.is_numeric_dtype(df[col])


def _safe_pct(n: float, d: float) -> float:
    return float(n / d * 100) if d else 0.0


def _format_pct(val: float) -> str:
    return f"{val:.1%}" if pd.notna(val) else "N/A"


def _format_num(val: float, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}" if pd.notna(val) else "N/A"


def _mode_timedelta_days(s: pd.Series) -> Optional[int]:
    if s is None or len(s) < 2:
        return None
    diffs = s.diff().dropna()
    if diffs.empty:
        return None
    mode = diffs.mode()
    if mode.empty:
        return None
    td = mode.iloc[0]
    try:
        return int(td.days)
    except Exception:
        return None


def _freq_label(freq_days: Optional[int]) -> str:
    if freq_days is None:
        return "Unknown"
    freq_map = {1: "Daily", 7: "Weekly", 14: "Biweekly", 28: "4-week", 30: "Monthly", 31: "Monthly"}
    return freq_map.get(freq_days, f"{freq_days} days")


def _series_lengths(df: pd.DataFrame, id_col: str, date_col: str) -> Optional[pd.Series]:
    if _exists(df, id_col) and _exists(df, date_col):
        return df.groupby(id_col)[date_col].count()
    return None


def _end_dates(df: pd.DataFrame, id_col: str, date_col: str) -> Optional[pd.Series]:
    if _exists(df, id_col) and _exists(df, date_col) and _is_datetime(df, date_col):
        return df.groupby(id_col)[date_col].max()
    return None


def _start_dates(df: pd.DataFrame, id_col: str, date_col: str) -> Optional[pd.Series]:
    if _exists(df, id_col) and _exists(df, date_col) and _is_datetime(df, date_col):
        return df.groupby(id_col)[date_col].min()
    return None


def _timeline_gap_stats(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    expected_freq_days: int = 7,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not (_exists(df, id_col) and _exists(df, date_col) and _is_datetime(df, date_col)):
        return (None, None, None)

    n_dups = int(df.duplicated(subset=[id_col, date_col]).sum())
    gaps_total = 0
    series_with_gaps = 0
    for _, g in df[[id_col, date_col]].sort_values([id_col, date_col]).groupby(id_col, sort=False):
        diffs = g[date_col].diff().dropna()
        if diffs.empty:
            continue
        gaps = (diffs.dt.days > expected_freq_days).sum()
        if gaps:
            gaps_total += int(gaps)
            series_with_gaps += 1

    return (gaps_total, series_with_gaps, n_dups)


# =============================================================================
# Q1: TARGET
# =============================================================================

def target_first_contact(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Q1 checks for 1.06: baseline target sanity."""
    date_col = kwargs.get("date_col", "ds")
    target_col = kwargs.get("target_col", "y")
    checks: List[Dict[str, Any]] = []

    _add(checks, "q1.ds_exists", "ds exists", "Yes" if _exists(df, date_col) else "No", "✓" if _exists(df, date_col) else "✗")
    if _exists(df, date_col):
        _add(checks, "q1.ds_dtype", "ds is datetime", str(df[date_col].dtype), "✓" if _is_datetime(df, date_col) else "✗")
        if _is_datetime(df, date_col):
            n_na = int(df[date_col].isna().sum())
            _add(checks, "q1.ds_nas", "No NAs in ds", f"{n_na:,}", "✓" if n_na == 0 else "✗")

    _add(checks, "q1.y_exists", "y exists", "Yes" if _exists(df, target_col) else "No", "✓" if _exists(df, target_col) else "✗")
    if _exists(df, target_col):
        _add(checks, "q1.y_numeric", "y is numeric", str(df[target_col].dtype), "✓" if _is_numeric(df, target_col) else "✗")
        n_na_y = int(df[target_col].isna().sum())
        _add(checks, "q1.y_nas", "NAs in y", f"{n_na_y:,} ({_safe_pct(n_na_y, len(df)):.1f}%)", "✓" if n_na_y == 0 else "⚠")
        if _is_numeric(df, target_col):
            zeros = int((df[target_col] == 0).sum())
            _add(checks, "q1.y_zeros", "Zeros", f"{_safe_pct(zeros, len(df)):.1f}%", "ℹ")
            neg = int((df[target_col] < 0).sum())
            _add(checks, "q1.y_negative", "Negative values", f"{neg:,}", "✓" if neg == 0 else "✗")

    return checks


def target_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Q1 checks for 1.08: ensure target readiness after prep."""
    date_col = kwargs.get("date_col", "ds")
    target_col = kwargs.get("target_col", "y")
    id_col = kwargs.get("id_col", "unique_id")
    checks: List[Dict[str, Any]] = []

    checks.extend(target_first_contact(df, date_col=date_col, target_col=target_col))

    if _exists(df, target_col) and _is_numeric(df, target_col):
        if _exists(df, id_col):
            nunique_y = df.groupby(id_col)[target_col].nunique(dropna=False)
            n_constant = int((nunique_y <= 1).sum())
            _add(checks, "q1.y_constant_series_pct", "Constant series",
                 f"{_safe_pct(n_constant, nunique_y.size):.1f}% ({n_constant:,}/{nunique_y.size:,})", "ℹ")

        y = df[target_col].astype(float)
        med = float(np.nanmedian(y))
        mad = float(np.nanmedian(np.abs(y - med)))
        if mad > 0:
            robust_z = np.abs(y - med) / (1.4826 * mad)
            n_extreme = int((robust_z > 10).sum())
            _add(checks, "q1.y_extreme_robust_z", "Extreme spikes (robust z>10)", f"{n_extreme:,}", "⚠" if n_extreme > 0 else "✓")

    if _exists(df, date_col) and _is_datetime(df, date_col):
        weekdays = df[date_col].dt.weekday
        mode_wd = int(weekdays.mode().iloc[0]) if not weekdays.mode().empty else -1
        pct_mode = _safe_pct((weekdays == mode_wd).sum(), len(weekdays))
        _add(checks, "q1.ds_weekday_consistency", "ds weekday consistency",
             f"{pct_mode:.1f}% on weekday={mode_wd} (0=Mon)", "✓" if pct_mode > 95 else "⚠")

    return checks


# =============================================================================
# Q2: METRIC
# =============================================================================

def metric_first_contact(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Q2 checks for 1.06."""
    date_col = kwargs.get("date_col", "ds")
    id_col = kwargs.get("id_col", "unique_id")
    checks: List[Dict[str, Any]] = []

    key_cols = [c for c in [date_col, id_col] if _exists(df, c)]
    n_dups = int(df.duplicated(subset=key_cols).sum()) if key_cols else 0
    _add(checks, "q2.duplicates", "Duplicates", f"{n_dups:,}", "✓" if n_dups == 0 else "✗")

    lens = _series_lengths(df, id_col, date_col)
    if lens is not None and not lens.empty:
        _add(checks, "q2.series_lengths", "Series lengths", f"{int(lens.min())} – {int(lens.max())}", "ℹ")

    end_dates = _end_dates(df, id_col, date_col)
    if end_dates is not None and not end_dates.empty:
        n_unique = int(end_dates.nunique())
        _add(checks, "q2.end_date_alignment", "End date alignment",
             "Aligned" if n_unique == 1 else f"{n_unique} different", "✓" if n_unique == 1 else "⚠")

    return checks


def metric_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Q2 checks for 1.08."""
    date_col = kwargs.get("date_col", "ds")
    id_col = kwargs.get("id_col", "unique_id")
    expected_freq_days = int(kwargs.get("expected_freq_days", 7))
    checks: List[Dict[str, Any]] = []

    key_cols = [c for c in [date_col, id_col] if _exists(df, c)]
    n_dups = int(df.duplicated(subset=key_cols).sum()) if key_cols else 0
    _add(checks, "q2.duplicates", "Duplicates", f"{n_dups:,}", "✓" if n_dups == 0 else "✗")

    if _exists(df, id_col) and _exists(df, date_col) and _is_datetime(df, date_col):
        sample_ids = df[id_col].drop_duplicates().head(500)
        sample = df[df[id_col].isin(sample_ids)].copy()
        matches = 0
        for _, g in sample.groupby(id_col):
            fd = _mode_timedelta_days(g[date_col].sort_values())
            if fd == expected_freq_days:
                matches += 1
        _add(checks, "q2.within_series_frequency", "Within-series frequency (sample)",
             f"{matches}/{len(sample_ids)} match {expected_freq_days}d", "✓" if matches == len(sample_ids) else "⚠")

    gaps_total, series_with_gaps, _ = _timeline_gap_stats(df, id_col, date_col, expected_freq_days)
    if gaps_total is not None:
        _add(checks, "q2.gaps_total", "Gaps remaining (>expected step)",
             f"{gaps_total:,} gaps across {series_with_gaps:,} series", "✓" if gaps_total == 0 else "✗")

    end_dates = _end_dates(df, id_col, date_col)
    if end_dates is not None:
        n_unique = int(end_dates.nunique())
        _add(checks, "q2.end_date_alignment", "End date alignment",
             "Aligned" if n_unique == 1 else f"{n_unique} different", "✓" if n_unique == 1 else "⚠")

    start_dates = _start_dates(df, id_col, date_col)
    if start_dates is not None:
        n_unique_start = int(start_dates.nunique())
        _add(checks, "q2.start_date_alignment", "Start date alignment", f"{n_unique_start} different", "ℹ")

    if _exists(df, id_col) and _exists(df, date_col):
        lens = df.groupby(id_col)[date_col].count()
        max_len = int(lens.max())
        p10 = int(np.percentile(lens, 10))
        p50 = int(np.percentile(lens, 50))
        _add(checks, "q2.series_coverage", "Series coverage (p10 / p50)",
             f"{int(p10/max_len*100)}% / {int(p50/max_len*100)}%", "ℹ")

    return checks


# =============================================================================
# Q3: STRUCTURE
# =============================================================================

def structure_first_contact(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Q3 checks for 1.06."""
    date_col = kwargs.get("date_col", "ds")
    id_col = kwargs.get("id_col", "unique_id")
    hierarchy_cols = kwargs.get("hierarchy_cols", ["state_id", "store_id", "cat_id", "dept_id"])
    checks: List[Dict[str, Any]] = []

    if _exists(df, date_col) and _is_datetime(df, date_col):
        min_d, max_d = df[date_col].min(), df[date_col].max()
        _add(checks, "q3.date_range", "Date range", f"{min_d.date()} → {max_d.date()}", "✓")
        n_weeks = int((max_d - min_d).days / 7)
        _add(checks, "q3.history_weeks", "History", f"{n_weeks} weeks ({n_weeks/52:.1f} yrs)", "ℹ")

        unique_dates = df[date_col].drop_duplicates().sort_values()
        freq_days = _mode_timedelta_days(unique_dates)
        _add(checks, "q3.frequency", "Frequency", _freq_label(freq_days), "✓" if freq_days else "⚠")

    if _exists(df, id_col):
        n_series = int(df[id_col].nunique())
        _add(checks, "q3.series_count", "Series", f"{n_series:,}", "✓")

    hier_parts = []
    for col in hierarchy_cols:
        if _exists(df, col):
            hier_parts.append(f"{df[col].nunique()} {col}")
    if hier_parts:
        _add(checks, "q3.hierarchy", "Hierarchy", " → ".join(hier_parts), "✓")

    if _exists(df, id_col) and _exists(df, date_col):
        series_per_date = df.groupby(date_col)[id_col].nunique()
        _add(checks, "q3.series_per_date", "Series per date",
             f"{int(series_per_date.min()):,} – {int(series_per_date.max()):,} (Variable)" 
             if series_per_date.min() != series_per_date.max() else f"{int(series_per_date.min()):,}", "ℹ")

    return checks


def structure_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Q3 checks for 1.08."""
    checks = structure_first_contact(df, **kwargs)
    date_col = kwargs.get("date_col", "ds")
    id_col = kwargs.get("id_col", "unique_id")

    if _exists(df, id_col) and _exists(df, date_col) and _is_datetime(df, date_col):
        sample_ids = df[id_col].drop_duplicates().head(500) if df[id_col].nunique() > 500 else df[id_col].unique()
        sample = df[df[id_col].isin(sample_ids)]
        mono_count = 0
        total = 0
        for uid, g in sample.groupby(id_col):
            total += 1
            if g[date_col].is_monotonic_increasing:
                mono_count += 1
        n_series = int(df[id_col].nunique())
        _add(checks, "q3.monotonic_within_series", "Timestamps sorted within series (sample)",
             f"{mono_count:,}/{total:,} monotonic", "✓" if mono_count == total else "⚠")

        series_per_date = df.groupby(date_col)[id_col].nunique()
        target = n_series
        _add(checks, "q3.full_panel", "Complete panel (series per date)",
             f"{int(series_per_date.min()):,} – {int(series_per_date.max()):,} (target {target:,})",
             "✓" if series_per_date.min() == series_per_date.max() == target else "⚠")

    return checks


# =============================================================================
# Q4: DRIVERS
# =============================================================================

def drivers_first_contact(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Q4 checks for 1.06."""
    date_col = kwargs.get("date_col", "ds")
    expected_freq_days = int(kwargs.get("expected_freq_days", 7))
    calendar_df = kwargs.get("calendar_df")
    checks: List[Dict[str, Any]] = []

    has_calendar = calendar_df is not None and isinstance(calendar_df, pd.DataFrame) and not calendar_df.empty
    _add(checks, "q4.calendar_available", "Calendar", "Available" if has_calendar else "Missing", "✓" if has_calendar else "✗")

    if has_calendar and date_col in calendar_df.columns and pd.api.types.is_datetime64_any_dtype(calendar_df[date_col]):
        unique_dates = calendar_df[date_col].drop_duplicates().sort_values()
        freq_days = _mode_timedelta_days(unique_dates)
        _add(checks, "q4.calendar_frequency", "Calendar frequency", _freq_label(freq_days),
             "✓" if freq_days == expected_freq_days else ("⚠" if freq_days is not None else "✗"))
    else:
        _add(checks, "q4.calendar_frequency", "Calendar frequency", "Unknown (target: Weekly)", "✗" if has_calendar else "ℹ")

    if has_calendar:
        event_cols = [c for c in ["event_name_1", "event_name_2"] if c in calendar_df.columns]
        if event_cols:
            n_events = int(pd.concat([calendar_df[c].dropna().astype(str) for c in event_cols]).nunique())
            _add(checks, "q4.events_unique", "Events", f"{n_events:,} unique", "✓")
        snap_cols = [c for c in calendar_df.columns if c.startswith("snap_")]
        _add(checks, "q4.snap_available", "SNAP", "Available" if snap_cols else "Missing", "✓" if snap_cols else "ℹ")

    return checks


def drivers_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Q4 checks for 1.08."""
    date_col = kwargs.get("date_col", "ds")
    checks: List[Dict[str, Any]] = []

    calendar_indicators = kwargs.get("calendar_indicators", ["wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI", "month"])
    has_calendar = any(c in df.columns for c in calendar_indicators)
    _add(checks, "q4.calendar_merged", "Calendar merged", "Features attached" if has_calendar else "No calendar cols", "✓" if has_calendar else "⚠")

    if has_calendar:
        sample_col = next((c for c in calendar_indicators if c in df.columns), None)
        if sample_col:
            null_pct = float(df[sample_col].isna().mean() * 100)
            _add(checks, "q4.merge_success", "Merge success", f"{100 - null_pct:.0f}% matched", "✓" if null_pct < 1 else "⚠")

    _add(checks, "q4.join_key", "Join key", "ds (week start)", "ℹ")

    if _exists(df, date_col) and _is_datetime(df, date_col) and has_calendar:
        tail_dates = df[date_col].drop_duplicates().sort_values().tail(8)
        tail_df = df[df[date_col].isin(tail_dates)]
        sample_col = next((c for c in calendar_indicators if c in df.columns), None)
        if sample_col:
            tail_null = float(tail_df[sample_col].isna().mean() * 100)
            _add(checks, "q4.calendar_tail_nulls", "Calendar nulls (last 8 dates)", f"{tail_null:.1f}%", "✓" if tail_null < 1 else "⚠")

    return checks


# =============================================================================
# Q5: READINESS
# =============================================================================

def readiness_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Q5 readiness gate for 1.08."""
    date_col = kwargs.get("date_col", "ds")
    target_col = kwargs.get("target_col", "y")
    id_col = kwargs.get("id_col", "unique_id")
    expected_freq_days = int(kwargs.get("expected_freq_days", 7))
    checks: List[Dict[str, Any]] = []

    gaps_total, _, _ = _timeline_gap_stats(df, id_col, date_col, expected_freq_days=expected_freq_days)
    _add(checks, "q5.timeline_complete", "Timeline complete",
         "All gaps filled" if gaps_total == 0 else f"{gaps_total:,} gaps remain", "✓" if gaps_total == 0 else "✗")

    na_count = int(df[target_col].isna().sum()) if _exists(df, target_col) else 0
    _add(checks, "q5.no_nas_in_target", "No NAs in target",
         "All values filled" if na_count == 0 else f"{na_count:,} remaining", "✓" if na_count == 0 else "✗")

    end_dates = _end_dates(df, id_col=id_col, date_col=date_col)
    if end_dates is not None:
        all_same_end = int(end_dates.nunique()) == 1
        _add(checks, "q5.series_aligned", "Series aligned",
             "All end same date" if all_same_end else f"{int(end_dates.nunique())} end dates", "✓" if all_same_end else "⚠")

    if _exists(df, id_col) and _exists(df, date_col):
        series_total = int(df[id_col].nunique())
        series_per_date = df.groupby(date_col)[id_col].nunique()
        ok = int(series_per_date.min()) == int(series_per_date.max()) == series_total
        _add(checks, "q5.full_panel", "Complete panel", "Yes" if ok else "No", "✓" if ok else "⚠")

    if _exists(df, date_col) and _is_datetime(df, date_col):
        unique_dates = df[date_col].drop_duplicates().sort_values()
        freq_days = _mode_timedelta_days(unique_dates)
        _add(checks, "q5.frequency_ready", "Frequency stable", _freq_label(freq_days), "✓" if freq_days == expected_freq_days else "✗")

    return checks


# =============================================================================
# 1.09 DIAGNOSTICS - NEW
# =============================================================================

def diagnostic_profile(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    Diagnostic profile for 1.09.
    Expects df to be the diagnostics dataframe with columns like:
    trend, seasonal_strength, entropy, adi, cv2
    """
    checks: List[Dict[str, Any]] = []
    
    # Thresholds for binning
    thresholds = kwargs.get("thresholds", {
        "trend": (0.3, 0.6),
        "seasonal_strength": (0.3, 0.6),
        "entropy": (0.5, 0.8),
        "adi": (1.32, 1.5),  # 1.32 = smooth/intermittent boundary
        "cv2": (0.5, 1.0),
    })
    
    def bin_metric(series: pd.Series, low_thresh: float, high_thresh: float) -> Dict[str, float]:
        total = len(series.dropna())
        if total == 0:
            return {"low": 0, "medium": 0, "high": 0}
        low = (series < low_thresh).sum()
        high = (series >= high_thresh).sum()
        medium = total - low - high
        return {
            "low": round(low / total * 100),
            "medium": round(medium / total * 100),
            "high": round(high / total * 100),
        }
    
    # Structure metrics (high = good)
    structure_metrics = ["trend", "seasonal_strength"]
    for metric in structure_metrics:
        if _exists(df, metric):
            lo, hi = thresholds.get(metric, (0.3, 0.6))
            bins = bin_metric(df[metric], lo, hi)
            _add(checks, f"diag.{metric}", metric.replace("_", " ").title(),
                 f"Low {bins['low']}% | Med {bins['medium']}% | High {bins['high']}%", "ℹ")
    
    # Chaos metrics (high = bad)
    chaos_metrics = ["entropy", "adi", "cv2"]
    for metric in chaos_metrics:
        if _exists(df, metric):
            lo, hi = thresholds.get(metric, (0.5, 1.0))
            bins = bin_metric(df[metric], lo, hi)
            _add(checks, f"diag.{metric}", metric.upper() if len(metric) <= 3 else metric.replace("_", " ").title(),
                 f"Low {bins['low']}% | Med {bins['medium']}% | High {bins['high']}%", "ℹ")
    
    return checks


def flags_diagnostics(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    Portfolio flags for 1.09 diagnostics.
    """
    checks: List[Dict[str, Any]] = []
    n_series = len(df)
    
    # High entropy (chaos)
    if _exists(df, "entropy"):
        high_entropy = (df["entropy"] > 0.8).sum()
        pct = _safe_pct(high_entropy, n_series)
        status = "⚠" if pct > 30 else "ℹ"
        _add(checks, "flag.high_entropy", "High entropy (>0.8)", f"{pct:.0f}% of series", status)
    
    # Sparse/intermittent (ADI > 1.32)
    if _exists(df, "adi"):
        sparse = (df["adi"] > 1.32).sum()
        pct = _safe_pct(sparse, n_series)
        status = "⚠" if pct > 20 else "ℹ"
        _add(checks, "flag.sparse", "Intermittent (ADI > 1.32)", f"{pct:.0f}% of series", status)
    
    # Strong seasonality (good signal)
    if _exists(df, "seasonal_strength"):
        strong_seas = (df["seasonal_strength"] > 0.6).sum()
        pct = _safe_pct(strong_seas, n_series)
        _add(checks, "flag.strong_seasonality", "Strong seasonality (>0.6)", f"{pct:.0f}% of series", "ℹ")
    
    # Weak structure (low trend + low seasonality)
    if _exists(df, "trend") and _exists(df, "seasonal_strength"):
        weak = ((df["trend"] < 0.3) & (df["seasonal_strength"] < 0.3)).sum()
        pct = _safe_pct(weak, n_series)
        status = "⚠" if pct > 25 else "ℹ"
        _add(checks, "flag.weak_structure", "Weak structure (trend + seas < 0.3)", f"{pct:.0f}% of series", status)
    
    return checks


# =============================================================================
# 1.10 LIE DETECTOR 6
# =============================================================================

def ld6_summary(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    LD6 metric summaries for 1.10.

    Returns median, IQR, and threshold-based percentages for each LD6 metric.
    """
    checks: List[Dict[str, Any]] = []

    # LD6 metric definitions
    ld6_metrics = {
        'trend': {'name': 'Trend', 'type': 'structure', 'high': 0.6, 'low': 0.3},
        'seasonal_strength': {'name': 'Seasonality', 'type': 'structure', 'high': 0.6, 'low': 0.3},
        'x_acf1': {'name': 'Autocorrelation', 'type': 'structure', 'high': 0.6, 'low': 0.3},
        'entropy': {'name': 'Entropy', 'type': 'chaos', 'high': 0.8, 'low': 0.4},
        'adi': {'name': 'ADI', 'type': 'chaos', 'high': 1.32, 'low': 1.0},
        'lumpiness': {'name': 'Lumpiness', 'type': 'chaos', 'high': None, 'low': None},
    }

    for col, meta in ld6_metrics.items():
        if _exists(df, col):
            vals = df[col].dropna()
            if len(vals) == 0:
                continue

            median = vals.median()
            p25 = vals.quantile(0.25)
            p75 = vals.quantile(0.75)

            # Calculate threshold percentages
            high_pct = (vals > meta['high']).mean() if meta['high'] is not None else None
            low_pct = (vals < meta['low']).mean() if meta['low'] is not None else None

            # Format value string
            iqr_str = f"[{p25:.2f}, {p75:.2f}]"
            high_str = _format_pct(high_pct) if high_pct is not None else "—"
            low_str = _format_pct(low_pct) if low_pct is not None else "—"

            value = f"med={median:.3f}  IQR={iqr_str}  hi={high_str}  lo={low_str}"

            _add(checks, f"ld6.{col}", meta['name'], value, "ℹ")

    return checks


def ld6_summary_table(df: pd.DataFrame, **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Return LD6 metrics as structured dict for table rendering.
    """
    ld6_metrics = {
        'trend': {'name': 'Trend', 'type': 'structure', 'high': 0.6, 'low': 0.3},
        'seasonal_strength': {'name': 'Seasonality', 'type': 'structure', 'high': 0.6, 'low': 0.3},
        'x_acf1': {'name': 'Autocorrelation', 'type': 'structure', 'high': 0.6, 'low': 0.3},
        'entropy': {'name': 'Entropy', 'type': 'chaos', 'high': 0.8, 'low': 0.4},
        'adi': {'name': 'ADI', 'type': 'chaos', 'high': 1.32, 'low': 1.0},
        'lumpiness': {'name': 'Lumpiness', 'type': 'chaos', 'high': None, 'low': None},
    }

    result = {}
    for col, meta in ld6_metrics.items():
        if _exists(df, col):
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            result[col] = {
                'name': meta['name'],
                'type': meta['type'],
                'median': float(vals.median()),
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'p25': float(vals.quantile(0.25)),
                'p75': float(vals.quantile(0.75)),
                'high_pct': float((vals > meta['high']).mean()) if meta['high'] else None,
                'low_pct': float((vals < meta['low']).mean()) if meta['low'] else None,
            }
    return result


def score_validation(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    Validate score distributions against known portfolio characteristics.
    """
    checks: List[Dict[str, Any]] = []

    structure_col = kwargs.get('structure_score_col', 'structure_score')
    chaos_col = kwargs.get('chaos_score_col', 'chaos_score')

    if not (_exists(df, structure_col) and _exists(df, chaos_col)):
        return checks

    structure = df[structure_col]
    chaos = df[chaos_col]

    # Score statistics
    structure_med = structure.median()
    structure_mean = structure.mean()
    structure_std = structure.std()
    structure_skew = structure.skew()

    chaos_med = chaos.median()
    chaos_mean = chaos.mean()
    chaos_std = chaos.std()
    chaos_skew = chaos.skew()

    corr = structure.corr(chaos)

    # Add score stats
    _add(checks, "score.structure_median", "Structure median", f"{structure_med:.3f}", "ℹ")
    _add(checks, "score.structure_stats", "Structure (mean/std/skew)",
         f"{structure_mean:.3f} / {structure_std:.3f} / {structure_skew:+.2f}", "ℹ")

    _add(checks, "score.chaos_median", "Chaos median", f"{chaos_med:.3f}", "ℹ")
    _add(checks, "score.chaos_stats", "Chaos (mean/std/skew)",
         f"{chaos_mean:.3f} / {chaos_std:.3f} / {chaos_skew:+.2f}", "ℹ")

    _add(checks, "score.correlation", "Correlation", f"{corr:+.3f}", "ℹ")

    # Validation checks
    # Structure validation based on known seasonal %
    seasonal_high_pct = (df['seasonal_strength'] > 0.6).mean() if _exists(df, 'seasonal_strength') else 0
    entropy_high_pct = (df['entropy'] > 0.8).mean() if _exists(df, 'entropy') else 0
    intermittent_pct = (df['adi'] > 1.32).mean() if _exists(df, 'adi') else 0

    # Structure should be low if few series have strong seasonality
    if 0.25 <= structure_med <= 0.45:
        _add(checks, "score.structure_valid", "Structure vs seasonality",
             f"✓ Consistent with {_format_pct(seasonal_high_pct)} strong seasonality", "✓")
    elif structure_med > 0.50:
        _add(checks, "score.structure_valid", "Structure vs seasonality",
             f"Higher than expected — only {_format_pct(seasonal_high_pct)} strong seasonal", "⚠")
    else:
        _add(checks, "score.structure_valid", "Structure vs seasonality",
             f"Lower than expected — verify computation", "⚠")

    # Chaos should be moderate-high if high entropy + intermittent
    if 0.35 <= chaos_med <= 0.55:
        _add(checks, "score.chaos_valid", "Chaos vs entropy/ADI",
             f"✓ Consistent with {_format_pct(entropy_high_pct)} high entropy + {_format_pct(intermittent_pct)} intermittent", "✓")
    elif chaos_med < 0.30:
        _add(checks, "score.chaos_valid", "Chaos vs entropy/ADI",
             f"Lower than expected — {_format_pct(entropy_high_pct)} had high entropy", "⚠")
    else:
        _add(checks, "score.chaos_valid", "Chaos vs entropy/ADI",
             f"Higher than expected — check for double-counting", "⚠")

    # Correlation validation
    if abs(corr) < 0.3:
        _add(checks, "score.corr_valid", "Correlation check",
             "✓ Low correlation — scores capture independent aspects", "✓")
    elif corr > 0.5:
        _add(checks, "score.corr_valid", "Correlation check",
             "High positive — check if metrics appear in both scores", "⚠")
    elif corr < -0.5:
        _add(checks, "score.corr_valid", "Correlation check",
             "Strong negative — structure and chaos are inverses", "ℹ")
    else:
        _add(checks, "score.corr_valid", "Correlation check",
             "Moderate correlation — acceptable", "ℹ")

    # Skewness validation
    skew_ok = structure_skew > 0  # Expect right-skewed for structure
    _add(checks, "score.structure_skew", "Structure skewness",
         f"{structure_skew:+.2f} (expect positive/right-skewed)", "✓" if skew_ok else "ℹ")

    _add(checks, "score.chaos_skew", "Chaos skewness",
         f"{chaos_skew:+.2f} (expect ~0 or negative)", "ℹ")

    # Boundary pile-up check
    at_bounds = (
        (structure == 0).mean() + (structure == 1).mean() +
        (chaos == 0).mean() + (chaos == 1).mean()
    ) / 2

    if at_bounds > 0.05:
        _add(checks, "score.boundary", "Boundary pile-up",
             f"{_format_pct(at_bounds)} at bounds — consider adjusting clip", "⚠")
    else:
        _add(checks, "score.boundary", "Boundary pile-up",
             "✓ No significant pile-up", "✓")

    return checks


def quadrant_distribution(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    S-B quadrant distribution for 1.10.
    """
    checks: List[Dict[str, Any]] = []
    quadrant_col = kwargs.get('quadrant_col', 'sb_quadrant')

    if not _exists(df, quadrant_col):
        return checks

    dist = df[quadrant_col].value_counts(normalize=True)
    counts = df[quadrant_col].value_counts()

    quadrant_order = ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']

    for q in quadrant_order:
        pct = dist.get(q, 0)
        count = counts.get(q, 0)
        bar_len = int(pct * 30)
        bar = '█' * bar_len
        _add(checks, f"quad.{q.lower()}", q, f"{count:>6,}  {pct:>6.1%}  {bar}", "ℹ")

    dominant = dist.idxmax()
    dominant_pct = dist.max()
    _add(checks, "quad.dominant", "Dominant quadrant", f"{dominant} ({_format_pct(dominant_pct)})", "ℹ")

    return checks


def portfolio_interpretation(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    Determine portfolio character and generate recommendations.
    """
    checks: List[Dict[str, Any]] = []

    structure_col = kwargs.get('structure_score_col', 'structure_score')
    chaos_col = kwargs.get('chaos_score_col', 'chaos_score')
    quadrant_col = kwargs.get('quadrant_col', 'sb_quadrant')

    if not (_exists(df, structure_col) and _exists(df, chaos_col)):
        return checks

    structure_med = df[structure_col].median()
    chaos_med = df[chaos_col].median()

    # Determine character
    if structure_med < 0.35 and chaos_med > 0.40:
        character = "CHAOS-DOMINANT"
        interpretation = "Weak signal + high instability → prioritize robust baselines"
    elif structure_med > 0.45 and chaos_med < 0.35:
        character = "STRUCTURE-DOMINANT"
        interpretation = "Strong patterns + low noise → good candidate for ML models"
    else:
        character = "MIXED"
        interpretation = "Neither dominates → segment by quadrant, use lane-specific strategies"

    _add(checks, "interp.character", "Portfolio character", character, "ℹ")
    _add(checks, "interp.implication", "Implication", interpretation, "ℹ")

    # Quadrant-specific recommendations
    if _exists(df, quadrant_col):
        dist = df[quadrant_col].value_counts(normalize=True)

        recommendations = {
            'Smooth': 'Standard ML (ETS, Prophet, LightGBM)',
            'Erratic': 'Robust methods, wide prediction intervals',
            'Intermittent': 'Croston, SBA, temporal aggregation',
            'Lumpy': 'Classification-first, simple baselines, aggregate up',
        }

        for q, rec in recommendations.items():
            pct = dist.get(q, 0)
            if pct > 0.05:  # Only show if >5%
                _add(checks, f"rec.{q.lower()}", f"{q} ({_format_pct(pct)})", rec, "ℹ")

    return checks


def flags_lie_detector(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    Flags for 1.10 Lie Detector.
    """
    checks: List[Dict[str, Any]] = []
    n_series = len(df)

    # High entropy
    if _exists(df, 'entropy'):
        high_entropy_pct = (df['entropy'] > 0.8).mean()
        if high_entropy_pct > 0.5:
            _add(checks, "flag.high_entropy", "High entropy (>0.8)",
                 f"{_format_pct(high_entropy_pct)} of series", "⚠")

    # Intermittent
    if _exists(df, 'adi'):
        intermittent_pct = (df['adi'] > 1.32).mean()
        if intermittent_pct > 0.3:
            _add(checks, "flag.intermittent", "Intermittent (ADI>1.32)",
                 f"{_format_pct(intermittent_pct)} of series", "⚠")

    # High CV²
    if _exists(df, 'cv2'):
        high_cv2_pct = (df['cv2'] > 0.49).mean()
        if high_cv2_pct > 0.5:
            _add(checks, "flag.high_cv2", "High variability (CV²>0.49)",
                 f"{_format_pct(high_cv2_pct)} of series", "⚠")

    # Weak seasonality
    if _exists(df, 'seasonal_strength'):
        weak_seasonal_pct = (df['seasonal_strength'] < 0.3).mean()
        if weak_seasonal_pct > 0.5:
            _add(checks, "flag.weak_seasonal", "Weak seasonality (<0.3)",
                 f"{_format_pct(weak_seasonal_pct)} of series", "ℹ")

    # Lumpy dominant
    if _exists(df, 'sb_quadrant'):
        lumpy_pct = (df['sb_quadrant'] == 'Lumpy').mean()
        if lumpy_pct > 0.3:
            _add(checks, "flag.lumpy_dominant", "Lumpy-dominant",
                 f"{_format_pct(lumpy_pct)} — consider if item-level forecasting adds value", "⚠")

    # Score issues
    if _exists(df, 'structure_score') and _exists(df, 'chaos_score'):
        corr = df['structure_score'].corr(df['chaos_score'])
        if abs(corr) > 0.7:
            _add(checks, "flag.score_correlation", "High score correlation",
                 f"r={corr:.2f} — scores may be redundant", "⚠")

    return checks


def format_decisions_1_10(decisions_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format decision log for 1.10 report.

    Expects decisions from DecisionLog.decisions
    """
    checks: List[Dict[str, Any]] = []

    for i, d in enumerate(decisions_list, 1):
        finding = d.get('finding', '')
        decision = d.get('decision', '')
        affected = d.get('affected_pct')

        value = f"{decision}"
        if affected:
            value += f" (affects {_format_pct(affected)})"

        _add(checks, f"decision.{i}", finding[:50] + "..." if len(finding) > 50 else finding,
             value, "📋")

    return checks


def compute_1_10_kwargs(
    scores_df: pd.DataFrame,
    weekly_df: pd.DataFrame = None,
    decisions_log=None,
) -> Dict[str, Any]:
    """
    Compute all kwargs needed for 1.10 report rendering.

    Call this before creating ModuleReport and pass as **kwargs.

    Example:
        kwargs = compute_1_10_kwargs(scores_df, weekly_df, decisions)
        report = ModuleReport("1.10", input_df, scores_df, **kwargs)
    """
    # LD6 summary
    ld6_data = ld6_summary_table(scores_df)

    # Score stats
    structure = scores_df['structure_score']
    chaos = scores_df['chaos_score']

    score_stats = {
        'structure': {
            'median': float(structure.median()),
            'mean': float(structure.mean()),
            'std': float(structure.std()),
            'skew': float(structure.skew()),
        },
        'chaos': {
            'median': float(chaos.median()),
            'mean': float(chaos.mean()),
            'std': float(chaos.std()),
            'skew': float(chaos.skew()),
        },
        'correlation': float(structure.corr(chaos)),
    }

    # Quadrant distribution
    quadrant_dist = scores_df['sb_quadrant'].value_counts(normalize=True).to_dict()
    quadrant_counts = scores_df['sb_quadrant'].value_counts().to_dict()
    dominant_quadrant = scores_df['sb_quadrant'].value_counts().idxmax()

    # Portfolio character
    structure_med = structure.median()
    chaos_med = chaos.median()

    if structure_med < 0.35 and chaos_med > 0.40:
        character = "CHAOS-DOMINANT"
        interpretations = [
            "Weak signal (low structure) combined with high instability (high chaos)",
            f"Consistent with {dominant_quadrant} being dominant ({quadrant_dist.get(dominant_quadrant, 0):.1%})",
            "Complex models will likely overfit — prioritize robust baselines"
        ]
    elif structure_med > 0.45 and chaos_med < 0.35:
        character = "STRUCTURE-DOMINANT"
        interpretations = [
            "Strong learnable patterns with manageable noise",
            "Good candidate for ML models (ETS, Prophet, LightGBM)",
            "Invest in feature engineering and model complexity"
        ]
    else:
        character = "MIXED"
        interpretations = [
            "Neither structure nor chaos clearly dominates",
            "Different series need different approaches",
            "Segment by quadrant and use lane-specific strategies"
        ]

    # Decisions
    decisions_logged = []
    if decisions_log is not None:
        if hasattr(decisions_log, 'decisions'):
            decisions_logged = [d.to_dict() if hasattr(d, 'to_dict') else d.__dict__
                              for d in decisions_log.decisions]
        elif isinstance(decisions_log, list):
            decisions_logged = decisions_log

    return {
        'ld6_summary': ld6_data,
        'score_stats': score_stats,
        'quadrant_dist': quadrant_dist,
        'quadrant_counts': quadrant_counts,
        'dominant_quadrant': dominant_quadrant,
        'character': character,
        'interpretations': interpretations,
        'decisions_logged': decisions_logged,
    }


def render_1_10_text(
    report,
    W: int = 65,
    section_header=None,
    get_memory_tier=None,
    render_checks_text=None,
) -> str:
    """
    Render Module 1.10 report as text.

    Called from ModuleReport.to_text() when module == "1.10".
    Formatter functions are passed in to avoid circular imports.
    """
    lines = []

    # Header
    lines.append("━" * W)
    lines.append(f"{report.module} · {report.module_title}")
    lines.append("━" * W)

    # Snapshot
    lines.append(section_header("SNAPSHOT", W))
    snapshot_cols = ["unique_id", "trend", "seasonal_strength", "entropy",
                    "adi", "cv2", "structure_score", "chaos_score", "sb_quadrant"]
    display_cols = [c for c in snapshot_cols if c in report.sample_df.columns]
    if display_cols:
        sample = report.sample_df[display_cols].head(3)
        lines.append(sample.to_string(index=False))

    # Portfolio Overview
    lines.append(section_header("PORTFOLIO OVERVIEW", W))
    lines.append(f"  Series:              {report.output.series:,}")
    lines.append(f"  Date Range:          {report.output.date_min} → {report.output.date_max}")
    lines.append(f"  History (weeks):     {report.output.n_weeks}")
    lines.append(f"  Source:              1.09 Diagnostics")

    # Memory
    lines.append(section_header("MEMORY", W))
    tier, note, status = get_memory_tier(report.output.memory_mb)
    lines.append(f"  {status} {report.output.memory_mb:.0f} MB ({tier}) — {note}")

    # LD6 Metric Summary
    lines.append(section_header("LD6 METRIC SUMMARY", W))

    # Structure metrics table
    lines.append("")
    lines.append("  STRUCTURE METRICS (↑ high = more signal)")
    lines.append(f"  {'Metric':<18} {'Median':>8} {'IQR':>14} {'High %':>8} {'Low %':>8}")
    lines.append(f"  {'-'*18} {'-'*8} {'-'*14} {'-'*8} {'-'*8}")

    ld6_data = report._ld6_summary_data

    for col in ['trend', 'seasonal_strength', 'x_acf1']:
        if col in ld6_data:
            d = ld6_data[col]
            iqr = f"[{d['p25']:.2f}, {d['p75']:.2f}]"
            high = f"{d['high_pct']:.1%}" if d['high_pct'] is not None else "—"
            low = f"{d['low_pct']:.1%}" if d['low_pct'] is not None else "—"
            lines.append(f"  {d['name']:<18} {d['median']:>8.3f} {iqr:>14} {high:>8} {low:>8}")

    # Chaos metrics table
    lines.append("")
    lines.append("  CHAOS METRICS (↑ high = less forecastable)")
    lines.append(f"  {'Metric':<18} {'Median':>8} {'IQR':>14} {'High %':>8} {'Low %':>8}")
    lines.append(f"  {'-'*18} {'-'*8} {'-'*14} {'-'*8} {'-'*8}")

    for col in ['entropy', 'adi', 'lumpiness']:
        if col in ld6_data:
            d = ld6_data[col]
            iqr = f"[{d['p25']:.2f}, {d['p75']:.2f}]"
            high = f"{d['high_pct']:.1%}" if d['high_pct'] is not None else "—"
            low = f"{d['low_pct']:.1%}" if d['low_pct'] is not None else "—"
            lines.append(f"  {d['name']:<18} {d['median']:>8.3f} {iqr:>14} {high:>8} {low:>8}")

    # Score Distributions
    lines.append(section_header("SCORE DISTRIBUTIONS", W))
    lines.append("")
    lines.append(f"  {'Score':<20} {'Median':>8} {'Mean':>8} {'Std':>8} {'Skew':>8}")
    lines.append(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    score_stats = report._score_stats
    if 'structure' in score_stats:
        s = score_stats['structure']
        lines.append(f"  {'Structure Score':<20} {s['median']:>8.3f} {s['mean']:>8.3f} {s['std']:>8.3f} {s['skew']:>+8.2f}")
    if 'chaos' in score_stats:
        s = score_stats['chaos']
        lines.append(f"  {'Chaos Score':<20} {s['median']:>8.3f} {s['mean']:>8.3f} {s['std']:>8.3f} {s['skew']:>+8.2f}")

    if 'correlation' in score_stats:
        lines.append(f"\n  Correlation:         {score_stats['correlation']:+.3f}")

    # Score Validation
    if report._score_validation:
        lines.append(section_header("SCORE VALIDATION", W))
        lines.extend(render_checks_text(report._score_validation, "  "))

    # S-B Quadrant Distribution
    lines.append(section_header("S-B QUADRANT DISTRIBUTION", W))
    lines.append("")
    lines.append(f"  {'Quadrant':<15} {'Count':>8} {'Pct':>8}  {'Bar'}")
    lines.append(f"  {'-'*15} {'-'*8} {'-'*8}  {'-'*25}")

    quadrant_data = report._quadrant_dist
    quadrant_counts = report._quadrant_counts
    dominant = report._dominant_quadrant

    for q in ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']:
        pct = quadrant_data.get(q, 0)
        count = quadrant_counts.get(q, 0)
        bar_len = int(pct * 35)
        bar_char = '█' if q == dominant else '▒'
        lines.append(f"  {q:<15} {count:>8,} {pct:>8.1%}  {bar_char * bar_len}")

    if dominant:
        lines.append(f"\n  Dominant: {dominant} ({quadrant_data.get(dominant, 0):.1%})")

    # Portfolio Interpretation
    lines.append(section_header("PORTFOLIO INTERPRETATION", W))
    lines.append("")

    character = report._character
    lines.append(f"  Character: {character}")
    lines.append("")

    interpretations = report._interpretations
    for interp in interpretations:
        lines.append(f"    • {interp}")

    # Modeling Recommendations
    lines.append(section_header("MODELING RECOMMENDATIONS BY QUADRANT", W))
    lines.append("")

    recommendations = {
        'Smooth': ('Standard ML', 'ETS, Prophet, LightGBM, ARIMA'),
        'Erratic': ('Robust methods', 'Theta, Robust ETS, wide prediction intervals'),
        'Intermittent': ('Specialized', 'Croston, SBA, TSB, temporal aggregation'),
        'Lumpy': ('Simple/Aggregate', 'Classification-first, naive, aggregate up'),
    }

    for q in ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']:
        pct = quadrant_data.get(q, 0)
        if pct > 0.01:  # Show if > 1%
            approach, models = recommendations[q]
            lines.append(f"  {q} ({pct:.1%}):")
            lines.append(f"    Approach: {approach}")
            lines.append(f"    Models:   {models}")
            lines.append("")

    # Flags
    lines.append(section_header("FLAGS & ALERTS", W))
    if report._flags:
        lines.extend(render_checks_text(report._flags, "  "))
    else:
        lines.append("  ✓ No critical flags")

    # Decisions
    if report._decisions_logged:
        lines.append(section_header("DECISIONS LOGGED", W))
        for i, d in enumerate(report._decisions_logged, 1):
            lines.append(f"\n  {i}. {d.get('finding', '')}")
            lines.append(f"     → {d.get('decision', '')}")
            if d.get('affected_pct'):
                lines.append(f"     → Affects {d['affected_pct']:.1%} of portfolio")

    # Footer
    lines.append("")
    lines.append("━" * W)
    lines.append(f"Generated: {report.generated_at}")
    lines.append("━" * W)

    return "\n".join(lines)


# =============================================================================
# FLAGS (consolidated for all modules)
# =============================================================================

def flags_first_contact(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Flags for 1.06."""
    checks: List[Dict[str, Any]] = []
    target_col = kwargs.get("target_col", "y")
    date_col = kwargs.get("date_col", "ds")
    
    # High zeros
    if _exists(df, target_col) and _is_numeric(df, target_col):
        zero_pct = _safe_pct((df[target_col] == 0).sum(), len(df))
        if zero_pct > 30:
            _add(checks, "flag.high_zeros", "High zeros", f"{zero_pct:.0f}% zeros in target", "⚠")
    
    # NAs blocking
    if _exists(df, target_col):
        na_pct = _safe_pct(df[target_col].isna().sum(), len(df))
        if na_pct > 0:
            _add(checks, "flag.nas_present", "NAs in target", f"{na_pct:.1f}% missing", "✗")
    
    return checks


def flags_data_prep(df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Flags for 1.08."""
    checks: List[Dict[str, Any]] = []
    id_col = kwargs.get("id_col", "unique_id")
    date_col = kwargs.get("date_col", "ds")
    
    # Uneven panel
    if _exists(df, id_col) and _exists(df, date_col):
        series_per_date = df.groupby(date_col)[id_col].nunique()
        if series_per_date.min() != series_per_date.max():
            _add(checks, "flag.uneven_panel", "Uneven panel", 
                 f"{int(series_per_date.min()):,} – {int(series_per_date.max()):,} series per date", "⚠")
    
    return checks



# =============================================================================
# MODULE REGISTRY
# =============================================================================

MODULE_CHECKS = {
    "1.06": {
        "target": target_first_contact,
        "metric": metric_first_contact,
        "structure": structure_first_contact,
        "drivers": drivers_first_contact,
        "readiness": lambda df, **kwargs: [],
        "flags": flags_first_contact,
    },
    "1.08": {
        "target": target_data_prep,
        "metric": metric_data_prep,
        "structure": structure_data_prep,
        "drivers": drivers_data_prep,
        "readiness": readiness_data_prep,
        "flags": flags_data_prep,
    },
    "1.09": {
        "target": lambda df, **kwargs: [],
        "metric": lambda df, **kwargs: [],
        "structure": lambda df, **kwargs: [],
        "drivers": lambda df, **kwargs: [],
        "readiness": lambda df, **kwargs: [],
        "diagnostic_profile": diagnostic_profile,
        "flags": flags_diagnostics,
    },
    "1.10": {
        "target": lambda df, **kwargs: [],
        "metric": lambda df, **kwargs: [],
        "structure": lambda df, **kwargs: [],
        "drivers": lambda df, **kwargs: [],
        "readiness": lambda df, **kwargs: [],
        "ld6_summary": ld6_summary,
        "score_validation": score_validation,
        "quadrant_distribution": quadrant_distribution,
        "portfolio_interpretation": portfolio_interpretation,
        "flags": flags_lie_detector,
    },
}

MODULE_TITLES = {
    "1.06": "First Contact",
    "1.07": "Data Structure",
    "1.08": "Data Preparation",
    "1.09": "Diagnostics",
    "1.10": "Lie Detector 6",
}

# Check catalog for automation
CHECK_CATALOG = {
    "1.06": {
        "Q1": ["q1.ds_exists", "q1.ds_dtype", "q1.ds_nas", "q1.y_exists", "q1.y_numeric", "q1.y_nas", "q1.y_zeros", "q1.y_negative"],
        "Q2": ["q2.duplicates", "q2.series_lengths", "q2.end_date_alignment"],
        "Q3": ["q3.date_range", "q3.history_weeks", "q3.frequency", "q3.series_count", "q3.hierarchy", "q3.series_per_date"],
        "Q4": ["q4.calendar_available", "q4.calendar_frequency", "q4.events_unique", "q4.snap_available"],
        "FLAGS": ["flag.high_zeros", "flag.nas_present"],
    },
    "1.08": {
        "Q1": ["q1.ds_exists", "q1.ds_dtype", "q1.ds_nas", "q1.y_exists", "q1.y_numeric", "q1.y_nas", "q1.y_zeros", "q1.y_negative", "q1.y_constant_series_pct", "q1.y_extreme_robust_z", "q1.ds_weekday_consistency"],
        "Q2": ["q2.duplicates", "q2.within_series_frequency", "q2.gaps_total", "q2.end_date_alignment", "q2.start_date_alignment", "q2.series_coverage"],
        "Q3": ["q3.date_range", "q3.history_weeks", "q3.frequency", "q3.series_count", "q3.hierarchy", "q3.series_per_date", "q3.monotonic_within_series", "q3.full_panel"],
        "Q4": ["q4.calendar_merged", "q4.merge_success", "q4.join_key", "q4.calendar_tail_nulls"],
        "Q5": ["q5.timeline_complete", "q5.no_nas_in_target", "q5.series_aligned", "q5.full_panel", "q5.frequency_ready"],
        "FLAGS": ["flag.uneven_panel"],
    },
    "1.09": {
        "DIAGNOSTIC_PROFILE": ["diag.trend", "diag.seasonal_strength", "diag.entropy", "diag.adi", "diag.cv2"],
        "FLAGS": ["flag.high_entropy", "flag.sparse", "flag.strong_seasonality", "flag.weak_structure"],
    },
    "1.10": {
        "LD6_SUMMARY": ["ld6.trend", "ld6.seasonal_strength", "ld6.x_acf1",
                       "ld6.entropy", "ld6.adi", "ld6.lumpiness"],
        "SCORE_VALIDATION": ["score.structure_median", "score.chaos_median",
                            "score.correlation", "score.structure_valid",
                            "score.chaos_valid", "score.corr_valid", "score.boundary"],
        "QUADRANTS": ["quad.smooth", "quad.erratic", "quad.intermittent",
                     "quad.lumpy", "quad.dominant"],
        "INTERPRETATION": ["interp.character", "interp.implication"],
        "FLAGS": ["flag.high_entropy", "flag.intermittent", "flag.high_cv2",
                 "flag.weak_seasonal", "flag.lumpy_dominant", "flag.score_correlation"],
    },
}


__all__ = [
    "MODULE_CHECKS",
    "MODULE_TITLES",
    "CHECK_CATALOG",
    # 1.10 Lie Detector functions
    "ld6_summary",
    "ld6_summary_table",
    "score_validation",
    "quadrant_distribution",
    "portfolio_interpretation",
    "flags_lie_detector",
    "format_decisions_1_10",
    "compute_1_10_kwargs",
    "render_1_10_text",]