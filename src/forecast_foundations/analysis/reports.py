"""
Module Report
=============

Single report class for all Forecast Academy modules.
Module ID drives which checks are run.

Usage:
    report = ModuleReport(
        "1.06", input_df, output_df,
        drivers={'calendar': calendar_df, 'prices': prices_df}
    )
    report.display()
    report.save("reports/1_06.md")
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Union


# =============================================================================
# DECISIONS LOADER
# =============================================================================

def _find_project_root(start_path: Path = None) -> Optional[Path]:
    """Find project root by looking for config/decisions.yaml or pyproject.toml."""
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()
    for _ in range(10):  # Max 10 levels up
        if (current / "config" / "decisions.yaml").exists():
            return current
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _load_decisions_from_yaml(module: str, root: Path = None) -> Optional[pd.DataFrame]:
    """Load decisions for a module from config/decisions.yaml."""
    if root is None:
        root = _find_project_root()

    if root is None:
        return None

    decisions_path = root / "config" / "decisions.yaml"
    if not decisions_path.exists():
        return None

    try:
        with open(decisions_path) as f:
            data = yaml.safe_load(f)

        if module not in data:
            return None

        module_data = data[module]
        decisions_list = module_data.get("decisions", [])

        if not decisions_list:
            return None

        return pd.DataFrame(decisions_list)
    except Exception:
        return None


# =============================================================================
# SNAPSHOT
# =============================================================================

@dataclass
class Snapshot:
    """Data state capture."""
    
    name: str
    rows: int
    columns: int
    series: int
    date_min: str
    date_max: str
    n_weeks: int
    frequency: str
    target_zeros_pct: float
    target_nas: int
    duplicates: int
    memory_mb: float
    
    @classmethod
    def from_df(
        cls, 
        df: pd.DataFrame, 
        name: str = "data",
        date_col: str = 'ds',
        target_col: str = 'y',
        id_col: str = 'unique_id'
    ) -> 'Snapshot':
        rows = len(df)
        columns = df.shape[1]
        series = df[id_col].nunique() if id_col in df.columns else 0
        
        date_min = date_max = 'N/A'
        n_weeks = 0
        frequency = 'N/A'
        
        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            min_d, max_d = df[date_col].min(), df[date_col].max()
            date_min = str(min_d.date())
            date_max = str(max_d.date())
            n_weeks = int((max_d - min_d).days / 7)
            
            unique_dates = df[date_col].drop_duplicates().sort_values()
            if len(unique_dates) > 1:
                freq_days = unique_dates.diff().mode().iloc[0].days
                freq_map = {1: 'Daily', 7: 'Weekly', 14: 'Biweekly', 30: 'Monthly', 31: 'Monthly'}
                frequency = freq_map.get(freq_days, f'{freq_days} days')
        
        target_zeros_pct = 0.0
        target_nas = 0
        if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            target_zeros_pct = (df[target_col] == 0).mean() * 100
            target_nas = int(df[target_col].isna().sum())
        
        key_cols = [c for c in [date_col, id_col] if c in df.columns]
        duplicates = int(df.duplicated(subset=key_cols).sum()) if key_cols else 0
        
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        return cls(
            name=name, rows=rows, columns=columns, series=series,
            date_min=date_min, date_max=date_max, n_weeks=n_weeks,
            frequency=frequency, target_zeros_pct=target_zeros_pct,
            target_nas=target_nas, duplicates=duplicates, memory_mb=memory_mb
        )


# =============================================================================
# CHECK FUNCTIONS (module-specific)
# =============================================================================

def _detect_frequency(df: pd.DataFrame, date_col: str = 'ds') -> str:
    """Detect frequency from a DataFrame's date column."""
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


def check_5q(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """5Q readiness checks for first contact."""
    
    date_col = kwargs.get('date_col', 'ds')
    target_col = kwargs.get('target_col', 'y')
    id_col = kwargs.get('id_col', 'unique_id')
    hierarchy_cols = kwargs.get('hierarchy_cols', [])
    drivers = kwargs.get('drivers', {}) or {}
    min_weeks = kwargs.get('min_weeks', 104)
    
    checks = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
    
    def add(q, check, value, status):
        checks[q].append({'check': check, 'value': value, 'status': status})
    
    # Q1: TARGET
    ds_exists = date_col in df.columns
    add('Q1', f'{date_col} exists', 'Yes' if ds_exists else 'Missing', '✓' if ds_exists else '✗')
    
    if ds_exists:
        ds_is_dt = pd.api.types.is_datetime64_any_dtype(df[date_col])
        add('Q1', f'{date_col} is datetime', str(df[date_col].dtype), '✓' if ds_is_dt else '✗')
        na_ds = df[date_col].isna().sum()
        add('Q1', f'No NAs in {date_col}', f'{na_ds:,}', '✓' if na_ds == 0 else '✗')
    
    y_exists = target_col in df.columns
    add('Q1', f'{target_col} exists', 'Yes' if y_exists else 'Missing', '✓' if y_exists else '✗')
    
    if y_exists:
        y_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        add('Q1', f'{target_col} is numeric', str(df[target_col].dtype), '✓' if y_numeric else '✗')
        na_y = df[target_col].isna().sum()
        na_pct = na_y / len(df) * 100 if len(df) > 0 else 0
        add('Q1', f'NAs in {target_col}', f'{na_y:,} ({na_pct:.1f}%)', 'ℹ' if na_y > 0 else '✓')
        if y_numeric:
            zeros_pct = (df[target_col] == 0).mean() * 100
            add('Q1', 'Zeros', f'{zeros_pct:.1f}%', 'ℹ')
            n_neg = (df[target_col] < 0).sum()
            add('Q1', 'Negative values', f'{n_neg:,}', '✓' if n_neg == 0 else '⚠')
    
    # Q2: METRIC
    key_cols = [c for c in [date_col, id_col] if c in df.columns]
    if key_cols:
        n_dups = df.duplicated(subset=key_cols).sum()
        add('Q2', 'Duplicates', f'{n_dups:,}', '✓' if n_dups == 0 else '✗')
    
    if id_col in df.columns and date_col in df.columns:
        lengths = df.groupby(id_col)[date_col].count()
        min_len, max_len = lengths.min(), lengths.max()
        add('Q2', 'Series lengths', f'{min_len:,} – {max_len:,}', '✓' if min_len == max_len else 'ℹ')
        
        end_dates = df.groupby(id_col)[date_col].max()
        n_end_dates = end_dates.nunique()
        add('Q2', 'End date alignment', 'Aligned' if n_end_dates == 1 else f'{n_end_dates} different',
            '✓' if n_end_dates == 1 else '⚠')
    
    # Q3: STRUCTURE
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        min_d, max_d = df[date_col].min(), df[date_col].max()
        span_weeks = (max_d - min_d).days / 7
        span_years = span_weeks / 52
        add('Q3', 'Date range', f'{min_d.date()} → {max_d.date()}', '✓')
        add('Q3', 'History for seasonality', f'{span_weeks:.0f} weeks ({span_years:.1f} yrs)',
            '✓' if span_weeks >= min_weeks else '⚠')
        
        unique_dates = df[date_col].drop_duplicates().sort_values()
        if len(unique_dates) > 1:
            freq_days = unique_dates.diff().mode().iloc[0].days
            freq_map = {1: 'Daily', 7: 'Weekly', 14: 'Biweekly', 30: 'Monthly', 31: 'Monthly'}
            add('Q3', 'Frequency', freq_map.get(freq_days, f'{freq_days} days'), '✓')
    
    if id_col in df.columns:
        add('Q3', 'Series', f'{df[id_col].nunique():,}', '✓')
    
    if hierarchy_cols:
        hier_counts = [f"{df[c].nunique()} {c}" for c in hierarchy_cols if c in df.columns]
        if hier_counts:
            add('Q3', 'Hierarchy', ' → '.join(hier_counts), '✓')
    
    # Q4: DRIVERS
    if not drivers:
        add('Q4', 'Drivers', 'None provided', 'ℹ')
    else:
        target_freq = _detect_frequency(df, date_col)

        for name, driver_df in drivers.items():
            if driver_df is None:
                add('Q4', name.title(), 'None', 'ℹ')
                continue

            add('Q4', name.title(), 'Available', '✓')

            # Detect driver frequency and compare to target
            driver_date_col = 'd' if 'd' in driver_df.columns else date_col
            driver_freq = _detect_frequency(driver_df, driver_date_col)
            freq_match = target_freq == driver_freq
            freq_status = '✓' if freq_match else '✗'
            add('Q4', f'{name.title()} frequency', f'{driver_freq} (target: {target_freq})', freq_status)

            # Calendar-specific checks
            if name == 'calendar':
                if 'event_name_1' in driver_df.columns:
                    n_events = len(driver_df['event_name_1'].dropna().unique())
                    add('Q4', 'Events', f'{n_events} unique', '✓')
                snap_cols = [c for c in driver_df.columns if 'snap' in c.lower()]
                if snap_cols:
                    add('Q4', 'SNAP', 'Available', '✓')

            # Prices-specific checks
            elif name == 'prices':
                price_cols = [c for c in driver_df.columns if 'price' in c.lower()]
                if price_cols:
                    add('Q4', 'Price columns', ', '.join(price_cols[:3]), '✓')
                if 'item_id' in driver_df.columns:
                    n_items = driver_df['item_id'].nunique()
                    add('Q4', 'Items with prices', f'{n_items:,}', '✓')
    
    return {'title': '5Q CHECKS', 'data': checks, 'type': '5q'}


def check_timeline(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Timeline health check."""
    
    date_col = kwargs.get('date_col', 'ds')
    id_col = kwargs.get('id_col', 'unique_id')
    
    series_per_date = df.groupby(date_col)[id_col].nunique()
    
    return {
        'title': 'TIMELINE HEALTH',
        'data': {
            'min_series': int(series_per_date.min()),
            'max_series': int(series_per_date.max()),
            'is_stable': series_per_date.std() / series_per_date.mean() < 0.05,
        },
        'type': 'timeline'
    }


def check_gaps(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Gap analysis for data prep module."""
    
    date_col = kwargs.get('date_col', 'ds')
    id_col = kwargs.get('id_col', 'unique_id')
    
    # Count series with gaps
    if 'is_gap' in df.columns:
        n_filled = df['is_gap'].sum()
        n_series_with_gaps = df[df['is_gap']].groupby(id_col).ngroups if n_filled > 0 else 0
    else:
        n_filled = 0
        n_series_with_gaps = 0
    
    return {
        'title': 'GAP ANALYSIS',
        'data': {
            'gaps_filled': n_filled,
            'series_with_gaps': n_series_with_gaps,
        },
        'type': 'gaps'
    }


def check_imputation(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Imputation summary for data prep module."""
    
    target_col = kwargs.get('target_col', 'y')
    
    return {
        'title': 'IMPUTATION',
        'data': {
            'remaining_nas': int(df[target_col].isna().sum()) if target_col in df.columns else 0,
        },
        'type': 'imputation'
    }


def check_features(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Feature summary for feature engineering module."""
    
    date_col = kwargs.get('date_col', 'ds')
    target_col = kwargs.get('target_col', 'y')
    id_col = kwargs.get('id_col', 'unique_id')
    
    core_cols = {date_col, target_col, id_col, 'is_gap'}
    feature_cols = [c for c in df.columns if c not in core_cols]
    
    return {
        'title': 'FEATURES',
        'data': {
            'n_features': len(feature_cols),
            'feature_cols': feature_cols[:10],  # First 10
        },
        'type': 'features'
    }


# =============================================================================
# MODULE REGISTRY
# =============================================================================

MODULE_CHECKS = {
    "1.06": [check_5q, check_timeline],
    "1.07": [check_5q],
    "1.08": [check_gaps, check_imputation],
    "1.09": [],
    "1.10": [check_features],
}

MODULE_TITLES = {
    "1.06": "First Contact",
    "1.08": "Data Preparation", 
    "1.10": "Feature Engineering",
}


# =============================================================================
# MODULE REPORT CLASS
# =============================================================================

def _parse_markdown_table(md: str) -> List[List[str]]:
    """Parse markdown table into rows of cells."""
    lines = [l.strip() for l in md.strip().split('\n') if l.strip()]
    rows = []
    for line in lines:
        if line.startswith('|') and not set(line.replace('|', '').strip()) <= {'-', ':'}:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            rows.append(cells)
    return rows


def _format_table(rows: List[List[str]], indent: str = "  ") -> str:
    """Format parsed table with aligned columns."""
    if not rows:
        return ""
    
    # Calculate column widths
    n_cols = len(rows[0])
    widths = [max(len(row[i]) if i < len(row) else 0 for row in rows) for i in range(n_cols)]
    
    lines = []
    
    # Header
    header = rows[0]
    header_line = indent + "  ".join(h.ljust(w) for h, w in zip(header, widths))
    lines.append(header_line)
    lines.append(indent + "  ".join("─" * w for w in widths))
    
    # Data rows
    for row in rows[1:]:
        line = indent + "  ".join(
            (row[i] if i < len(row) else "").ljust(widths[i]) 
            for i in range(n_cols)
        )
        lines.append(line)
    
    return "\n".join(lines)


class ModuleReport:
    """
    Single report class for all modules.

    Decisions are auto-loaded from config/decisions.yaml if not provided.

    Parameters
    ----------
    module : str
        Module ID (e.g., "1.06", "1.08")
    input_df : pd.DataFrame
        Data before transformations
    output_df : pd.DataFrame
        Data after transformations
    decisions : str, pd.DataFrame, or None
        Explicit decisions (markdown string or DataFrame).
        If None, auto-loads from config/decisions.yaml using module ID.
    drivers : dict[str, pd.DataFrame], optional
        Driver datasets to validate. Known keys ('calendar', 'prices') get
        specific checks; others get generic frequency validation.

    Examples
    --------
    >>> report = ModuleReport(
    ...     "1.06", daily_sales, weekly_sales,
    ...     drivers={'calendar': cal_df, 'prices': prices_df}
    ... )
    >>> report.display()
    """
    
    def __init__(
        self,
        module: str,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        decisions: Union[str, pd.DataFrame, None] = None,
        drivers: Dict[str, pd.DataFrame] = None,
        date_col: str = 'ds',
        target_col: str = 'y',
        id_col: str = 'unique_id',
        sample_df: pd.DataFrame = None,
        hierarchy_cols: List[str] = None,
        **kwargs
    ):
        self.module = module
        self.module_title = MODULE_TITLES.get(module, "")
        self.input = Snapshot.from_df(input_df, "Input", date_col, target_col, id_col)
        self.output = Snapshot.from_df(output_df, "Output", date_col, target_col, id_col)
        self.sample_df = sample_df if sample_df is not None else output_df
        self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Load decisions: explicit > yaml > none
        if decisions is not None:
            if isinstance(decisions, pd.DataFrame):
                self.decisions_df = decisions
            else:
                self.decisions_df = None
                self._decisions_md = decisions.strip()
        else:
            # Auto-load from config/decisions.yaml
            self.decisions_df = _load_decisions_from_yaml(module)
            self._decisions_md = None
        
        # Store params for checks
        if hierarchy_cols is None:
            hierarchy_cols = ['state_id', 'store_id', 'cat_id', 'dept_id']

        self.params = {
            'date_col': date_col,
            'target_col': target_col,
            'id_col': id_col,
            'hierarchy_cols': hierarchy_cols,
            'drivers': drivers or {},
            **kwargs
        }
        
        # Run module-specific checks
        self.sections = []
        check_funcs = MODULE_CHECKS.get(module, [])
        for check_func in check_funcs:
            result = check_func(output_df, **self.params)
            self.sections.append(result)
    
    def _section_header(self, title: str, width: int = 65) -> str:
        return f"\n{title}\n{'─' * width}"
    
    def _render_5q(self, data: dict) -> List[str]:
        """Render 5Q checks section."""
        lines = []
        q_names = {'Q1': 'Target', 'Q2': 'Metric', 'Q3': 'Structure', 'Q4': 'Drivers'}
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            if q in data and data[q]:
                lines.append(f"\n  {q} · {q_names[q]}")
                for c in data[q]:
                    lines.append(f"    {c['status']} {c['check']:<25} {c['value']}")
        return lines
    
    def _render_timeline(self, data: dict) -> List[str]:
        """Render timeline health section."""
        stable = "Stable" if data.get('is_stable') else "Variable"
        return [f"  {'Series/date':<15} {data['min_series']:,} – {data['max_series']:,} ({stable})"]
    
    def _render_gaps(self, data: dict) -> List[str]:
        """Render gap analysis section."""
        return [
            f"  {'Gaps filled':<15} {data['gaps_filled']:,}",
            f"  {'Series affected':<15} {data['series_with_gaps']:,}",
        ]
    
    def _render_imputation(self, data: dict) -> List[str]:
        """Render imputation section."""
        return [f"  {'Remaining NAs':<15} {data['remaining_nas']:,}"]
    
    def _render_features(self, data: dict) -> List[str]:
        """Render features section."""
        lines = [f"  {'Features added':<15} {data['n_features']}"]
        if data['feature_cols']:
            lines.append(f"  {'Columns':<15} {', '.join(data['feature_cols'][:5])}...")
        return lines

    def _render_decisions_df(self, df: pd.DataFrame, indent: str = "  ") -> str:
        """Render decisions DataFrame as aligned table."""
        # Normalize column names
        col_map = {'step': 'Step', 'decision': 'Decision', 'why': 'Why', 'rev': 'Rev'}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Convert rev column to symbols
        if 'Rev' in df.columns:
            df = df.copy()
            df['Rev'] = df['Rev'].map(lambda x: '✓' if x is True else ('—' if x is None or pd.isna(x) else str(x)))

        # Build rows list
        cols = [c for c in ['Step', 'Decision', 'Why', 'Rev'] if c in df.columns]
        rows = [cols] + df[cols].values.tolist()

        return _format_table(rows, indent)

    def _decisions_df_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert decisions DataFrame to markdown table."""
        col_map = {'step': 'Step', 'decision': 'Decision', 'why': 'Why', 'rev': 'Rev'}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if 'Rev' in df.columns:
            df = df.copy()
            df['Rev'] = df['Rev'].map(lambda x: '✓' if x is True else ('—' if x is None or pd.isna(x) else str(x)))

        cols = [c for c in ['Step', 'Decision', 'Why', 'Rev'] if c in df.columns]
        lines = ['| ' + ' | '.join(cols) + ' |']
        lines.append('|' + '|'.join(['------'] * len(cols)) + '|')
        for _, row in df[cols].iterrows():
            lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
        return '\n'.join(lines)

    def _get_blocking_issues(self) -> List[str]:
        """Extract blocking issues from 5Q checks."""
        blocking = []
        for section in self.sections:
            if section['type'] == '5q':
                for q, items in section['data'].items():
                    for c in items:
                        if c['status'] == '✗':
                            blocking.append(f"{c['check']}: {c['value']}")
        return blocking
    
    def to_text(self) -> str:
        """Generate text report."""
        w = 65
        lines = []
        
        # Header
        title = f"{self.module} · {self.module_title}" if self.module_title else self.module
        lines.append("━" * w)
        lines.append(title)
        lines.append("━" * w)
        
        # Snapshot
        lines.append(self._section_header("SNAPSHOT"))
        lines.append(self.sample_df.head(3).to_string(index=False))
        
        # Data Summary
        lines.append(self._section_header("DATA SUMMARY"))
        lines.append(f"  {'Rows':<15} {self.output.rows:,}")
        lines.append(f"  {'Series':<15} {self.output.series:,}")
        lines.append(f"  {'Dates':<15} {self.output.date_min} → {self.output.date_max}")
        lines.append(f"  {'Frequency':<15} {self.output.frequency}")
        lines.append(f"  {'History':<15} {self.output.n_weeks} weeks ({self.output.n_weeks/52:.1f} yrs)")
        lines.append(f"  {'Target zeros':<15} {self.output.target_zeros_pct:.1f}%")

        # Memory assessment
        lines.append(self._section_header("MEMORY"))
        mem_mb = self.output.memory_mb
        mem_gb = mem_mb / 1024

        if mem_mb < 100:
            mem_tier = "Small"
            mem_status = "✓"
            mem_note = "Fits easily in memory"
        elif mem_mb < 1024:
            mem_tier = "Medium"
            mem_status = "✓"
            mem_note = "Fine for most operations"
        elif mem_mb < 10240:
            mem_tier = "Large"
            mem_status = "⚠"
            mem_note = "May need chunking for CV/grid search"
        else:
            mem_tier = "Very Large"
            mem_status = "✗"
            mem_note = "Consider sampling or distributed processing"

        if mem_gb >= 1:
            lines.append(f"  {mem_status} {mem_gb:.1f} GB ({mem_tier}) — {mem_note}")
        else:
            lines.append(f"  {mem_status} {mem_mb:.0f} MB ({mem_tier}) — {mem_note}")

        # Module-specific sections
        renderers = {
            '5q': self._render_5q,
            'timeline': self._render_timeline,
            'gaps': self._render_gaps,
            'imputation': self._render_imputation,
            'features': self._render_features,
        }
        
        for section in self.sections:
            lines.append(self._section_header(section['title']))
            renderer = renderers.get(section['type'])
            if renderer:
                lines.extend(renderer(section['data']))
        
        # Blocking issues (only for modules with 5Q checks)
        blocking = self._get_blocking_issues()
        if any(s['type'] == '5q' for s in self.sections):
            lines.append(self._section_header("BLOCKING ISSUES"))
            if blocking:
                for b in blocking:
                    lines.append(f"  ✗ {b}")
            else:
                lines.append("  None ✓")
        
        # Decisions (from DataFrame or markdown string)
        if self.decisions_df is not None and not self.decisions_df.empty:
            lines.append(self._section_header("DECISIONS"))
            lines.append(self._render_decisions_df(self.decisions_df))
        elif hasattr(self, '_decisions_md') and self._decisions_md:
            lines.append(self._section_header("DECISIONS"))
            rows = _parse_markdown_table(self._decisions_md)
            if rows:
                lines.append(_format_table(rows))
            else:
                lines.append(self._decisions_md)
        
        # Changes
        lines.append(self._section_header("CHANGES"))
        row_delta = self.output.rows - self.input.rows
        row_pct = (row_delta / self.input.rows * 100) if self.input.rows > 0 else 0
        lines.append(f"  {'Rows':<12} {self.input.rows:>12,} → {self.output.rows:>12,}  ({row_pct:+.0f}%)")
        
        if self.input.target_nas > 0 or self.output.target_nas > 0:
            if self.input.target_nas > 0 and self.output.target_nas == 0:
                lines.append(f"  {'NAs (y)':<12} {self.input.target_nas:>12,} → {self.output.target_nas:>12,}  ✓ Fixed")
            else:
                lines.append(f"  {'NAs (y)':<12} {self.input.target_nas:>12,} → {self.output.target_nas:>12,}")
        
        if self.input.duplicates > 0 and self.output.duplicates == 0:
            lines.append(f"  {'Duplicates':<12} {self.input.duplicates:>12,} → {self.output.duplicates:>12,}  ✓ Fixed")
        
        if self.input.frequency != self.output.frequency:
            lines.append(f"  {'Frequency':<12} {self.input.frequency:>12} → {self.output.frequency:>12}")

        # Memory - show reduction percentage
        if self.input.memory_mb > 0:
            mem_reduction = ((self.input.memory_mb - self.output.memory_mb) / self.input.memory_mb * 100)
            if self.output.memory_mb < self.input.memory_mb:
                lines.append(f"  {'Memory':<12} {self.input.memory_mb:>10.1f} MB → {self.output.memory_mb:>8.1f} MB  (-{mem_reduction:.0f}%) ✓ Reduced")
            else:
                lines.append(f"  {'Memory':<12} {self.input.memory_mb:>10.1f} MB → {self.output.memory_mb:>8.1f} MB  (+{-mem_reduction:.0f}%)")

        # Footer
        lines.append("\n" + "━" * w)
        lines.append(f"Generated: {self.generated_at}")
        lines.append("━" * w)
        
        return "\n".join(lines)
    
    def _render_5q_markdown(self, data: dict) -> List[str]:
        """Render 5Q checks as markdown."""
        lines = []
        q_names = {'Q1': 'Target', 'Q2': 'Metric', 'Q3': 'Structure', 'Q4': 'Drivers'}
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            if q in data and data[q]:
                lines.append(f"\n### {q}: {q_names[q]}\n")
                lines.append("| Check | Value | Status |")
                lines.append("|-------|-------|--------|")
                for c in data[q]:
                    lines.append(f"| {c['check']} | {c['value']} | {c['status']} |")
        return lines

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []

        title = f"{self.module} · {self.module_title}" if self.module_title else self.module
        lines.append(f"# {title}")

        # Snapshot
        lines.append("\n## Snapshot\n")
        lines.append("```")
        lines.append(self.sample_df.head(3).to_string(index=False))
        lines.append("```")

        # Data Summary
        lines.append("\n## Data Summary\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Rows | {self.output.rows:,} |")
        lines.append(f"| Series | {self.output.series:,} |")
        lines.append(f"| Date range | {self.output.date_min} → {self.output.date_max} |")
        lines.append(f"| Frequency | {self.output.frequency} |")
        lines.append(f"| History | {self.output.n_weeks} weeks ({self.output.n_weeks/52:.1f} yrs) |")
        lines.append(f"| Target zeros | {self.output.target_zeros_pct:.1f}% |")

        # Memory assessment
        lines.append("\n## Memory\n")
        mem_mb = self.output.memory_mb
        mem_gb = mem_mb / 1024
        if mem_mb < 100:
            mem_tier, mem_note = "Small", "Fits easily in memory"
        elif mem_mb < 1024:
            mem_tier, mem_note = "Medium", "Fine for most operations"
        elif mem_mb < 10240:
            mem_tier, mem_note = "Large", "May need chunking for CV/grid search"
        else:
            mem_tier, mem_note = "Very Large", "Consider sampling or distributed processing"
        mem_str = f"{mem_gb:.1f} GB" if mem_gb >= 1 else f"{mem_mb:.0f} MB"
        lines.append(f"**{mem_str}** ({mem_tier}) — {mem_note}")

        # 5Q Checks
        for section in self.sections:
            if section['type'] == '5q':
                lines.append("\n## 5Q Checks")
                lines.extend(self._render_5q_markdown(section['data']))
            elif section['type'] == 'timeline':
                lines.append("\n## Timeline Health\n")
                data = section['data']
                stable = "Stable" if data.get('is_stable') else "Variable"
                lines.append(f"- **Series per date:** {data['min_series']:,} – {data['max_series']:,} ({stable})")
            elif section['type'] == 'gaps':
                lines.append("\n## Gap Analysis\n")
                data = section['data']
                lines.append(f"- **Gaps filled:** {data['gaps_filled']:,}")
                lines.append(f"- **Series affected:** {data['series_with_gaps']:,}")
            elif section['type'] == 'imputation':
                lines.append("\n## Imputation\n")
                lines.append(f"- **Remaining NAs:** {section['data']['remaining_nas']:,}")
            elif section['type'] == 'features':
                lines.append("\n## Features\n")
                data = section['data']
                lines.append(f"- **Features added:** {data['n_features']}")
                if data['feature_cols']:
                    lines.append(f"- **Columns:** {', '.join(data['feature_cols'][:5])}...")

        # Blocking issues
        blocking = self._get_blocking_issues()
        if any(s['type'] == '5q' for s in self.sections):
            lines.append("\n## Blocking Issues\n")
            if blocking:
                for b in blocking:
                    lines.append(f"- ❌ {b}")
            else:
                lines.append("None ✓")

        # Decisions
        if self.decisions_df is not None and not self.decisions_df.empty:
            lines.append("\n## Decisions\n")
            lines.append(self._decisions_df_to_markdown(self.decisions_df))
        elif hasattr(self, '_decisions_md') and self._decisions_md:
            lines.append("\n## Decisions\n")
            lines.append(self._decisions_md)

        # Changes
        lines.append("\n## Changes\n")
        lines.append("| Metric | Before | After | Δ |")
        lines.append("|--------|--------|-------|---|")
        row_pct = ((self.output.rows - self.input.rows) / self.input.rows * 100) if self.input.rows > 0 else 0
        lines.append(f"| Rows | {self.input.rows:,} | {self.output.rows:,} | {row_pct:+.0f}% |")

        if self.input.target_nas > 0 or self.output.target_nas > 0:
            fixed = " ✓" if self.output.target_nas == 0 and self.input.target_nas > 0 else ""
            lines.append(f"| NAs (y) | {self.input.target_nas:,} | {self.output.target_nas:,} | Fixed{fixed} |")

        if self.input.frequency != self.output.frequency:
            lines.append(f"| Frequency | {self.input.frequency} | {self.output.frequency} | — |")

        if self.input.duplicates > 0:
            lines.append(f"| Duplicates | {self.input.duplicates:,} | {self.output.duplicates:,} | ✓ Fixed |")

        # Memory change - show reduction percentage
        if self.input.memory_mb > 0:
            mem_reduction = ((self.input.memory_mb - self.output.memory_mb) / self.input.memory_mb * 100)
            if self.output.memory_mb < self.input.memory_mb:
                lines.append(f"| Memory | {self.input.memory_mb:.1f} MB | {self.output.memory_mb:.1f} MB | -{mem_reduction:.0f}% ✓ |")
            else:
                lines.append(f"| Memory | {self.input.memory_mb:.1f} MB | {self.output.memory_mb:.1f} MB | +{-mem_reduction:.0f}% |")

        lines.append(f"\n---\n*Generated: {self.generated_at}*")

        return "\n".join(lines)
    
    def display(self):
        """Display in notebook."""
        print(self.to_text())

    def to_html(self) -> str:
        """Generate styled HTML report (for PDF conversion)."""
        text_content = self.to_text()

        # Escape HTML special characters
        import html
        escaped = html.escape(text_content)

        title = f"{self.module} · {self.module_title}" if self.module_title else self.module

        html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @page {{
            size: letter;
            margin: 0.75in;
        }}
        body {{
            font-family: "SF Mono", "Monaco", "Inconsolata", "Fira Code", "Courier New", monospace;
            font-size: 10pt;
            line-height: 1.4;
            color: #1a1a1a;
            background: #ffffff;
            max-width: 8.5in;
            margin: 0 auto;
            padding: 20px;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
            padding: 0;
        }}
        /* Status indicators */
        .content {{
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="content">
        <pre>{escaped}</pre>
    </div>
</body>
</html>'''
        return html_template

    def to_pdf(self, path: str):
        """
        Generate PDF report.

        Requires weasyprint: pip install weasyprint
        """
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError(
                "PDF export requires weasyprint. Install with: pip install weasyprint"
            )

        html_content = self.to_html()
        HTML(string=html_content).write_pdf(path)
        print(f"✓ Report saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'ModuleReport':
        """
        Load a saved report from file.

        Parameters
        ----------
        path : str
            Path to saved report file (.txt, .md, .html)

        Returns
        -------
        ModuleReport
            A minimal report object with parsed summary data
        """
        path = Path(path)

        with open(path, 'r') as f:
            content = f.read()

        # Create a minimal report-like object
        report = object.__new__(cls)
        report._raw_content = content
        report.module = "loaded"
        report.module_title = ""
        report.generated_at = ""
        report.sections = []
        report.decisions_df = None
        report._decisions_md = None

        # Parse all sections from content
        parsed = cls._parse_all_sections(content)
        report._parsed_summary = parsed.get('summary', {})
        report._parsed_snapshot = parsed.get('snapshot', "")
        report._parsed_memory = parsed.get('memory', {})
        report._parsed_checks = parsed.get('checks', {})
        report._parsed_timeline = parsed.get('timeline', {})
        report._parsed_blocking = parsed.get('blocking', [])
        report._parsed_decisions = parsed.get('decisions', "")
        report._parsed_changes = parsed.get('changes', {})

        # Create minimal Snapshot for output
        report.output = Snapshot(
            name="loaded",
            rows=0, columns=0, series=0,
            date_min="", date_max="",
            n_weeks=0, frequency="",
            target_zeros_pct=0.0, target_nas=0,
            duplicates=0, memory_mb=0.0
        )
        report.input = report.output
        report.sample_df = pd.DataFrame()
        report.params = {}

        return report

    @staticmethod
    def _parse_all_sections(content: str) -> dict:
        """Parse all sections from saved report text."""
        result = {
            'summary': {},
            'snapshot': "",
            'memory': {},
            'checks': {},
            'timeline': {},
            'blocking': [],
            'decisions': "",
            'changes': {},
        }

        lines = content.split('\n')
        current_section = None
        section_lines = []

        section_markers = {
            'SNAPSHOT': 'snapshot',
            'DATA SUMMARY': 'summary',
            'STRUCTURE': 'summary',
            'MEMORY': 'memory',
            '5Q CHECKS': 'checks',
            'TIMELINE HEALTH': 'timeline',
            'BLOCKING ISSUES': 'blocking',
            'DECISIONS': 'decisions',
            'CHANGES': 'changes',
        }

        for line in lines:
            # Check for section header
            new_section = None
            for marker, section_name in section_markers.items():
                if marker in line and '─' not in line:
                    new_section = section_name
                    break

            if new_section:
                # Save previous section
                if current_section and section_lines:
                    result[current_section] = ModuleReport._parse_section(current_section, section_lines)
                current_section = new_section
                section_lines = []
            elif current_section and not line.startswith('─') and not line.startswith('━'):
                section_lines.append(line)

        # Save last section
        if current_section and section_lines:
            result[current_section] = ModuleReport._parse_section(current_section, section_lines)

        return result

    @staticmethod
    def _parse_section(section_name: str, lines: List[str]):
        """Parse a specific section from its lines."""
        if section_name == 'snapshot':
            return '\n'.join(line for line in lines if line.strip())

        elif section_name == 'summary':
            summary = {}
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    if 'Rows' in line:
                        summary['Rows'] = parts[-1]
                    elif 'Series' in line and 'per' not in line.lower():
                        summary['Series'] = parts[-1]
                    elif 'Dates' in line and '→' in line:
                        idx = line.index('2') if '2' in line else -1
                        if idx > 0:
                            summary['Dates'] = line[idx:].strip()
                    elif 'Frequency' in line:
                        summary['Frequency'] = parts[-1]
                    elif 'History' in line and len(parts) > 1:
                        hist_start = line.find(parts[1])
                        if hist_start > 0:
                            summary['History'] = line[hist_start:].strip()
                    elif 'Target zeros' in line or 'zeros' in line.lower():
                        summary['Target zeros'] = parts[-1]
            return summary

        elif section_name == 'memory':
            for line in lines:
                if line.strip():
                    # Parse "✓ 134 MB (Medium) — Fine for most operations"
                    if 'MB' in line or 'GB' in line:
                        parts = line.split()
                        size = None
                        tier = None
                        for i, p in enumerate(parts):
                            if p in ('MB', 'GB') and i > 0:
                                try:
                                    size = float(parts[i-1])
                                    if p == 'GB':
                                        size *= 1024
                                except ValueError:
                                    pass
                            if p.startswith('(') and p.endswith(')'):
                                tier = p[1:-1]
                        note_idx = line.find('—')
                        note = line[note_idx+1:].strip() if note_idx > 0 else ""
                        return {'size_mb': size or 0, 'tier': tier or "", 'note': note}
            return {}

        elif section_name == 'checks':
            # Return raw text for now - complex to parse
            return '\n'.join(lines)

        elif section_name == 'timeline':
            for line in lines:
                if 'Series/date' in line or 'series' in line.lower():
                    # Parse "Series/date     6,979 – 30,490 (Variable)"
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if '–' in p or '-' in p:
                            try:
                                min_val = int(parts[i-1].replace(',', ''))
                                max_val = int(parts[i+1].replace(',', ''))
                                stable = 'Stable' in line
                                return {'min_series': min_val, 'max_series': max_val, 'is_stable': stable}
                            except (ValueError, IndexError):
                                pass
            return {}

        elif section_name == 'blocking':
            issues = []
            for line in lines:
                if line.strip() and '✗' in line:
                    issues.append(line.strip().replace('✗', '').strip())
            return issues

        elif section_name == 'decisions':
            return '\n'.join(line for line in lines if line.strip())

        elif section_name == 'changes':
            changes = {}
            for line in lines:
                if not line.strip():
                    continue
                if 'Rows' in line and '→' in line:
                    parts = line.split()
                    try:
                        before = int(parts[1].replace(',', ''))
                        after = int(parts[3].replace(',', ''))
                        changes['rows'] = {'before': before, 'after': after}
                    except (ValueError, IndexError):
                        pass
                elif 'NAs' in line and '→' in line:
                    parts = line.split()
                    try:
                        before = int(parts[1].replace(',', ''))
                        after = int(parts[3].replace(',', ''))
                        changes['nas'] = {'before': before, 'after': after}
                    except (ValueError, IndexError):
                        pass
                elif 'Frequency' in line and '→' in line:
                    parts = line.split()
                    try:
                        before = parts[1]
                        after = parts[3]
                        changes['frequency'] = {'before': before, 'after': after}
                    except IndexError:
                        pass
                elif 'Memory' in line and '→' in line:
                    parts = line.split()
                    try:
                        before = float(parts[1])
                        after = float(parts[4])
                        changes['memory'] = {'before_mb': before, 'after_mb': after}
                    except (ValueError, IndexError):
                        pass
            return changes

        return None

    @property
    def summary(self) -> dict:
        """Return the DATA SUMMARY section as a dictionary."""
        if hasattr(self, '_parsed_summary') and self._parsed_summary:
            return self._parsed_summary
        return {
            'Rows': f"{self.output.rows:,}",
            'Series': f"{self.output.series:,}",
            'Dates': f"{self.output.date_min} → {self.output.date_max}",
            'Frequency': self.output.frequency,
            'History': f"{self.output.n_weeks} weeks ({self.output.n_weeks/52:.1f} yrs)",
            'Target zeros': f"{self.output.target_zeros_pct:.1f}%",
        }

    @property
    def snapshot(self) -> str:
        """Return the SNAPSHOT section (sample data preview)."""
        if hasattr(self, '_parsed_snapshot') and self._parsed_snapshot:
            return self._parsed_snapshot
        return self.sample_df.head(3).to_string(index=False)

    @property
    def memory(self) -> dict:
        """Return the MEMORY assessment."""
        if hasattr(self, '_parsed_memory') and self._parsed_memory:
            return self._parsed_memory
        mem_mb = self.output.memory_mb
        if mem_mb < 100:
            tier, note = "Small", "Fits easily in memory"
        elif mem_mb < 1024:
            tier, note = "Medium", "Fine for most operations"
        elif mem_mb < 10240:
            tier, note = "Large", "May need chunking for CV/grid search"
        else:
            tier, note = "Very Large", "Consider sampling or distributed processing"
        return {
            'size_mb': round(mem_mb, 1),
            'tier': tier,
            'note': note,
        }

    @property
    def checks(self) -> dict:
        """Return the 5Q CHECKS section."""
        if hasattr(self, '_parsed_checks') and self._parsed_checks:
            return self._parsed_checks
        for section in self.sections:
            if section['type'] == '5q':
                return section['data']
        return {}

    @property
    def timeline(self) -> dict:
        """Return the TIMELINE HEALTH section."""
        if hasattr(self, '_parsed_timeline') and self._parsed_timeline:
            return self._parsed_timeline
        for section in self.sections:
            if section['type'] == 'timeline':
                return section['data']
        return {}

    @property
    def blocking_issues(self) -> list:
        """Return list of blocking issues from 5Q checks."""
        if hasattr(self, '_parsed_blocking') and self._parsed_blocking is not None:
            return self._parsed_blocking
        return self._get_blocking_issues()

    @property
    def decisions(self) -> str:
        """Return the DECISIONS section."""
        if hasattr(self, '_parsed_decisions') and self._parsed_decisions:
            return self._parsed_decisions
        if self.decisions_df is not None and not self.decisions_df.empty:
            return self._render_decisions_df(self.decisions_df)
        if hasattr(self, '_decisions_md') and self._decisions_md:
            return self._decisions_md
        return ""

    @property
    def changes(self) -> dict:
        """Return the CHANGES section."""
        if hasattr(self, '_parsed_changes') and self._parsed_changes:
            return self._parsed_changes
        row_delta = self.output.rows - self.input.rows
        row_pct = (row_delta / self.input.rows * 100) if self.input.rows > 0 else 0
        result = {
            'rows': {'before': self.input.rows, 'after': self.output.rows, 'pct': round(row_pct, 0)},
        }
        if self.input.target_nas > 0 or self.output.target_nas > 0:
            result['nas'] = {'before': self.input.target_nas, 'after': self.output.target_nas}
        if self.input.frequency != self.output.frequency:
            result['frequency'] = {'before': self.input.frequency, 'after': self.output.frequency}
        if self.input.memory_mb > 0:
            mem_pct = ((self.input.memory_mb - self.output.memory_mb) / self.input.memory_mb * 100)
            result['memory'] = {
                'before_mb': round(self.input.memory_mb, 1),
                'after_mb': round(self.output.memory_mb, 1),
                'pct': round(mem_pct, 0)
            }
        return result

    @property
    def full_report(self) -> str:
        """Return the full report as text."""
        if hasattr(self, '_raw_content') and self._raw_content:
            return self._raw_content
        return self.to_text()

    def save(self, path: str, format: str = 'auto'):
        """
        Save report to file.

        Parameters
        ----------
        path : str
            Output file path
        format : str, default='auto'
            Output format: 'auto' (detect from extension), 'txt', 'md', 'html', 'pdf'
        """
        path = Path(path)
        fmt = path.suffix.lstrip('.') if format == 'auto' else format

        if fmt == 'pdf':
            self.to_pdf(str(path))
        elif fmt == 'html':
            with open(path, 'w') as f:
                f.write(self.to_html())
            print(f"✓ Report saved: {path}")
        elif fmt in ('md', 'markdown'):
            with open(path, 'w') as f:
                f.write(self.to_markdown())
            print(f"✓ Report saved: {path}")
        else:  # txt or any other
            with open(path, 'w') as f:
                f.write(self.to_text())
            print(f"✓ Report saved: {path}")


# =============================================================================
# PLOT FUNCTION (kept separate - user calls if needed)
# =============================================================================

def plot_timeline_health(
    df: pd.DataFrame,
    date_col: str = 'ds',
    id_col: str = 'unique_id',
    figsize: tuple = (12, 3),
    title: str = 'Timeline Health: Series Reporting per Date'
):
    """Single plot: series count per date. Flat = good."""
    import matplotlib.pyplot as plt
    
    series_per_date = df.groupby(date_col)[id_col].nunique()
    
    fig, ax = plt.subplots(figsize=figsize)
    series_per_date.plot(ax=ax, color='#2596be', linewidth=1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Series count')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(series_per_date.max(), color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig, ax


__all__ = ['ModuleReport', 'Snapshot', 'plot_timeline_health']