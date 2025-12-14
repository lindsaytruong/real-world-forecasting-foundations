"""
First Contact Report
====================

Data quality assessment with auto-detected changes from prior module.

Features:
- Schema/completeness/validity/integrity checks
- Auto-detected changes when prior_report provided
- Clean artifact display
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class FirstContactReport:
    """
    Structured report card for time series data.
    
    Attributes
    ----------
    checks : pd.DataFrame
        Individual check results
    summary : dict
        Summary statistics
    dataset_name : str
        Name for display
    generated_at : str
        Timestamp
    prior_report : FirstContactReport, optional
        For computing changes
        
    Examples
    --------
    >>> # Observation module (no prior)
    >>> report = first_contact_check(df, dataset_name='1.06 Weekly M5')
    >>> report.table()
    
    >>> # Transformation module (with prior)
    >>> report = first_contact_check(df, dataset_name='1.08 Prepared', prior_report=prior)
    >>> report.changes()  # Auto-detected diff
    """
    checks: pd.DataFrame
    summary: dict
    dataset_name: str = "dataset"
    generated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    prior_report: Optional['FirstContactReport'] = None
    
    def __repr__(self):
        passed = (self.checks['Status'] == '✓').sum()
        failed = (self.checks['Status'] == '✗').sum()
        arrow = f"{self.prior_report.dataset_name} → " if self.prior_report else ""
        return f"FirstContactReport({arrow}'{self.dataset_name}': {passed} passed, {failed} failed)"
    
    def _repr_html_(self):
        """Auto-display in Jupyter."""
        return self.table()._repr_html_()
    
    # =========================================================================
    # CHANGES (auto-detected)
    # =========================================================================
    
    def changes(self):
        """
        Show what changed from prior report.
        
        Returns styled comparison table, or None if no prior.
        """
        if self.prior_report is None:
            print("ℹ No prior report to compare against.")
            return None
        
        prior = self.prior_report
        
        def parse_num(val):
            """Extract number from summary string."""
            if isinstance(val, str):
                clean = val.replace(',', '').split()[0].split('(')[0]
                try:
                    return float(clean)
                except ValueError:
                    return None
            return val
        
        rows = []
        
        # Metrics to compare
        metrics = ['Rows', 'Columns', 'Series', 'Unique dates', 'Memory']
        
        for key in metrics:
            prior_val = prior.summary.get(key, '—')
            curr_val = self.summary.get(key, '—')
            
            if prior_val == '—' and curr_val == '—':
                continue
            
            prior_num = parse_num(prior_val)
            curr_num = parse_num(curr_val)
            
            if prior_num is not None and curr_num is not None:
                diff = curr_num - prior_num
                if diff > 0:
                    change = f"+{diff:,.0f}"
                elif diff < 0:
                    change = f"{diff:,.0f}"
                else:
                    change = "—"
            elif prior_val != curr_val:
                change = "Changed"
            else:
                change = "—"
            
            rows.append({
                'Metric': key,
                'Before': prior_val,
                'After': curr_val,
                'Δ': change
            })
        
        # NA comparison
        prior_na = prior.checks[prior.checks['Check'].str.contains('NAs: y', na=False)]
        curr_na = self.checks[self.checks['Check'].str.contains('NAs: y', na=False)]
        
        if len(prior_na) > 0 and len(curr_na) > 0:
            prior_na_val = prior_na.iloc[0]['Value']
            curr_na_val = curr_na.iloc[0]['Value']
            
            prior_has_na = not str(prior_na_val).startswith('0')
            curr_has_na = not str(curr_na_val).startswith('0')
            
            if prior_has_na and not curr_has_na:
                change = "✓ Fixed"
            elif prior_na_val != curr_na_val:
                change = "Changed"
            else:
                change = "—"
            
            rows.append({
                'Metric': 'NAs (target)',
                'Before': prior_na_val,
                'After': curr_na_val,
                'Δ': change
            })
        
        df = pd.DataFrame(rows)
        
        def style_change(val):
            if '✓' in str(val):
                return 'color: #06A77D; font-weight: bold'
            elif str(val).startswith('+'):
                return 'color: #2596be'
            elif str(val).startswith('-'):
                return 'color: #c41e3a'
            return ''
        
        return (df.style
            .applymap(style_change, subset=['Δ'])
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left')]},
                {'selector': 'caption', 'props': [('font-size', '14px'), ('font-weight', 'bold')]},
            ])
            .set_caption(f"📈 Changes: {prior.dataset_name} → {self.dataset_name}")
            .hide(axis='index')
        )
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def table(self):
        """Styled report card table."""
        df = self.checks.copy()
        
        def style_row(row):
            colors = {
                '✓': 'background-color: #d4edda',
                '✗': 'background-color: #f8d7da',
                '⚠': 'background-color: #fff3cd',
                'ℹ': 'background-color: #e2e3e5',
            }
            return [colors.get(row['Status'], '')] * len(row)
        
        return (df.style
            .apply(style_row, axis=1)
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left'), ('font-weight', 'bold')]},
                {'selector': 'caption', 'props': [('font-size', '14px'), ('font-weight', 'bold')]},
            ])
            .set_caption(f"📋 Data Quality: {self.dataset_name}")
            .hide(axis='index')
        )
    
    def summary_table(self):
        """Summary stats as styled table."""
        df = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in self.summary.items()])
        return (df.style
            .set_properties(**{'text-align': 'left'})
            .set_caption(f"📊 Summary: {self.dataset_name}")
            .hide(axis='index')
        )
    
    def blocking_issues(self) -> pd.DataFrame:
        """Return only failing checks."""
        return self.checks[self.checks['Status'] == '✗'].copy()
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def save(self, path: str):
        """Save to JSON."""
        path = Path(path)
        data = {
            'dataset_name': self.dataset_name,
            'generated_at': self.generated_at,
            'prior_dataset': self.prior_report.dataset_name if self.prior_report else None,
            'prior_summary': self.prior_report.summary if self.prior_report else None,
            'summary': self.summary,
            'checks': self.checks.to_dict(orient='records'),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FirstContactReport':
        """Load from JSON."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct minimal prior for changes display
        prior_report = None
        if data.get('prior_summary') and data.get('prior_dataset'):
            prior_report = cls(
                checks=pd.DataFrame(),
                summary=data['prior_summary'],
                dataset_name=data['prior_dataset']
            )
        
        return cls(
            checks=pd.DataFrame(data['checks']),
            summary=data['summary'],
            dataset_name=data['dataset_name'],
            generated_at=data['generated_at'],
            prior_report=prior_report
        )


def first_contact_check(
    df: pd.DataFrame,
    date_col: str = 'ds',
    target_col: str = 'y',
    dataset_name: str = 'dataset',
    prior_report: Optional[FirstContactReport] = None,
) -> FirstContactReport:
    """
    Run first-contact checks and return a report.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    date_col : str
        Date column name
    target_col : str
        Target column name
    dataset_name : str
        Name for the report
    prior_report : FirstContactReport, optional
        Prior module's report (for auto-detecting changes)
    
    Returns
    -------
    FirstContactReport
    
    Examples
    --------
    >>> # Simple
    >>> report = first_contact_check(df, dataset_name='1.06 Weekly')
    
    >>> # With prior (auto-detects changes)
    >>> report = first_contact_check(df, dataset_name='1.08 Prepared', prior_report=prior)
    >>> report.changes()
    """
    
    checks = []
    id_cols = [c for c in df.columns if c not in [date_col, target_col]]
    n_rows = len(df)
    
    def add(category, check, status, value, notes=""):
        checks.append({
            'Category': category,
            'Check': check,
            'Status': status,
            'Value': value,
            'Notes': notes
        })
    
    # --- SCHEMA ---
    has_date = date_col in df.columns
    has_target = target_col in df.columns
    
    add('Schema', f'Column: {date_col}', 
        '✓' if has_date else '✗', 
        'Present' if has_date else 'Missing',
        'Required')
    add('Schema', f'Column: {target_col}', 
        '✓' if has_target else '✗',
        'Present' if has_target else 'Missing',
        'Required')
    
    if not (has_date and has_target):
        return FirstContactReport(
            checks=pd.DataFrame(checks),
            summary={'Error': 'Missing required columns'},
            dataset_name=dataset_name,
            prior_report=prior_report
        )
    
    date_is_dt = pd.api.types.is_datetime64_any_dtype(df[date_col])
    target_is_num = pd.api.types.is_numeric_dtype(df[target_col])
    
    add('Schema', f'Type: {date_col}',
        '✓' if date_is_dt else '✗',
        str(df[date_col].dtype),
        '' if date_is_dt else 'Use pd.to_datetime()')
    add('Schema', f'Type: {target_col}',
        '✓' if target_is_num else '✗',
        str(df[target_col].dtype),
        '' if target_is_num else 'Convert to numeric')
    
    # --- COMPLETENESS ---
    na_date = df[date_col].isna().sum()
    na_target = df[target_col].isna().sum()
    na_pct = na_target / n_rows * 100 if n_rows > 0 else 0
    
    add('Completeness', f'NAs: {date_col}',
        '✓' if na_date == 0 else '✗',
        f'{na_date:,}',
        '' if na_date == 0 else 'Dates cannot be null')
    add('Completeness', f'NAs: {target_col}',
        '✓' if na_target == 0 else 'ℹ',
        f'{na_target:,} ({na_pct:.1f}%)',
        '' if na_target == 0 else 'Handle in imputation')
    
    if id_cols:
        na_ids = df[id_cols].isna().any(axis=1).sum()
        add('Completeness', 'NAs: ID columns',
            '✓' if na_ids == 0 else '✗',
            f'{na_ids:,}',
            '' if na_ids == 0 else 'IDs cannot be null')
    
    # --- VALIDITY ---
    if date_is_dt:
        dates = df[date_col]
        min_date, max_date = dates.min(), dates.max()
        today = pd.Timestamp.today()
        
        n_old = (dates < '1900-01-01').sum()
        n_future = (dates > today).sum()
        
        add('Validity', 'Dates ≥ 1900',
            '✓' if n_old == 0 else '✗',
            f'{n_old:,} invalid',
            '' if n_old == 0 else 'Check for errors')
        add('Validity', 'No future dates',
            '✓' if n_future == 0 else '⚠',
            f'{n_future:,} future',
            '' if n_future == 0 else 'May need filtering')
    
    if target_is_num:
        n_neg = (df[target_col] < 0).sum()
        add('Validity', f'{target_col} ≥ 0',
            'ℹ',
            f'{n_neg:,} negative',
            'Review if unexpected')
    
    # --- INTEGRITY ---
    key_cols = [date_col] + id_cols[:5]
    n_dups = df.duplicated(subset=key_cols).sum()
    add('Integrity', 'No duplicate keys',
        '✓' if n_dups == 0 else '✗',
        f'{n_dups:,}',
        '' if n_dups == 0 else f'Keys: {", ".join(key_cols[:2])}...')
    
    if date_is_dt:
        unique_dates = df[date_col].drop_duplicates().sort_values()
        if len(unique_dates) > 1:
            freq = unique_dates.diff().mode().iloc[0]
            freq_str = str(freq).replace('0 days ', '').replace('00:00:00', '').strip()
            if not freq_str:
                freq_str = 'Daily'
            add('Integrity', 'Frequency',
                'ℹ',
                freq_str,
                'Check for gaps')
    
    # --- SUMMARY ---
    summary = {
        'Rows': f'{n_rows:,}',
        'Columns': f'{df.shape[1]}',
    }
    
    if id_cols:
        if 'unique_id' in id_cols:
            n_series = df['unique_id'].nunique()
        elif len(id_cols) == 1:
            n_series = df[id_cols[0]].nunique()
        else:
            n_series = df.groupby(id_cols[:3], sort=False).ngroups
        summary['Series'] = f'{n_series:,}'
    
    if date_is_dt:
        summary['Date range'] = f'{min_date.date()} → {max_date.date()}'
        summary['Unique dates'] = f'{df[date_col].nunique():,}'
    
    if target_is_num:
        summary[f'{target_col} mean'] = f'{df[target_col].mean():,.2f}'
        n_zeros = (df[target_col] == 0).sum()
        summary[f'{target_col} zeros'] = f'{n_zeros:,} ({n_zeros/n_rows:.1%})'
    
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    summary['Memory'] = f'{mem_mb:.1f} MB'
    
    return FirstContactReport(
        checks=pd.DataFrame(checks),
        summary=summary,
        dataset_name=dataset_name,
        prior_report=prior_report
    )


__all__ = ['first_contact_check', 'FirstContactReport']