"""
Core report classes: ModuleReport, Snapshot.

ModuleReport provides a consistent interface across all modules:
    report.summary      # dict
    report.target       # list[check] - Q1
    report.metric       # list[check] - Q2
    report.structure    # list[check] - Q3
    report.drivers      # list[check] - Q4
    report.readiness    # list[check] - final status (optional)
    report.decisions    # str
    report.changes      # dict
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

from .checks import MODULE_CHECKS, MODULE_TITLES
from .formatters import (
    format_table,
    format_decisions_df,
    decisions_df_to_markdown,
    render_checks_text,
    render_checks_markdown,
    section_header,
    get_memory_tier,
)


# =============================================================================
# DECISIONS LOADER
# =============================================================================

def _find_project_root(start_path: Path = None) -> Optional[Path]:
    """Find project root by looking for config/decisions.yaml or pyproject.toml."""
    if start_path is None:
        start_path = Path.cwd()
    
    current = start_path.resolve()
    for _ in range(10):
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
# MODULE REPORT
# =============================================================================

class ModuleReport:
    """
    Unified report class for all modules.
    
    Provides consistent properties across all modules:
        report.summary, report.target, report.metric, report.structure,
        report.drivers, report.readiness, report.decisions, report.changes
    
    Parameters
    ----------
    module : str
        Module ID (e.g., "1.06", "1.08")
    input_df : pd.DataFrame
        Data before transformations
    output_df : pd.DataFrame
        Data after transformations
    decisions : str, pd.DataFrame, or None
        Explicit decisions. If None, auto-loads from config/decisions.yaml.
    drivers : dict[str, pd.DataFrame], optional
        Driver datasets (calendar, prices) for validation.
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
        
        # Load decisions
        if decisions is not None:
            if isinstance(decisions, pd.DataFrame):
                self.decisions_df = decisions
                self._decisions_md = None
            else:
                self.decisions_df = None
                self._decisions_md = decisions.strip()
        else:
            self.decisions_df = _load_decisions_from_yaml(module)
            self._decisions_md = None
        
        # Params for check functions
        if hierarchy_cols is None:
            hierarchy_cols = ['state_id', 'store_id', 'cat_id', 'dept_id']
        
        params = {
            'date_col': date_col,
            'target_col': target_col,
            'id_col': id_col,
            'hierarchy_cols': hierarchy_cols,
            'drivers': drivers or {},
            'input_df': input_df,
            **kwargs
        }
        
        # Run module-specific checks
        module_checks = MODULE_CHECKS.get(module, {})
        self._target = module_checks.get('target', lambda *a, **k: [])(output_df, **params)
        self._metric = module_checks.get('metric', lambda *a, **k: [])(output_df, **params)
        self._structure = module_checks.get('structure', lambda *a, **k: [])(output_df, **params)
        self._drivers = module_checks.get('drivers', lambda *a, **k: [])(output_df, **params)
        self._readiness = module_checks.get('readiness', lambda *a, **k: [])(output_df, **params)
    
    # -------------------------------------------------------------------------
    # Properties - Consistent API
    # -------------------------------------------------------------------------
    
    @property
    def summary(self) -> dict:
        """DATA SUMMARY as dict."""
        return {
            'Rows': f"{self.output.rows:,}",
            'Series': f"{self.output.series:,}",
            'Dates': f"{self.output.date_min} → {self.output.date_max}",
            'Frequency': self.output.frequency,
            'History': f"{self.output.n_weeks} weeks ({self.output.n_weeks/52:.1f} yrs)",
            'Target zeros': f"{self.output.target_zeros_pct:.1f}%",
        }
    
    @property
    def target(self) -> List[Dict]:
        """Q1: TARGET checks."""
        return self._target
    
    @property
    def metric(self) -> List[Dict]:
        """Q2: METRIC checks."""
        return self._metric
    
    @property
    def structure(self) -> List[Dict]:
        """Q3: STRUCTURE checks."""
        return self._structure
    
    @property
    def drivers(self) -> List[Dict]:
        """Q4: DRIVERS checks."""
        return self._drivers
    
    @property
    def readiness(self) -> List[Dict]:
        """READINESS checks (prep modules only)."""
        return self._readiness
    
    @property
    def decisions(self) -> str:
        """DECISIONS as formatted string."""
        if self.decisions_df is not None and not self.decisions_df.empty:
            return format_decisions_df(self.decisions_df)
        if self._decisions_md:
            return self._decisions_md
        return ""
    
    @property
    def changes(self) -> dict:
        """CHANGES vs input."""
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
    def blocking_issues(self) -> List[str]:
        """List of blocking issues (status == '✗')."""
        issues = []
        for checks in [self._target, self._metric, self._structure, self._drivers]:
            for c in checks:
                if c.get('status') == '✗':
                    issues.append(f"{c['check']}: {c['value']}")
        return issues
    
    @property
    def memory(self) -> dict:
        """Memory assessment."""
        tier, note, status = get_memory_tier(self.output.memory_mb)
        return {
            'size_mb': round(self.output.memory_mb, 1),
            'tier': tier,
            'note': note,
            'status': status,
        }
    
    # -------------------------------------------------------------------------
    # Output Methods
    # -------------------------------------------------------------------------
    
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
        lines.append(section_header("SNAPSHOT"))
        lines.append(self.sample_df.head(3).to_string(index=False))
        
        # Data Summary
        lines.append(section_header("DATA SUMMARY"))
        for k, v in self.summary.items():
            lines.append(f"  {k:<15} {v}")
        
        # Memory
        lines.append(section_header("MEMORY"))
        mem = self.memory
        lines.append(f"  {mem['status']} {mem['size_mb']:.0f} MB ({mem['tier']}) — {mem['note']}")
        
        # 5Q Sections
        q_sections = [
            ('Q1 · Target', self._target),
            ('Q2 · Metric', self._metric),
            ('Q3 · Structure', self._structure),
            ('Q4 · Drivers', self._drivers),
        ]
        
        has_checks = any(checks for _, checks in q_sections)
        if has_checks:
            lines.append(section_header("5Q CHECKS"))
            for name, checks in q_sections:
                if checks:
                    lines.append(f"\n  {name}")
                    lines.extend(render_checks_text(checks, indent="    "))
        
        # Readiness (if present)
        if self._readiness:
            lines.append(section_header("READINESS"))
            lines.extend(render_checks_text(self._readiness))
        
        # Blocking Issues
        if has_checks:
            lines.append(section_header("BLOCKING ISSUES"))
            if self.blocking_issues:
                for b in self.blocking_issues:
                    lines.append(f"  ✗ {b}")
            else:
                lines.append("  None ✓")
        
        # Decisions
        if self.decisions:
            lines.append(section_header("DECISIONS"))
            lines.append(self.decisions)
        
        # Changes
        lines.append(section_header("CHANGES"))
        c = self.changes
        row_pct = c['rows']['pct']
        sign = '+' if row_pct >= 0 else ''
        lines.append(f"  {'Rows':<15} {c['rows']['before']:,} → {c['rows']['after']:,}  ({sign}{row_pct:.0f}%)")
        if 'nas' in c:
            fixed = "✓ Fixed" if c['nas']['after'] == 0 and c['nas']['before'] > 0 else ""
            lines.append(f"  {'NAs (y)':<15} {c['nas']['before']:,} → {c['nas']['after']:,}  {fixed}")
        if 'frequency' in c:
            lines.append(f"  {'Frequency':<15} {c['frequency']['before']} → {c['frequency']['after']}")
        if 'memory' in c:
            mem_sign = '+' if c['memory']['pct'] < 0 else '-'
            lines.append(f"  {'Memory':<15} {c['memory']['before_mb']:.1f} MB → {c['memory']['after_mb']:.1f} MB  ({mem_sign}{abs(c['memory']['pct']):.0f}%)")
        
        # Footer
        lines.append("\n" + "━" * w)
        lines.append(f"Generated: {self.generated_at}")
        lines.append("━" * w)
        
        return '\n'.join(lines)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []
        
        # Header
        title = f"{self.module} · {self.module_title}" if self.module_title else self.module
        lines.append(f"# {title}\n")
        
        # Summary
        lines.append("## Data Summary\n")
        for k, v in self.summary.items():
            lines.append(f"- **{k}:** {v}")
        
        # Memory
        mem = self.memory
        lines.append(f"\n**Memory:** {mem['status']} {mem['size_mb']:.0f} MB ({mem['tier']}) — {mem['note']}\n")
        
        # 5Q Sections
        q_sections = [
            ('Q1: Target', self._target),
            ('Q2: Metric', self._metric),
            ('Q3: Structure', self._structure),
            ('Q4: Drivers', self._drivers),
        ]
        
        for name, checks in q_sections:
            if checks:
                lines.append(f"\n## {name}\n")
                lines.extend(render_checks_markdown(checks))
        
        # Readiness
        if self._readiness:
            lines.append("\n## Readiness\n")
            lines.extend(render_checks_markdown(self._readiness))
        
        # Blocking Issues
        lines.append("\n## Blocking Issues\n")
        if self.blocking_issues:
            for b in self.blocking_issues:
                lines.append(f"- ❌ {b}")
        else:
            lines.append("None ✓")
        
        # Decisions
        if self.decisions_df is not None and not self.decisions_df.empty:
            lines.append("\n## Decisions\n")
            lines.append(decisions_df_to_markdown(self.decisions_df))
        elif self._decisions_md:
            lines.append("\n## Decisions\n")
            lines.append(self._decisions_md)
        
        # Changes
        lines.append("\n## Changes\n")
        lines.append("| Metric | Before | After | Δ |")
        lines.append("|--------|--------|-------|---|")
        c = self.changes
        lines.append(f"| Rows | {c['rows']['before']:,} | {c['rows']['after']:,} | {c['rows']['pct']:+.0f}% |")
        if 'nas' in c:
            fixed = " ✓" if c['nas']['after'] == 0 else ""
            lines.append(f"| NAs (y) | {c['nas']['before']:,} | {c['nas']['after']:,} | Fixed{fixed} |")
        if 'frequency' in c:
            lines.append(f"| Frequency | {c['frequency']['before']} | {c['frequency']['after']} | — |")
        if 'memory' in c:
            lines.append(f"| Memory | {c['memory']['before_mb']:.0f} MB | {c['memory']['after_mb']:.0f} MB | {c['memory']['pct']:+.0f}% |")
        
        lines.append(f"\n---\n*Generated: {self.generated_at}*")
        
        return '\n'.join(lines)
    
    def display(self):
        """Print text report."""
        print(self.to_text())
    
    def to_dict(self) -> dict:
        """Serialize report to dictionary for storage."""
        return {
            'module': self.module,
            'module_title': self.module_title,
            'generated_at': self.generated_at,
            'input': self.input.__dict__,
            'output': self.output.__dict__,
            'target': self._target,
            'metric': self._metric,
            'structure': self._structure,
            'drivers': self._drivers,
            'readiness': self._readiness,
            'decisions_md': self._decisions_md,
            'decisions_df': self.decisions_df.to_dict('records') if self.decisions_df is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ModuleReport':
        """Reconstruct report from dictionary."""
        report = object.__new__(cls)
        report.module = data['module']
        report.module_title = data.get('module_title', '')
        report.generated_at = data['generated_at']
        report.input = Snapshot(**data['input'])
        report.output = Snapshot(**data['output'])
        report._target = data.get('target', [])
        report._metric = data.get('metric', [])
        report._structure = data.get('structure', [])
        report._drivers = data.get('drivers', [])
        report._readiness = data.get('readiness', [])
        report._decisions_md = data.get('decisions_md')
        report.decisions_df = pd.DataFrame(data['decisions_df']) if data.get('decisions_df') else None
        report.sample_df = pd.DataFrame()  # Not stored, empty placeholder
        return report

    @classmethod
    def load(cls, path: str) -> 'ModuleReport':
        """Load report from txt file by parsing the text format."""
        import re
        path = Path(path)

        with open(path, 'r') as f:
            content = f.read()

        # Parse module from header
        module_match = re.search(r'^([\d.]+)\s*·\s*(.*)$', content, re.MULTILINE)
        module = module_match.group(1) if module_match else ''
        module_title = module_match.group(2).strip() if module_match else ''

        # Parse DATA SUMMARY section
        summary_section = re.search(r'DATA SUMMARY\n─+\n(.*?)(?=\n[A-Z]|\n━)', content, re.DOTALL)
        summary_data = {}
        if summary_section:
            for line in summary_section.group(1).strip().split('\n'):
                match = re.match(r'\s*(\S+(?:\s+\S+)?)\s{2,}(.+)$', line.strip())
                if match:
                    key, value = match.groups()
                    summary_data[key.strip()] = value.strip()

        # Parse output snapshot data from summary
        rows = int(summary_data.get('Rows', '0').replace(',', ''))
        series = int(summary_data.get('Series', '0').replace(',', ''))
        dates = summary_data.get('Dates', 'N/A → N/A')
        date_parts = dates.split(' → ')
        date_min = date_parts[0] if len(date_parts) > 0 else 'N/A'
        date_max = date_parts[1] if len(date_parts) > 1 else 'N/A'
        frequency = summary_data.get('Frequency', 'N/A')
        history = summary_data.get('History', '0 weeks')
        n_weeks_match = re.search(r'(\d+)\s*weeks', history)
        n_weeks = int(n_weeks_match.group(1)) if n_weeks_match else 0
        zeros_str = summary_data.get('Target zeros', '0%')
        target_zeros_pct = float(zeros_str.replace('%', ''))

        # Parse MEMORY section
        memory_match = re.search(r'MEMORY\n─+\n\s*[✓⚠✗ℹ]\s*(\d+)\s*MB', content)
        memory_mb = float(memory_match.group(1)) if memory_match else 0.0

        # Parse 5Q CHECKS
        def parse_checks(section_name: str) -> List[Dict]:
            pattern = rf'{section_name}\n(.*?)(?=\n\s*Q\d|BLOCKING|READINESS|DECISIONS|CHANGES|\n━)'
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                return []
            checks = []
            for line in match.group(1).strip().split('\n'):
                check_match = re.match(r'\s*([✓⚠✗ℹ])\s+(\S.*?)\s{2,}(.+)$', line.strip())
                if check_match:
                    status, check, value = check_match.groups()
                    checks.append({'status': status, 'check': check.strip(), 'value': value.strip()})
            return checks

        target_checks = parse_checks('Q1 · Target')
        metric_checks = parse_checks('Q2 · Metric')
        structure_checks = parse_checks('Q3 · Structure')
        drivers_checks = parse_checks('Q4 · Drivers')
        readiness_checks = parse_checks('READINESS')

        # Parse CHANGES to get input data
        changes_section = re.search(r'CHANGES\n─+\n(.*?)(?=\n━)', content, re.DOTALL)
        input_rows = rows
        if changes_section:
            rows_match = re.search(r'Rows\s+([\d,]+)\s*→', changes_section.group(1))
            if rows_match:
                input_rows = int(rows_match.group(1).replace(',', ''))

        # Parse generated_at
        gen_match = re.search(r'Generated:\s*(.+)$', content, re.MULTILINE)
        generated_at = gen_match.group(1).strip() if gen_match else ''

        # Build report object
        report = object.__new__(cls)
        report.module = module
        report.module_title = module_title
        report.generated_at = generated_at
        report.input = Snapshot(
            name='Input', rows=input_rows, columns=0, series=series,
            date_min=date_min, date_max=date_max, n_weeks=n_weeks,
            frequency=frequency, target_zeros_pct=target_zeros_pct,
            target_nas=0, duplicates=0, memory_mb=0.0
        )
        report.output = Snapshot(
            name='Output', rows=rows, columns=0, series=series,
            date_min=date_min, date_max=date_max, n_weeks=n_weeks,
            frequency=frequency, target_zeros_pct=target_zeros_pct,
            target_nas=0, duplicates=0, memory_mb=memory_mb
        )
        report._target = target_checks
        report._metric = metric_checks
        report._structure = structure_checks
        report._drivers = drivers_checks
        report._readiness = readiness_checks
        report._decisions_md = None
        report.decisions_df = None
        report.sample_df = pd.DataFrame()
        return report

    def save(self, path: str):
        """Save report to txt file."""
        path = Path(path)
        with open(path, 'w') as f:
            f.write(self.to_text())
        print(f"✓ Report saved: {path}")
    
    def __repr__(self):
        return f"ModuleReport({self.module}, {self.output.rows:,} rows, {self.output.series:,} series)"


# =============================================================================
# PLOT FUNCTION
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
