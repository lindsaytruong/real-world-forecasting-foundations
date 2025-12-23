"""
Core report classes: ModuleReport, Snapshot.

ModuleReport provides a consistent interface across all modules.
For Module 1.10, all statistics are computed internally — no external kwargs needed.

Usage:
    report = ModuleReport("1.10", input_df=diagnostics, output_df=scores_df)
    report.display()
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

from .checks import (
    MODULE_CHECKS, 
    MODULE_TITLES, 
    ld6_summary_table,
    render_1_10_text,
)
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
        series = df[id_col].nunique() if id_col in df.columns else rows
        
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
    
    For Module 1.10, all statistics are computed internally from output_df.
    No external kwargs needed.
    
    Parameters
    ----------
    module : str
        Module ID (e.g., "1.06", "1.08", "1.10")
    input_df : pd.DataFrame
        Data before transformations
    output_df : pd.DataFrame
        Data after transformations (for 1.10: scores DataFrame)
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
        
        # Store drivers
        if drivers is None:
            drivers = {}
        
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
        
        # Get module-specific check functions
        checks_funcs = MODULE_CHECKS.get(module, {})
        
        # Common kwargs for checks
        check_kwargs = {
            "date_col": date_col,
            "target_col": target_col,
            "id_col": id_col,
            "hierarchy_cols": hierarchy_cols or ["state_id", "store_id", "cat_id", "dept_id"],
            "calendar_df": drivers.get("calendar") if drivers else None,
            **kwargs
        }
        
        # Run standard 5Q checks
        self._target = checks_funcs.get("target", lambda df, **kw: [])(output_df, **check_kwargs)
        self._metric = checks_funcs.get("metric", lambda df, **kw: [])(output_df, **check_kwargs)
        self._structure = checks_funcs.get("structure", lambda df, **kw: [])(output_df, **check_kwargs)
        self._drivers = checks_funcs.get("drivers", lambda df, **kw: [])(output_df, **check_kwargs)
        self._readiness = checks_funcs.get("readiness", lambda df, **kw: [])(output_df, **check_kwargs)
        
        # Run diagnostic profile (1.09+)
        self._diagnostic_profile = checks_funcs.get("diagnostic_profile", lambda df, **kw: [])(output_df, **check_kwargs)

        # Run flags (all modules)
        self._flags = checks_funcs.get("flags", lambda df, **kw: [])(output_df, **check_kwargs)

        # =====================================================================
        # 1.10-specific: Compute all statistics internally
        # =====================================================================
        if module == "1.10":
            self._compute_1_10_stats(output_df, check_kwargs)
        else:
            # Initialize empty for non-1.10 modules
            self._ld6_summary_data = {}
            self._score_stats = {}
            self._score_validation = []
            self._quadrant_dist = {}
            self._quadrant_counts = {}
            self._dominant_quadrant = ""
            self._character = ""
            self._interpretations = []
            self._decisions_logged = []
    
    def _compute_1_10_stats(self, df: pd.DataFrame, check_kwargs: dict):
        """
        Compute all 1.10-specific statistics from the scores DataFrame.
        Called automatically during __init__ for module 1.10.
        """
        # LD6 summary table
        self._ld6_summary_data = ld6_summary_table(df)
        
        # Score statistics
        structure = df['structure_score'] if 'structure_score' in df.columns else pd.Series()
        chaos = df['chaos_score'] if 'chaos_score' in df.columns else pd.Series()
        
        if len(structure) > 0 and len(chaos) > 0:
            self._score_stats = {
                'structure': {
                    'median': float(structure.median()),
                    'mean': float(structure.mean()),
                    'std': float(structure.std()),
                    'skew': float(structure.skew()),
                    'min': float(structure.min()),
                    'max': float(structure.max()),
                },
                'chaos': {
                    'median': float(chaos.median()),
                    'mean': float(chaos.mean()),
                    'std': float(chaos.std()),
                    'skew': float(chaos.skew()),
                    'min': float(chaos.min()),
                    'max': float(chaos.max()),
                },
                'correlation': float(structure.corr(chaos)),
            }
        else:
            self._score_stats = {}
        
        # Score validation checks
        checks_funcs = MODULE_CHECKS.get("1.10", {})
        self._score_validation = checks_funcs.get("score_validation", lambda df, **kw: [])(df, **check_kwargs)
        
        # Quadrant distribution
        if 'sb_quadrant' in df.columns:
            self._quadrant_dist = df['sb_quadrant'].value_counts(normalize=True).to_dict()
            self._quadrant_counts = df['sb_quadrant'].value_counts().to_dict()
            self._dominant_quadrant = df['sb_quadrant'].value_counts().idxmax()
        else:
            self._quadrant_dist = {}
            self._quadrant_counts = {}
            self._dominant_quadrant = ""
        
        # Portfolio character
        structure_med = self._score_stats.get('structure', {}).get('median', 0.5)
        chaos_med = self._score_stats.get('chaos', {}).get('median', 0.5)
        
        if structure_med < 0.35 and chaos_med > 0.40:
            self._character = "CHAOS-DOMINANT"
            self._interpretations = [
                "Weak signal (low structure) combined with high instability (high chaos)",
                f"Consistent with {self._dominant_quadrant} being dominant ({self._quadrant_dist.get(self._dominant_quadrant, 0):.1%})",
                "Complex models will likely overfit — prioritize robust baselines"
            ]
        elif structure_med > 0.45 and chaos_med < 0.35:
            self._character = "STRUCTURE-DOMINANT"
            self._interpretations = [
                "Strong learnable patterns with manageable noise",
                "Good candidate for ML models (ETS, LightGBM)",
                "Invest in feature engineering and model complexity"
            ]
        else:
            self._character = "MIXED"
            self._interpretations = [
                "Neither structure nor chaos clearly dominates",
                "Different series need different approaches",
                "Segment by quadrant and use lane-specific strategies"
            ]

        # Initialize decisions logged (empty by default)
        self._decisions_logged = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def summary(self) -> Dict[str, str]:
        """Key metrics for quick reference."""
        o = self.output
        summary = {
            "Rows": f"{o.rows:,}",
            "Series": f"{o.series:,}",
        }
        if o.date_min != 'N/A':
            summary["Dates"] = f"{o.date_min} → {o.date_max}"
            summary["Frequency"] = o.frequency
            summary["History"] = f"{o.n_weeks} weeks ({o.n_weeks/52:.1f} yrs)"
        if o.target_zeros_pct > 0:
            summary["Target zeros"] = f"{o.target_zeros_pct:.1f}%"
        return summary
    
    @property
    def target(self) -> List[Dict]:
        return self._target
    
    @property
    def metric(self) -> List[Dict]:
        return self._metric
    
    @property
    def structure(self) -> List[Dict]:
        return self._structure
    
    @property
    def drivers(self) -> List[Dict]:
        return self._drivers
    
    @property
    def readiness(self) -> List[Dict]:
        return self._readiness
    
    @property
    def diagnostic_profile(self) -> List[Dict]:
        return self._diagnostic_profile
    
    @property
    def flags(self) -> List[Dict]:
        return self._flags
    
    @property
    def blocking_issues(self) -> List[str]:
        """Extract all blocking (✗) issues."""
        all_checks = self._target + self._metric + self._structure + self._drivers + self._readiness + self._flags
        return [f"{c['check']}: {c['value']}" for c in all_checks if c.get('status') == '✗']
    
    @property
    def changes(self) -> Dict[str, Any]:
        """Compute before/after deltas."""
        i, o = self.input, self.output
        changes = {
            "rows": {"before": i.rows, "after": o.rows, "pct": ((o.rows - i.rows) / i.rows * 100) if i.rows else 0},
            "columns": {"before": i.columns, "after": o.columns},
            "memory": {"before_mb": i.memory_mb, "after_mb": o.memory_mb, "pct": ((o.memory_mb - i.memory_mb) / i.memory_mb * 100) if i.memory_mb else 0},
        }
        if i.target_nas != o.target_nas:
            changes["nas"] = {"before": i.target_nas, "after": o.target_nas}
        if i.frequency != o.frequency:
            changes["frequency"] = {"before": i.frequency, "after": o.frequency}
        return changes
    
    # -------------------------------------------------------------------------
    # 1.10-specific properties
    # -------------------------------------------------------------------------
    
    @property
    def ld6_summary(self) -> Dict[str, Dict]:
        """LD6 metric statistics (1.10 only)."""
        return self._ld6_summary_data
    
    @property
    def score_stats(self) -> Dict[str, Any]:
        """Structure/Chaos score statistics (1.10 only)."""
        return self._score_stats
    
    @property
    def quadrant_distribution(self) -> Dict[str, float]:
        """S-B quadrant percentages (1.10 only)."""
        return self._quadrant_dist
    
    @property
    def character(self) -> str:
        """Portfolio character: CHAOS-DOMINANT, STRUCTURE-DOMINANT, or MIXED (1.10 only)."""
        return self._character
    
    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    
    def to_text(self) -> str:
        """Render full text report."""
        W = 65

        # Use specialized renderer for 1.10
        if self.module == "1.10":
            return render_1_10_text(
                self, W,
                section_header=section_header,
                get_memory_tier=get_memory_tier,
                render_checks_text=render_checks_text,
            )

        # Standard rendering for other modules
        lines = []

        # Header
        lines.append("━" * W)
        lines.append(f"{self.module} · {self.module_title}")
        lines.append("━" * W)
        
        # Snapshot
        lines.append(section_header("SNAPSHOT", W))
        sample_cols = self._get_snapshot_cols()
        if not self.sample_df.empty and sample_cols:
            display_cols = [c for c in sample_cols if c in self.sample_df.columns]
            if display_cols:
                sample = self.sample_df[display_cols].head(3)
                lines.append(sample.to_string(index=False))
        
        # Data Summary
        lines.append(section_header("DATA SUMMARY", W))
        if self.module == "1.09":
            lines.append(f"  Series diagnosed  {self.output.series:,}")
            lines.append(f"  Metrics computed  {self.output.columns}")
            lines.append(f"  Source            1.08 Data Preparation")
        else:
            for k, v in self.summary.items():
                lines.append(f"  {k:<16} {v}")
        
        # Memory
        lines.append(section_header("MEMORY", W))
        tier, note, status = get_memory_tier(self.output.memory_mb)
        lines.append(f"  {status} {self.output.memory_mb:.0f} MB ({tier}) — {note}")
        
        # 5Q Checks (1.06, 1.08) or Diagnostic Profile (1.09)
        if self.module in ["1.06", "1.08"]:
            lines.append(section_header("5Q CHECKS", W))
            
            if self._target:
                lines.append("\n  Q1 · Target")
                lines.extend(render_checks_text(self._target, "    "))
            
            if self._metric:
                lines.append("\n  Q2 · Metric")
                lines.extend(render_checks_text(self._metric, "    "))
            
            if self._structure:
                lines.append("\n  Q3 · Structure")
                lines.extend(render_checks_text(self._structure, "    "))
            
            if self._drivers:
                lines.append("\n  Q4 · Drivers")
                lines.extend(render_checks_text(self._drivers, "    "))
            
            if self._readiness:
                lines.append(section_header("READINESS", W))
                lines.extend(render_checks_text(self._readiness, "  "))
        
        elif self.module == "1.09":
            lines.append(section_header("DIAGNOSTIC PROFILE", W))
            lines.append("                        Low      Medium     High")
            lines.append("  STRUCTURE (▲ high = more signal)")
            for c in self._diagnostic_profile:
                if c['key'].startswith('diag.') and c['key'].split('.')[1] in ['trend', 'seasonal_strength']:
                    lines.append(f"    {c['check']:<20} {c['value']}")
            lines.append("  CHAOS (▼ high = less forecastable)")
            for c in self._diagnostic_profile:
                if c['key'].startswith('diag.') and c['key'].split('.')[1] in ['entropy', 'adi', 'cv2']:
                    lines.append(f"    {c['check']:<20} {c['value']}")
            
            # List all diagnostic columns
            lines.append(section_header("METRICS COMPUTED", W))
            exclude_cols = {'unique_id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'}
            metric_cols = [c for c in self.sample_df.columns if c not in exclude_cols]
            # Display in rows of 4
            for i in range(0, len(metric_cols), 4):
                row_cols = metric_cols[i:i+4]
                lines.append("  " + "  ".join(f"{c:<16}" for c in row_cols))
        
        # Flags (all modules)
        lines.append(section_header("FLAGS", W))
        if self._flags:
            lines.extend(render_checks_text(self._flags, "  "))
        elif self.blocking_issues:
            for b in self.blocking_issues:
                lines.append(f"  ✗ {b}")
        else:
            lines.append("  None ✓")
        
        # Changes (skip for 1.09 — transformation is obvious)
        if self.module != "1.09":
            lines.append(section_header("CHANGES", W))
            c = self.changes
            lines.append(f"  Rows              {c['rows']['before']:,} → {c['rows']['after']:,}  ({c['rows']['pct']:+.0f}%)")
            if 'nas' in c:
                fixed = " ✓ Fixed" if c['nas']['after'] == 0 else ""
                lines.append(f"  NAs (y)           {c['nas']['before']:,} → {c['nas']['after']:,}{fixed}")
            if 'frequency' in c:
                lines.append(f"  Frequency         {c['frequency']['before']} → {c['frequency']['after']}")
            lines.append(f"  Memory            {c['memory']['before_mb']:.1f} MB → {c['memory']['after_mb']:.1f} MB  ({c['memory']['pct']:+.0f}%)")
        
        # Footer
        lines.append("\n" + "━" * W)
        lines.append(f"Generated: {self.generated_at}")
        lines.append("━" * W)
        
        return "\n".join(lines)
    
    def _get_snapshot_cols(self) -> List[str]:
        """Get columns to show in snapshot based on module."""
        if self.module == "1.10":
            return ["unique_id", "trend", "seasonal_strength", "entropy",
                    "adi", "cv2", "structure_score", "chaos_score", "sb_quadrant"]
        elif self.module == "1.09":
            return ["unique_id", "trend", "seasonal_strength", "entropy", "adi", "cv2"]
        else:
            return ["unique_id", "ds", "y"]
    
    def display(self):
        """Print text report."""
        print(self.to_text())
    
    def to_markdown(self) -> str:
        """Render report as markdown."""
        lines = []
        
        lines.append(f"# {self.module} · {self.module_title}")
        lines.append(f"\n*Generated: {self.generated_at}*\n")
        
        # Summary
        lines.append("## Summary\n")
        for k, v in self.summary.items():
            lines.append(f"- **{k}:** {v}")
        
        # 5Q Checks or Diagnostic Profile
        if self.module in ["1.06", "1.08"]:
            lines.append("\n## 5Q Checks\n")
            if self._target:
                lines.append("### Q1 · Target\n")
                lines.extend(render_checks_markdown(self._target))
            if self._metric:
                lines.append("\n### Q2 · Metric\n")
                lines.extend(render_checks_markdown(self._metric))
            if self._structure:
                lines.append("\n### Q3 · Structure\n")
                lines.extend(render_checks_markdown(self._structure))
            if self._drivers:
                lines.append("\n### Q4 · Drivers\n")
                lines.extend(render_checks_markdown(self._drivers))
        elif self.module == "1.09":
            lines.append("\n## Diagnostic Profile\n")
            lines.extend(render_checks_markdown(self._diagnostic_profile))
        
        # Flags
        lines.append("\n## Flags\n")
        if self._flags:
            lines.extend(render_checks_markdown(self._flags))
        else:
            lines.append("None ✓")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> dict:
        """Serialize report to dictionary."""
        result = {
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
            'diagnostic_profile': self._diagnostic_profile,
            'flags': self._flags,
            'decisions_md': self._decisions_md,
            'decisions_df': self.decisions_df.to_dict('records') if self.decisions_df is not None else None,
        }
        # 1.10-specific
        if self.module == "1.10":
            result['ld6_summary'] = self._ld6_summary_data
            result['score_stats'] = self._score_stats
            result['score_validation'] = self._score_validation
            result['quadrant_dist'] = self._quadrant_dist
            result['quadrant_counts'] = self._quadrant_counts
            result['dominant_quadrant'] = self._dominant_quadrant
            result['character'] = self._character
            result['interpretations'] = self._interpretations
        return result
    
    def to_handoff(self) -> dict:
        """
        Generate handoff dictionary for downstream modules (1.10 only).
        
        Returns a dict suitable for saving as JSON for Module 1.12.
        """
        if self.module != "1.10":
            return {}
        
        return {
            'module': '1.10',
            'series_count': self.output.series,
            'score_stats': self._score_stats,
            'quadrant_distribution': self._quadrant_dist,
            'dominant_quadrant': self._dominant_quadrant,
            'character': self._character,
            'interpretations': self._interpretations,
            'thresholds': {
                'adi_intermittent': 1.32,
                'cv2_erratic': 0.49,
                'entropy_high': 0.8,
                'seasonality_strong': 0.6,
            },
            'flags': {
                'high_entropy_pct': float((self.sample_df['entropy'] > 0.8).mean()) if 'entropy' in self.sample_df.columns else None,
                'intermittent_pct': float((self.sample_df['adi'] > 1.32).mean()) if 'adi' in self.sample_df.columns else None,
                'high_cv2_pct': float((self.sample_df['cv2'] > 0.49).mean()) if 'cv2' in self.sample_df.columns else None,
                'low_structure_pct': float((self.sample_df['structure_score'] < 0.3).mean()) if 'structure_score' in self.sample_df.columns else None,
                'high_chaos_pct': float((self.sample_df['chaos_score'] > 0.6).mean()) if 'chaos_score' in self.sample_df.columns else None,
            }
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
        report._diagnostic_profile = data.get('diagnostic_profile', [])
        report._flags = data.get('flags', [])
        report._decisions_md = data.get('decisions_md')
        report.decisions_df = pd.DataFrame(data['decisions_df']) if data.get('decisions_df') else None
        report.sample_df = pd.DataFrame()
        # 1.10-specific
        report._ld6_summary_data = data.get('ld6_summary', {})
        report._score_stats = data.get('score_stats', {})
        report._score_validation = data.get('score_validation', [])
        report._quadrant_dist = data.get('quadrant_dist', {})
        report._quadrant_counts = data.get('quadrant_counts', {})
        report._dominant_quadrant = data.get('dominant_quadrant', '')
        report._character = data.get('character', '')
        report._interpretations = data.get('interpretations', [])
        return report
    
    def save(self, path: str):
        """Save report to txt file."""
        path = Path(path)
        with open(path, 'w') as f:
            f.write(self.to_text())
        print(f"✓ Report saved: {path}")
    
    def __repr__(self):
        return f"ModuleReport({self.module}, {self.output.rows:,} rows, {self.output.series:,} series)"


__all__ = ['ModuleReport', 'Snapshot']