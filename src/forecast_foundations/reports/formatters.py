"""
Formatting utilities for report rendering.

Handles table parsing, text alignment, and output generation.
"""

import pandas as pd
from typing import List, Dict, Any


# =============================================================================
# TABLE PARSING & FORMATTING
# =============================================================================

def parse_markdown_table(md: str) -> List[List[str]]:
    """Parse markdown table into rows of cells."""
    lines = [l.strip() for l in md.strip().split('\n') if l.strip()]
    rows = []
    for line in lines:
        if line.startswith('|') and not set(line.replace('|', '').strip()) <= {'-', ':'}:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            rows.append(cells)
    return rows


def format_table(rows: List[List[str]], indent: str = "  ") -> str:
    """Format parsed table with aligned columns."""
    if not rows:
        return ""
    
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


def format_decisions_df(df: pd.DataFrame, indent: str = "  ") -> str:
    """Render decisions DataFrame as aligned table."""
    col_map = {'step': 'Step', 'decision': 'Decision', 'why': 'Why', 'rev': 'Rev'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    if 'Rev' in df.columns:
        df = df.copy()
        df['Rev'] = df['Rev'].map(lambda x: '✓' if x is True else ('—' if x is None or pd.isna(x) else str(x)))
    
    cols = [c for c in ['Step', 'Decision', 'Why', 'Rev'] if c in df.columns]
    rows = [cols] + df[cols].values.tolist()
    
    return format_table(rows, indent)


def decisions_df_to_markdown(df: pd.DataFrame) -> str:
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


# =============================================================================
# CHECK RENDERING
# =============================================================================

def render_checks_text(checks: List[Dict], indent: str = "  ") -> List[str]:
    """Render list of checks as text lines."""
    if not checks:
        return [f"{indent}No checks for this section"]
    
    lines = []
    for c in checks:
        status = c.get('status', '?')
        check = c.get('check', '')
        value = c.get('value', '')
        lines.append(f"{indent}{status} {check:<22} {value}")
    return lines


def render_checks_markdown(checks: List[Dict]) -> List[str]:
    """Render list of checks as markdown lines."""
    if not checks:
        return ["*No checks for this section*"]
    
    lines = []
    for c in checks:
        status = c.get('status', '?')
        check = c.get('check', '')
        value = c.get('value', '')
        lines.append(f"- {status} **{check}:** {value}")
    return lines


# =============================================================================
# SECTION HEADERS
# =============================================================================

def section_header(title: str, width: int = 65) -> str:
    """Create a section header with underline."""
    return f"\n{title}\n{'─' * width}"


def section_header_md(title: str, level: int = 2) -> str:
    """Create a markdown section header."""
    return f"\n{'#' * level} {title}\n"


# =============================================================================
# MEMORY TIER
# =============================================================================

def get_memory_tier(mb: float) -> tuple:
    """Return (tier, note, status) for memory size."""
    if mb < 100:
        return "Small", "Fits easily in memory", "✓"
    elif mb < 1024:
        return "Medium", "Fine for most operations", "✓"
    elif mb < 10240:
        return "Large", "May need chunking for CV/grid search", "⚠"
    else:
        return "Very Large", "Consider sampling or distributed", "⚠"


__all__ = [
    'parse_markdown_table',
    'format_table', 
    'format_decisions_df',
    'decisions_df_to_markdown',
    'render_checks_text',
    'render_checks_markdown',
    'section_header',
    'section_header_md',
    'get_memory_tier',
]
