"""
Helper Utilities
================

Small, reusable utility functions for path handling and notebook detection.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

from forecast_foundations.plots import theme

def find_project_root(marker_files=('.git', 'pyproject.toml')) -> Path:
    """
    Walk up from this module's location until we find a directory with marker files.
    
    Parameters
    ----------
    marker_files : tuple
        Files that indicate project root
        
    Returns
    -------
    Path
        Project root directory
        
    Raises
    ------
    FileNotFoundError
        If project root cannot be found
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise FileNotFoundError("Could not find project root")


def get_notebook_name() -> Optional[str]:
    """
    Try to detect the current Jupyter notebook name.

    Returns the notebook name without the .ipynb extension, or None if not in a notebook
    or if detection fails. Works with VS Code and JupyterLab.
    
    Returns
    -------
    str or None
        Notebook name (e.g., '1_06_first_contact') or None
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return None

        # Check for VS Code's notebook variable
        if hasattr(ipython, 'user_ns') and '__vsc_ipynb_file__' in ipython.user_ns:
            nb_path = ipython.user_ns['__vsc_ipynb_file__']
            return Path(nb_path).stem

    except Exception:
        pass

    return None


def get_module_from_notebook() -> Optional[str]:
    """
    Extract module identifier (first 4 chars) from notebook name.

    Returns
    -------
    str or None
        Module identifier (e.g., '1_06') or None

    Examples
    --------
    >>> # In notebook "1_06_first_contact.ipynb"
    >>> get_module_from_notebook()
    '1_06'
    """
    nb_name = get_notebook_name()
    return nb_name[:4] if nb_name and len(nb_name) >= 4 else None


def get_notebook_path() -> Optional[Path]:
    """
    Get the full path to the current notebook.

    Returns
    -------
    Path or None
        Full path to notebook file, or None if not in a notebook
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return None

        if hasattr(ipython, 'user_ns') and '__vsc_ipynb_file__' in ipython.user_ns:
            return Path(ipython.user_ns['__vsc_ipynb_file__'])

    except Exception:
        pass

    return None


def get_artifact_subfolder() -> Optional[str]:
    """
    Derive artifact subfolder from notebook's parent directory.

    Strips 'module_' prefix from the parent directory name.

    Returns
    -------
    str or None
        Subfolder name (e.g., '01_foundations') or None

    Examples
    --------
    >>> # In notebook at notebooks/module_01_foundations/1_06_first_contact.ipynb
    >>> get_artifact_subfolder()
    '01_foundations'
    """
    nb_path = get_notebook_path()
    if nb_path is None:
        return None

    parent_name = nb_path.parent.name
    # Strip 'module_' prefix if present
    if parent_name.startswith('module_'):
        return parent_name[7:]  # len('module_') == 7
    return parent_name

def find_contrasting_pair(scores_df, quadrant='Erratic', structure_gap=0.25,
                          adi_tolerance=0.3, cv2_tolerance=0.3):
    """Find two series in same S-B quadrant with different structure scores."""
    quad_df = scores_df[scores_df['sb_quadrant'] == quadrant].copy()

    if len(quad_df) < 100:
        # Fallback for small quadrants
        if len(quad_df) < 20:
            return None, None
        high = quad_df.nlargest(len(quad_df)//10, 'structure_score').iloc[0]
        low = quad_df.nsmallest(len(quad_df)//10, 'structure_score').iloc[0]
        return high, low

    structure_33 = quad_df['structure_score'].quantile(0.33)
    structure_67 = quad_df['structure_score'].quantile(0.67)

    high_pool = quad_df[quad_df['structure_score'] >= structure_67]
    low_pool = quad_df[quad_df['structure_score'] <= structure_33]

    best_pair = None
    best_diff = float('inf')

    high_sample = high_pool.sample(min(100, len(high_pool)), random_state=42)
    low_sample = low_pool.sample(min(100, len(low_pool)), random_state=42)

    for _, high in high_sample.iterrows():
        for _, low in low_sample.iterrows():
            if high['structure_score'] - low['structure_score'] < structure_gap:
                continue
            adi_diff = abs(high['adi'] - low['adi'])
            cv2_diff = abs(high['cv2'] - low['cv2'])
            if adi_diff <= adi_tolerance and cv2_diff <= cv2_tolerance:
                total_diff = adi_diff + cv2_diff
                if total_diff < best_diff:
                    best_diff = total_diff
                    best_pair = (high, low)

    if best_pair is None:
        high = high_pool.iloc[len(high_pool)//2]
        low = low_pool.iloc[len(low_pool)//2]
        return high, low

    return best_pair


def plot_ld6_vs_sb(weekly_df, scores_df, quadrant='Erratic'):
    """Plot high vs low structure series from same S-B quadrant."""
    high_ex, low_ex = find_contrasting_pair(scores_df, quadrant)
    if high_ex is None:
        print(f"Not enough {quadrant} series to compare")
        return None, None

    high_id = high_ex['unique_id']
    low_id = low_ex['unique_id']

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.9, 0.3], hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    fig.suptitle(f'Same S-B Quadrant ({quadrant}), Different LD6 Profiles',
                 fontsize=14, fontweight='bold')

    # Top Left: High Structure
    high_data = weekly_df[weekly_df['unique_id'] == high_id].sort_values('ds')
    ax1.plot(high_data['ds'], high_data['y'], color=theme.STRUCTURE, linewidth=1.2)
    ax1.fill_between(high_data['ds'], high_data['y'], alpha=0.1, color=theme.STRUCTURE)
    ax1.set_title("SERIES A (High Structure)", fontsize=12, color=theme.STRUCTURE, fontweight='bold')
    ax1.set_ylabel('Weekly Sales', fontsize=10)
    ax1.text(0.02, 0.98,
             f"Structure: {high_ex['structure_score']:.2f}\nChaos: {high_ex['chaos_score']:.2f}\n"
             f"─────────\nSeasonal: {high_ex['seasonal_strength']:.2f}\nEntropy: {high_ex['entropy']:.2f}\n"
             f"ADI: {high_ex['adi']:.2f}\nCV²: {high_ex['cv2']:.2f}",
             transform=ax1.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=theme.STRUCTURE, alpha=0.95))

    # Top Right: Low Structure
    low_data = weekly_df[weekly_df['unique_id'] == low_id].sort_values('ds')
    ax2.plot(low_data['ds'], low_data['y'], color=theme.CHAOS, linewidth=1.2)
    ax2.fill_between(low_data['ds'], low_data['y'], alpha=0.1, color=theme.CHAOS)
    ax2.set_title("SERIES B (Low Structure)", fontsize=12, color=theme.CHAOS, fontweight='bold')
    ax2.set_ylabel('Weekly Sales', fontsize=10)
    ax2.text(0.02, 0.98,
             f"Structure: {low_ex['structure_score']:.2f}\nChaos: {low_ex['chaos_score']:.2f}\n"
             f"─────────\nSeasonal: {low_ex['seasonal_strength']:.2f}\nEntropy: {low_ex['entropy']:.2f}\n"
             f"ADI: {low_ex['adi']:.2f}\nCV²: {low_ex['cv2']:.2f}",
             transform=ax2.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=theme.CHAOS, alpha=0.95))

    # Bottom Left: S-B View
    ax3.scatter(scores_df['adi'], scores_df['cv2'], alpha=0.15, s=8, c='gray')
    if quadrant == 'Erratic':
        ax3.axvspan(0, 1.32, ymin=0.49/2, ymax=1, alpha=0.08, color='orange')
    ax3.scatter(high_ex['adi'], high_ex['cv2'], s=250, c=theme.STRUCTURE,
                marker='*', label='Series A', zorder=5, edgecolors='white', linewidth=2)
    ax3.scatter(low_ex['adi'], low_ex['cv2'], s=250, c=theme.CHAOS,
                marker='*', label='Series B', zorder=5, edgecolors='white', linewidth=2)
    ax3.axvline(1.32, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(0.49, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.text(0.6, 0.25, 'Smooth', fontsize=9, ha='center', alpha=0.5)
    ax3.text(0.6, 1.0, 'Erratic', fontsize=9, ha='center', alpha=0.5, fontweight='bold')
    ax3.text(2.5, 0.25, 'Intermittent', fontsize=9, ha='center', alpha=0.5)
    ax3.text(2.5, 1.0, 'Lumpy', fontsize=9, ha='center', alpha=0.5)
    ax3.set_xlabel('ADI (demand interval)', fontsize=10)
    ax3.set_ylabel('CV² (variability)', fontsize=10)
    ax3.set_title('S-B View: Both in SAME Quadrant', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 2)
    ax3.annotate('', xy=(high_ex['adi'], high_ex['cv2']),
                 xytext=(low_ex['adi'], low_ex['cv2']),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5, ls='--'))

    # Bottom Right: LD6 View
    ax4.scatter(scores_df['structure_score'], scores_df['chaos_score'], alpha=0.15, s=8, c='gray')
    ax4.scatter(high_ex['structure_score'], high_ex['chaos_score'], s=250, c=theme.STRUCTURE,
                marker='*', label='Series A', zorder=5, edgecolors='white', linewidth=2)
    ax4.scatter(low_ex['structure_score'], low_ex['chaos_score'], s=250, c=theme.CHAOS,
                marker='*', label='Series B', zorder=5, edgecolors='white', linewidth=2)
    ax4.axvline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax4.text(0.75, 0.25, 'Forecastable\n(high structure,\nlow chaos)',
             fontsize=8, ha='center', va='center', alpha=0.5)
    ax4.text(0.25, 0.75, 'Difficult\n(low structure,\nhigh chaos)',
             fontsize=8, ha='center', va='center', alpha=0.5)
    ax4.set_xlabel('Structure Score', fontsize=10)
    ax4.set_ylabel('Chaos Score', fontsize=10)
    ax4.set_title('LD6 View: DIFFERENT Forecastability', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.annotate('', xy=(high_ex['structure_score'], high_ex['chaos_score']),
                 xytext=(low_ex['structure_score'], low_ex['chaos_score']),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))

    # Bottom: Interpretation
    ax5.axis('off')
    interpretation = (
        f"S-B Classification: Both series are \"{quadrant}\" (ADI <= 1.32, CV² > 0.49)\n\n"
        f"Series A: ADI={high_ex['adi']:.2f}, CV²={high_ex['cv2']:.2f}  ->  "
        f"Structure={high_ex['structure_score']:.2f}, Chaos={high_ex['chaos_score']:.2f}  ->  INVEST in model complexity\n"
        f"Series B: ADI={low_ex['adi']:.2f}, CV²={low_ex['cv2']:.2f}  ->  "
        f"Structure={low_ex['structure_score']:.2f}, Chaos={low_ex['chaos_score']:.2f}  ->  KEEP IT SIMPLE (baseline only)\n\n"
        f"S-B says: \"Same type\" - both are Erratic.\n"
        f"LD6 says: \"Different forecastability\" - A has patterns to learn, B is mostly noise."
    )
    ax5.text(0.5, 0.5, interpretation, transform=ax5.transAxes,
             fontsize=10, fontfamily='monospace', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('ld6_vs_sb_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    return high_ex, low_ex


def plot_contrasting_series(weekly_df, scores_df, quadrant='Erratic'):
    """Plot high vs low structure series from same S-B quadrant (simpler version)."""
    high_ex, low_ex = find_contrasting_pair(scores_df, quadrant)
    if high_ex is None:
        print(f"Not enough {quadrant} series to compare")
        return None, None

    high_id = high_ex['unique_id']
    low_id = low_ex['unique_id']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Same S-B Quadrant ({quadrant}), Different LD6 Profiles',
                 fontsize=14, fontweight='bold', y=1.02)

    # Top Left: High Structure
    ax1 = axes[0, 0]
    high_data = weekly_df[weekly_df['unique_id'] == high_id].sort_values('ds')
    ax1.plot(high_data['ds'], high_data['y'], color=theme.STRUCTURE, linewidth=1)
    ax1.set_title(f"HIGH STRUCTURE: {high_id[:35]}...", fontsize=11, color=theme.STRUCTURE)
    ax1.set_ylabel('Sales')
    ax1.text(0.02, 0.98,
             f"Structure: {high_ex['structure_score']:.3f}\nChaos: {high_ex['chaos_score']:.3f}\n"
             f"Seasonal: {high_ex['seasonal_strength']:.2f}\nEntropy: {high_ex['entropy']:.2f}",
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Top Right: Low Structure
    ax2 = axes[0, 1]
    low_data = weekly_df[weekly_df['unique_id'] == low_id].sort_values('ds')
    ax2.plot(low_data['ds'], low_data['y'], color=theme.CHAOS, linewidth=1)
    ax2.set_title(f"LOW STRUCTURE: {low_id[:35]}...", fontsize=11, color=theme.CHAOS)
    ax2.set_ylabel('Sales')
    ax2.text(0.02, 0.98,
             f"Structure: {low_ex['structure_score']:.3f}\nChaos: {low_ex['chaos_score']:.3f}\n"
             f"Seasonal: {low_ex['seasonal_strength']:.2f}\nEntropy: {low_ex['entropy']:.2f}",
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Bottom Left: S-B View
    ax3 = axes[1, 0]
    ax3.scatter(scores_df['adi'], scores_df['cv2'], alpha=0.1, s=5, c='gray')
    ax3.scatter(high_ex['adi'], high_ex['cv2'], s=200, c=theme.STRUCTURE,
                marker='*', label='High structure', zorder=5, edgecolors='white', linewidth=2)
    ax3.scatter(low_ex['adi'], low_ex['cv2'], s=200, c=theme.CHAOS,
                marker='*', label='Low structure', zorder=5, edgecolors='white', linewidth=2)
    ax3.axvline(1.32, color='black', linestyle='--', alpha=0.5)
    ax3.axhline(0.49, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('ADI (demand interval)')
    ax3.set_ylabel('CV² (variability)')
    ax3.set_title('S-B View: Both in SAME Quadrant', fontsize=11)
    ax3.legend(loc='upper right')
    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 2)

    # Bottom Right: LD6 View
    ax4 = axes[1, 1]
    ax4.scatter(scores_df['structure_score'], scores_df['chaos_score'], alpha=0.1, s=5, c='gray')
    ax4.scatter(high_ex['structure_score'], high_ex['chaos_score'], s=200, c=theme.STRUCTURE,
                marker='*', label='High structure', zorder=5, edgecolors='white', linewidth=2)
    ax4.scatter(low_ex['structure_score'], low_ex['chaos_score'], s=200, c=theme.CHAOS,
                marker='*', label='Low structure', zorder=5, edgecolors='white', linewidth=2)
    ax4.axvline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Structure Score')
    ax4.set_ylabel('Chaos Score')
    ax4.set_title('LD6 View: DIFFERENT Forecastability', fontsize=11)
    ax4.legend(loc='upper right')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    return high_ex, low_ex


def ld6_vs_sb_summary(high_ex, low_ex, quadrant='Erratic'):
    """Print comparison table showing LD6 vs S-B insight."""
    print(f"""
SAME S-B QUADRANT ({quadrant}), DIFFERENT LD6 PROFILES
{'='*60}

                    SERIES A          SERIES B
                  (High Structure)  (Low Structure)
{'─'*60}
S-B INPUTS
  ADI               {high_ex['adi']:.2f}              {low_ex['adi']:.2f}
  CV²               {high_ex['cv2']:.2f}              {low_ex['cv2']:.2f}
{'─'*60}
LD6 SCORES
  Structure         {high_ex['structure_score']:.2f}              {low_ex['structure_score']:.2f}
  Chaos             {high_ex['chaos_score']:.2f}              {low_ex['chaos_score']:.2f}
{'─'*60}
WHY DIFFERENT?
  Seasonality       {high_ex['seasonal_strength']:.2f}              {low_ex['seasonal_strength']:.2f}
  Entropy           {high_ex['entropy']:.2f}              {low_ex['entropy']:.2f}
{'─'*60}
RECOMMENDATION      Invest in         Use simple
                    complexity        baseline
{'='*60}
S-B alone can't distinguish A from B. LD6 can.
""")


__all__ = [
    'find_project_root',
    'get_notebook_name',
    'get_notebook_path',
    'get_module_from_notebook',
    'get_artifact_subfolder',
    'plot_ld6_vs_sb',
    'ld6_vs_sb_summary',
]
