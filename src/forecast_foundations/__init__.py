"""
Forecast Academy
================

Core modules for the Real-World Forecasting Foundations course.

Structure:
- src.data: M5 data loading, preprocessing, messification
- src.cache: CacheManager for data caching with lineage
- src.report: FirstContactReport for data quality assessment
- src.helpers: Utility functions (path finding, notebook detection)

Quick Start:
    from src import load_m5, CacheManager, first_contact_check
    
    cache = CacheManager(Path('data/cache'))
    df = load_m5(Path('data'), cache=cache, messify=True)
    report = first_contact_check(df, dataset_name='M5 Sales')
"""

# =============================================================================
# Data Loading
# =============================================================================
from .loaders import (
    # Loading
    load_m5,
    load_m5_with_feedback,
    load_m5_calendar,
    has_m5_cache,
    # Preprocessing
    create_unique_id,
    expand_hierarchy,
    create_subset,
    messify_m5_data,
    # Constants
    HIERARCHY_COLS,
    M5_AVAILABLE,
)

from .features import aggregate_calendar_to_weekly
from .version import __version__
# =============================================================================
# Cache Management
# =============================================================================
from .cache.cache import CacheManager, ArtifactManager, cached

# =============================================================================
# Reporting & Analysis
# =============================================================================
from .analysis.profile import (
    acf_summary,
    profile_series,
    profile_dataframe,
    summarize_profiles,
    interpret_strength,
    calc_stl_strength,
    calc_acf_metrics,
    calc_intermittency_metrics,
    calc_distribution_metrics,
    calc_volatility_metrics,
    calc_outlier_metrics,
    ADI_THRESHOLD,
    CV2_THRESHOLD,
)

# =============================================================================
# Reports
# =============================================================================
from .reports import (
    ModuleReport,
    Snapshot,
    # plot_timeline_health,
    MODULE_CHECKS,
    MODULE_TITLES,
)

# =============================================================================
# Helpers
# =============================================================================
from .utils.helpers import (
    find_project_root,
    get_notebook_name,
    get_notebook_path,
    get_module_from_notebook,
    get_artifact_subfolder,
    plot_ld6_vs_sb,
    ld6_vs_sb_summary,
)

# =============================================================================
# Plotting / Theme
# =============================================================================
from .plots import theme


__all__ = [
    # Data Loading
    'load_m5',
    'load_m5_with_feedback',
    'load_m5_calendar',
    'has_m5_cache',
    # Preprocessing
    'create_unique_id',
    'expand_hierarchy',
    'create_subset',
    'messify_m5_data',
    # Inspection
    'aggregate_calendar_to_weekly',
    # Constants
    'HIERARCHY_COLS',
    'M5_AVAILABLE',
    # Cache
    'CacheManager',
    'ArtifactManager',
    'cached',
    # Report & Analysis
    'first_contact_check',
    'FirstContactReport',
    'profile_series',
    'profile_dataframe',
    'summarize_profiles',
    'interpret_strength',
    'calc_stl_strength',
    'calc_acf_metrics',
    'calc_intermittency_metrics',
    'calc_distribution_metrics',
    'calc_volatility_metrics',
    'calc_outlier_metrics',
    'ADI_THRESHOLD',
    'CV2_THRESHOLD',
    # Reports
    'ModuleReport',
    'Snapshot',
    'plot_timeline_health',
    'MODULE_CHECKS',
    'MODULE_TITLES',
    # Helpers
    'find_project_root',
    'get_notebook_name',
    'get_notebook_path',
    'get_module_from_notebook',
    'get_artifact_subfolder',
    'plot_ld6_vs_sb',
    'ld6_vs_sb_summary',
    # Theme
    'theme',
]

# Bootstrap (notebook setup)
from .utils.bootstrap import setup_notebook, NotebookEnvironment
