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

# =============================================================================
# Cache Management
# =============================================================================
from .cache.cache import CacheManager, ArtifactManager

# =============================================================================
# Reporting
# =============================================================================
from .analysis.reports import first_contact_check, FirstContactReport

# =============================================================================
# Helpers
# =============================================================================
from .utils.helpers import (
    find_project_root,
    get_notebook_name,
    get_notebook_path,
    get_module_from_notebook,
    get_artifact_subfolder,
)


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
    # Report
    'first_contact_check',
    'FirstContactReport',
    # Helpers
    'find_project_root',
    'get_notebook_name',
    'get_notebook_path',
    'get_module_from_notebook',
    'get_artifact_subfolder',
]
