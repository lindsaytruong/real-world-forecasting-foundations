"""
M5 Data Loading
===============

Core functions for loading M5 time series data:
- load_m5: Main entry point with caching, messification, and hierarchy options
- load_m5_with_feedback: Lower-level loader with progress feedback
- load_m5_calendar: Load M5 calendar/events data
- has_m5_cache: Check if M5 data is already cached
"""

import time
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Union, TYPE_CHECKING

from .constants import HIERARCHY_COLS, M5_AVAILABLE, M5
from .hierarchy import expand_hierarchy, create_unique_id
from .subset import create_subset
from .messify import messify_m5_data
from ..utils.helpers import find_project_root, get_module_from_notebook

if TYPE_CHECKING:
    from ..cache.cache import CacheManager


def has_m5_cache(data_dir: Path) -> bool:
    """
    Quick check for M5 cache files.
    
    Parameters
    ----------
    data_dir : Path
        Directory where M5 data would be cached
        
    Returns
    -------
    bool
        True if M5 cache files are found
    """
    cache_files = ['M5.parquet', 'M5.csv', 'm5.parquet', 'm5.csv', 'm5.p', 'M5.p']
    cache_subdirs = ['cache', 'm5', 'M5', 'm5-forecasting-accuracy', 'm5/datasets', 'M5/datasets']

    if any((data_dir / f).exists() for f in cache_files):
        return True

    for subdir in cache_subdirs:
        subdir_path = data_dir / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            if any((subdir_path / f).exists() for f in cache_files):
                return True

    return False


def load_m5_calendar(data_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load M5 calendar data with date information and events.

    Parameters
    ----------
    data_dir : Path
        Directory containing M5 data
    verbose : bool, default=True
        Print loading information

    Returns
    -------
    pd.DataFrame
        Calendar DataFrame with date and event information
    """
    data_dir = Path(data_dir)

    search_paths = [
        data_dir / 'calendar.csv',
        data_dir / 'm5' / 'calendar.csv',
        data_dir / 'M5' / 'calendar.csv',
        data_dir / 'm5-forecasting-accuracy' / 'calendar.csv',
        data_dir / 'datasets' / 'calendar.csv',
        data_dir / 'm5' / 'datasets' / 'calendar.csv',
        data_dir / 'M5' / 'datasets' / 'calendar.csv',
    ]

    calendar_path = None
    for path in search_paths:
        if path.exists():
            calendar_path = path
            break

    if calendar_path is None:
        raise FileNotFoundError(
            f"calendar.csv not found. Searched:\n" +
            "\n".join(f"  - {p}" for p in search_paths[:4])
        )

    if verbose:
        print(f"Loading calendar from: {calendar_path}")

    calendar = pd.read_csv(calendar_path)

    if verbose:
        print(f"  Shape: {calendar.shape[0]:,} rows × {calendar.shape[1]} columns")

    return calendar


def load_m5_with_feedback(
    data_dir: Path,
    verbose: bool = True,
    return_additional: bool = False
) -> Tuple:
    """
    Load M5 dataset with informative feedback.
    
    Parameters
    ----------
    data_dir : Path
        Directory for M5 data cache
    verbose : bool, default=True
        Print progress messages
    return_additional : bool, default=False
        If True, returns (Y_df, X_df, S_df). If False, returns just Y_df.
        
    Returns
    -------
    pd.DataFrame or tuple
        Y_df only, or (Y_df, X_df, S_df) tuple
    """
    if not M5_AVAILABLE:
        raise ImportError(
            "datasetsforecast is required to load M5 data.\n"
            "Install it with: pip install datasetsforecast"
        )
    
    cache_exists = has_m5_cache(data_dir)
    
    if verbose:
        if cache_exists:
            print("✓ M5 cache detected. Loading from local files...")
        else:
            print("⚠ No M5 cache found. First download will take ~30-60s (~200MB)...")
    
    start_time = time.time()
    result = M5.load(directory=str(data_dir))
    load_time = time.time() - start_time
    
    df = result[0]
    
    if verbose:
        print(f"✓ Loaded in {load_time:.1f}s")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  Memory: {memory_mb:,.1f} MB")
    
    if return_additional:
        return result
    else:
        return df


def load_m5(
    data_dir: Path,
    # Caching
    cache: Optional['CacheManager'] = None,
    cache_key: str = 'm5_data',
    module: Optional[str] = None,
    force_refresh: bool = False,
    # Data source
    from_parquet: Optional[Path] = None,
    n_series: Optional[int] = None,
    random_state: int = 42,
    # Messification
    messify: bool = False,
    messify_config: Optional[dict] = None,
    # Output format
    include_hierarchy: bool = False,
    unique_id_columns: Optional[Union[bool, List[str]]] = None,
    # Other
    m5_data_dir: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load M5 time series data with optional messification and caching.
    
    Parameters
    ----------
    data_dir : Path
        Directory for data operations
    cache : CacheManager, optional
        If provided, enables caching
    cache_key : str, default='m5_data'
        Identifier for this dataset in the cache
    module : str, optional
        Module identifier. Auto-detects from notebook.
    force_refresh : bool, default=False
        Ignore cache and regenerate
    from_parquet : Path, optional
        Load from parquet instead of raw M5
    n_series : int, optional
        Subset to this many series
    random_state : int, default=42
        Random seed
    messify : bool, default=False
        Apply messification
    messify_config : dict, optional
        Override default messification parameters
    include_hierarchy : bool, default=False
        Expand unique_id to hierarchy columns
    unique_id_columns : bool or list of str, optional
        Create unique_id from hierarchy columns using create_unique_id().
        If True, uses default columns ['item_id', 'store_id'].
        If list of str, uses those columns.
        If None (default), no unique_id creation.
    m5_data_dir : Path, optional
        Directory for raw M5 data
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    pd.DataFrame
        M5 data
    """
    
    if module is None:
        module = get_module_from_notebook() or 'unknown'
    
    # Build messify config
    _messify_config = {
        'random_state': random_state,
        'zeros_to_na_frac': 0.15,
        'zeros_drop_frac': 0.02,
        'zeros_drop_gaps_frac': None,
        'duplicates_add_n': 150,
        'na_drop_frac': None,
        'dtypes_corrupt': True,
    }
    if messify_config:
        _messify_config.update(messify_config)
    
    # Full config for cache
    full_config = {
        'from_parquet': str(from_parquet) if from_parquet else None,
        'n_series': n_series,
        'random_state': random_state,
        'messify': messify,
        'include_hierarchy': include_hierarchy,
        'unique_id_columns': unique_id_columns,
        **_messify_config
    }
    
    # Check cache
    if cache is not None and not force_refresh:
        df = cache.load(cache_key, config=full_config, verbose=verbose)
        if df is not None:
            return df
        if verbose:
            print(f"🔄 Cache miss for '{cache_key}' - creating fresh...")
    
    # Load data
    if m5_data_dir is None:
        project_root = find_project_root()
        m5_data_dir = project_root / 'data'
    m5_data_dir = Path(m5_data_dir)
    m5_data_dir.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print("=" * 70)
        print(f"LOADING {'FROM PARQUET' if from_parquet else 'M5 DATA'}")
        print("=" * 70)
    
    S_df = None
    
    if from_parquet:
        from_parquet = Path(from_parquet)
        if not from_parquet.exists():
            raise FileNotFoundError(f"Parquet not found: {from_parquet}")
        df = pd.read_parquet(from_parquet)
        if verbose:
            print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        if include_hierarchy:
            Y_df, _, S_df = load_m5_with_feedback(m5_data_dir, verbose=verbose, return_additional=True)
        else:
            Y_df = load_m5_with_feedback(m5_data_dir, verbose=verbose, return_additional=False)
        df = Y_df
    
    has_hierarchy = all(c in df.columns for c in ['item_id', 'store_id'])
    
    # Subset
    if n_series is not None:
        if verbose:
            print(f"\n📊 Subsetting to {n_series} series...")
        df = create_subset(df, n_series=n_series, random_state=random_state, verbose=verbose)
    
    # Messify
    if messify:
        if verbose:
            print(f"\n🔧 Applying messification...")
        df = messify_m5_data(df, **_messify_config, verbose=verbose)
    
    # Expand hierarchy
    if include_hierarchy and not has_hierarchy:
        if verbose:
            print(f"\n🏗️ Expanding hierarchy...")
        df = expand_hierarchy(df, S_df=S_df, drop_unique_id=False, verbose=verbose)

    # Create unique_id from hierarchy columns
    if unique_id_columns is not None:
        if verbose:
            print(f"\n🔑 Creating unique_id...")
        columns = None if unique_id_columns is True else unique_id_columns
        df = create_unique_id(df, columns=columns, verbose=verbose)

    # Save to cache
    if cache is not None:
        cache.save(
            df=df,
            key=cache_key,
            config=full_config,
            module=module,
            source='raw_m5' if not from_parquet else Path(from_parquet).name
        )
    
    if verbose:
        print("\n" + "=" * 70)
        print("LOAD COMPLETE")
        print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
        print("=" * 70)
    
    return df


__all__ = [
    'load_m5',
    'load_m5_with_feedback',
    'load_m5_calendar',
    'has_m5_cache',
]
