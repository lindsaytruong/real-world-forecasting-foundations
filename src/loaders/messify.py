"""
Data Messification Utilities
============================

Functions for introducing realistic data quality issues into clean M5 data
for training and testing data quality workflows.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def messify_m5_data(
    df: pd.DataFrame,
    id_col: str = 'unique_id',
    date_col: str = 'ds',
    target_col: str = 'y',
    random_state: int = 42,
    # --- ZEROS HANDLING ---
    zeros_to_na_frac: Optional[float] = 0.15,
    zeros_drop_frac: Optional[float] = 0.02,
    zeros_drop_gaps_frac: Optional[float] = None,
    # --- DUPLICATES ---
    duplicates_add_n: Optional[int] = 150,
    # --- NA HANDLING ---
    na_drop_frac: Optional[float] = None,
    # --- DATA TYPES ---
    dtypes_corrupt: bool = True,
    # --- CACHING ---
    cache_dir: Optional[Path] = None,
    cache_tag: Optional[str] = None,
    force_refresh: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Messify clean M5 data to simulate real-world data quality issues.
    
    Introduces common data problems found in real forecasting pipelines:
    
    1. zeros_to_na_frac: Convert zeros → NAs (simulates missing reporting)
    2. zeros_drop_frac: Remove zero rows entirely (simulates sparse reporting)
    3. zeros_drop_gaps_frac: Remove zeros from MIDDLE of series (creates internal gaps)
    4. duplicates_add_n: Add duplicate rows (simulates faulty ETL/merges)
    5. na_drop_frac: Drop some NA rows (simulates partial data recovery)
    6. dtypes_corrupt: Convert dates/numbers to strings (simulates CSV round-trips)
    
    Results are cached to speed up repeated runs with the same parameters.
    
    OPTIMIZED: Uses vectorized operations instead of slow groupby-apply patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean M5 dataset from datasetsforecast
    id_col : str, default='unique_id'
        Name of the ID column
    date_col : str, default='ds'
        Name of the date column
    target_col : str, default='y'
        Name of the target column
    random_state : int, default=42
        Random seed for reproducibility
        
    zeros_to_na_frac : float, optional, default=0.15
        Fraction of zero values to convert to NA (0.0 to 1.0).
        Simulates missing data where zeros weren't reported.
        Set to None to disable.
    zeros_drop_frac : float, optional, default=0.02
        Fraction of zero-value rows to drop entirely (0.0 to 1.0).
        Simulates sparse reporting where zero-demand periods aren't recorded.
        Set to None to disable.
    zeros_drop_gaps_frac : float, optional, default=None
        Fraction of zero-value rows to drop from MIDDLE of each series only
        (never first or last row). Creates true internal gaps for testing
        gap detection. Set to None to disable.
    duplicates_add_n : int, optional, default=150
        Number of duplicate rows to add. Simulates faulty ETL or merge issues.
        Set to None to disable.
    na_drop_frac : float, optional, default=None
        Fraction of NA rows to drop (0.0 to 1.0). Applied after zeros_to_na
        creates NAs. Simulates partial data recovery efforts.
        Set to None to disable.
    dtypes_corrupt : bool, default=True
        Whether to corrupt data types by converting date and target columns
        to strings. Simulates CSV round-trips or poorly typed databases.
        
    cache_dir : Path, optional
        Directory to cache the messified data. If None, no caching.
    cache_tag : str, optional
        Optional tag to add to cache filename.
    force_refresh : bool, default=False
        If True, regenerate messified data even if cache exists.
    verbose : bool, default=True
        Whether to print summary of changes.
        
    Returns
    -------
    pd.DataFrame
        Messified version of the input data
        
    Examples
    --------
    >>> # Default messification
    >>> df_messy = messify_m5_data(df_clean)
    
    >>> # Heavy messification with internal gaps
    >>> df_messy = messify_m5_data(
    ...     df_clean,
    ...     zeros_to_na_frac=0.30,
    ...     zeros_drop_gaps_frac=0.10,
    ...     duplicates_add_n=200,
    ...     cache_dir=Path('data')
    ... )
    
    >>> # Light messification (dtype corruption only)
    >>> df_messy = messify_m5_data(
    ...     df_clean,
    ...     zeros_to_na_frac=None,
    ...     zeros_drop_frac=None,
    ...     duplicates_add_n=None,
    ...     dtypes_corrupt=True
    ... )
    """
    # Generate cache filename based on parameters
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        # Save cached data to a 'cache' subfolder within the data directory
        cache_subdir = cache_dir / 'cache'
        cache_subdir.mkdir(exist_ok=True, parents=True)

        # Create filename that reflects the messification parameters
        n_series = df[id_col].nunique()
        tag_prefix = f"{cache_tag}_" if cache_tag else ""
        cache_filename = (
            f"{tag_prefix}m5_messy_"
            f"n{n_series}_"
            f"rs{random_state}_"
            f"z2na{int(zeros_to_na_frac*100) if zeros_to_na_frac else 0}_"
            f"zdrp{int(zeros_drop_frac*100) if zeros_drop_frac else 0}_"
            f"zgap{int(zeros_drop_gaps_frac*100) if zeros_drop_gaps_frac else 0}_"
            f"dup{duplicates_add_n if duplicates_add_n else 0}_"
            f"nadrp{int(na_drop_frac*100) if na_drop_frac else 0}_"
            f"dtype{1 if dtypes_corrupt else 0}"
            f".parquet"
        )
        cache_path = cache_subdir / cache_filename
        
        # Check if cache exists and should be used
        if cache_path.exists() and not force_refresh:
            if verbose:
                print("=" * 70)
                print("LOADING CACHED MESSIFIED DATA")
                print("=" * 70)
                print(f"\n📁 Cache file: {cache_path.name}")
                print("   Using cached version (skip messification)")
                print("\n💡 To regenerate: set force_refresh=True")
            
            df_messy = pd.read_parquet(cache_path)
            
            if verbose:
                print(f"\n✓ Loaded {df_messy.shape[0]:,} rows × {df_messy.shape[1]} columns")
                print("=" * 70)
            
            return df_messy
        
        if verbose and cache_path.exists():
            print(f"\n⚠️  Cache exists but force_refresh=True, regenerating...\n")
    
    # Perform messification
    np.random.seed(random_state)
    df_messy = df.copy()
    changes_log = []
    
    # Step 1: Convert some zeros to NAs
    if zeros_to_na_frac is not None and zeros_to_na_frac > 0 and target_col in df_messy.columns:
        if verbose:
            print("Step 1/6: Converting zeros to NAs...")
        
        zero_mask = df_messy[target_col] == 0
        n_zeros = zero_mask.sum()
        n_to_convert = int(n_zeros * zeros_to_na_frac)
        
        if n_to_convert > 0:
            zero_indices = df_messy[zero_mask].index
            na_indices = np.random.choice(zero_indices, size=n_to_convert, replace=False)
            df_messy.loc[na_indices, target_col] = np.nan
            
            changes_log.append(f"Converted {n_to_convert:,} zeros to NAs ({zeros_to_na_frac*100:.0f}% of zeros)")
            if verbose:
                print(f"  ✓ Converted {n_to_convert:,} zeros to NAs")
    elif verbose:
        print("Step 1/6: Converting zeros to NAs... [SKIPPED]")
    
    # Step 2: Add duplicate rows
    if duplicates_add_n is not None and duplicates_add_n > 0:
        if verbose:
            print("Step 2/6: Adding duplicate rows...")
        
        n_duplicates = min(duplicates_add_n, len(df_messy))
        if n_duplicates > 0:
            duplicate_indices = np.random.choice(df_messy.index, size=n_duplicates, replace=False)
            duplicates = df_messy.loc[duplicate_indices].copy()
            df_messy = pd.concat([df_messy, duplicates], ignore_index=True)
            
            changes_log.append(f"Added {n_duplicates:,} duplicate rows")
            if verbose:
                print(f"  ✓ Added {n_duplicates:,} duplicate rows")
    elif verbose:
        print("Step 2/6: Adding duplicate rows... [SKIPPED]")
    
    # Step 3: Remove some zero-demand rows (sparse reporting)
    if zeros_drop_frac is not None and zeros_drop_frac > 0:
        if verbose:
            print("Step 3/6: Dropping zero-demand rows (sparse reporting)...")
        
        # Only target actual zeros (not NAs created in step 1)
        zero_mask = df_messy[target_col] == 0
        zero_indices = df_messy[zero_mask].index
        
        n_to_remove = int(len(zero_indices) * zeros_drop_frac)
        if n_to_remove > 0:
            removal_indices = np.random.choice(zero_indices, size=n_to_remove, replace=False)
            df_messy = df_messy.drop(removal_indices).reset_index(drop=True)
            
            changes_log.append(f"Dropped {n_to_remove:,} zero-demand rows ({zeros_drop_frac*100:.0f}% of zeros)")
            if verbose:
                print(f"  ✓ Dropped {n_to_remove:,} zero-demand rows")
    elif verbose:
        print("Step 3/6: Dropping zero-demand rows... [SKIPPED]")
    
    # Step 4: Drop fraction of NA rows if requested
    if na_drop_frac is not None and na_drop_frac > 0:
        if verbose:
            print("Step 4/6: Dropping NA rows...")

        na_mask = df_messy[target_col].isna()
        n_na = na_mask.sum()

        if n_na > 0:
            n_to_drop = int(n_na * na_drop_frac)
            na_indices = df_messy[na_mask].index
            drop_indices = np.random.choice(na_indices, size=n_to_drop, replace=False)
            df_messy = df_messy.drop(drop_indices).reset_index(drop=True)

            changes_log.append(f"Dropped {n_to_drop:,} NA rows ({na_drop_frac*100:.0f}% of NAs)")
            if verbose:
                print(f"  ✓ Dropped {n_to_drop:,} of {n_na:,} NA rows")
        elif verbose:
            print("  No NA rows found to drop")
    elif verbose:
        print("Step 4/6: Dropping NA rows... [SKIPPED]")

    # Step 5: Create internal gaps by dropping ZERO rows from middle of each series
    # OPTIMIZED: Uses vectorized operations instead of slow groupby-apply
    if zeros_drop_gaps_frac is not None and zeros_drop_gaps_frac > 0:
        if verbose:
            print("Step 5/6: Creating internal gaps (dropping middle zeros)...")

        n_before = len(df_messy)
        
        # Sort by id and date (required for identifying first/last per group)
        df_messy = df_messy.sort_values([id_col, date_col]).reset_index(drop=True)
        
        # Vectorized identification of first/last rows per group using shift
        is_first = df_messy[id_col] != df_messy[id_col].shift(1)
        is_last = df_messy[id_col] != df_messy[id_col].shift(-1)
        
        # Middle rows are neither first nor last AND have zero demand
        is_zero = df_messy[target_col] == 0
        is_middle_zero = ~is_first & ~is_last & is_zero
        middle_zero_indices = df_messy.index[is_middle_zero].values
        
        # Sample from middle zero indices
        n_to_drop = int(len(middle_zero_indices) * zeros_drop_gaps_frac)
        
        if n_to_drop > 0 and len(middle_zero_indices) > 0:
            drop_indices = np.random.choice(middle_zero_indices, size=n_to_drop, replace=False)
            df_messy = df_messy.drop(drop_indices).reset_index(drop=True)
        
        n_dropped = n_before - len(df_messy)

        changes_log.append(f"Created internal gaps: dropped {n_dropped:,} middle zeros ({zeros_drop_gaps_frac*100:.0f}%)")
        if verbose:
            print(f"  ✓ Dropped {n_dropped:,} zeros from middle of series")
    elif verbose:
        print("Step 5/6: Creating internal gaps... [SKIPPED]")

    # Step 6: Corrupt data types
    if dtypes_corrupt:
        if verbose:
            print("Step 6/6: Corrupting data types...")

        if date_col in df_messy.columns:
            df_messy[date_col] = df_messy[date_col].astype(str)
            changes_log.append(f"Converted {date_col} to string dtype")
            if verbose:
                print(f"  ✓ Converted {date_col} to string")

        if target_col in df_messy.columns:
            df_messy[target_col] = df_messy[target_col].astype(str)
            changes_log.append(f"Converted {target_col} to string dtype")
            if verbose:
                print(f"  ✓ Converted {target_col} to string")
    elif verbose:
        print("Step 6/6: Corrupting data types... [SKIPPED]")

    # Save to cache if requested
    if cache_dir is not None:
        if verbose:
            print(f"\n💾 Caching messified data...")
            print(f"   → {cache_path.name}")
        
        df_messy.to_parquet(cache_path, index=False)
        
        if verbose:
            cache_size_mb = cache_path.stat().st_size / 1024**2
            print(f"   ✓ Cached ({cache_size_mb:.1f} MB)")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("DATA MESSIFICATION SUMMARY")
        print("=" * 70)
        print(f"\nOriginal shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Messified shape: {df_messy.shape[0]:,} rows × {df_messy.shape[1]} columns")
        
        if changes_log:
            print(f"\nChanges applied ({len(changes_log)}):")
            for i, change in enumerate(changes_log, 1):
                print(f"  {i}. {change}")
        else:
            print("\nNo changes applied (all steps skipped)")
        
        print("\n" + "=" * 70)
        print("✓ Data successfully messified!")
        print("=" * 70)
    
    return df_messy


__all__ = ['messify_m5_data']
