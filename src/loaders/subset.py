"""
Subset Creation
===============

Create smaller subsets of M5 data for faster iteration.
"""

import numpy as np
import pandas as pd


def create_subset(
    df: pd.DataFrame,
    n_series: int = 100,
    id_col: str = 'unique_id',
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create a random subset of series for faster processing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full M5 dataset
    n_series : int, default=100
        Number of series to sample
    id_col : str, default='unique_id'
        Name of the series ID column
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print subset information
        
    Returns
    -------
    pd.DataFrame
        Subset of the original data
    """
    np.random.seed(random_state)
    
    all_series = df[id_col].unique()
    total_series = len(all_series)
    
    if n_series > total_series:
        if verbose:
            print(f"⚠ Requested {n_series} series but only {total_series} available")
        n_series = total_series
    
    sample_series = np.random.choice(all_series, size=n_series, replace=False)
    df_subset = df[df[id_col].isin(sample_series)].copy()
    
    if verbose:
        print(f"✓ Subset: {len(df_subset):,} rows, {n_series:,} series")
        print(f"  ({len(df_subset) / len(df) * 100:.1f}% of original)")
    
    return df_subset


__all__ = ['create_subset']
