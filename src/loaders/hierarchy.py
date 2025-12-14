"""
M5 Hierarchy Utilities
======================

Functions for working with M5 hierarchy columns:
- expand_hierarchy: Convert unique_id back to item_id, dept_id, etc.
- create_unique_id: Create unique_id from hierarchy columns
"""

import pandas as pd
from typing import Optional, List

from .constants import HIERARCHY_COLS


def create_unique_id(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    separator: str = '_',
    target_col: str = 'unique_id',
    inplace: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create a unique_id column by concatenating multiple columns.

    Nixtla libraries require a 'unique_id' column to identify each time series.
    This function creates it by joining specified columns with a separator.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list of str, optional
        Columns to concatenate. If None, uses ['item_id', 'store_id'] (M5 default)
    separator : str, default='_'
        Separator between column values
    target_col : str, default='unique_id'
        Name of the output column
    inplace : bool, default=False
        If True, modify DataFrame in place
    verbose : bool, default=True
        Print summary

    Returns
    -------
    pd.DataFrame
        DataFrame with new unique_id column
    """
    if not inplace:
        df = df.copy()

    # Auto-detect columns if not specified
    if columns is None:
        if 'item_id' in df.columns and 'store_id' in df.columns:
            columns = ['item_id', 'store_id']
        else:
            exclude = {'ds', 'y', 'date', 'value', 'target', target_col}
            columns = [
                c for c in df.columns
                if c not in exclude and
                (df[c].dtype == 'object' or df[c].dtype.name == 'category')
            ]
            if not columns:
                raise ValueError(
                    "No columns specified and could not auto-detect. "
                    "Please provide columns=['col1', 'col2', ...]"
                )

    # Validate columns exist
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Create unique_id by concatenating columns
    df[target_col] = df[columns[0]].astype(str)
    for col in columns[1:]:
        df[target_col] = df[target_col] + separator + df[col].astype(str)

    if verbose:
        n_unique = df[target_col].nunique()
        print(f"Created {target_col}: {n_unique:,} unique series")
        print(f"Sample: {df[target_col].iloc[0]}")

    return df


def expand_hierarchy(
    df: pd.DataFrame,
    S_df: pd.DataFrame = None,
    id_col: str = 'unique_id',
    drop_unique_id: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Replace unique_id with original M5 hierarchy columns.
    
    The unique_id in M5 (e.g., "FOODS_1_001_CA_1") encodes the hierarchy:
    - item_id: FOODS_1_001
    - dept_id: FOODS_1
    - cat_id: FOODS
    - store_id: CA_1
    - state_id: CA
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with unique_id column
    S_df : pd.DataFrame, optional
        Static hierarchy dataframe. If provided, uses merge.
        If None, parses the unique_id string directly.
    id_col : str, default='unique_id'
        Name of the ID column
    drop_unique_id : bool, default=True
        Whether to drop the unique_id column after expansion
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        DataFrame with hierarchy columns
    """
    if id_col not in df.columns:
        if verbose:
            print(f"⚠ Column '{id_col}' not found, returning unchanged")
        return df
    
    df_result = df.copy()
    
    if S_df is not None:
        if verbose:
            print("  Expanding hierarchy via S_df merge...")
        
        hierarchy_cols_present = [c for c in HIERARCHY_COLS if c in S_df.columns]
        merge_cols = [id_col] + hierarchy_cols_present
        
        df_result = df_result.merge(
            S_df[merge_cols].drop_duplicates(),
            on=id_col,
            how='left'
        )
    else:
        if verbose:
            print("  Expanding hierarchy via vectorized ID parsing...")
        
        unique_ids = df_result[id_col].unique()
        id_mapping = pd.DataFrame({id_col: unique_ids})
        
        parts = id_mapping[id_col].str.split('_', expand=True)
        
        id_mapping['cat_id'] = parts[0]
        id_mapping['dept_id'] = parts[0] + '_' + parts[1]
        id_mapping['item_id'] = parts[0] + '_' + parts[1] + '_' + parts[2]
        id_mapping['state_id'] = parts[3]
        id_mapping['store_id'] = parts[3] + '_' + parts[4]
        
        df_result = df_result.merge(id_mapping, on=id_col, how='left')
    
    # Reorder columns
    other_cols = [c for c in df_result.columns if c not in HIERARCHY_COLS + [id_col]]
    if drop_unique_id:
        new_order = HIERARCHY_COLS + other_cols
    else:
        new_order = HIERARCHY_COLS + [id_col] + other_cols
    
    new_order = [c for c in new_order if c in df_result.columns]
    df_result = df_result[new_order]
    
    if verbose:
        print(f"  ✓ Added hierarchy columns: {HIERARCHY_COLS}")
    
    return df_result


__all__ = ['create_unique_id', 'expand_hierarchy', 'HIERARCHY_COLS']
