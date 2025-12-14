"""
Shared constants for M5 data loading utilities.
"""

# M5 hierarchy columns in order (from most specific to least)
HIERARCHY_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

# Check if datasetsforecast is available
try:
    from datasetsforecast.m5 import M5
    M5_AVAILABLE = True
except ImportError:
    M5_AVAILABLE = False
    M5 = None

__all__ = ['HIERARCHY_COLS', 'M5_AVAILABLE', 'M5']
