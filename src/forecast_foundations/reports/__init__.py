"""
Reports package for Forecast Academy.

Usage:
    from reports import ModuleReport
    
    report = ModuleReport("1.06", input_df, output_df)
    report.display()
    
    # Consistent API across all modules:
    report.summary              # dict
    report.target               # list[check] - Q1
    report.metric               # list[check] - Q2
    report.structure            # list[check] - Q3
    report.drivers              # list[check] - Q4
    report.readiness            # list[check] - Q5
    report.diagnostic_profile   # list[check] - 1.09+
    report.flags                # list[check] - consolidated flags
    report.changes              # dict
"""

from .core import ModuleReport, Snapshot
from .checks import MODULE_CHECKS, MODULE_TITLES, CHECK_CATALOG

__all__ = [
    'ModuleReport',
    'Snapshot',
    'MODULE_CHECKS',
    'MODULE_TITLES',
    'CHECK_CATALOG',
]
