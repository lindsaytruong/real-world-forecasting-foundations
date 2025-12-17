"""
Reports package for Forecast Academy.

Usage:
    from reports import ModuleReport, plot_timeline_health
    
    report = ModuleReport("1.06", input_df, output_df)
    report.display()
    
    # Consistent API across all modules:
    report.summary      # dict
    report.target       # list[check] - Q1
    report.metric       # list[check] - Q2
    report.structure    # list[check] - Q3
    report.drivers      # list[check] - Q4
    report.readiness    # list[check] - final status
    report.decisions    # str
    report.changes      # dict
"""

from .core import ModuleReport, Snapshot, plot_timeline_health
from .checks import MODULE_CHECKS, MODULE_TITLES

__all__ = [
    'ModuleReport',
    'Snapshot', 
    'plot_timeline_health',
    'MODULE_CHECKS',
    'MODULE_TITLES',
]
